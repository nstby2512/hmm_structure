import csv
import logging
import os
import pickle
from typing import Optional, Tuple, Dict, List

import numpy as np
from sklearn.cluster import k_means
import torch
import torch.nn.functional as F
from transformers import (
    BertTokenizerFast,
    BertModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import DatasetDict
from tqdm import tqdm

from utils.preprocess_dataset import *
from utils.utils import calculate_v_measure, calculate_variation_of_information

logger = logging.getLogger()


class KmeansPipeline:
    def __init__(
        self,
        tag_name: str,
        max_epochs: int,
        save_path: str,
        subset: Optional[int] = None,
        word_embedding_path: Optional[str] = None,
        model_for_embeddings: str = "bert-base-uncased",
        layer_for_embeddings: str = "last",
        embedding_dim: Optional[int] = None,
    ) -> None:
        """K-means clustering pipeline for POS tagging.

        Args:
            tag_name: The name of the POS tag set to use (e.g., "upos" or "xpos").
            subset: The subset of the dataset to use.
            max_epochs: The maximum number of epochs to run the clustering.
            save_path: The path to save checkpoints and results.
            word_embedding_path: Optional path to precomputed word embeddings.
            model_for_embeddings: Huggingface model name for generating word embeddings.
            layer_for_embeddings: Which layer to extract embeddings from.
            embedding_dim: The dimension of the embeddings.
        """
        assert save_path is not None, "save_path should not be None"
        self.save_path = save_path
        self.checkpoint_path = os.path.join(self.save_path, "checkpoints")
        self.res_path = os.path.join(self.save_path, "results")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.res_path, exist_ok=True)
        self.max_epochs = max_epochs
        self.tag_mapping, obs_mapping, dataset_splits = self._load_dataset(
            tag_name, subset
        )
        self.dataset_splits = dataset_splits
        self.all_embeddings = []

        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if word_embedding_path is None:
            logger.info(
                "Word embedding path is not provided. Generating word embeddings..."
            )
            embedding_save_path = os.path.join(save_path, "word_embedding.pkl")

            # Create model and tokenizer
            if model_for_embeddings == "bert-base-uncased":
                self.tokenizer = BertTokenizerFast.from_pretrained(model_for_embeddings)
                self.model = BertModel.from_pretrained(model_for_embeddings).to(device)
            elif model_for_embeddings == "facebook/opt-125m":
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_for_embeddings, add_prefix_space=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_for_embeddings
                ).to(device)
            elif model_for_embeddings in [
                "Qwen/Qwen3-Embedding-0.6B",
                "Qwen/Qwen3-0.6B",
            ]:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_for_embeddings, use_fast=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_for_embeddings
                ).to(device)
            else:
                raise ValueError(
                    f"Unsupported model for embeddings: {model_for_embeddings}"
                )

            # Generate embeddings
            with torch.no_grad():
                for data_sample in tqdm(dataset_splits["train"]):
                    sentence = data_sample["form"]
                    inputs = self.tokenizer(
                        sentence, return_tensors="pt", is_split_into_words=True
                    ).to(device)
                    outputs = self.model(**inputs, output_hidden_states=True)

                    # Extract embeddings
                    if layer_for_embeddings == "first":
                        embeddings = outputs.hidden_states[0][0]
                    elif layer_for_embeddings == "last":
                        embeddings = outputs.hidden_states[-1][0]
                    elif layer_for_embeddings == "middle":
                        layer_idx = len(outputs.hidden_states) // 2
                        embeddings = outputs.hidden_states[layer_idx][0]

                    if embedding_dim is not None:
                        embeddings = embeddings[:, :embedding_dim]

                    # Convert token embeddings to word embeddings
                    word_token_mapping = {
                        i: [] for i in range(len(data_sample["form"]))
                    }
                    for token_idx, word_idx in enumerate(inputs.word_ids()):
                        if word_idx is not None:
                            word_token_mapping[word_idx].append(embeddings[token_idx])
                    word_embeddings = []

                    # For each word, average its token embeddings to get the word embedding
                    for idx in range(len(data_sample["form"])):
                        v = torch.stack(word_token_mapping[idx])
                        mean_v = torch.mean(v, dim=0)
                        # Normalize the embedding
                        mean_v = F.normalize(mean_v, p=2, dim=0)
                        word_embeddings.append(mean_v)
                    self.all_embeddings.append(
                        torch.stack(word_embeddings).cpu().numpy()
                    )
                with open(embedding_save_path, "wb") as f:
                    pickle.dump(self.all_embeddings, f)
            logger.info(f"Embeddings are saved to {embedding_save_path}")

        else:
            logger.info(f"Loading word embeddings from {word_embedding_path}...")
            with open(word_embedding_path, "rb") as f:
                self.all_embeddings = pickle.load(f)
            logger.info(f"Embeddings loaded.")

        self.lengths = [s.shape[0] for s in self.all_embeddings]
        self.all_embeddings = np.concatenate(self.all_embeddings, axis=0)

    def _load_dataset(
        self, tag_name: str, subset: Optional[int] = None
    ) -> Tuple[Dict, Dict, DatasetDict]:
        logger.warning(f"Using {tag_name} as tag")
        # Load and wrap PTB dataset
        sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
        dataset = wrap_dataset(sentences)

        tag_mapping = {
            "upos": create_tag_mapping(upos_set),
            "xpos": create_tag_mapping(xpos_set),
        }[tag_name]
        obs_mapping = create_obs_mapping(sentences)

        def map_tag_and_token(examples):
            input_ids = []
            for token in examples["form"]:
                input_ids.append(obs_mapping[token])
            examples["input_ids"] = input_ids
            # Using UPoS as tags
            tags = []
            for tag in examples[tag_name]:
                tags.append(tag_mapping[tag])
            examples["tags"] = tags
            return examples

        dataset = dataset.map(map_tag_and_token, desc="Mapping tokens and tags")

        dataset_splits = DatasetDict({"train": dataset, "test": dataset})

        return tag_mapping, obs_mapping, dataset_splits

    def _run(self, max_iter: int) -> Tuple[List, np.ndarray]:
        """Run K-means clustering.

        Args:
            max_iter: Maximum number of iterations for K-means.

        Returns:
            pred_tags: List of predicted tags for each sentence.
            centroid: Centroids of the clusters.
        """
        # Run sklearn K-means
        centroid, label, _, best_n_iter = k_means(
            self.all_embeddings,
            n_clusters=len(self.tag_mapping),
            max_iter=max_iter,
            return_n_iter=True,
        )

        # Split labels back into sentences
        pred_tags = []
        prev = 0
        for l in self.lengths:
            pred_tags.append(label[prev : prev + l])
            prev += l
        logger.info(f"Best_n_iter: {best_n_iter}")

        return pred_tags, centroid

    def run_kmeans(self) -> None:
        """Run the K-means clustering pipeline."""
        logger.info("Running K-means...")

        # Outer loop for multiple epochs
        if len(self.max_epochs) > 1:
            for i in tqdm(range(self.max_epochs[0]), "Outer train loop"):
                pred_tags, centroid = self._run(self.max_epochs[1] * (i + 1))

                t_path = os.path.join(self.checkpoint_path, f"checkpoint_{i}.npy")
                logger.info(f"Saving K-means centroid to {t_path}")
                np.save(t_path, centroid)

                # Evaluation
                t_path = os.path.join(self.res_path, f"results_{i}.csv")
                with torch.no_grad():
                    self.eval(pred_tags, t_path, self.dataset_splits["test"])
        # Single run
        else:
            pred_tags, centroid = self._run(self.max_epochs[0])
            t_path = os.path.join(self.checkpoint_path, "checkpoint.npy")
            logger.info(f"Saving K-means centroid to {t_path}")
            np.save(t_path, centroid)

            # Evaluation
            t_path = os.path.join(self.res_path, "results.csv")
            with torch.no_grad():
                self.eval(pred_tags, t_path, self.dataset_splits["test"])

    def eval(
        self,
        labels,
        res_path,
        dataset_split,
    ):
        # Evaluate
        num_samples = len(dataset_split)
        results = []
        homo_sum = 0.0
        comp_sum = 0.0
        v_score_sum = 0.0
        vi_sum = 0.0
        normalized_vi_sum = 0.0
        true_labels = torch.tensor([])
        pred_labels = torch.tensor([])
        for i, example in enumerate(
            tqdm(dataset_split, "Evaluating K-means...", num_samples)
        ):
            forms = example["form"]
            true_tags = example["tags"]
            pred_tags = labels[i]

            sentence = " ".join(forms)

            # Compute per-example V-measure and VI
            homo_score, comp_score, v_score = calculate_v_measure(true_tags, pred_tags)
            vi, normalized_vi = calculate_variation_of_information(true_tags, pred_tags)

            homo_sum += homo_score
            comp_sum += comp_score
            v_score_sum += v_score
            vi_sum += vi
            normalized_vi_sum += normalized_vi
            results.append(
                [i + 1, sentence, vi, normalized_vi, homo_score, comp_score, v_score]
            )

            # Record true and predicted labels for computing whole-dataset V-measure and VI
            true_labels = torch.hstack([true_labels, torch.tensor(true_tags)])
            pred_labels = torch.hstack([pred_labels, torch.tensor(pred_tags)])

        # Compute whole-dataset V-measure and VI
        logger.info("Computing whole-dataset V-measure")
        homo_score_whole, comp_score_whole, v_score_whole = calculate_v_measure(
            true_labels.tolist(), pred_labels.tolist()
        )
        logger.info("Computing whole-dataset VI")
        vi_whole, normalized_vi_whole = calculate_variation_of_information(
            true_labels.tolist(), pred_labels.tolist()
        )

        logger.info(
            f"| Homogeneity score: {homo_score_whole}\n"
            f"| Completeness score: {comp_score_whole}\n"
            f"| V-measure: {v_score_whole}\n"
            f"| Variation of information: {vi_whole}\n"
            f"| Normalized VI: {normalized_vi_whole}\n"
        )

        # Save results to CSV
        logger.info(f"Saving results to {res_path}")
        with open(res_path, "w+", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "sentence",
                    "VI",
                    "normalized-VI",
                    "homogeneity",
                    "completeness",
                    "V-score",
                ]
            )
            # Save whole-dataset results as id==0
            writer.writerow(
                [
                    0,
                    "-",
                    vi_whole,
                    normalized_vi_whole,
                    homo_score_whole,
                    comp_score_whole,
                    v_score_whole,
                ]
            )
            # Save per-example results
            writer.writerows(results)
