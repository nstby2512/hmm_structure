import csv
import logging
import os
from typing import List, Union, Optional

import torch
from datasets import DatasetDict
from tqdm import tqdm

from pos_tagging.hmm import HMMClassifier
from pos_tagging.nhmm import NeuralHMMClassifier
from utils.preprocess_dataset import *
from utils.utils import calculate_v_measure, calculate_variation_of_information

logger = logging.getLogger()


def train_hmm(
    method: str,
    dataset_splits: DatasetDict,
    max_epochs: int,
    num_states: int,
    num_obs: int,
    save_path: str,
    nhmm_hidden_dim: int = 128,
    sEM_alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
    nhmm_num_inner_loop_updates: Optional[int] = None,
):
    logger.info("Training HMM")
    if method == "nhmm":
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert batch_size is not None, "Batch size is required for neural HMM"
        assert (
            nhmm_num_inner_loop_updates is not None
        ), "Number of inner loop updates is required for neural HMM"
        hmm = NeuralHMMClassifier(
            num_states=num_states,
            hidden_dim=nhmm_hidden_dim,
            num_obs=num_obs,
            device=device,
            batch_size=batch_size,
            num_inner_loop_updates=nhmm_num_inner_loop_updates,
        )
    else:
        hmm = HMMClassifier(
            num_states=num_states,
            num_obs=num_obs,
            sEM_alpha=sEM_alpha,
            sEM_batch_size=batch_size,
        )

    # Training
    hmm.train(inputs=dataset_splits["train"], epochs=max_epochs, method=method)

    # Save HMM parameters
    checkpoint_path = os.path.join(save_path, "checkpoints", "checkpoint.pt")
    logger.info(f"Saving HMM model to {checkpoint_path}")
    torch.save(hmm, checkpoint_path)

    logger.info("HMM training done")

    return hmm


def train_hmm_stage(
    method: str,
    dataset_splits: DatasetDict,
    max_epochs: List[int],
    num_states: int,
    num_obs: int,
    save_path: str,
    nhmm_hidden_dim: int = 128,
    sEM_alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
    nhmm_num_inner_loop_updates: Optional[int] = None,
):
    logger.info("Training HMM by stages")
    if method == "nhmm":
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert batch_size is not None, "Batch size is required for neural HMM"
        assert (
            nhmm_num_inner_loop_updates is not None
        ), "Number of inner loop updates is required for neural HMM"
        hmm = NeuralHMMClassifier(
            num_states=num_states,
            num_obs=num_obs,
            hidden_dim=nhmm_hidden_dim,
            device=device,
            batch_size=batch_size,
            num_inner_loop_updates=nhmm_num_inner_loop_updates,
        )
    else:
        hmm = HMMClassifier(
            num_states=num_states,
            num_obs=num_obs,
            sEM_alpha=sEM_alpha,
            sEM_batch_size=batch_size,
        )

    # Training
    f = False  # Continue training flag
    N = max_epochs[0]  # Number of stages (outer loop)
    checkpoint_path = os.path.join(save_path, "checkpoints")
    res_path = os.path.join(save_path, "results")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(res_path, exist_ok=True)

    for i in tqdm(range(N), "Outer train loop", N):
        hmm.train(
            inputs=dataset_splits["train"],
            epochs=max_epochs[1],
            method=method,
            continue_training=f,
        )
        f = True  # Set continue training to True after the first iteration

        # Save HMM parameters
        t_path = os.path.join(checkpoint_path, f"checkpoint_{i}.pt")
        logger.info(f"Saving HMM model to {t_path}")
        torch.save(hmm, t_path)

        t_path = os.path.join(res_path, f"results_{i}.csv")
        with torch.no_grad():
            eval_hmm(
                dataset_splits["test"].select(
                    range(round(len(dataset_splits["test"]) * 0.05))
                ),
                res_path=t_path,
                hmm=hmm,
                is_neural=(method == "nhmm"),
            )

    logger.info("HMM training done")

    return hmm


def eval_hmm(
    dataset_split: Dataset,
    res_path: str,
    hmm: Optional[Union[HMMClassifier, NeuralHMMClassifier]] = None,
    load_path: str = None,
    is_neural: bool = False,
):
    if hmm is None:
        if load_path is None:
            raise ValueError(
                "At least one of HMM model and load_path should be provided"
            )
        # Load HMM parameters
        logger.info(f"Loading HMM model from {load_path}")
        hmm: Union[HMMClassifier, NeuralHMMClassifier] = torch.load(
            load_path, weights_only=False
        )

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
    if is_neural:
        hmm.get_probabilities_for_eval()

    for i, example in enumerate(tqdm(dataset_split, "HMM testing", num_samples)):
        input_ids = example["input_ids"]
        forms = example["form"]
        true_tags = example["tags"]
        pred_tags = hmm.inference(input_ids)

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


def train_and_test(
    method: str,
    dataset: str,
    tag_name: str,
    max_epochs: List[int],
    save_path: str,
    nhmm_hidden_dim: int = 128,
    subset: Optional[int] = None,
    load_path: Optional[str] = None,
    sEM_alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
    nhmm_num_inner_loop_updates: Optional[int] = None,
):
    assert len(max_epochs) <= 2
    logger.info(f"Using {tag_name} as tag")

    # 增加选择数据集
    logger.info(f"Using {dataset} as datasets")
    if dataset == 'ptb-trains':
        # Load and wrap dataset
        sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
        dataset = wrap_dataset(sentences)
        tag_mapping = {
        "upos": create_tag_mapping(upos_set),
        "xpos": create_tag_mapping(xpos_set),
        }[tag_name]
        obs_mapping = create_obs_mapping(sentences)
    else:
        sentences, tag_set = load_shrg_dataset(line_num=subset)
        dataset = wrap_dataset(sentences)
        tag_mapping = create_tag_mapping(tag_set)
        obs_mapping = create_obs_mapping(sentences)



    def map_tag_and_token(examples):
        input_ids = []
        for token in examples["form"]:
            input_ids.append(obs_mapping[token])
        examples["input_ids"] = input_ids
        # Using UPoS as tags
        # 临时修改了examples里用“mixrule”
        tags = []
        for tag in examples[tag_name]:
            tags.append(tag_mapping[tag])
        examples["tags"] = tags
        return examples

    dataset = dataset.map(map_tag_and_token, desc="Mapping tokens and tags")

    # Unsupervised learning, so train and test on the same dataset
    dataset_splits = DatasetDict({"train": dataset, "test": dataset})

    if len(max_epochs) == 1:
        hmm = train_hmm(
            method=method,
            dataset_splits=dataset_splits,
            max_epochs=max_epochs[0],
            num_states=len(tag_mapping),
            num_obs=len(obs_mapping),
            save_path=save_path,
            nhmm_hidden_dim=nhmm_hidden_dim,
            sEM_alpha=sEM_alpha,
            batch_size=batch_size,
            nhmm_num_inner_loop_updates=nhmm_num_inner_loop_updates,
        )
    else:
        hmm = train_hmm_stage(
            method=method,
            dataset_splits=dataset_splits,
            max_epochs=max_epochs,
            num_states=len(tag_mapping),
            num_obs=len(obs_mapping),
            save_path=save_path,
            nhmm_hidden_dim=nhmm_hidden_dim,
            sEM_alpha=sEM_alpha,
            batch_size=batch_size,
            nhmm_num_inner_loop_updates=nhmm_num_inner_loop_updates,
        )

    res_path = os.path.join(save_path, "results", "final_results.csv")
    with torch.no_grad():
        eval_hmm(
            dataset_splits["test"],
            res_path,
            hmm,
            load_path=load_path,
            is_neural=(method == "nhmm"),
        )


def test(
    method: str,
    dataset: str, 
    tag_name: str,
    save_path: str,
    subset: Optional[int] = None,
    load_path: Optional[str] = None,
):
    logger.warning(f"Using {tag_name} as tag")

    # 增加选择数据集
    logger.info(f"Using {dataset} as datasets")
    if dataset == 'ptb-trains':
        # Load and wrap dataset
        sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
        dataset = wrap_dataset(sentences)
        tag_mapping = {
        "upos": create_tag_mapping(upos_set),
        "xpos": create_tag_mapping(xpos_set),
        }[tag_name]
        obs_mapping = create_obs_mapping(sentences)
    else:
        sentences, tag_set = load_shrg_dataset(line_num=subset)
        dataset = wrap_dataset(sentences)
        tag_mapping = create_tag_mapping(tag_set)
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

    res_path = os.path.join(save_path, "results", "final_results.csv")
    with torch.no_grad():
        eval_hmm(
            dataset_splits["test"],
            res_path,
            load_path=load_path,
            is_neural=(method == "nhmm"),
        )
