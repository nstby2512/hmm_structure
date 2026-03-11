import json
import logging
import os
import random
import warnings
from datetime import datetime

import torch

from pos_tagging import hmm_pipeline
from pos_tagging.kmeans import KmeansPipeline
from utils.argparser import arg_parsing
from utils.logging_nlp import set_logging_verbosity, setup_log_file


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    set_logging_verbosity("info")
    args = arg_parsing()

    # Suppress warnings in evaluation
    warnings.filterwarnings(
        "ignore", message=".*number of unique classes is greater than 50%.*"
    )
    warnings.filterwarnings(
        "ignore", message=".*invalid value encountered in scalar divide.*"
    )

    # Create the experiment directory
    assert (
        args["experiment_name"] is not None
    ), "Please provide an experiment name using --experiment-name"
    save_path = os.path.join(
        "./logs",
        args["experiment_name"] + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(save_path, exist_ok=True)
    checkpoint_path = os.path.join(save_path, "checkpoints")
    res_path = os.path.join(save_path, "results")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(res_path, exist_ok=True)

    # Set up logging
    log_path = os.path.join(save_path, "log.log")
    setup_log_file(log_path)
    logger = logging.getLogger()
    logger.info(f"Logging setup complete. Logs will be saved to {log_path}.")

    # Save the configuration
    with open(f"{save_path}/configs.json", "w") as f:
        json.dump(args, f)

    if args["model"] == "kmeans":
        kmeans_pipeline = KmeansPipeline(
            tag_name=args["tag"],
            max_epochs=args["max_epochs"],
            save_path=save_path,
            subset=args["subset"],
            word_embedding_path=args["word_embedding_path"],
            model_for_embeddings=args["model_for_embeddings"],
            layer_for_embeddings=args["layer_for_embeddings"],
            embedding_dim=args["word_embedding_dim"],
        )
        kmeans_pipeline.run_kmeans()
    else:
        method = args["model"].split("-")[-1]
        if args["action"] == "train-test":
            hmm_pipeline.train_and_test(
                method=method,
                dataset=args["dataset"],
                tag_name=args["tag"],
                max_epochs=args["max_epochs"],
                save_path=save_path,
                nhmm_hidden_dim=args["nhmm_hidden_dim"],
                subset=args["subset"],
                load_path=args["load_path"],
                sEM_alpha=args["sEM_alpha"],
                batch_size=args["batch_size"],
                nhmm_num_inner_loop_updates=args["nhmm_num_inner_loop_updates"],
            )
        else:
            hmm_pipeline.test(
                method=method,
                tag_name=args["tag"],
                save_path=save_path,
                subset=args["subset"],
                load_path=args["load_path"],
            )
