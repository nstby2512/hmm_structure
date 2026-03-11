import argparse
import os


def arg_parsing():
    argparser = argparse.ArgumentParser()
    ACTIONS = ["train-test", "test"]
    MODELS = ["hmm-mle", "hmm-EM", "hmm-sEM", "hmm-hardEM", "nhmm", "kmeans"]
    TAGS = ["upos", "xpos"]
    DATASETS = ["ptb-trains", "cds-mature"]
    argparser.add_argument(
        "action",
        choices=ACTIONS,
        help=f"action to perform. One of {'(' + '|'.join(ACTIONS) + ')'}",
        metavar="ACTION",
    )
    argparser.add_argument(
        "model",
        choices=MODELS,
        help=f"model and method to run. One of {'(' + '|'.join(MODELS) + ')'}",
        metavar="MODEL",
    )
    argparser.add_argument(
        "tag",
        choices=TAGS,
        help=f"part-of-speech tag to use. One of {'(' + '|'.join(TAGS) + ')'}",
        metavar="TAG",
    )
    argparser.add_argument(
        "dataset",
        choices=DATASETS,
        help=f"dataset to use. One of {'(' + '|'.join(DATASETS) + ')'}",
        metavar="DATASET"
    )
    argparser.add_argument(
        "--subset",
        dest="subset",
        type=int,
        default=None,
        help="number of data rows to use. Default using all data",
        metavar="INT",
    )
    argparser.add_argument(
        "--max-epochs",
        dest="max_epochs",
        nargs="+",
        type=int,
        default=[50],
        help="""maximum training iterations; for HMM, provide two numbers if using validation
        for the number of validation and validation iteration interval. Default to 50""",
        metavar="INT",
    )
    argparser.add_argument(
        "--load-path",
        dest="load_path",
        type=_valid_dir_or_file_path,
        default=None,
        help="path to load the model checkpoint from for testing.",
        metavar="PATH",
    )
    argparser.add_argument(
        "--experiment-name",
        dest="experiment_name",
        default=None,
        help="Experiment name",
    )
    argparser.add_argument(
        "--word-embedding-path",
        dest="word_embedding_path",
        default=None,
        help="path to load or save the BERT outputs for K-means method, use `.pt`.",
        metavar="PATH",
    )
    argparser.add_argument(
        "--sEM-alpha",
        dest="sEM_alpha",
        type=float,
        default=None,
        help="alpha for sEM algorithm",
    )
    argparser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=256,
        help="Batch size for sEM or neural HMM",
    )
    argparser.add_argument(
        "--nhmm-num-inner-loop-updates",
        dest="nhmm_num_inner_loop_updates",
        type=int,
        default=6,
        help="Number of inner loop updates for neural HMM",
    )
    argparser.add_argument(
        "--nhmm-hidden-dim",
        dest="nhmm_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension size for neural HMM",
    )
    argparser.add_argument(
        "--model-for-embeddings",
        dest="model_for_embeddings",
        type=str,
        default="bert-base-uncased",
        choices=[
            "bert-base-uncased",
            "facebook/opt-125m",
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-0.6B",
        ],
        help="Huggingface model name for generating word embeddings",
    )
    argparser.add_argument(
        "--layer-for-embeddings",
        dest="layer_for_embeddings",
        type=str,
        default="last",
        choices=["first", "last", "middle"],
        help="Which layer to extract embeddings from",
    )
    argparser.add_argument(
        "--word-embedding-dim",
        dest="word_embedding_dim",
        type=int,
        default=None,
        help="Number of dimensions for word embeddings. Full dimensions by default.",
    )
    args = argparser.parse_args()
    args = vars(args)
    return args


def _valid_dir_or_file_path(path: str):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("file or directory not found")
    return os.path.abspath(path)
