import os

import pandas as pd
import matplotlib.pyplot as plt


def get_v_measure(filepath):
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath, header=None)
    return float(df.iloc[1, -1])


def retrieve_and_plot(paths, ax):
    dims = paths.keys()
    v_measures = []

    for dim in dims:
        path = os.path.join(paths[dim], "results", "results.csv")
        val = get_v_measure(path)
        v_measures.append(val)

    ax.plot(dims, v_measures, marker="o", ls="--", linewidth=1.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("Word Embedding Dimension")


if __name__ == "__main__":
    # Plot Figure 5 in the report

    # Set the tag, can be "XPOS" or "UPOS"
    tag_to_plot = "UPOS"
    # Set save path for plots
    save_path = f"figs/kmeans_embedding_dim_{tag_to_plot.lower()}.pdf"
    # Set the paths to the result directories for each method
    # For UPOS tags
    upos_paths = {
        "16": "logs/kmeans_upos_qwen_embedding_dim_16_20260119_172555",
        "32": "logs/kmeans_upos_qwen_embedding_dim_32_20260119_181526",
        "64": "logs/kmeans_upos_qwen_embedding_dim_64_20260119_190429",
        "128": "logs/kmeans_upos_qwen_embedding_dim_128_20260119_195438",
        "256": "logs/kmeans_upos_qwen_embedding_dim_256_20260119_204256",
        "384": "logs/kmeans_upos_qwen_embedding_dim_384_20260119_213201",
        "512": "logs/kmeans_upos_qwen_embedding_dim_512_20260116_094611",
        "640": "logs/kmeans_upos_qwen_embedding_dim_640_20260116_102605",
        "768": "logs/kmeans_upos_qwen_embedding_dim_768_20260116_110411",
        "896": "logs/kmeans_upos_qwen_embedding_dim_896_20260116_114124",
        "1024": "logs/kmeans_upos_qwen_embedding_dim_1024_20260116_122007",
    }

    # Plot settings
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot the lines
    retrieve_and_plot(upos_paths, ax)

    # Formatting
    ax.set_ylabel("V-measure")
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
