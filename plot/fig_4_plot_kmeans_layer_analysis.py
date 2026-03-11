import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_v_measure(filepath):
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath, header=None)
    return float(df.iloc[1, -1])


def retrieve_and_plot(paths, ax, colors, width=0.25, num_models=3):
    plot_data = {}
    layers = paths.keys()

    for layer in layers:
        v_measure = []
        for model in paths[layer]:
            file_path = paths[layer][model]
            file_path = os.path.join(file_path, "results", "results.csv")
            val = get_v_measure(file_path)
            v_measure.append(val)
        plot_data[layer] = v_measure

    x = np.arange(num_models)

    for i, layer in enumerate(layers):
        offset = (i - 1) * width
        rects = ax.bar(
            x + offset, plot_data[layer], width, label=layer, color=colors[layer]
        )
        ax.bar_label(rects, padding=3, fmt="%.2f", fontsize=9)

    ax.set_ylabel("V-measure")
    ax.set_xticks(x)
    ax.set_xticklabels(list(paths["First Layer"].keys()))
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend()


if __name__ == "__main__":
    # Plot Figure 4 in the report

    # Set the tag, can be "XPOS" or "UPOS"
    tag_to_plot = "UPOS"
    # Set save path for plots
    save_path = f"figs/embedding_layers_analysis_{tag_to_plot.lower()}.pdf"
    # Set the paths to the result directories for each method
    upos_paths = {
        "First Layer": {
            "BERT-base-uncased": "logs/kmeans_upos_bert_first_20260116_062144",
            "OPT-125M": "logs/kmeans_upos_opt_first_20260116_070519",
            "Qwen3-0.6B": "logs/kmeans_upos_qwen_first_20260116_075212",
        },
        "Middle Layer": {
            "BERT-base-uncased": "logs/kmeans_upos_bert_middle_20260116_063630",
            "OPT-125M": "logs/kmeans_upos_opt_middle_20260116_072134",
            "Qwen3-0.6B": "logs/kmeans_upos_qwen_middle_20260116_082903",
        },
        "Last Layer": {
            "BERT-base-uncased": "logs/kmeans_upos_bert_last_20260116_065109",
            "OPT-125M": "logs/kmeans_upos_opt_last_20260116_073609",
            "Qwen3-0.6B": "logs/kmeans_upos_qwen_last_20260116_090510",
        },
    }

    colors = {
        "First Layer": "#a1c9f4",
        "Middle Layer": "#4878d0",
        "Last Layer": "#133979",
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot the lines
    retrieve_and_plot(upos_paths, ax, colors)

    # Formatting
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
