import os

import pandas as pd
import matplotlib.pyplot as plt


def get_v_measure(filepath):
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath, header=None)
    return float(df.iloc[1, -1])


def retrieve_and_plot(paths, ax):
    plot_data = {}
    methods = paths.keys()
    epochs = list(range(5, 55, 5))

    for method in methods:
        folder_path = os.path.join(paths[method], "results")
        v_measures = []

        for i, _ in enumerate(epochs):
            file_path = os.path.join(folder_path, f"results.{i}.csv")
            val = get_v_measure(file_path)
            if val is None:
                file_path = os.path.join(folder_path, f"results_{i}.csv")
                val = get_v_measure(file_path)

            v_measures.append(val)

        plot_data[method] = v_measures

    for idx, method in enumerate(methods):
        ax.plot(
            epochs, plot_data[method], marker="o", ls="--", label=method, linewidth=1.5
        )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("Training Epoch")


if __name__ == "__main__":
    # Plot Figure 2 in the report

    # Set the tag, can be "XPOS" or "UPOS"
    tag_to_plot = "UPOS"
    # Set save path for plots
    save_path = f"figs/stepwise_EM_batch_sizes_{tag_to_plot.lower()}.pdf"
    # Set the paths to the result directories for each method
    # For UPOS tags
    upos_paths = {
        "32": "logs/stepwise_EM_upos_1_0_bs_32_20260115_072648",
        "64": "logs/stepwise_EM_upos_1_0_bs_64_20260115_112753",
        "128": "logs/stepwise_EM_upos_1_0_bs_128_20260115_152934",
        "256": "logs/stepwise_EM_upos_1_0_bs_256_20260115_192923",
        "512": "logs/stepwise_EM_upos_1_0_bs_512_20260115_232821",
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
    plt.legend(title="Batch Size", ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
