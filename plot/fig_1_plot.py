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

    for _, method in enumerate(methods):
        ax.plot(
            epochs, plot_data[method], marker="o", ls="--", label=method, linewidth=1.5
        )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("Training Epoch")


if __name__ == "__main__":
    # Plot Figure 1 in the report

    # Set the tag, can be "XPOS" or "UPOS"
    tag_to_plot = "XPOS"
    # Set save path for plots
    save_path = f"figs/training_curves.pdf"
    # Set the paths to the result directories for each method
    # For UPOS tags
    upos_paths = {
        "Standard EM": "logs/standardEM_upos_20260113_123818",
        "Hard EM": "logs/hard_EM_upos_20260113_164150",
        "Stepwise EM (α = 0.6)": "logs/stepwise_EM_upos_0_6_20260113_184016",
        "Stepwise EM (α = 0.8)": "logs/stepwise_EM_upos_0_8_20260113_223839",
        "Stepwise EM (α = 1.0)": "logs/stepwise_EM_upos_1_0_20260114_024004",
        "Neural HMM": "logs/neural_hmm_upos_20260114_063848",
    }
    # For XPOS tags
    xpos_paths = {
        "Standard EM": "logs/standardEM_xpos_20260114_083340",
        "Hard EM": "logs/hard_EM_xpos_20260114_130027",
        "Stepwise EM (α = 0.6)": "logs/stepwise_EM_xpos_0_6_20260114_160152",
        "Stepwise EM (α = 0.8)": "logs/stepwise_EM_xpos_0_8_20260114_202824",
        "Stepwise EM (α = 1.0)": "logs/stepwise_EM_xpos_1_0_20260115_005337",
        "Neural HMM": "logs/neural_hmm_xpos_20260115_051848",
    }

    # Plot settings
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Plot the lines
    retrieve_and_plot(upos_paths, ax1)
    retrieve_and_plot(xpos_paths, ax2)

    # Formatting
    ax1.set_ylabel("V-measure")
    ax2.set_ylabel("")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, format="pdf")
