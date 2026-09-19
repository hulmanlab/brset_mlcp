# %%
import argparse
import gc
import os
from matplotlib.patches import Patch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# %%
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metrics", type=str, default="S", choices=["S", "V"], help="Metrics to plot")
    parser.add_argument("-f", "--folder", type=str, default="mBRSET_EX_b", choices=["BRSET_TL_b", "mBRSET_EX_b", "mBRSET_TL_b"], help="Folder containing the results")
    return parser.parse_args()


# %%
def main(metrics, folder):
    DATASET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    summary_root = os.path.join(DATASET, "output/predicted_probabilities", folder, "summary")

    df_results = pd.read_csv(os.path.join(summary_root, f'cm.csv'))
    df_recalib = pd.read_csv(os.path.join(summary_root, f'cm_recalib.csv'))

    df_results["calibration"] = "before"
    df_recalib["calibration"] = "after"
    df = pd.concat([df_results, df_recalib], ignore_index=True)

    colors = {"Head fine-tune": "#E66100", "Full fine-tune": "#5D3A9B",}

    markers = {"before": "o", "after": "^",}

    # Small horizontal offsets so the 4 points at each threshold don't overlap
    offsets = {
        ("Head fine-tune", "before"): -0.023,
        ("Head fine-tune", "after"): -0.008,
        ("Full fine-tune", "before"): 0.008,
        ("Full fine-tune", "after"): 0.023,
    }

    models = ["DINOv3", "RETFound", "VisionFM"]
    metrics = ["Sen", "Spe"] if metrics == "S" else ["PPV", "NPV"]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(7, 8),
        sharex=True,
        sharey=True,
    )

    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            df_model = df[df["model"].str.contains(model)]
            for _, row in df_model.iterrows():
                x = row["threshold"] + offsets[(row["mode"], row["calibration"])]
                
                ax.errorbar(
                    x,
                    row[f'{metric}_mean'],
                    yerr=[[row[f"{metric}_mean"] - row[f"{metric}_lower"]], [row[f"{metric}_upper"] - row[f"{metric}_mean"]]],
                    color=colors[row["mode"]],
                    marker=markers[row["calibration"]],
                    markersize=5,
                    capsize=5,
                    ecolor='gray'
                )
            metric_title = "Sensitivity" if metric == "Sen" else "Specificity" if metric == "Spe" else metric 
            ax.set_title(metric_title if i == 0 else "", size=10)
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_ylabel(model if j == 0 else "", rotation=90, va="center", labelpad=10)
            ax.set_xlabel("Threshold" if i == 2 else "")
            ax.set_xticks(sorted(df["threshold"].unique()))

    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="darkgray",
            markeredgecolor="none",
            markersize=8,
            label="Before recalibration"
        ),
        Line2D(
            [0], [0],
            marker="^",
            color="none",
            markerfacecolor="darkgray",
            markeredgecolor="none",
            markersize=8,
            label="After recalibration"
        ),
        Patch(
            color="#E66100",
            label="Head fine-tune"
        ),
        Patch(
            color="#5D3A9B",
            label="Full fine-tune"
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        frameon=False,
    )

    fig.subplots_adjust(right=0.80)

    plt.savefig(os.path.join(summary_root, f"{metrics[0]}_{metrics[1]}_plot.png"), dpi=300, bbox_inches='tight', pad_inches=0.02)

    plt.tight_layout()
    plt.show()

    gc.collect()

#%%
if __name__ == "__main__":
    args = get_args()
    main(args.metrics, args.folder)

