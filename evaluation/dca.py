import argparse
import gc
import pandas as pd
import numpy as np
from dcurves import dca, plot_graphs
import os
import matplotlib.pyplot as plt
# %%
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prob_root", default="BRSET_TL_b", help="Path to predicted probabilities directory")
path = parser.parse_args().prob_root

DATASET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prob_root = os.path.join(DATASET, "output/predicted_probabilities", path)
models = [
    "retfound_d2_s",
    "retfound_d2_m",
    "retfound",
    "dinov3_large",
    "visionfm",
    "dinov2",
    "eyeclip"
]
files = [x for x in os.listdir(prob_root) if any(model in x for model in models)]
files.sort()
# print(files)
# %%

for filename in files:
    # print(f"File: {filename}")
    df = pd.read_csv(os.path.join(prob_root, filename))
    model = 'RETFound DINOv2 Shanghai' if 'retfound_d2_s' in filename else \
            'RETFound DINOv2 MEH' if 'retfound_d2_m' in filename else \
            'RETFound' if 'retfound' in filename else \
            'DINOv3 Large' if 'dinov3_large' in filename else \
            'VisionFM' if 'visionfm' in filename else \
            'DINOv2 Large' if 'dinov2' in filename else \
            'EyeCLIP' if 'eyeclip' in filename else \
            'Unknown Model'

    mode = 'Head fine-tune' if 'eval' in filename else 'Full fine-tune'
    dca_multi_df = dca(
        data=df,
        outcome="y_test",
        modelnames=["y_pred"],
        thresholds=np.arange(0, 0.5, 0.05),
    )
    os.makedirs(os.path.join(prob_root, "dca_plots"), exist_ok=True)
    plot_graphs(plot_df=dca_multi_df, y_limits=[-0.3, 0.3], graph_type="net_benefit", file_name=os.path.join(prob_root, "dca_plots", f"{model}_{mode}.png"))
    plt.close('all')
    

#%%
# Assemble DCA plots into a single figure


# arrange by model (rows) and mode (columns)
modes = ["Head fine-tune", "Full fine-tune"]
plot_dir = os.path.join(prob_root, "dca_plots")
plot_files = [f for f in os.listdir(plot_dir) if f.endswith(".png")]

# discover model names by stripping the mode suffix from filenames
models_set = set()
for fn in plot_files:
    for m in modes:
        token = f"_{m}.png"
        if token in fn:
            models_set.add(fn.replace(token, ""))
            break
models = sorted(models_set)  # adjust ordering if you want a specific order

nrows = max(1, len(models))
ncols = len(modes)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 2 * nrows))

# normalize axes to 2D array
if nrows == 1 and ncols == 1:
    axes = np.array([[axes]])
elif nrows == 1:
    axes = axes[np.newaxis, :]
elif ncols == 1:
    axes = axes[:, np.newaxis]

for i, model in enumerate(models):
    for j, mode in enumerate(modes):
        ax = axes[i, j]
        fname = os.path.join(plot_dir, f"{model}_{mode}.png")
        if os.path.exists(fname):
            img = plt.imread(fname)
            h, w = img.shape[:2]
            # central crop (keep 80% of each dimension)
            crop = img[int(0.08*h):int(0.94*h), int(0.04*w):int(0.91*w)]
            ax.imshow(crop)
            # ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center")
        ax.axis("off")
        # column headings: mode on the top row
        if i == 0:
            ax.text(0.5, 0.98, mode, transform=ax.transAxes, fontsize=8,
                ha='center', va='bottom')
        # row labels: model names on the leftmost column
        if j == 0:
            ax.text(0, 0.5, model, transform=ax.transAxes, fontsize=8,
                rotation=90, ha='right', va='center')

# reduce margins between subplots
plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0, right=1, top=0.98, bottom=0.01)
plt.savefig(os.path.join(prob_root, "summary", "DCA_Comparison.png"), dpi=300)
plt.show()
plt.close()
gc.collect()
# %%
