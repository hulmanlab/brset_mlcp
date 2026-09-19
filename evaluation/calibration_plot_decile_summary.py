#%%
# Assemble calibration plot decile into a single figure
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import gc

#%%

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prob_root", default="BRSET_TL_b", help="Path to predicted probabilities directory")
path = parser.parse_args().prob_root
# path = 'mBRSET_EX_b'

DATASET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prob_root = os.path.join(DATASET, "output/predicted_probabilities", path)

#%%

# arrange by model (rows) and mode (columns)
modes = ["Head Fine-tune", "Full Fine-tune"]
plot_dir = os.path.join(prob_root, "calibration_plots_decile")
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
models = ['DINOv3', 'RETFound', 'VisionFM']
nrows = max(1, len(models))
ncols = len(modes)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 2 * nrows))

legend_elements = [
    Patch(facecolor='lightgray', edgecolor='none', label='Predicted'),
    Patch(facecolor='black', edgecolor='none', label='Observed')
]

fig.legend(handles=legend_elements,
           loc='center right',
           bbox_to_anchor=(1, 0.5),
           fontsize=10,
           frameon=False)

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
            #[top:bottom, left:right]
            crop = img[0:int(0.92*h), int(0.05*w):int(0.8*w)]
            ax.imshow(crop)
            # ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center")
        ax.axis("off")
        # column headings: mode on the top row
        if i == 0:
            ax.text(0.5, 0.98, mode, transform=ax.transAxes, fontsize=10,
                ha='center', va='bottom')
        if i == nrows - 1:
            ax.text(0.5, -0.08, 'Deciles of predicted risk', transform=ax.transAxes, fontsize=9,
                ha='center', va='bottom')
            
        # row labels: model names and Observed risk (%) on the leftmost column
        if j == 0:
            ax.text(-0.08, 0.5, model, transform=ax.transAxes, fontsize=10,
                rotation=90, ha='right', va='center')
            ax.text(0, 0.5, 'Observed risk (%)', transform=ax.transAxes, fontsize=9,
                rotation=90, ha='right', va='center')

# reduce margins between subplots
plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.07, right=0.8, top=0.98, bottom=0.03)
plt.savefig(os.path.join(prob_root, "summary", "Calibration_decile.png"), dpi=300)
plt.show()
plt.close()
gc.collect()
# %%
