#%%
import os
import pandas as pd
from functools import reduce
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%%

binary = ['BRSET_TL_b', 'mBRSET_EX_b', 'mBRSET_TL_b']
ternary = ['BRSET_TL', 'mBRSET_EXEVAL', 'mBRSET_TL']

items = ['AUROC', 'ECE', 'Brier']
# base_dir = f"/home/livieymli/brset_analysis/BRSET/output/predicted_probabilities/{binary[0]}/summary"


#%%

fig, axes = plt.subplots(3, 1, figsize=(9, 8))
prob_root = '/home/livieymli/brset_analysis/BRSET/output'
case = 'B'
image_paths = [f"/home/livieymli/brset_analysis/BRSET/output/predicted_probabilities/{folder}/summary/{items[0]}_forest_plot_main_per_person.png" for folder in binary]
subtitles = ["Experiment 1: Fine-tuning pre-trained models to BRSET", "Experiment 2: External validation on mBRSET", "Experiment 3: Fine-tuning models to mBRSET"]

for i, (ax, img_path, subtitle) in enumerate(zip(axes, image_paths, subtitles)):
    img = mpimg.imread(img_path)
    h, w = img.shape[:2]

    img = img[int(w * 0.04):,
            int(w * 0.01):int(w * 0.99)]

    ax.imshow(img)
    ax.set_title(subtitle, fontsize = 11)
    ax.axis("off")


# Main title
fig.suptitle("Area Under the ROC Curve", fontsize=12, y=0.93)


plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(prob_root, 'figures', f'{case}_{items[0]}_3_1_per_person.png'), bbox_inches='tight', dpi=300)
plt.show()

# %%

