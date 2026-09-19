# %% [markdown]
# ### Setup Environment:

# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 6"


import pandas as pd
import matplotlib.pyplot as plt


import torch
import pandas as pd

import numpy as np
import umap
import os


# %%
# Constants:
DATASET = os.path.dirname(os.path.realpath(__name__))
# Load save_dict from the .pt file
TIMEPOINT = 'b' 
MODEL = 'visionfm'
Canon_embeddings = torch.load(os.path.join(DATASET, f'output/extracted_feature/{'FTed_' if TIMEPOINT == 'a' else ''}Canon_embeddings_{MODEL}_fine_tune.pt'), weights_only=False)
Nikon_embeddings = torch.load(os.path.join(DATASET, f'output/extracted_feature/{'FTed_' if TIMEPOINT == 'a' else ''}Nikon_embeddings_{MODEL}_fine_tune.pt'), weights_only=False)
print("Loaded image_id:", len(Canon_embeddings['name']), len(Nikon_embeddings['name']))
print("Loaded features shape:", Canon_embeddings['features'].shape, Nikon_embeddings['features'].shape)
Canon_df = pd.read_csv(os.path.join(DATASET, 'data/labels_brset_Canon.csv'))
Nikon_df = pd.read_csv(os.path.join(DATASET, 'data/labels_brset_NIKON.csv'))
# Map numeric ICDR labels to categories
mapping = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}

def _map_icdr_column(df, col='DR_ICDR'):
    nums = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # robustly parse numeric-like values
    mapped = nums.map(mapping)  # map to 0,1,2
    return mapped.astype('Int64')  # keep nullable integer dtype with <NA> for unmapped

Canon_df['DR_ICDR'] = _map_icdr_column(Canon_df, 'DR_ICDR')
Nikon_df['DR_ICDR'] = _map_icdr_column(Nikon_df, 'DR_ICDR')

# Combine the DataFrames
Canon_img_df = pd.DataFrame({'image_id': Canon_embeddings['name']})
Canon_img_df = Canon_img_df.merge(Canon_df, on='image_id', how='left')
Nikon_img_df = pd.DataFrame({'image_id': Nikon_embeddings['name']})
Nikon_img_df = Nikon_img_df.merge(Nikon_df, on='image_id', how='left')

# Combine features from both datasets
features = np.concatenate([Canon_embeddings['features'], Nikon_embeddings['features']], axis=0)
ids = [0] * len(Canon_embeddings['name']) + [1] * len(Nikon_embeddings['name'])
icdr = np.concatenate([Canon_img_df['DR_ICDR'].values, Nikon_img_df['DR_ICDR'].values])
# Shuffle the rows of the features array
indices = np.random.permutation(features.shape[0])
features = features[indices]
image_id = [ids[i] for i in indices]
img_icdr = icdr[indices]


# Reduce dimensionality for visualization 
umap_model = umap.UMAP( n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, ) 
features_2d_uni = umap_model.fit_transform(features) 
# features_2d_Canon = umap_model.transform(Canon_embeddings["features"]) 
# features_2d_Nikon = umap_model.transform(Nikon_embeddings["features"])

def plot_umap_visualizations(features_2d_uni, image_id, dataset, tp, cmap='viridis', legend_labels=None):
    # Plot UMAP visualization by dataset
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(features_2d_uni[:, 0], features_2d_uni[:, 1], c=image_id, cmap=cmap, alpha=0.3, s=5)
    # plt.title(f'UMAP Visualization of Features \n Extracted using Pre-trained RETFound from {dataset}')
    plt.xlabel('UMAP x')
    if legend_labels is not None:
        plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Class", loc=1, fontsize=12, title_fontsize=13, markerscale=2.0)
    else:
        plt.legend(*scatter.legend_elements(), title="Class", loc=1, fontsize=12, title_fontsize=13, markerscale=2.0)
    plt.ylabel('UMAP y')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(DATASET, f'output/extracted_feature/{'Head' if tp == 'b' else 'Full'}_{MODEL}_{dataset}.png'), dpi=300, bbox_inches='tight')
    print(f"Saved UMAP figure to: {'Head' if tp == 'b' else 'Full'}_{MODEL}_{dataset}.png")
    # plt.colorbar(scatter)
    plt.show()


# MMD
def rbf_kernel(x, y, gamma=None):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    sqdist = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return np.exp(-gamma * sqdist)

def compute_mmd(x, y, **kwargs):
    k_xx = rbf_kernel(x, x, **kwargs)
    k_yy = rbf_kernel(y, y, **kwargs)
    k_xy = rbf_kernel(x, y, **kwargs)
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

def compute_mmd_for_locations(img_df, embeddings, loc_col, loc1, loc2, **kwargs):
    ids1 = set(img_df.loc[img_df[loc_col] == loc1, 'filename'])
    ids2 = set(img_df.loc[img_df[loc_col] == loc2, 'filename'])
    idx1 = [i for i, img_id in enumerate(embeddings['image_ids']) if img_id in ids1]
    idx2 = [i for i, img_id in enumerate(embeddings['image_ids']) if img_id in ids2]
    feats1 = np.asarray(embeddings['features'][idx1])
    feats2 = np.asarray(embeddings['features'][idx2])
    return compute_mmd(feats1, feats2, **kwargs)

# Compute Maximum Mean Discrepancy between Canon and Nikon features

Canon_features = features_2d_uni[np.asarray(image_id) == 0]
Nikon_features = features_2d_uni[np.asarray(image_id) == 1]

mmd_value = compute_mmd(Canon_features, Nikon_features)
print(f"Maximum Mean Discrepancy (Canon vs Nikon): {mmd_value:.6f}")


# %%
# Example usage:
# Plot UMAP visualization by dataset
plot_umap_visualizations(features_2d_uni, image_id, dataset='', tp=TIMEPOINT, cmap='viridis', legend_labels=['Canon', 'Nikon'])
# # Plot UMAP visualization by ICDR label
# plot_umap_visualizations(features_2d_uni, img_icdr, dataset='Combined Dataset by ICDR', tp=TIMEPOINT, cmap='viridis_r', legend_labels=['Normal', 'NPDR', 'PDR'])
# # Plot UMAP visualization for DK dataset by ICDR label
# plot_umap_visualizations(features_2d_Canon, Canon_img_df['DR_ICDR'].values, dataset='Canon subset by ICDR', tp=TIMEPOINT, cmap='viridis_r', legend_labels=['Normal', 'NPDR', 'PDR'])
# # Plot UMAP visualization for GL dataset by ICDR label
# plot_umap_visualizations(features_2d_Nikon, Nikon_img_df['DR_ICDR'].values, dataset='Nikon subset by ICDR', tp=TIMEPOINT, cmap='viridis_r', legend_labels=['Normal', 'NPDR', 'PDR'])



#%%
import os
import matplotlib.pyplot as plt
from PIL import Image

FIG_DIR = os.path.join(DATASET, "output/extracted_feature")  # adjust if needed

fig_names = {
    r"$\mathbf{a.}$ backbone of head fine-tuned RETFound": 
        "Frozen_Combined Dataset by Camera.png",
    r"$\mathbf{b.}$ backbone of full fine-tuned RETFound": 
        "Fine-tuned_Combined Dataset by Camera.png",
}

# col_titles = [
#     "",
# ]

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(8, 4),
    constrained_layout=True,
)

for col_idx, (column_name, fname) in enumerate(fig_names.items()):
        ax = axes[col_idx]
        img = Image.open(os.path.join(FIG_DIR, fname))
        ax.imshow(img)
        ax.axis("off")

        ax.set_title(column_name, fontsize=13)


# fig.text(-0.01, 0.5, "Combined", va="center", rotation=90, fontsize=12)
fig.text(0.5, 1.02, "Extracted feature visualization", ha="center", fontsize=15)
# fig.text(-0.01, 0.25, "Fine-tuned", va="center", rotation=90, fontsize=13)


plt.savefig(
    os.path.join(FIG_DIR, "UMAP_Frozen_vs_Finetuned_c.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
