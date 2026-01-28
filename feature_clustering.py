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
from sklearn.manifold import TSNE
import umap
import os


# %%
# Constants:
DATASET = os.path.dirname(os.path.realpath(__name__))
# Load save_dict from the .pt file
TIMEPOINT = 'a'
Canon_embeddings = torch.load(os.path.join(DATASET, f'output/extracted_feature/{'FTed_' if TIMEPOINT == 'a' else ''}Canon_embeddings_retfound_fine_tune.pt'), weights_only=False)
Nikon_embeddings = torch.load(os.path.join(DATASET, f'output/extracted_feature/{'FTed_' if TIMEPOINT == 'a' else ''}Nikon_embeddings_retfound_fine_tune.pt'), weights_only=False)
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

# %%
# Combine the DataFrames
Canon_img_df = pd.DataFrame({'image_id': Canon_embeddings['name']})
Canon_img_df = Canon_img_df.merge(Canon_df, on='image_id', how='left')
Nikon_img_df = pd.DataFrame({'image_id': Nikon_embeddings['name']})
Nikon_img_df = Nikon_img_df.merge(Nikon_df, on='image_id', how='left')
# %%
# Combine features from both datasets
features = np.concatenate([Canon_embeddings['features'], Nikon_embeddings['features']], axis=0)
ids = [0] * len(Canon_embeddings['name']) + [1] * len(Nikon_embeddings['name'])
icdr = np.concatenate([Canon_img_df['DR_ICDR'].values, Nikon_img_df['DR_ICDR'].values])
# Shuffle the rows of the features array
indices = np.random.permutation(features.shape[0])
features = features[indices]
image_id = [ids[i] for i in indices]
img_icdr = icdr[indices]

# %%
# Reduce dimensionality for visualization
# tsne = TSNE(n_components=2, random_state=42)
# features_2d_uni = tsne.fit_transform(features)
# features_2d_Canon = tsne.fit_transform(Canon_embeddings['features'])
# features_2d_Nikon = tsne.fit_transform(Nikon_embeddings['features'])

# Reduce dimensionality for visualization 
umap_model = umap.UMAP( n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, ) 
features_2d_uni = umap_model.fit_transform(features) 
features_2d_Canon = umap_model.transform(Canon_embeddings["features"]) 
features_2d_Nikon = umap_model.transform(Nikon_embeddings["features"])

# %%
def plot_tsne_visualizations(features_2d_uni, image_id, dataset, tp, cmap='viridis', legend_labels=None):
    # Plot UMAP visualization by dataset
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(features_2d_uni[:, 0], features_2d_uni[:, 1], c=image_id, cmap=cmap, alpha=0.3, s=5)
    # plt.title(f'UMAP Visualization of Features \n Extracted using Pre-trained RETFound from {dataset}')
    plt.xlabel('UMAP x')
    if legend_labels is not None:
        plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Class", loc=1)
    else:
        plt.legend(*scatter.legend_elements(), title="Class", loc=1)
    plt.ylabel('UMAP y')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(DATASET, f'output/extracted_feature/{'Frozen' if tp == 'b' else 'Fine-tuned'}_{dataset}.png'), dpi=300, bbox_inches='tight')
    print(f"Saved UMAP figure to: {'Frozen' if tp == 'b' else 'Fine-tuned'}_{dataset}.png")
    # plt.colorbar(scatter)
    plt.show()



    

# %%
# Example usage:
# Plot UMAP visualization by dataset
plot_tsne_visualizations(features_2d_uni, image_id, dataset='Combined Dataset by Camera', tp=TIMEPOINT, cmap='viridis', legend_labels=['Canon', 'Nikon'])
# Plot UMAP visualization by ICDR label
plot_tsne_visualizations(features_2d_uni, img_icdr, dataset='Combined Dataset by ICDR', tp=TIMEPOINT, cmap='viridis_r', legend_labels=['Normal', 'NPDR', 'PDR'])
# Plot UMAP visualization for DK dataset by ICDR label
plot_tsne_visualizations(features_2d_Canon, Canon_img_df['DR_ICDR'].values, dataset='Canon subset by ICDR', tp=TIMEPOINT, cmap='viridis_r', legend_labels=['Normal', 'NPDR', 'PDR'])
# Plot UMAP visualization for GL dataset by ICDR label
plot_tsne_visualizations(features_2d_Nikon, Nikon_img_df['DR_ICDR'].values, dataset='Nikon subset by ICDR', tp=TIMEPOINT, cmap='viridis_r', legend_labels=['Normal', 'NPDR', 'PDR'])
# %%

import os
import matplotlib.pyplot as plt
from PIL import Image

FIG_DIR = os.path.join(DATASET, "output/extracted_feature")  # adjust if needed

fig_names = {
    "Frozen": [
        "Frozen_Combined Dataset by Camera.png",
        "Frozen_Combined Dataset by ICDR.png",
        "Frozen_Canon subset by ICDR.png",
        "Frozen_Nikon subset by ICDR.png",
    ],
    "Fine-tuned": [
        "Fine-tuned_Combined Dataset by Camera.png",
        "Fine-tuned_Combined Dataset by ICDR.png",
        "Fine-tuned_Canon subset by ICDR.png",
        "Fine-tuned_Nikon subset by ICDR.png",
    ],
}

col_titles = [
    "Combined (Camera)",
    "Combined (ICDR)",
    "Canon (ICDR)",
    "Nikon (ICDR)",
]

fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(16, 8),
    constrained_layout=True,
)

for row_idx, (row_name, files) in enumerate(fig_names.items()):
    for col_idx, fname in enumerate(files):
        ax = axes[row_idx, col_idx]
        img = Image.open(os.path.join(FIG_DIR, fname))
        ax.imshow(img)
        ax.axis("off")

        if row_idx == 0:
            ax.set_title(col_titles[col_idx], fontsize=12)

        if col_idx == 0:
            ax.set_ylabel(row_name, fontsize=12, rotation=90)
fig.text(-0.01, 0.75, "Frozen", va="center", rotation=90, fontsize=13)
fig.text(-0.01, 0.25, "Fine-tuned", va="center", rotation=90, fontsize=13)


plt.savefig(
    os.path.join(FIG_DIR, "UMAP_Frozen_vs_Finetuned.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
