# %%
import argparse
import gc
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import resample
from itertools import combinations
from tqdm import tqdm
from torch import tensor
from torchmetrics.classification import BinaryCalibrationError, MulticlassCalibrationError


# %%
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prob_root", default="BRSET_TL", help="Path to predicted probabilities directory")
path = parser.parse_args().prob_root

DATASET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prob_root = os.path.join(DATASET, "output/predicted_probabilities", path)
files = os.listdir(prob_root)
files = [f for f in files if f.startswith('y_') and f.endswith('.csv') and not any(x in f for x in ('convnextv2', 'resnet', 'recalib'))]
files.sort()
# print(files)

# %%
def bootstrap_ensemble(df, n_iterations=1000):
    # Extract person ID from image ID
    if 'person_id' not in df.columns:
        df["person_id"] = df["image_ids"].astype(str).str.split(".").str[0]

    # Optional: store as numeric
    df["person_id"] = pd.to_numeric(df["person_id"])

    # Get unique persons
    persons = df["person_id"].unique()
    n_persons = len(persons)
    ece_score = []

    # Per-person bootstrap sample
    for i in range(n_iterations):
        sampled_persons = np.random.choice(
            persons,
            size=n_persons,
            replace=True
        )
        # Build bootstrap DataFrame
        boot_df = pd.concat(
            [df[df["person_id"] == person_id] for person_id in sampled_persons],
            ignore_index=True
        )
        if 'y_pred' not in boot_df.columns:
            y_test = tensor(np.argmax(boot_df.to_numpy()[:, :3].astype(int), axis=1))
            y_pred = tensor(boot_df.to_numpy()[:, 3:6].astype(float))
            metric = MulticlassCalibrationError(num_classes=3, n_bins=10, norm='l1')
        else:
            y_test = tensor(np.array(boot_df['y_test']))
            y_pred = tensor(np.array(boot_df['y_pred']))
            metric = BinaryCalibrationError(n_bins=10, norm='l1')
        ece = metric(y_pred, y_test).item()

        ece_score.append(ece)
        
        
    return ece_score

def CI95(scores):
    mean = np.mean(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return mean, lower, upper

# %%
df_result = pd.DataFrame(columns=['model', 'mode', 'ece'])
df_plot = pd.DataFrame(columns=['model', 'mode', 'mean_ece', 'lower_ece', 'upper_ece'])
for filename in tqdm(files):
    if filename.endswith(".csv") and filename.startswith("y_") and 'reproduce' not in filename:
        # print(f"File: {filename}")
        df = pd.read_csv(os.path.join(prob_root, filename))
        model = 'RETFound DINOv2 Shanghai' if 'retfound_d2_s' in filename else \
                'RETFound DINOv2 MEH' if 'retfound_d2_m' in filename else \
                'RETFound' if 'retfound' in filename else \
                'DINOv3 Large' if 'dinov3_large' in filename else \
                'VisionFM' if 'visionfm' in filename else \
                'DINOv2 Large' if 'dinov2' in filename else \
                'EyeCLIP' if 'eyeclip' in filename else \
                'ConvNeXt' if 'convnext' in filename else \
                'ResNet200d' if 'resnet200d' in filename else \
                'ResNet50' if 'resnet50' in filename else \
                'Unknown Model'

        mode = 'Head fine-tune' if 'eval' in filename else 'Full fine-tune'
        # Function to process and append results
        def process_and_append(df_sub, name_prefix=""):
            
            ece_score= bootstrap_ensemble(df_sub)

            mean_ece, lower_ece, upper_ece = CI95(ece_score)

            new_results_row = {
                'model': model,
                'mode': mode,
                'ece': f'{mean_ece:.2f} [{lower_ece:.2f}, {upper_ece:.2f}]'
            }
            new_plot_row = {
                'model': model,
                'mode': mode,
                'mean_ece': mean_ece,
                'lower_ece': lower_ece,
                'upper_ece': upper_ece
            }
            global df_plot
            df_plot = pd.concat([df_plot, pd.DataFrame([new_plot_row])], ignore_index=True)
            global df_result
            df_result = pd.concat([df_result, pd.DataFrame([new_results_row])], ignore_index=True)
        
        if 'y_camera' in df.columns:
            df = df.drop(columns=['y_camera'])
        process_and_append(df)
    
# print(df_result)
#%% 

df_result.to_csv(os.path.join(prob_root, 'summary', 'ECE_results_per_person.csv'), index=False)



# %%
# import matplotlib.pyplot as plt
# # Create a label combining model and mode for y-axis
# df_plot = df_plot[~df_plot['model'].str.contains('resnet|convnext', case=False)]
# df_plot['label'] = df_plot['model'] + ' - ' + df_plot['mode']
# df_plot[['mean_ece', 'lower_ece', 'upper_ece']] = df_plot[['mean_ece', 'lower_ece', 'upper_ece']].astype(float)
# # Assign color: green if 'Head' in mode, else blue
# df_plot['color'] = df_plot['mode'].apply(lambda x: '#E66100' if 'Head' in x else '#5D3A9B')
# # Sort by model and mode for better grouping
# df_plot_sorted = df_plot.sort_values(['model', 'mode'], ascending=[False, True])

# fig, ax = plt.subplots(1, 1, figsize=(9, 4))
# ax.set_ymargin(0.15)

# # --- ECE plot ---
# for _, row in df_plot_sorted.iterrows():
#     ax.errorbar(
#         row['mean_ece'],
#         row['label'],
#         xerr=[[row['mean_ece'] - row['lower_ece']], [row['upper_ece'] - row['mean_ece']]],
#         fmt='o',
#         ls='-.',
#         color=row['color'],
#         ecolor='gray',
#         capsize=7
#     )
#     ax.text(
#         0.51,
#         row['label'],
#         f"{row['mean_ece']:.2f} [{row['lower_ece']:.2f}, {row['upper_ece']:.2f}]",
#         va='center',
#         fontsize=12
#     )
# yd = 0.14
# ys = [0.15, 0.15+yd, 0.15+2*yd, 0.15+3*yd, 0.15+4*yd, 0.85]
# labels = df_plot_sorted['model'].unique()
# for m, y in zip(labels, ys):
#     ax.text(
#         -0.05, y, m,
#         transform=ax.transAxes,
#         va="center",
#         ha="right",
#         fontsize=10
#     )
# ax.set_yticks([])
# ax.set_xlim(0, 0.5)
# ax.set_title("Expected Calibration Error", fontsize=14, pad=10)
# # Custom legend
# import matplotlib.patches as mpatches
# legend_handles = [
#     mpatches.Patch(color='#E66100', label='Head fine-tune'),
#     mpatches.Patch(color='#5D3A9B', label='Full fine-tune')
# ]
# ax.legend(handles=legend_handles, title='Training mode', bbox_to_anchor=(0.04, 0.96), loc='upper left', borderaxespad=0., ncol=1, frameon=True)

# plt.tight_layout(rect=[0.05, 0, 1, 1])
# plt.show()
# fig.savefig(os.path.join(prob_root, 'summary', 'ECE_forest_plot.png'), bbox_inches='tight', dpi=300)
gc.collect()

# %%
