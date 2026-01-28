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
parser.add_argument("-p", "--prob_root", default="BRSET_TL_b", help="Path to predicted probabilities directory")
path = parser.parse_args().prob_root

DATASET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prob_root = os.path.join(DATASET, "output/predicted_probabilities", path)
files = os.listdir(prob_root)
files = [f for f in files if f.startswith('y_') and f.endswith('.csv') and not any(x in f for x in ('convnextv2', 'resnet'))]
files.sort()
# print(files)

# %%
def bootstrap_ensemble(df, n_iterations=1000):
    arr = df.to_numpy()
    if arr.shape[1] > 3:
        y_true = np.array(arr[:, :3].astype(int))
        y_true = np.argmax(y_true, axis=1)  # Convert one-hot to class labels
        y_score = arr[:, 3:]
    else:
        y_true = np.array(arr[:, 0].astype(int))
        y_score = arr[:, 1]
    y_score = np.where(y_score == 0, 0.000001, y_score)
    y_score = np.where(y_score == 1, 0.999999, y_score) # shape: (n, 4)
    
    # Run bootstrap
    ece_score = []


    for _ in range(n_iterations):
        # Generate a random sample of indices
        y_resample, y_score_resample = resample(y_true, y_score, replace=True)
        preds = tensor(y_score_resample)
        target = tensor(y_resample)
        if arr.shape[1] > 3:
            metric = MulticlassCalibrationError(num_classes=3, n_bins=10, norm='l1')
        else:
            metric = BinaryCalibrationError(n_bins=10, norm='l1')
        ece = metric(preds, target).item()

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
for filename in files:
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

df_result.to_csv(os.path.join(prob_root, 'summary', 'ECE_results.csv'), index=False)



# %%
import matplotlib.pyplot as plt
# Create a label combining model and mode for y-axis
df_plot = df_plot[~df_plot['model'].str.contains('resnet|convnext', case=False)]
df_plot['label'] = df_plot['model'] + ' - ' + df_plot['mode']
df_plot[['mean_ece', 'lower_ece', 'upper_ece']] = df_plot[['mean_ece', 'lower_ece', 'upper_ece']].astype(float)
# Assign color: green if 'Head' in mode, else blue
df_plot['color'] = df_plot['mode'].apply(lambda x: '#E1BE6A' if 'Head' in x else '#40B0A6')
# Sort by model and mode for better grouping
df_plot_sorted = df_plot.sort_values(['model', 'mode'], ascending=[False, True])

fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.set_ymargin(0.15)

# --- ECE plot ---
for _, row in df_plot_sorted.iterrows():
    ax.errorbar(
        row['mean_ece'],
        row['label'],
        xerr=[[row['mean_ece'] - row['lower_ece']], [row['upper_ece'] - row['mean_ece']]],
        fmt='o',
        ls='-.',
        color=row['color'],
        ecolor='gray',
        capsize=7
    )
    ax.text(
        0.51,
        row['label'],
        f"{row['mean_ece']:.2f} [{row['lower_ece']:.2f}, {row['upper_ece']:.2f}]",
        va='center',
        fontsize=12
    )
yd = 0.14
ys = [0.15, 0.15+yd, 0.15+2*yd, 0.15+3*yd, 0.15+4*yd, 0.85]
labels = df_plot_sorted['model'].unique()
for m, y in zip(labels, ys):
    ax.text(
        -0.05, y, m,
        transform=ax.transAxes,
        va="center",
        ha="right",
        fontsize=10
    )
ax.set_yticks([])
ax.set_xlim(0, 0.5)
ax.set_title("Expected Calibration Error", fontsize=14, pad=10)
# Custom legend
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color='#E1BE6A', label='Head fine-tune'),
    mpatches.Patch(color='#40B0A6', label='Full fine-tune')
]
ax.legend(handles=legend_handles, title='Training mode', bbox_to_anchor=(0.04, 0.96), loc='upper left', borderaxespad=0., ncol=1, frameon=True)

plt.tight_layout(rect=[0.05, 0, 1, 1])
plt.show()
fig.savefig(os.path.join(prob_root, 'summary', 'ECE_forest_plot.png'), bbox_inches='tight', dpi=300)
gc.collect()

# %%
