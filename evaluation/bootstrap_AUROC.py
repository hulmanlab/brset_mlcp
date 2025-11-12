# %%
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from itertools import combinations
from tqdm import tqdm


# %%
DATASET = os.path.dirname(os.path.abspath(__name__))
prob_root = os.path.join(DATASET, "output/predicted_probabilities/mBRSET_EXEVAL")
files = os.listdir(prob_root)
files.sort()
print(files)

# %%
def bootstrap_ensemble(df, n_iterations=1000):
    arr = df.to_numpy()
    y_true = np.array(arr[:, :3].astype(int))
    y_score = arr[:, 3:]
    y_score = np.where(y_score == 0, 0.000001, y_score)
    y_score = np.where(y_score == 1, 0.999999, y_score) # shape: (n, 4)
    
    # Run bootstrap
    # acc_scores = []
    auc_scores = []
    auc_0_scores = []
    auc_1_scores = []
    auc_2_scores = []


    for _ in range(n_iterations):
        # Generate a random sample of indices
        y_resample, y_score_resample = resample(y_true, y_score, replace=True)
        # acc = accuracy_score(np.argmax(y_resample,axis=1), np.argmax(y_score_resample, axis=1))
        # acc_scores.append(acc)
        auc = roc_auc_score(y_resample, y_score_resample, multi_class='ovr', average='macro')
        auc_scores.append(auc)
        

        # Compute class-wise AUCs
        class_auc = roc_auc_score(y_resample, y_score_resample, average=None, multi_class='ovr')
        auc_0_scores.append(class_auc[0])
        auc_1_scores.append(class_auc[1])
        auc_2_scores.append(class_auc[2])

        
        
    # return  acc_scores, auc_scores
    return auc_scores, auc_0_scores, auc_1_scores, auc_2_scores

def CI95(scores):
    mean_auc = np.mean(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return mean_auc, lower, upper

# %%
# if 'ensemble_results.csv' in files:
#     df_plot = pd.read_csv(os.path.join(prob_root, 'ensemble_results.csv'))
# else:
df_result = pd.DataFrame(columns=['model', 'mode', 'auc_macro', 'auc_0', 'auc_1', 'auc_2'])
df_plot = pd.DataFrame(columns=['model', 'mode', 'mean_auc', 'lower_auc', 'upper_auc'])
for filename in files:
    if filename.endswith(".csv") and filename.startswith("y_") and 'reproduce' not in filename:
        print(f"File: {filename}")
        df = pd.read_csv(os.path.join(prob_root, filename))
        model = filename.split('_')[1]
        mode = 'Head fine-tune' if 'eval' in filename else 'Full fine-tuned'
        # Function to process and append results
        def process_and_append(df_sub, name_prefix=""):
            # acc_scores, auc_scores = bootstrap_ensemble(df_sub)
            # mean_acc, lower_acc, upper_acc = CI95(acc_scores)
            auc_scores, auc_0_scores, auc_1_scores, auc_2_scores = bootstrap_ensemble(df_sub)

            mean_auc, lower_auc, upper_auc = CI95(auc_scores)
            mean_auc_0, lower_auc_0, upper_auc_0 = CI95(auc_0_scores)
            mean_auc_1, lower_auc_1, upper_auc_1 = CI95(auc_1_scores)
            mean_auc_2, lower_auc_2, upper_auc_2 = CI95(auc_2_scores)

            new_results_row = {
                'model': model,
                'mode': mode,
                'auc_macro': f'{mean_auc:.2f} [{lower_auc:.2f}-{upper_auc:.2f}]',
                'auc_0': f'{mean_auc_0:.2f} [{lower_auc_0:.2f}-{upper_auc_0:.2f}]',
                'auc_1': f'{mean_auc_1:.2f} [{lower_auc_1:.2f}-{upper_auc_1:.2f}]',
                'auc_2': f'{mean_auc_2:.2f} [{lower_auc_2:.2f}-{upper_auc_2:.2f}]'
            }

            new_plot_row = {
                'model': model,
                'mode': mode,
                'mean_auc': mean_auc,
                'lower_auc': lower_auc,
                'upper_auc': upper_auc
            }
            global df_result
            df_result = pd.concat([df_result, pd.DataFrame([new_results_row])], ignore_index=True)
            global df_plot
            df_plot = pd.concat([df_plot, pd.DataFrame([new_plot_row])], ignore_index=True)
        
        if 'y_camera' in df.columns:
            df = df.drop(columns=['y_camera'])
        process_and_append(df)
    
# print(df_result)
# df_result.to_csv(os.path.join(prob_root, 'auroc_results.csv'), index=False)
# pdi_result = pd.read_csv(os.path.join(prob_root, 'pdi_results.csv'))
# df_plot = df_plot.merge(pdi_result, on=['model', 'mode'], how='left')
# df_plot.to_csv(os.path.join(prob_root, 'ensemble_results.csv'), index=False)

# %%
import matplotlib.pyplot as plt
# Create a label combining model and mode for y-axis
df_plot = df_plot[~df_plot['model'].str.contains('resnet|convnext', case=False)]
df_plot['label'] = df_plot['model'] + ' - ' + df_plot['mode']

# Assign color: green if 'Head' in mode, else blue
df_plot['color'] = df_plot['mode'].apply(lambda x: '#E1BE6A' if 'Head' in x else '#40B0A6')

# Sort by model and mode for better grouping
df_plot_sorted = df_plot.sort_values(['model', 'mode'], ascending=[False, True])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2), sharey=True)
ax1.set_ymargin(0.15)
ax2.set_ymargin(0.15)
# --- AUC plot ---
for _, row in df_plot_sorted.iterrows():
    ax1.errorbar(
        row['mean_auc'],
        row['label'],
        xerr=[[row['mean_auc'] - row['lower_auc']], [row['upper_auc'] - row['mean_auc']]],
        fmt='o',
        ls='-.',
        color=row['color'],
        ecolor='gray',
        capsize=5
    )
    ax1.text(
        1.01,
        row['label'],
        f"{row['mean_auc']:.2f} [{row['lower_auc']:.2f}, {row['upper_auc']:.2f}]",
        va='center',
        fontsize=10
    )
# ax1.set_xlabel('AUROC (95% CI)')
ax1.set_yticks([])
ax1.set_xlim(0.5, 1)
# ax1.set_title('AUROC')

# --- PDI plot ---
for _, row in df_plot_sorted.iterrows():
    ax2.errorbar(
        row['mean_pdi'],
        row['label'],
        xerr=[[row['mean_pdi'] - row['lower_pdi']], [row['upper_pdi'] - row['mean_pdi']]],
        fmt='o',
        ls='-',
        color=row['color'],
        ecolor='gray',
        capsize=5
    )
    ax2.text(
        1.01,
        row['label'],
        f"{row['mean_pdi']:.2f} [{row['lower_pdi']:.2f}, {row['upper_pdi']:.2f}]",
        va='center',
        fontsize=10
    )
# ax2.set_xlabel('PDI (95% CI)')
ax2.set_yticks([])
ax2.set_xlim(0.33, 1)
# ax2.set_title('PDI')

# Shared y-label
# fig.text(0.04, 0.5, 'Model', va='center', rotation='vertical', fontsize=12)

# Custom legend
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color='#E1BE6A', label='Head fine-tune'),
    mpatches.Patch(color='#40B0A6', label='Full fine-tuned')
]
# Set ncol=1 for a vertical legend
ax2.legend(handles=legend_handles, title='Training mode', bbox_to_anchor=(1.45, 1), loc='upper left', borderaxespad=0., ncol=1, frameon=True)
# ax2.legend(handles=legend_handles, title='Training mode', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Add a text box annotation to the figure (example: add to ax1)
# ax1.text(
#     -0.2,  # x position in data coordinates
#     0.5,   # y position in axes fraction (0=bottom, 1=top)
#     "Dinov2\n\n\n\nRETFound\n\n\n\nVisionFM", 
#     transform=ax1.get_yaxis_transform(),  # so y is in axes fraction
#     fontsize=8
#     # bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
# )

plt.tight_layout(rect=[0.05, 0, 1, 1])
plt.show()
fig.savefig(os.path.join(prob_root, 'forest_plot.png'), bbox_inches='tight', dpi=300)





