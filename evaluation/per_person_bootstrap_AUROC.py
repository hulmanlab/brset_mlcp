# %%
import gc
import os
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# %%
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prob_root", default="BRSET_TL", help="Path to predicted probabilities directory")
    return parser.parse_args()


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
    auc_scores = []
    auc_0 = []
    auc_1 = []
    auc_2 = []
    # Per-person bootstrap sample
    for _ in range(n_iterations):
        boot_df = pd.DataFrame()

        sampled_persons = resample(
            persons,
            replace=True,
            n_samples=n_persons
        )

        # Build bootstrap DataFrame
        boot_df = pd.concat(
            [df[df["person_id"] == person_id] for person_id in sampled_persons],
            ignore_index=True
        )
        
        # Generate a random sample of indices
        # y_resample, y_score_resample = resample(boot_df['y_test'], boot_df['y_pred'], replace=True)
        # acc = accuracy_score(np.argmax(y_resample,axis=1), np.argmax(y_score_resample, axis=1))
        # acc_scores.append(acc)
        if 'y_pred' not in boot_df.columns:
            y_test = boot_df.to_numpy()[:, :3].astype(int)
            y_pred = boot_df.to_numpy()[:, 3:6]
            class_auc = roc_auc_score(y_test, y_pred, average=None, multi_class='ovr')
            auc_1.append(class_auc[0])
            auc_0.append(class_auc[1])
            auc_2.append(class_auc[2])

        else:
            y_test = boot_df['y_test']
            y_pred = boot_df['y_pred']

        auc = roc_auc_score(y_test, y_pred, multi_class='ovr', average='macro')
        auc_scores.append(auc)
        
    return auc_scores, auc_0, auc_1, auc_2

def CI95(scores):
    mean_auc = np.mean(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return mean_auc, lower, upper


def process_and_append(df_sub, model, mode):
    # acc_scores, auc_scores = bootstrap_ensemble(df_sub)
    # mean_acc, lower_acc, upper_acc = CI95(acc_scores)
    auc_scores, auc_0, auc_1, auc_2 = bootstrap_ensemble(df_sub)
    mean_auc, lower_auc, upper_auc = CI95(auc_scores)
    if auc_0 and auc_1 and auc_2:
        mean_auc_0, lower_auc_0, upper_auc_0 = CI95(auc_0)
        mean_auc_1, lower_auc_1, upper_auc_1 = CI95(auc_1)
        mean_auc_2, lower_auc_2, upper_auc_2 = CI95(auc_2)
        new_results_row = {
            'model': model,
            'mode': mode,
            'auc_macro': f'{mean_auc:.2f} [{lower_auc:.2f}, {upper_auc:.2f}]',
            'auc_0': f'{mean_auc_0:.2f} [{lower_auc_0:.2f}, {upper_auc_0:.2f}]',
            'auc_1': f'{mean_auc_1:.2f} [{lower_auc_1:.2f}, {upper_auc_1:.2f}]',
            'auc_2': f'{mean_auc_2:.2f} [{lower_auc_2:.2f}, {upper_auc_2:.2f}]'
        }
    else:
        new_results_row = {
                'model': model,
                'mode': mode,
                'auc_macro': f'{mean_auc:.2f} [{lower_auc:.2f}, {upper_auc:.2f}]',
            }
    new_plot_row = {
        'model': model,
        'mode': mode,
        'mean_auc': mean_auc,
        'lower_auc': lower_auc,
        'upper_auc': upper_auc
    }
    return new_results_row, new_plot_row


#### plot function 
def plot(df_plot_sorted, figsize, prob_root, name = ''):

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_ymargin(0.15)

    # --- AUC plot ---
    for _, row in df_plot_sorted.iterrows():
        ax.errorbar(
            row['mean_auc'],
            row['label'],
            xerr=[[row['mean_auc'] - row['lower_auc']], [row['upper_auc'] - row['mean_auc']]],
            fmt='o',
            ls='-.',
            color=row['color'],
            ecolor='gray',
            capsize=7
        )
        ax.text(
            1.01,
            row['label'],
            f"{row['mean_auc']:.2f} [{row['lower_auc']:.2f}, {row['upper_auc']:.2f}]",
            va='center',
            fontsize=12
        )
    yd = 0.7/(len(df_plot_sorted)/2-1)
    ys = [0.15+i*yd for i in range(len(df_plot_sorted))]
    # ys = [0.15, 0.15+yd, 0.15+2*yd, 0.15+3*yd, 0.15+4*yd, 0.85]
    labels = df_plot_sorted['model'].unique()
    for m, y in zip(labels, ys):
        ax.text(
            -0.05, y, m,
            transform=ax.transAxes,
            va="center",
            ha="right",
            fontsize=12
        )

    ax.set_yticks([])
    ax.set_xlim(0.5, 1)
    ax.grid(axis='x')
    ax.set_title("Area Under the ROC Curve", fontsize=14, pad=10)
    # Custom legend
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='#E66100', label='Head fine-tune'),
        mpatches.Patch(color='#5D3A9B', label='Full fine-tune')
    ]
    ax.legend(handles=legend_handles, title='Training mode', bbox_to_anchor=(0.04, 0.96), loc='upper left', borderaxespad=0., ncol=1, frameon=True)

    plt.tight_layout(rect=[0.05, 0, 1, 1])
    plt.show()
    fig.savefig(os.path.join(prob_root, 'summary', f'AUROC_forest_plot{name}.png'), bbox_inches='tight', dpi=300)
    gc.collect()
# %%
def main(path):
    DATASET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prob_root = os.path.join(DATASET, "output/predicted_probabilities", path) 
    files = os.listdir(prob_root)
    files = [f for f in files if f.startswith('y_') and f.endswith('.csv') and not any(x in f for x in ('convnextv2', 'resnet', 'recalib'))]
    files.sort()

    df_result = pd.DataFrame(columns=['model', 'mode', 'auc_macro', 'auc_0', 'auc_1', 'auc_2'])
    df_plot = pd.DataFrame(columns=['model', 'mode', 'mean_auc', 'lower_auc', 'upper_auc'])
    for filename in tqdm(files):
        if filename.endswith(".csv") and filename.startswith("y_") and 'reproduce' not in filename:
            # print(f"File: {filename}")
            df = pd.read_csv(os.path.join(prob_root, filename))
            model = 'RETFound DINOv2 Shanghai' if 'retfound_d2_s' in filename else \
                    'RETFound DINOv2 MEH' if 'retfound_d2_m' in filename else \
                    'RETFound' if 'retfound' in filename else \
                    'DINOv3' if 'dinov3_large' in filename else \
                    'VisionFM' if 'visionfm' in filename else \
                    'DINOv2' if 'dinov2' in filename else \
                    'EyeCLIP' if 'eyeclip' in filename else \
                    'ConvNeXt' if 'convnext' in filename else \
                    'ResNet200d' if 'resnet200d' in filename else \
                    'ResNet50' if 'resnet50' in filename else \
                    'Unknown Model'

            mode = 'Head fine-tune' if 'eval' in filename else 'Full fine-tune'
            # Function to process and append results
            if 'y_camera' in df.columns:
                df = df.drop(columns=['y_camera'])
            new_results_row, new_plot_row = process_and_append(df, model, mode)
            
            df_result = pd.concat([df_result, pd.DataFrame([new_results_row])], ignore_index=True)
            # if not df.shape[1] > 3:
            #     df_result = df_result[['model', 'mode', 'auc_macro']]
            df_plot = pd.concat([df_plot, pd.DataFrame([new_plot_row])], ignore_index=True)
            
        
    # # print(df_result)
    os.makedirs(os.path.join(prob_root, 'summary'), exist_ok=True)
    df_result.to_csv(os.path.join(prob_root, 'summary', 'AUROC_results_per_person.csv'), index=False)

    # Create a label combining model and mode for y-axis
    df_plot = df_plot[~df_plot['model'].str.contains('resnet|convnext', case=False)]
    df_plot['label'] = df_plot['model'] + ' - ' + df_plot['mode']
    df_plot[['mean_auc', 'lower_auc', 'upper_auc']] = df_plot[['mean_auc', 'lower_auc', 'upper_auc']].astype(float)
    # Assign color: green if 'Head' in mode, else blue
    df_plot['color'] = df_plot['mode'].apply(lambda x: '#E66100' if 'Head' in x else '#5D3A9B')
    # Sort by model and mode so rows are reversed than df_plot
    df_plot_sorted = df_plot.sort_values(['model', 'mode'], ascending=[False, True])
    df_plot_sorted_main = df_plot_sorted[df_plot_sorted['model'].isin(['DINOv3', 'RETFound', 'VisionFM'])]
    

    plot(df_plot_sorted, (9, 4), prob_root, '_per_person')
    plot(df_plot_sorted_main, (9, 3), prob_root, '_main_per_person')

# %%
if __name__ == "__main__":
    args = get_args()
    main(args.prob_root)
