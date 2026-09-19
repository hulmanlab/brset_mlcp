# %%
import argparse
import gc
import os
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# %%
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", default="BRSET_TL_b", required=False, help="Path to predicted probabilities directory")
path = parser.parse_args().path
path = 'mBRSET_EX_b'
DATASET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prob_root = os.path.join(DATASET, "output/predicted_probabilities", path)
files = os.listdir(prob_root)
files = [f for f in files if f.startswith('y_') and f.endswith('.csv') and not any(x in f for x in ('convnextv2', 'resnet', 'recalib', 'retfound_d2'))]
files.sort()

pairs = [
    (f, f.replace('_fine_tune_binary.csv', '_eval_binary.csv')
    .replace('_fine_tune_3class.csv', '_eval_3class.csv'))
    for f in files
    if f.endswith(('_fine_tune_binary.csv', '_fine_tune_3class.csv'))
]


#%%
def metadata_from_filename(filename):
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
    return model, mode
# %%
def bootstrap_ensemble(df1, df2, n_iterations=1000):
    if 'person_id' not in df1.columns:
        df1["person_id"] = df1["image_ids"].astype(str).str.split(".").str[0]
        df2["person_id"] = df2["image_ids"].astype(str).str.split(".").str[0]
    # Optional: store as numeric
    df1["person_id"] = pd.to_numeric(df1["person_id"])
    df2["person_id"] = pd.to_numeric(df2["person_id"])

    # Get unique persons
    persons = df1["person_id"].unique()
    n_persons = len(persons)
    d_auc = []
    d_auc_0 = []
    d_auc_1 = []
    d_auc_2 = []
    # Per-person bootstrap sample
    for _ in range(n_iterations):
        boot_df1 = pd.DataFrame()
        boot_df2 = pd.DataFrame()

        sampled_persons = resample(
            persons,
            replace=True,
            n_samples=n_persons
        )

        # Build bootstrap DataFrame
        boot_df1 = pd.concat(
            [df1[df1["person_id"] == person_id] for person_id in sampled_persons],
            ignore_index=True
        )
        boot_df2 = pd.concat(
            [df2[df2["person_id"] == person_id] for person_id in sampled_persons],
            ignore_index=True
        )

        if 'y_pred' not in boot_df1.columns:
            y_test1 = boot_df1.to_numpy()[:, :3].astype(int)
            y_pred1 = boot_df1.to_numpy()[:, 3:6]
            y_test2 = boot_df2.to_numpy()[:, :3].astype(int)
            y_pred2 = boot_df2.to_numpy()[:, 3:6]
            class_auc1 = roc_auc_score(y_test1, y_pred1, average=None, multi_class='ovr')
            class_auc2 = roc_auc_score(y_test2, y_pred2, average=None, multi_class='ovr')
            d_auc_1.append(class_auc1[0]-class_auc2[0])
            d_auc_0.append(class_auc1[1]-class_auc2[1])
            d_auc_2.append(class_auc1[2]-class_auc2[2])

        else:
            y_test1 = boot_df1['y_test']
            y_pred1 = boot_df1['y_pred']
            y_test2 = boot_df2['y_test']
            y_pred2 = boot_df2['y_pred']

        auc1 = roc_auc_score(y_test1, y_pred1, multi_class='ovr', average='macro')
        auc2 = roc_auc_score(y_test2, y_pred2, multi_class='ovr', average='macro')
        d_auc.append(auc1-auc2)

    # return  acc_scores, auc_scores
    return d_auc, d_auc_0, d_auc_1, d_auc_2


def CI95(scores):
    mean_auc = np.mean(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return mean_auc, lower, upper

# %%
# if 'ensemble_results.csv' in files:
#     df_plot = pd.read_csv(os.path.join(prob_root, 'ensemble_results.csv'))
# else:
df_result = pd.DataFrame(columns=['model', 'mode', 'd_auc_macro', 'd_auc_0', 'd_auc_1', 'd_auc_2'])
# df_plot = pd.DataFrame(columns=['model', 'mode', 'mean_auc', 'lower_auc', 'upper_auc'])

for pair in tqdm(pairs):
    file1, file2 = pair
    df1 = pd.read_csv(os.path.join(prob_root, file1))
    df2 = pd.read_csv(os.path.join(prob_root, file2))
    model, mode1 = metadata_from_filename(file1)
    _, mode2 = metadata_from_filename(file2)
    mode = f"{mode1} vs {mode2}"
    d_auc_scores, d_auc_0_scores, d_auc_1_scores, d_auc_2_scores = bootstrap_ensemble(df1, df2)
    mean_d_auc, lower_d_auc, upper_d_auc = CI95(d_auc_scores)
    if 'y_pred' not in df1.columns:
        mean_d_auc_0, lower_d_auc_0, upper_d_auc_0 = CI95(d_auc_0_scores)
        mean_d_auc_1, lower_d_auc_1, upper_d_auc_1 = CI95(d_auc_1_scores)
        mean_d_auc_2, lower_d_auc_2, upper_d_auc_2 = CI95(d_auc_2_scores)
        new_results_row = {
            'model': model,
            'mode': mode,
            'd_auc_macro': f'{mean_d_auc:.2f} [{lower_d_auc:.2f}, {upper_d_auc:.2f}]',
            'd_auc_0': f'{mean_d_auc_0:.2f} [{lower_d_auc_0:.2f}, {upper_d_auc_0:.2f}]',
            'd_auc_1': f'{mean_d_auc_1:.2f} [{lower_d_auc_1:.2f}, {upper_d_auc_1:.2f}]',
            'd_auc_2': f'{mean_d_auc_2:.2f} [{lower_d_auc_2:.2f}, {upper_d_auc_2:.2f}]'
        }
    else:
        new_results_row = {
            'model': model,
            'mode': mode,
            'd_auc_macro': f'{mean_d_auc:.2f} [{lower_d_auc:.2f}, {upper_d_auc:.2f}]'
        }
    df_result = pd.concat([df_result, pd.DataFrame([new_results_row])], ignore_index=True)
        
        
        
    
#%%
df_result.to_csv(os.path.join(prob_root, 'summary', 'AUROC_diff_mode_results_per_person.csv'), index=False)
gc.collect()
#%%