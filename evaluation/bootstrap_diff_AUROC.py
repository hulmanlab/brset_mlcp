# %%
import argparse
import gc
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from itertools import combinations
from tqdm import tqdm


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
eval_lists = [f for f in files if 'eval' in f]
full_lists = [f for f in files if 'eval' not in f]
eval_pairs = list(combinations(eval_lists, 2))
full_pairs = list(combinations(full_lists, 2))

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
    for df in (df1, df2):
        df.drop(columns=['y_camera'], inplace=True, errors='ignore')
    arr1 = df1.to_numpy()
    arr2 = df2.to_numpy()
    if arr1.shape[1] > 3:
        y_true = np.array(arr1[:, :3].astype(int))
        y_score1 = arr1[:, 3:]
    else:
        y_true = np.array(arr1[:, 0].astype(int))
        y_score1 = arr1[:, 1]
    y_score1 = np.where(y_score1 == 0, 0.000001, y_score1)
    y_score1 = np.where(y_score1 == 1, 0.999999, y_score1) # shape: (n, 4)
    if arr2.shape[1] > 3:
        y_score2 = arr2[:, 3:]
    else:
        y_score2 = arr2[:, 1]
    y_score2 = np.where(y_score2 == 0, 0.000001, y_score2)
    y_score2 = np.where(y_score2 == 1, 0.999999, y_score2) # shape: (n, 4)

    # Run bootstrap
    # acc_scores = []
    d_auc = []
    d_auc_0 = []
    d_auc_1 = []
    d_auc_2 = []


    for _ in range(n_iterations):
        # Generate a random sample of indices
        random_state = np.random.randint(0, 1e6)
        y_resample, y_score_resample1 = resample(y_true, y_score1, replace=True, random_state=random_state)
        _, y_score_resample2 = resample(y_true, y_score2, replace=True, random_state=random_state)
        # acc = accuracy_score(np.argmax(y_resample,axis=1), np.argmax(y_score_resample, axis=1))
        # acc_scores.append(acc)
        auc1 = roc_auc_score(y_resample, y_score_resample1, multi_class='ovr', average='macro')
        auc2 = roc_auc_score(y_resample, y_score_resample2, multi_class='ovr', average='macro')
        d_auc.append(auc1-auc2)
        

        # Compute class-wise AUCs
        if arr1.shape[1] > 3:
            class_auc1 = roc_auc_score(y_resample, y_score_resample1, average=None, multi_class='ovr')
            class_auc2 = roc_auc_score(y_resample, y_score_resample2, average=None, multi_class='ovr')
            d_auc_0.append(class_auc1[0] - class_auc2[0])
            d_auc_1.append(class_auc1[1] - class_auc2[1])
            d_auc_2.append(class_auc1[2] - class_auc2[2])
        else:
            d_auc_0.append(np.nan)
            d_auc_1.append(np.nan)
            d_auc_2.append(np.nan)

        
        
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

for pair in tqdm(eval_pairs + full_pairs):
    file1, file2 = pair
    df1 = pd.read_csv(os.path.join(prob_root, file1))
    df2 = pd.read_csv(os.path.join(prob_root, file2))
    model1, mode = metadata_from_filename(file1)
    model2, _ = metadata_from_filename(file2)
    model = f"{model1} vs {model2}"
    d_auc_scores, d_auc_0_scores, d_auc_1_scores, d_auc_2_scores = bootstrap_ensemble(df1, df2)
    mean_d_auc, lower_d_auc, upper_d_auc = CI95(d_auc_scores)
    if df1.shape[1] > 3:
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
df_result.to_csv(os.path.join(prob_root, 'summary', 'AUROC_diff_results.csv'), index=False)
gc.collect()