# %%
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from itertools import combinations
from tqdm import tqdm
from torch import tensor
from torchmetrics.classification import MulticlassCalibrationError


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
    y_true = np.argmax(y_true, axis=1)  # Convert one-hot to class labels
    y_score = arr[:, 3:]
    y_score = np.where(y_score == 0, 0.000001, y_score)
    y_score = np.where(y_score == 1, 0.999999, y_score) # shape: (n, 4)
    
    # Run bootstrap
    ece_score = []


    for _ in range(n_iterations):
        # Generate a random sample of indices
        y_resample, y_score_resample = resample(y_true, y_score, replace=True)
        preds = tensor(y_score_resample)
        target = tensor(y_resample)
        metric = MulticlassCalibrationError(num_classes=3, n_bins=10, norm='l1')
        ece = metric(preds, target).item()

        ece_score.append(ece)
        
        
    # return  acc_scores, auc_scores
    return ece_score

def CI95(scores):
    mean = np.mean(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return mean, lower, upper

# %%
df_result = pd.DataFrame(columns=['model', 'mode', 'ece'])
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
            ece_score= bootstrap_ensemble(df_sub)

            mean_ece, lower_ece, upper_ece = CI95(ece_score)

            new_results_row = {
                'model': model,
                'mode': mode,
                'ece': f'{mean_ece:.2f} [{lower_ece:.2f}-{upper_ece:.2f}]'
            }
            
            global df_result
            df_result = pd.concat([df_result, pd.DataFrame([new_results_row])], ignore_index=True)
        
        if 'y_camera' in df.columns:
            df = df.drop(columns=['y_camera'])
        process_and_append(df)
    
print(df_result)
df_result.to_csv(os.path.join(prob_root, 'ece_results.csv'), index=False)


