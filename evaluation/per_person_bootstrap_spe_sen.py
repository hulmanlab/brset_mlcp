# %%
import argparse
import gc
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix

# %%
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default='mBRSET_EX_b', choices=['BRSET_TL_b', 'mBRSET_EX_b', 'mBRSET_TL_b'])
    parser.add_argument("-r", "--recalibration", type=bool, default=False, help="Whether to use recalibrated probabilities")
    return parser.parse_args()


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


#%%
def bootstrap_cm(df, threshold, n_iterations=1000):
    # Extract person ID from image ID
    if 'person_id' not in df.columns:
        df["person_id"] = df["image_ids"].astype(str).str.split(".").str[0]

    # Optional: store as numeric
    df["person_id"] = pd.to_numeric(df["person_id"])

    # Get unique persons
    persons = df["person_id"].unique()
    n_persons = len(persons)
        
    # Run bootstrap
    sen, spe, ppv, npv = [], [], [], []


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

        y_true = boot_df.iloc[:, 0].astype(int).to_numpy()
        y_score = boot_df.iloc[:, 1].to_numpy()
        y_pred = (y_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sen.append(tp / (tp + fn) if (tp + fn) else np.nan)
        spe.append(tn / (tn + fp) if (tn + fp) else np.nan)
        ppv.append(tp / (tp + fp) if (tp + fp) else np.nan)
        npv.append(tn / (tn + fn) if (tn + fn) else np.nan)
        
    return sen, spe, ppv, npv

def CI95(scores):
    mean = np.mean(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return mean, lower, upper

# %%
def main(path, recalibration):
    DATASET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    prob_root = os.path.join(DATASET, "output/predicted_probabilities", path)

    files = [
        f for f in os.listdir(prob_root)
        if f.startswith('y_')
        and f.endswith('.csv')
        and not any(x in f for x in ('convnextv2', 'resnet', 'dinov2', 'retfound_d2'))
    ]

    if recalibration:
        files = [f for f in files if 'recalib' in f]
    else:
        files = [f for f in files if 'recalib' not in f]

    df_results = pd.DataFrame(
        columns=['threshold','model', 'mode', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Sen_mean', 'Sen_lower', 'Sen_upper', 'Spe_mean', 'Spe_lower', 'Spe_upper', 'PPV_mean', 'PPV_lower', 'PPV_upper', 'NPV_mean', 'NPV_lower', 'NPV_upper']
    )

    for filename in tqdm(files):
        model, mode = metadata_from_filename(filename)
        df = pd.read_csv(os.path.join(prob_root, filename))
        for threshold in [0.1, 0.2, 0.3]:
            sen, spe, ppv, npv = bootstrap_cm(df, threshold)
            mean_sen, lower_sen, upper_sen = CI95(sen)
            mean_spe, lower_spe, upper_spe = CI95(spe)
            mean_ppv, lower_ppv, upper_ppv = CI95(ppv)
            mean_npv, lower_npv, upper_npv = CI95(npv)
            
            new_results_row = {
                    'threshold': threshold,
                    'model': model,
                    'mode': mode,
                    'Sensitivity': f'{mean_sen:.2f} [{lower_sen:.2f}, {upper_sen:.2f}]',
                    'Specificity': f'{mean_spe:.2f} [{lower_spe:.2f}, {upper_spe:.2f}]',
                    'PPV': f'{mean_ppv:.2f} [{lower_ppv:.2f}, {upper_ppv:.2f}]',
                    'NPV': f'{mean_npv:.2f} [{lower_npv:.2f}, {upper_npv:.2f}]',
                    'Sen_mean': mean_sen,
                    'Sen_lower': lower_sen,
                    'Sen_upper': upper_sen,
                    'Spe_mean': mean_spe,
                    'Spe_lower': lower_spe,
                    'Spe_upper': upper_spe,
                    'PPV_mean': mean_ppv,
                    'PPV_lower': lower_ppv,
                    'PPV_upper': upper_ppv,
                    'NPV_mean': mean_npv,
                    'NPV_lower': lower_npv,
                    'NPV_upper': upper_npv
                }
            df_results = pd.concat([df_results, pd.DataFrame([new_results_row])], ignore_index=True)
        
    df_results = df_results.sort_values(['threshold', 'model', 'mode'], ascending=[True, True, False])
    df_results.to_csv(
        os.path.join(prob_root, 'summary', f'cm{'_recalib' if recalibration else ""}_per_person.csv'),
        index=False
    )
    gc.collect()

#%%
if __name__ == "__main__":
    args = get_args()
    main(args.path, args.recalibration)

#%%
