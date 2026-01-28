#%% Probability preprocessing
import os
import pandas as pd


#%%
def prob_rev(df):
    df = df.copy()
    df['y_test'] = 1 - df['y_test']
    df['y_pred'] = 1 - df['y_pred']
    return df
# %%
DATASET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
paths = ['BRSET_TL_b', 'mBRSET_EX_b', 'mBRSET_TL_b']

for path in paths:
    prob_root = os.path.join(DATASET, "output/predicted_probabilities", path) 
    files = sorted(os.listdir(prob_root))
    for filename in files:
        if filename.endswith(".csv") and filename.startswith("y_"):
            # print(f"File: {filename}")
            df = pd.read_csv(os.path.join(prob_root, filename))
            df = prob_rev(df)
            df.to_csv(os.path.join(prob_root, filename), index=False)



# %%
