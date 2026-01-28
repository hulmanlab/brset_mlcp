#%%
import os
import pandas as pd
from functools import reduce

#%%

binary = ['BRSET_TL_b', 'mBRSET_EX_b', 'mBRSET_TL_b']
ternary = ['BRSET_TL', 'mBRSET_EXEVAL', 'mBRSET_TL']

items = ['AUROC', 'ECE', 'Brier']
diff_items = ['AUROC_diff', 'Brier_diff', 'ECE_diff']
diff_mode_items = ['AUROC_diff_mode', 'Brier_diff_mode', 'ECE_diff_mode']
# base_dir = f"/home/livieymli/brset_analysis/BRSET/output/predicted_probabilities/{binary[0]}/summary"

# %%
def load_and_merge(base_dir, items, suffix=""):
    dfs = []

    for item in items:
        df = pd.read_csv(os.path.join(base_dir, f"{item}_results.csv"))
        df = df[~df['model'].str.contains(r'ConvNext|ResNet', case=False, regex=True)]
        df = df.sort_values(['model', 'mode'], ascending=[True, False])

        metric = df.iloc[:, 2].rename(item)
        dfs.append(
            pd.concat([df[['model', 'mode']], metric], axis=1)
        )

    return reduce(
        lambda left, right: left.merge(
            right, on=['model', 'mode'], how='outer'
        ),
        dfs
    )
# %%
# for folder in binary:
for folder in ternary:
    
    base_dir = f"/home/livieymli/brset_analysis/BRSET/output/predicted_probabilities/{folder}/summary"
    if '_b' not in folder:
        items = ['AUROC', 'PDI', 'ECE', 'Brier']
    final_df = load_and_merge(base_dir, items).sort_values(['model', 'mode'], ascending=[True, False])
    final_diff_df = load_and_merge(base_dir, diff_items).sort_values(['model', 'mode'], ascending=[True, False])
    final_diff_mode_df = load_and_merge(base_dir, diff_mode_items).sort_values(['model', 'mode'], ascending=[True, False])
    if '_b' in folder:
        cm_df = pd.read_csv(os.path.join(base_dir, 'cm_40.csv')).sort_values(['model'], ascending=True)
        int_df = pd.read_csv(os.path.join(base_dir, 'calibration_intercept&slope_results.csv')).sort_values(['model', 'mode'], ascending=[True, False])[['model', 'mode', 'intercept', 'slope']]
    int_df = pd.read_csv(os.path.join(base_dir, 'calibration_intercept&slope_results.csv'))
   
    out_path = os.path.join(base_dir, "metrics_summary.xlsx")

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        final_df.to_excel(writer, sheet_name="metrics", index=False)
        final_diff_df.to_excel(writer, sheet_name="diff_metrics", index=False)
        final_diff_mode_df.to_excel(writer, sheet_name = 'diff mode metrics', index=False)
        if '_b' in folder:
            cm_df.to_excel(writer, sheet_name='cm_40', index=False)
        int_df.to_excel(writer, sheet_name='intercept&slope', index=False)
# %%
