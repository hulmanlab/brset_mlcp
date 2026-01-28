import argparse
import gc
import pandas as pd
import os
import matplotlib.pyplot as plt
# 
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prob_root", default="BRSET_TL_b", help="Path to predicted probabilities directory")
path = parser.parse_args().prob_root

DATASET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prob_root = os.path.join(DATASET, "output/predicted_probabilities", path)
df_plot = pd.read_csv(os.path.join(prob_root, 'summary', 'Brier_results.csv'))
# Create a label combining model and mode for y-axis
df_plot = df_plot[~df_plot['model'].str.contains('resnet|convnext', case=False)]
df_plot['label'] = df_plot['model'] + ' - ' + df_plot['mode']
df_plot[['mean_brier', 'lower_brier', 'upper_brier']] = df_plot[['mean_brier', 'lower_brier', 'upper_brier']].astype(float)
# Assign color: green if 'Head' in mode, else blue
df_plot['color'] = df_plot['mode'].apply(lambda x: '#E1BE6A' if 'Head' in x else '#40B0A6')
# Sort by model and mode for better grouping
df_plot_sorted = df_plot.sort_values(['model', 'mode'], ascending=[False, True])

fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.set_ymargin(0.15)

# --- ECE plot ---
for _, row in df_plot_sorted.iterrows():
    ax.errorbar(
        row['mean_brier'],
        row['label'],
        xerr=[[row['mean_brier'] - row['lower_brier']], [row['upper_brier'] - row['mean_brier']]],
        fmt='o',
        ls='-.',
        color=row['color'],
        ecolor='gray',
        capsize=7
    )
    ax.text(
        0.51,
        row['label'],
        f"{row['mean_brier']:.2f} [{row['lower_brier']:.2f}, {row['upper_brier']:.2f}]",
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
ax.set_title("Brier Score", fontsize=14, pad=10)
# Custom legend
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color='#E1BE6A', label='Head fine-tune'),
    mpatches.Patch(color='#40B0A6', label='Full fine-tune')
]
ax.legend(handles=legend_handles, title='Training mode', bbox_to_anchor=(0.04, 0.96), loc='upper left', borderaxespad=0., ncol=1, frameon=True)

plt.tight_layout(rect=[0.05, 0, 1, 1])
plt.show()
fig.savefig(os.path.join(prob_root, 'summary', 'Brier_forest_plot.png'), bbox_inches='tight', dpi=300)

gc.collect()

# %%
