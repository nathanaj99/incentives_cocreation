"""
Figure S10: robustness of the AI adoption measure -- correlations between
alternative adoption-rate definitions (n-gram windows and embedding-based),
(a) Study 1, (b) Study 2.

Output: figs/figure_s10.png
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import processed, save_fig, set_style

set_style()

RENAME = {
    "adoption_5_10": "n = 5-10",
    "adoption_10": "n = 10",
    "adoption_total": "n = 1-10",
    "adoption_3": "n = 3",
    "adoption_embedding": "embedding",
}
COLS = list(RENAME.values())

fig = plt.figure(figsize=(13, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.35)
axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
cbar_ax = fig.add_subplot(gs[2])

for i, (ax, study, title) in enumerate(zip(axes, (1, 2), ["(a) Study 1", "(b) Study 2"])):
    corr = pd.read_csv(processed(study, "ai_reliance.csv")).rename(columns=RENAME)[COLS].corr()
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0,
        square=True, cbar=(i == 1), cbar_ax=cbar_ax if i == 1 else None,
        annot_kws={"fontsize": 11},
    )
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelrotation=45, labelsize=11)
    ax.tick_params(axis="y", labelrotation=0, labelsize=11)

save_fig(fig, "figure_s10", dpi=300)
