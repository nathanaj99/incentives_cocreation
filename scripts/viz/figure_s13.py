"""
Figure S13: similarity trajectories while writing (snapshots every 30 s,
texts >= 200 words), by group, for every metric. top = Study 1,
bottom = Study 2. Zeros in the snapshot files mark missing values.

Outputs: figs/figure_s13_top.png, figs/figure_s13_bottom.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import (GROUP_ORDER, TRAJECTORY_MAX_MINUTES, load_trajectory, metric_cols,
                    palette_map, pretty_map, processed, relabel_group_legend, save_fig,
                    set_style)

# Small multiples: designed at a smaller base font so ticks don't collide
set_style(font_size=15)


def plot_study(study, name, xlim=None):
    y_cols = metric_cols(pd.read_csv(processed(study, "Y.csv")))
    fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(24, 8), sharex=True)

    for idx, metric in enumerate(y_cols):
        a = ax[idx // 6][idx % 6]
        df_long = load_trajectory(study, metric)
        df_long["value"] = df_long["value"].replace(0, np.nan)
        sns.lineplot(x="minutes", y="value", data=df_long[df_long["minutes"] < 26],
                     hue="group_key", errorbar="se", hue_order=GROUP_ORDER,
                     palette=palette_map, ax=a, linewidth=2)
        a.set_title(pretty_map[metric], fontsize=18)
        if idx == 0:
            relabel_group_legend(a)
        else:
            a.legend_.remove()
        a.set_ylabel("Similarity" if idx % 6 == 0 else "", fontsize=16)
        a.grid(True, axis="y")
        # x-label once, centered under the bottom row
        a.set_xlabel("Minutes spent writing" if idx in (8, 9) else "", fontsize=16)
        if metric == "self_bleu":
            a.tick_params(axis="y", labelsize=12.5)
        if xlim is not None:
            a.set_xlim(*xlim)

    plt.subplots_adjust(wspace=0.25)
    save_fig(fig, name)


plot_study(1, "figure_s13_top")
plot_study(2, "figure_s13_bottom", xlim=(0, TRAJECTORY_MAX_MINUTES[2]))
