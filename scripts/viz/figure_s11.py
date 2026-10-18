"""
Figure S11: brainstorm baseline (Study 1).

  top    -- similarity of the pre-writing brainstorms, by group, for every
            metric (violin + points + mean +/- SE)
  bottom -- baseline-adjusted change: difference-in-differences
            (final - brainstorm) between groups, with a leave-one-
            participant-out jackknife that drops the same participant from
            both the brainstorm and final U-statistics. Stars use
            BH-adjusted one-sided p-values.

Outputs: figs/figure_s11_top.png, figs/figure_s11_bottom.png
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import (FINAL_COMPARISONS, add_bh, add_block_labels, add_metric_split,
                    alternative_for, annotate_stars, effect_matrix, jackknife_var,
                    make_muted_diverging_cmap, metric_cols, plot_similarity_distribution,
                    pretty_map, processed, relabel_group_legend, save_fig, set_style,
                    style_heatmap_grid, u_loo, wrap_labels)

set_style()

STUDY = 1
Y = pd.read_csv(processed(STUDY, "Y.csv"))
Y_brainstorm = pd.read_csv(processed(STUDY, "Y_brainstorm.csv"))
y_cols = metric_cols(Y)

# ============================================================
# TOP: brainstorm similarity by group
# ============================================================

fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(24, 8))
for idx, col in enumerate(y_cols):
    a = ax[idx // 6][idx % 6]
    plot_similarity_distribution(a, Y_brainstorm[["participant_id", "group_key", col]], col,
                                 x="group_key", dodge=False, width=0.5)
    a.set_ylabel("Similarity" if idx % 6 == 0 else "", fontsize=16)
    a.set_title(pretty_map[col], fontsize=18)
    a.grid(True, axis="y")
    if idx == 5:  # legend on the top-right panel only
        relabel_group_legend(a)
    else:
        a.legend_.remove()
    a.set_xlabel("")
    a.set_xticklabels("")

plt.subplots_adjust(wspace=0.25, hspace=0.25)
save_fig(fig, "figure_s11_top")

# ============================================================
# BOTTOM: difference-in-differences heatmap
# ============================================================

def jackknife_did(group1, group2, y_col, alternative):
    """DiD = (post1 - pre1) - (post2 - pre2); pre/post matched by participant."""

    def group_change(group):
        post = Y.loc[Y["group_key"] == group, ["participant_id", y_col]].rename(columns={y_col: "post"})
        pre = Y_brainstorm.loc[Y_brainstorm["group_key"] == group, ["participant_id", y_col]].rename(columns={y_col: "pre"})
        df = post.merge(pre, on="participant_id", how="inner", validate="one_to_one").dropna()
        if len(df) <= 2:
            raise ValueError("Need at least 3 matched participants per group.")
        post, pre = df["post"].to_numpy(float), df["pre"].to_numpy(float)
        # Leave the SAME participant out of both statistics
        change_loo = u_loo(post) - u_loo(pre)
        return post.mean() - pre.mean(), jackknife_var(change_loo)

    c1, v1 = group_change(group1)
    c2, v2 = group_change(group2)
    did = c1 - c2
    z = did / np.sqrt(v1 + v2)  # independent groups
    p = stats.norm.sf(z) if alternative == "greater" else stats.norm.cdf(z)
    return {"effect": did, "p_one_sided": p}


res_did = {
    m: {name: jackknife_did(g1, g2, m, alternative_for(m)) for name, (g1, g2) in FINAL_COMPARISONS.items()}
    for m in y_cols
}
add_bh(res_did, y_cols, FINAL_COMPARISONS)

COLS = ("DIV", "CONV", "AI", "SELF")
LABELS = ("(AI - SELF) O", "(AI - SELF) T", "AI (T−O)", "SELF (T−O)")

# This panel was designed at a smaller base font size
with matplotlib.rc_context({"font.size": 15}):
    M, A = effect_matrix(res_did, y_cols, COLS, p_key="p_one_sided_bh")
    fig, ax = plt.subplots(figsize=(12, 7.5))
    im = ax.imshow(M, aspect="auto", cmap=make_muted_diverging_cmap(), vmin=-0.05, vmax=0.05)

    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(wrap_labels(LABELS, width=16), rotation=0, ha="center",
                       fontweight="bold", fontsize=20)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_yticks(range(len(y_cols)), [pretty_map[m] for m in y_cols])

    style_heatmap_grid(ax, len(y_cols), len(COLS))
    add_metric_split(ax, y_cols)
    add_block_labels(ax, y_cols, x=-0.28, fontsize=20)
    annotate_stars(ax, A, fontsize=20)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, extend="both")
    cbar.ax.set_title("Effect\nsize", pad=8, fontsize=18)
    ax.set_title("Baseline-adjusted change in similarity", pad=15)
    fig.tight_layout()
    save_fig(fig, "figure_s11_bottom")
