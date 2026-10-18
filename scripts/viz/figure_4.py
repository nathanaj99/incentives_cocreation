"""
Figure 4: AI-vs-SELF effect sizes stratified by Low / High AI adoption.

Only the AI groups are split by adoption; each AI stratum is compared to
the full SELF group with the same incentive. Stars use unadjusted one-sided
jackknife p-values.

Output: figs/figure_4.png
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import (add_block_labels, add_metric_split, annotate_stars,
                    compute_effects_by_adoption, effect_matrix, load_Y_with_reliance,
                    make_muted_diverging_cmap, pretty_map, save_fig, set_style,
                    style_heatmap_grid)

set_style()

COLS = ("DIV", "CONV")
COL_TITLES = {"DIV": "AI-O − S-O", "CONV": "AI-T − S-T"}


def draw_heatmap_row(fig, subplot_spec, res_low, res_high, y_cols, panel_label,
                     vmin=-0.05, vmax=0.05, annot_fontsize=18, xtick_fontsize=20,
                     ytick_fontsize=20, title_fontsize=22, wspace=0.10, show_row_labels=True):
    """One study: one heatmap per contrast, columns = Low / High adoption."""
    metric_names = [pretty_map.get(k, k.replace("_", " ")) for k in y_cols]
    cmap = make_muted_diverging_cmap()
    inner = subplot_spec.subgridspec(1, len(COLS), wspace=wspace)
    axes = [fig.add_subplot(inner[0, j]) for j in range(len(COLS))]

    images = []
    for ax, comp in zip(axes, COLS):
        M_low, A_low = effect_matrix(res_low, y_cols, [comp])
        M_high, A_high = effect_matrix(res_high, y_cols, [comp])
        M = np.hstack([M_low, M_high])
        A = np.hstack([A_low, A_high])

        images.append(ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low", "High"], rotation=0, ha="center",
                           fontfamily="serif", fontsize=xtick_fontsize)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_yticks(range(len(metric_names)))
        if ax is axes[0] and show_row_labels:
            ax.set_yticklabels(metric_names, fontsize=ytick_fontsize)
        else:
            ax.set_yticklabels([])

        style_heatmap_grid(ax, len(y_cols), 2)
        add_metric_split(ax, y_cols, x_right=1.5)
        ax.set_title(COL_TITLES[comp], fontsize=title_fontsize, pad=12, fontweight="bold")
        annotate_stars(ax, A, fontsize=annot_fontsize)

    if show_row_labels:
        add_block_labels(axes[0], y_cols, x=-0.63, fontsize=20)

    bbox = axes[0].get_position()
    fig.text(bbox.x0 - 0.06, bbox.y1 + 0.077, panel_label, fontsize=24, fontweight="bold",
             ha="left", va="bottom")
    return axes, images


# ============================================================
# EFFECTS (median split computed separately within each study)
# ============================================================

Y1, y_cols1 = load_Y_with_reliance(1)
Y2, y_cols2 = load_Y_with_reliance(2)
assert y_cols1 == y_cols2, "Study 1 and Study 2 metric columns differ."

res_low1, res_high1 = (compute_effects_by_adoption(Y1, y_cols1, lvl) for lvl in ("Low", "High"))
res_low2, res_high2 = (compute_effects_by_adoption(Y2, y_cols2, lvl) for lvl in ("Low", "High"))

# ============================================================
# FIGURE: Study 1 left, Study 2 right
# ============================================================

fig = plt.figure(figsize=(22, 9))
outer = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05)

axes1, images1 = draw_heatmap_row(fig, outer[0], res_low1, res_high1, y_cols1,
                                  "(a) Study 1", wspace=0.05, show_row_labels=True)
axes2, _ = draw_heatmap_row(fig, outer[1], res_low2, res_high2, y_cols2,
                            "(b) Study 2", wspace=0.10, show_row_labels=False)

fig.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.08)

# Separator between studies
x_sep = (axes1[-1].get_position().x1 + axes2[0].get_position().x0) / 2
fig.add_artist(Line2D([x_sep, x_sep], [0.08, 0.9], transform=fig.transFigure,
                      color="black", linewidth=1.5, alpha=0.5))

cbar = fig.colorbar(images1[0], cax=fig.add_axes([0.93, 0.14, 0.012, 0.72]), extend="max")
cbar.ax.set_title("Effect size", pad=8, fontsize=18)

save_fig(fig, "figure_4", dpi=300)
