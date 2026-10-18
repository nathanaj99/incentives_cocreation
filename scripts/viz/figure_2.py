"""
Figure 2: effect-size heatmaps for both studies.

For each study and similarity metric (rows):
  left block  -- final-submission contrasts: (AI-SELF) O, (AI-SELF) T,
                 AI (T-O), SELF (T-O)
  right block -- first AI draft minus final submission, within AI-O / AI-T
Stars use unadjusted one-sided jackknife p-values.

Output: figs/figure_2.png
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import (FINAL_COMPARISONS, add_block_labels, add_metric_split, annotate_stars,
                    effect_matrix, load_Y_with_first_draft, make_muted_diverging_cmap,
                    pretty_map, run_comparisons, save_fig, set_style, style_heatmap_grid,
                    wrap_labels)

set_style()

# First AI draft vs. final submission, within each AI group
FIRST_VS_FINAL = {
    "AI-DIV": ("AI-DIV_first", "AI-DIV"),
    "AI-CONV": ("AI-CONV_first", "AI-CONV"),
}


def analyze_study(study):
    Y, y_cols = load_Y_with_first_draft(study)
    return run_comparisons(Y, y_cols, FINAL_COMPARISONS), run_comparisons(Y, y_cols, FIRST_VS_FINAL), y_cols


def plot_effect_heatmaps_on_spec(
    fig, subplot_spec, res, res_first_draft, metric_keys, study_label,
    p_key="p_one_sided",
    finals_cols=("DIV", "CONV", "AI", "SELF"),
    first_cols=("AI-DIV", "AI-CONV"),
    finals_labels=("(AI - SELF) O", "(AI - SELF) T", "AI (T−O)", "SELF (T−O)"),
    first_labels=("AI-O", "AI-T"),
    vmin=-0.05, vmax=0.05,
    annot_fontsize=20, xtick_fontsize=20, ytick_fontsize=18,
    gap_width=0.10, wspace=0.03,
    show_row_labels=True,
):
    """Draw one study (final block | gap | first-draft block) into subplot_spec."""
    metric_names = [pretty_map.get(k, k.replace("_", " ")) for k in metric_keys]
    M_final, A_final = effect_matrix(res, metric_keys, finals_cols, p_key)
    M_first, A_first = effect_matrix(res_first_draft, metric_keys, first_cols, p_key)
    cmap = make_muted_diverging_cmap()

    inner = subplot_spec.subgridspec(1, 3, width_ratios=[len(finals_cols), gap_width, len(first_cols)],
                                     wspace=wspace)
    axL = fig.add_subplot(inner[0, 0])
    fig.add_subplot(inner[0, 1]).axis("off")
    axR = fig.add_subplot(inner[0, 2])

    imL = axL.imshow(M_final, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axR.imshow(M_first, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Column labels on top
    for ax, labels in [(axL, finals_labels), (axR, first_labels)]:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(wrap_labels(labels, width=16), rotation=0, ha="center",
                           fontweight="bold", fontfamily="serif", fontsize=xtick_fontsize)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_yticks(range(len(metric_names)))

    # Row labels only on the left-most block of Study 1
    if show_row_labels:
        axL.set_yticklabels(metric_names, fontsize=ytick_fontsize)
    else:
        axL.set_yticklabels([])
        axL.tick_params(axis="y", left=False)
    axR.set_yticklabels([])
    axR.tick_params(axis="y", left=False)

    for ax, A, ncol in [(axL, A_final, len(finals_cols)), (axR, A_first, len(first_cols))]:
        style_heatmap_grid(ax, len(metric_keys), ncol)
        add_metric_split(ax, metric_keys)
        annotate_stars(ax, A, fontsize=annot_fontsize)
    if show_row_labels:
        add_block_labels(axL, metric_keys, x=-0.32, fontsize=20)

    axL.set_title("Final submission", pad=12, fontsize=20)
    axR.set_title("First draft − Final submission", pad=12, fontsize=20)

    bbox = axL.get_position()
    fig.text(bbox.x0, bbox.y1 + 0.09, study_label, fontsize=24, fontweight="bold",
             ha="left", va="bottom")
    return axL, axR, imL


res1, res_first1, y_cols1 = analyze_study(1)
res2, res_first2, y_cols2 = analyze_study(2)

# ============================================================
# FIGURE: Study 1 | Study 2 | colorbar
# ============================================================

fig = plt.figure(figsize=(27, 9))
outer = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.025], wspace=0.04,
                         left=0.08, right=0.96, top=0.88, bottom=0.08)

ax1L, ax1R, im1 = plot_effect_heatmaps_on_spec(
    fig, outer[0, 0], res1, res_first1, y_cols1,
    study_label="(a) Study 1: Short Story (n=200)", wspace=0.02, show_row_labels=True,
)
ax2L, ax2R, _ = plot_effect_heatmaps_on_spec(
    fig, outer[0, 1], res2, res_first2, y_cols2,
    study_label="(b) Study 2: Product Pitch (n=400)", wspace=0.02, show_row_labels=False,
)

# Shared colorbar
cbar = fig.colorbar(im1, cax=fig.add_subplot(outer[0, 2]), extend="max")
cbar.ax.set_title("Effect size", pad=8, fontsize=19)
cbar.ax.tick_params(labelsize=15)

# Vertical separator between the studies
x_sep = (ax1R.get_position().x1 + ax2L.get_position().x0) / 2
fig.add_artist(Line2D([x_sep, x_sep], [ax1L.get_position().y0, ax1L.get_position().y1 + 0.1],
                      transform=fig.transFigure, color="black", linewidth=2, alpha=0.5))

save_fig(fig, "figure_2", dpi=300)
