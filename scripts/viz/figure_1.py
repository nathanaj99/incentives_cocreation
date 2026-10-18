"""
Figure 1 (Study 1, all-MiniLM-L12-v2):
  (a) similarity of the first AI draft vs. the final submission, by group
  (b) similarity trajectory while writing, with AI request timing below
  (c) 2-D PCA of final submissions and AI drafts, with two common
      AI-draft openings annotated

Output: figs/figure_1.png
Note: panel (c) re-embeds all texts with sentence-transformers.
"""

import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import (GROUP_ORDER, build_text_corpus, embed_and_pca, load_trajectory,
                    palette_map, pca_stage_handles, plot_request_kde,
                    plot_similarity_distribution, processed, relabel_group_legend,
                    save_fig, scatter_pca, set_style)

set_style()

STUDY = 1
METRIC = "all-MiniLM-L12-v2"

# ============================================================
# DATA
# ============================================================

# (a) final submissions + first AI drafts
Y = pd.read_csv(processed(STUDY, "Y.csv"))
Y_drafts = pd.read_csv(processed(STUDY, "Y_first_draft.csv"))
Y["type"] = "Final Submission"
Y_drafts["type"] = "AI 1st Draft"
cols_subset = ["participant_id", "group_key", "type", METRIC]
merged = pd.concat([Y[cols_subset], Y_drafts[cols_subset]])

# (b) trajectories + AI request times
df_long = load_trajectory(STUDY, METRIC)
ai_requests = pd.read_csv(processed(STUDY, "ai_requests_labeled.csv"))

# (c) texts -> embeddings -> PCA
texts_list, group, stage, first_drafts = build_text_corpus(STUDY)
X2 = embed_and_pca(texts_list)

# Flags for the two annotated AI-draft openings (first drafts only).
# "I am [character]..." drafts were identified by hand (indices into the
# list of non-missing first drafts); "I woke up..." drafts by their opening.
IAM_INDEX = [0, 1, 4, 5, 8, 14, 36, 38, 45, 72, 75, 77, 80, 88, 85]
is_draft = stage == "draft"
iam = np.zeros(len(texts_list), dtype=bool)
iwoke = np.zeros(len(texts_list), dtype=bool)
iam[np.where(is_draft)[0][IAM_INDEX]] = True
iwoke[is_draft] = [("I woke" in t[:100]) or ("I wake" in t[:100]) for t in first_drafts]

# ============================================================
# LAYOUT
# ============================================================

fig = plt.figure(figsize=(30, 7))

# Middle column is stacked: (b) on top, request KDE below
gs = gridspec.GridSpec(
    2, 5,
    width_ratios=[2, 0.6, 3.5, 0.6, 4],
    height_ratios=[1.5, 0.5],
    hspace=0.05,
    wspace=0.0,
)
ax0 = fig.add_subplot(gs[:, 0])               # (a)
ax1 = fig.add_subplot(gs[0, 2])               # (b)
ax1b = fig.add_subplot(gs[1, 2], sharex=ax1)  # (b) bottom: request KDE
ax2 = fig.add_subplot(gs[:, 4])               # (c)

# ============================================================
# (a) violin distributions + mean +/- SE
# ============================================================

plot_similarity_distribution(ax0, merged, METRIC, x="type",
                             order=["AI 1st Draft", "Final Submission"], point_scale=1.75)
ax0.grid(True, axis="y", alpha=0.3)
ax0.margins(x=0.15)
relabel_group_legend(ax0, fontsize=24)
ax0.set_ylabel("Avg Cosine Similarity", fontsize=24)
ax0.set_xlabel("")
ax0.set_ylim(0.08, 0.52)

# Nudge the left category label right so the two labels don't collide
xticklabels = ax0.get_xticklabels()
xticklabels[0].set_x(1.5)
xticklabels[0].set_ha("right")

# ============================================================
# (b) similarity trajectory
# ============================================================

sns.lineplot(
    x="minutes", y="value", data=df_long[df_long["minutes"] < 26],
    hue="group_key", errorbar="se", hue_order=GROUP_ORDER,
    palette=palette_map, ax=ax1, linewidth=4,
)

# Sample sizes at selected time points (AI = AI-O + AI-T, SELF = SELF-O + SELF-T)
sample_times = [5, 10, 15, 20, 25]
tmp = df_long[df_long["minutes"].isin(sample_times)].dropna(subset=["value"]).copy()
tmp["condition"] = np.where(tmp["group_key"].str.startswith("AI-"), "AI", "SELF")
n_by_time = (
    tmp.groupby(["condition", "minutes"]).size()
    .unstack("minutes")
    .reindex(index=["AI", "SELF"], columns=sample_times)
)

y_ai, y_self = 0.075, 0.025  # rows near the bottom of (b), in axes coords
for t in sample_times:
    for cond, y in [("AI", y_ai), ("SELF", y_self)]:
        ax1.text(t, y, f"{int(n_by_time.loc[cond, t])}", transform=ax1.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=14)
ax1.text(0.15, y_ai, r"$n_{\mathrm{AI}} =$", transform=ax1.transAxes, ha="right", va="bottom", fontsize=14)
ax1.text(0.15, y_self, r"$n_{\mathrm{SELF}} =$", transform=ax1.transAxes, ha="right", va="bottom", fontsize=14)

ax1.set_yticks([0.20, 0.24, 0.28, 0.32])
ax1.grid(True, axis="both")
ax1.set_ylabel("Avg Cosine Similarity")
ax1.set_xlabel("")
ax1.legend_.remove()
ax1.tick_params(axis="x", labelbottom=False)  # shared x is labeled on the KDE axis

# (b) bottom: when editing / drafting requests were made
plot_request_kde(ax1b, ai_requests, xmax=25, bw_method=0.25)
ax1b.set_xlim(0, 25)
ax1b.set_xlabel("Minutes spent writing", fontsize=24)
ax1b.set_ylabel("# Req.")
ax1b.grid(True, axis="both")
ax1b.legend(frameon=False)

# ============================================================
# (c) PCA
# ============================================================

scatter_pca(ax2, X2, group, stage, s=100)

# Drafts beginning with each opening, restricted to the cluster region
pts_iam = X2[iam & (X2[:, 0] > 0) & (X2[:, 1] > 0)]
pts_iwoke = X2[iwoke & (X2[:, 0] > 0) & (X2[:, 1] < 0.1)]


def add_shadow_box(ax, xy_axes, text, fontsize=18):
    """Text box with a drop shadow at axes coordinates."""
    txt = ax.text(
        xy_axes[0], xy_axes[1], text, transform=ax.transAxes, ha="right", va="top",
        fontsize=fontsize, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.4", lw=0.8),
        zorder=30,
    )
    txt.get_bbox_patch().set_path_effects([
        pe.SimplePatchShadow(offset=(2, -2), alpha=0.35, shadow_rgbFace=(0, 0, 0)),
        pe.Normal(),
    ])
    return txt


def point_to_box(ax, pts, anchor_axes, max_arrows=12, arrowstyle="-"):
    """Draw pointer lines from points to an anchor (axes coords)."""
    if len(pts) == 0:
        return
    anchor_data = ax.transData.inverted().transform(ax.transAxes.transform(anchor_axes))
    # Cap the number of pointers: keep the points closest to the anchor
    if len(pts) > max_arrows:
        d = np.sum((pts - anchor_data) ** 2, axis=1)
        pts = pts[np.argsort(d)[:max_arrows]]
    for x, y in pts:
        ax.annotate(
            "", xy=anchor_data, xytext=(x, y),
            arrowprops=dict(arrowstyle=arrowstyle, lw=0.8, color="0.25", alpha=0.7,
                            shrinkA=0, shrinkB=6),
            zorder=20,
        )


add_shadow_box(ax2, (1.05, 1.03), "\"I am [character name]: [description of\npersonality and motivation]…\"", fontsize=20)
add_shadow_box(ax2, (1.1, 0.8), "\"I woke up…\"", fontsize=20)
point_to_box(ax2, pts_iam, (0.90, 0.97), max_arrows=10)
point_to_box(ax2, pts_iwoke, (0.99, 0.75), max_arrows=10)

ax2.set_xlabel("PC1", fontsize=24)
ax2.set_ylabel("PC2", fontsize=24)

# Shrink (c) to leave room on the right for the annotation boxes
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width * 0.72, box.height])
ax2.legend(handles=pca_stage_handles(markersize=12), title=None,
           loc="upper left", bbox_to_anchor=(0.7, 0.3), fontsize=24)

# Panel labels
fig.text(0.08, 0.88, "(a)", fontweight="bold", fontsize=28)
fig.text(0.275, 0.88, "(b)", fontweight="bold", fontsize=28)
fig.text(0.57, 0.88, "(c)", fontweight="bold", fontsize=28)

save_fig(fig, "figure_1")
