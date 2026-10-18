"""
Figure S15: Figure 1 for Study 2 (all-MiniLM-L12-v2).

Differences from Figure 1: no jittered points in (a), no sample-size row
or opening annotations, a 12-minute time axis, and (c) is computed after
removing every word containing "snap" (the product name, which otherwise
dominates the embedding geometry) before re-embedding.

Output: figs/figure_s15.png
Note: panel (c) re-embeds all texts with sentence-transformers.
"""

import re

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import (GROUP_ORDER, TRAJECTORY_MAX_MINUTES, build_text_corpus, embed_and_pca,
                    load_trajectory, palette_map, pca_stage_handles, plot_request_kde,
                    plot_similarity_distribution, processed, relabel_group_legend, save_fig,
                    scatter_pca, set_style)

set_style()

STUDY = 2
METRIC = "all-MiniLM-L12-v2"
XMAX = TRAJECTORY_MAX_MINUTES[STUDY]

# ============================================================
# DATA
# ============================================================

Y = pd.read_csv(processed(STUDY, "Y.csv"))
Y_drafts = pd.read_csv(processed(STUDY, "Y_first_draft.csv"))
Y["type"] = "Final Submission"
Y_drafts["type"] = "AI 1st Draft"
cols_subset = ["participant_id", "group_key", "type", METRIC]
merged = pd.concat([Y[cols_subset], Y_drafts[cols_subset]])

df_long = load_trajectory(STUDY, METRIC)
ai_requests = pd.read_csv(processed(STUDY, "ai_requests_labeled.csv"))

texts_list, group, stage, _ = build_text_corpus(STUDY)
texts_no_snap = [re.sub(r"\s+", " ", re.sub(r"\b\w*snap\w*\b", "", t, flags=re.IGNORECASE)).strip()
                 for t in texts_list]
X2 = embed_and_pca(texts_no_snap)

# ============================================================
# LAYOUT (same as Figure 1)
# ============================================================

fig = plt.figure(figsize=(30, 7))
gs = gridspec.GridSpec(2, 5, width_ratios=[2, 0.6, 3.5, 0.6, 4], height_ratios=[1.5, 0.5],
                       hspace=0.05, wspace=0.0)
ax0 = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 2])
ax1b = fig.add_subplot(gs[1, 2], sharex=ax1)
ax2 = fig.add_subplot(gs[:, 4])

# (a)
plot_similarity_distribution(ax0, merged, METRIC, x="type",
                             order=["AI 1st Draft", "Final Submission"],
                             point_scale=1.75, show_strip=False)
ax0.grid(True, axis="y", alpha=0.3)
ax0.margins(x=0.15)
relabel_group_legend(ax0, fontsize=24)
ax0.set_ylabel("Avg Cosine Similarity", fontsize=24)
ax0.set_xlabel("")
ax0.set_ylim(0.4, 0.85)

# Nudge the left category label right so the two labels don't collide (as in Figure 1)
xticklabels = ax0.get_xticklabels()
xticklabels[0].set_x(1.5)
xticklabels[0].set_ha("right")

# (b)
sns.lineplot(x="minutes", y="value", data=df_long[df_long["minutes"] < XMAX],
             hue="group_key", errorbar="se", hue_order=GROUP_ORDER,
             palette=palette_map, ax=ax1, linewidth=4)
ax1.grid(True, axis="both")
ax1.set_ylabel("Avg Cosine Similarity")
ax1.set_xlabel("")
ax1.legend_.remove()
ax1.tick_params(axis="x", labelbottom=False)

plot_request_kde(ax1b, ai_requests, xmax=XMAX, bw_method=0.25)
ax1b.set_xlim(0, XMAX)
ax1b.set_xlabel("Minutes spent writing", fontsize=24)
ax1b.set_ylabel("# Req.")
ax1b.grid(True, axis="both")
ax1b.legend(frameon=False)

# (c)
scatter_pca(ax2, X2, group, stage, s=100)
ax2.set_xlabel("PC1", fontsize=24)
ax2.set_ylabel("PC2", fontsize=24)
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width * 0.72, box.height])
ax2.legend(handles=pca_stage_handles(markersize=12), title=None,
           loc="upper left", bbox_to_anchor=(0.7, 0.3), fontsize=24)

fig.text(0.08, 0.88, "(a)", fontweight="bold", fontsize=28)
fig.text(0.275, 0.88, "(b)", fontweight="bold", fontsize=28)
fig.text(0.57, 0.88, "(c)", fontweight="bold", fontsize=28)

save_fig(fig, "figure_s15")
