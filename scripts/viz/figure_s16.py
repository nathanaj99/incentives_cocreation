"""
Figure S16: Study 2 AI adoption measured with the n = 10 n-gram window
(adoption_10, as in Study 1) instead of the default n = 3.

  (a) adoption-rate distribution (AI-T left, AI-O right; dashed = means)
  (b) chat time, (c) prompt length, (d) # requests vs. adoption

Output: figs/figure_s16.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import load_time_spent, palette_map, processed, save_fig, set_style

set_style()

STUDY = 2
RELIANCE_COL = "adoption_10"

reliance = pd.read_csv(processed(STUDY, "ai_reliance.csv"))
time_spent = load_time_spent(STUDY, RELIANCE_COL)

fig, ax = plt.subplots(1, 4, figsize=(24, 7), width_ratios=[4, 3, 3, 3])

# ------------------------------------------------------------
# (a) back-to-back histogram of adoption
# ------------------------------------------------------------
conv = reliance.loc[reliance["group_key"] == "AI-CONV", RELIANCE_COL].dropna().to_numpy()
div = reliance.loc[reliance["group_key"] == "AI-DIV", RELIANCE_COL].dropna().to_numpy()
bins = np.linspace(0, 1, 21)
conv_hist, edges = np.histogram(conv, bins=bins)
div_hist, _ = np.histogram(div, bins=bins)
centers = (edges[:-1] + edges[1:]) / 2
heights = np.diff(edges)

ax[0].barh(centers, -conv_hist, height=heights, label="AI-T", color=palette_map["AI-CONV"],
           edgecolor="black", linewidth=1)
ax[0].barh(centers, div_hist, height=heights, label="AI-O", color=palette_map["AI-DIV"],
           edgecolor="black", linewidth=1)
ax[0].axvline(0, color="k", linewidth=1)
ax[0].axhline(conv.mean(), linestyle="--", xmin=0, xmax=0.5, color="black")
ax[0].axhline(div.mean(), linestyle="--", xmin=0.5, xmax=1, color="black")

xmax = max(conv_hist.max(), div_hist.max()) * 1.05
ax[0].set_xlim(-xmax, xmax)
xticks = ax[0].get_xticks()
ax[0].set_xticks(xticks, [f"{abs(t)}" for t in xticks])  # counts on both sides
ax[0].set_ylim(0, 1)
ax[0].set_ylabel(r"Adoption rate $A_i$")
ax[0].set_xlabel("Number of participants")
ax[0].grid(axis="x")
ax[0].legend(title=None)

# ------------------------------------------------------------
# (b)-(d) AI use vs. adoption
# ------------------------------------------------------------
for a, y, ylabel, ylim in [
    (ax[1], "chat_time", "Chat time (mins)", (0, 8)),
    (ax[2], "num_words_hundreds", "# words in prompt (hundreds)", (0, 2)),
    (ax[3], "num_requests", "# requests", (0, 6)),
]:
    for g in ["AI-DIV", "AI-CONV"]:
        sns.regplot(x=RELIANCE_COL, y=y, data=time_spent[time_spent["group_key"] == g], ax=a,
                    scatter=True, seed=0, color=palette_map[g], scatter_kws={"s": 15, "alpha": 0.5})
    a.set_ylim(*ylim)
    a.set_ylabel(ylabel)
    a.set_xlabel("Adoption")
    a.grid(axis="y")

plt.subplots_adjust(wspace=0.35)
for x, label in [(0.08, "(a)"), (0.33, "(b)"), (0.51, "(c)"), (0.715, "(d)")]:
    fig.text(x, 0.87, label, fontweight="bold", fontsize=26)

save_fig(fig, "figure_s16")
