"""
Figure 5: technical quality and quality-conditional similarity.

  (a)/(c) distribution of LLM-rated technical quality by group
          (Study 1 / Study 2), mean +/- SE
  (b)/(d) all-MiniLM-L12-v2 similarity among AI-O / AI-T participants
          within each quality score (jackknife SE), with stratum sizes

Similarity within a quality stratum comes from Y_by_quality.csv, where Y_i
is computed only against participants in the same group x quality cell.
Study 1 quality = median over 3 LLM runs (technical_quality_median.csv).

Output: figs/figure_5.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import (GROUP_LABELS, GROUP_ORDER, jackknife_u_stats, palette_map, processed,
                    raw, save_fig, set_style)

set_style()

Y_COL = "all-MiniLM-L12-v2"
QUALITY_SCORES = [5, 6, 7, 8]
POSITIONS = np.arange(len(GROUP_ORDER))


def load_quality(study, filename):
    quality = pd.read_csv(processed(study, filename))
    submissions = pd.read_csv(raw(study, "submissions.csv"))
    return quality.merge(submissions[["participant_id", "group_key"]], on="participant_id", how="inner")


def jackknife_summary(x):
    """Mean pairwise similarity and jackknife SE within one stratum (n < 3 -> NaN)."""
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return pd.Series({"mean": np.nan, "se": np.nan, "n": len(x)})
    s = jackknife_u_stats(x)
    return pd.Series({"mean": s["U"], "se": s["se"], "n": s["n"]})


def make_plot_df(Y):
    """Summarize each group x quality stratum."""
    out = (
        Y.dropna(subset=[Y_COL, "technical_quality_score", "group_key"])
        .groupby(["group_key", "technical_quality_score"])[Y_COL]
        .apply(jackknife_summary)
        .unstack()
        .reset_index()
    )
    out["condition"] = out["group_key"].str.split("-").str[0]
    out["incentive"] = out["group_key"].str.split("-").str[1]
    return out


def add_panel_label(ax, label):
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=14, fontweight="bold",
            ha="left", va="top")


def plot_quality_distribution(ax, results, panel_label):
    """Horizontal violins + jittered points + mean +/- SE of quality, by group."""
    df = results[["group_key", "technical_quality_score"]].dropna()
    summary = df.groupby("group_key")["technical_quality_score"].agg(["mean", "std", "count"]).reindex(GROUP_ORDER)
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    data = [df.loc[df["group_key"] == g, "technical_quality_score"].values for g in GROUP_ORDER]

    vp = ax.violinplot(data, positions=POSITIONS, vert=False, widths=0.72,
                       showmeans=False, showmedians=False, showextrema=False)
    for body, g in zip(vp["bodies"], GROUP_ORDER):
        body.set_facecolor(palette_map[g])
        body.set_edgecolor("none")
        body.set_alpha(0.35)

    rng = np.random.default_rng(42)
    for pos, g, vals in zip(POSITIONS, GROUP_ORDER, data):
        ax.scatter(vals, rng.normal(loc=pos, scale=0.05, size=len(vals)), s=12, alpha=0.18,
                   color=palette_map[g], edgecolors="none", zorder=2)

    ax.errorbar(summary["mean"].values, POSITIONS, xerr=summary["se"].values, fmt="none",
                ecolor="black", elinewidth=1.3, capsize=3, zorder=4)
    ax.scatter(summary["mean"].values, POSITIONS, s=80, c=[palette_map[g] for g in GROUP_ORDER],
               edgecolor="black", linewidth=0.8, zorder=5)

    ax.set_yticks(POSITIONS)
    ax.set_yticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Technical Quality Score", fontsize=13)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    add_panel_label(ax, panel_label)


def plot_similarity_by_quality(ax, plot_df, panel_label, show_legend=True):
    """AI-O / AI-T similarity by quality score, with a small n table below."""
    sub = plot_df[(plot_df["condition"] == "AI") & plot_df["technical_quality_score"].isin(QUALITY_SCORES)]

    style = {
        "DIV": dict(offset=-0.06, marker="o", label="AI-O", color=palette_map["AI-DIV"]),
        "CONV": dict(offset=0.06, marker="s", label="AI-T", color=palette_map["AI-CONV"]),
    }
    for incentive, st in style.items():
        d = sub[sub["incentive"] == incentive].sort_values("technical_quality_score")
        ax.errorbar(d["technical_quality_score"].to_numpy() + st["offset"], d["mean"].to_numpy(),
                    yerr=d["se"].to_numpy(), fmt=st["marker"], color=st["color"], ecolor=st["color"],
                    capsize=3, markersize=7, markeredgecolor="black", markeredgewidth=0.8,
                    elinewidth=1.3, label=st["label"], zorder=4)

    ax.set_xlim(4.5, 8.5)
    ax.set_xticks(QUALITY_SCORES)
    ax.set_xlabel("Technical Quality Score", fontsize=13, labelpad=7)
    ax.set_ylabel(f"Avg Cosine Similarity\n{Y_COL}", fontsize=12)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    if show_legend:
        ax.legend(title="Group", frameon=False)

    # Stratum sizes under the x-axis
    n_table = sub.pivot_table(index="incentive", columns="technical_quality_score", values="n",
                              aggfunc="first").reindex(index=["DIV", "CONV"], columns=QUALITY_SCORES)
    cell_text = [["" if pd.isna(v) else f"{int(v)}" for v in n_table.loc[inc]] for inc in ["DIV", "CONV"]]
    table = ax.table(cellText=cell_text, rowLabels=["AI-O", "AI-T"], cellLoc="center",
                     rowLoc="center", bbox=[0, -0.4, 1, 0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for cell in table.get_celld().values():
        cell.set_linewidth(0)
        cell.set_facecolor("none")
    ax.text(-0.15, -0.35, r"$n =$", transform=ax.transAxes, fontsize=10, ha="right", va="center")
    add_panel_label(ax, panel_label)


# ============================================================
# DATA
# ============================================================

quality1 = load_quality(1, "technical_quality_median.csv")
quality2 = load_quality(2, "technical_quality.csv")
plot_df1 = make_plot_df(pd.read_csv(processed(1, "Y_by_quality.csv")))
plot_df2 = make_plot_df(pd.read_csv(processed(2, "Y_by_quality.csv")))

# ============================================================
# FIGURE: 1 x 4
# ============================================================

fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(15, 3))

plot_quality_distribution(ax0, quality1, "(a)")
plot_similarity_by_quality(ax1, plot_df1, "(b)", show_legend=False)
plot_quality_distribution(ax2, quality2, "(c)")
plot_similarity_by_quality(ax3, plot_df2, "(d)", show_legend=True)

for ax, label in [(ax0, "Study 1"), (ax2, "Study 2")]:
    ax.text(-0.2, 1.15, label, transform=ax.transAxes, fontsize=15, fontweight="bold",
            ha="left", va="top")

fig.subplots_adjust(left=0.05, right=0.995, top=0.94, bottom=0.25, wspace=0.25)

# Separator between studies
x_sep = (ax1.get_position().x1 + ax2.get_position().x0) / 2 - 0.02
fig.add_artist(Line2D([x_sep, x_sep], [0.05, 0.96], transform=fig.transFigure,
                      color="black", linewidth=1.0, alpha=0.5))

save_fig(fig, "figure_5", dpi=300)
