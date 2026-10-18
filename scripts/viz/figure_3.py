"""
Figure 3: AI adoption and AI use.

  Row 1 (Study 1): (a) adoption-rate distribution, (b) chat time vs.
                   adoption, (c) # requests vs. adoption,
                   (d) request timing for HIGH adopters
  Row 2 (Study 2): (f), (g), (h) as (a)-(c);
                   (e) Study 1 request timing for LOW adopters

High/Low adoption is a median split within each study (see ADOPTION_COL).

Output: figs/figure_3.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from common import (ADOPTION_COL, add_reliance_type, load_time_spent, palette_map,
                    plot_request_kde, processed, raw, save_fig, set_style)

set_style()

MEDIAN_COLOR = "#176B45"  # dark green


def load_study(study):
    """Adoption rates, per-request labels (with High/Low + group), time use."""
    reliance_col = ADOPTION_COL[study]
    reliance = add_reliance_type(pd.read_csv(processed(study, "ai_reliance.csv")), reliance_col)
    submissions = pd.read_csv(raw(study, "submissions.csv"))

    requests = pd.read_csv(processed(study, "ai_requests_labeled.csv"))
    requests = requests.merge(
        reliance[["participant_id", "reliance_type"]].rename(columns={"reliance_type": "adoption_level"}),
        on="participant_id", how="left",
    )
    requests = requests.merge(submissions[["participant_id", "group_key"]], on="participant_id", how="left")

    return reliance, requests, load_time_spent(study, reliance_col), reliance_col


# ============================================================
# PANEL HELPERS
# ============================================================

def plot_adoption_distribution(ax, reliance_df, reliance_col):
    """Back-to-back histogram of A_i (AI-T left, AI-O right) with group medians."""
    conv = reliance_df.loc[reliance_df["group_key"] == "AI-CONV", reliance_col].dropna().to_numpy()
    div = reliance_df.loc[reliance_df["group_key"] == "AI-DIV", reliance_col].dropna().to_numpy()

    bins = np.linspace(0, 1, 21)
    conv_hist, edges = np.histogram(conv, bins=bins)
    div_hist, _ = np.histogram(div, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    heights = np.diff(edges)

    ax.barh(centers, -conv_hist, height=heights, color=palette_map["AI-CONV"],
            edgecolor="black", linewidth=1, label="AI-T")
    ax.barh(centers, div_hist, height=heights, color=palette_map["AI-DIV"],
            edgecolor="black", linewidth=1, label="AI-O")
    ax.axvline(0, color="black", linewidth=1)

    # Group medians (dashed), each drawn over its own half
    med_conv = np.median(conv) if len(conv) else np.nan
    med_div = np.median(div) if len(div) else np.nan
    for med, (xmin, xmax_frac) in [(med_conv, (0, 0.5)), (med_div, (0.5, 1))]:
        if np.isfinite(med):
            ax.axhline(med, linestyle=(0, (5, 3)), xmin=xmin, xmax=xmax_frac,
                       color=MEDIAN_COLOR, linewidth=2.0, zorder=5)

    xmax = max(conv_hist.max(), div_hist.max(), 1) * 1.15
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(0, 1)
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(t):g}" for t in xticks])  # counts on both sides
    ax.set_xlabel("Number of participants")
    ax.set_ylabel(r"Adoption rate $A_i$")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")

    # Median values toward the outer edge of each half
    label_bbox = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5)
    for med, x in [(med_conv, -0.75 * xmax), (med_div, 0.75 * xmax)]:
        if np.isfinite(med):
            ax.annotate(f"{med:.2f}", xy=(x, med), xytext=(0, 5), textcoords="offset points",
                        fontsize=14, fontweight="bold", color=MEDIAN_COLOR, ha="center",
                        va="bottom", bbox=label_bbox, zorder=7)

    # High / Low indicators around the AI-O median, near the center
    if np.isfinite(med_div):
        for text, dy, va in [("↑ High", 22, "bottom"), ("↓ Low", -22, "top")]:
            ax.annotate(text, xy=(0.23 * xmax, med_div), xytext=(0, dy), textcoords="offset points",
                        color=MEDIAN_COLOR, fontsize=14, fontweight="bold", ha="left", va=va,
                        bbox=label_bbox, zorder=7)


def plot_vs_adoption(ax, time_df, reliance_col, y, ylabel):
    """Scatter + linear fit of `y` against adoption, separately for AI-O / AI-T."""
    for g in ["AI-DIV", "AI-CONV"]:
        sns.regplot(x=reliance_col, y=y, data=time_df[time_df["group_key"] == g], ax=ax,
                    scatter=True, seed=0, color=palette_map[g], scatter_kws={"s": 15, "alpha": 0.5})
    ax.set_xlabel("Adoption")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def plot_request_kde_by_adoption(ax, requests, adoption_level, title):
    """Request-timing KDE for one adoption level, normalized per participant."""
    subset = requests[requests["adoption_level"] == adoption_level]
    plot_request_kde(ax, subset, xmax=25, bw_method=0.2,
                     normalize_by=subset["participant_id"].nunique())
    ax.set_title(title)
    ax.set_xlabel("Mins from Start")
    ax.set_ylabel("# Req. (normalized)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)


# ============================================================
# FIGURE
# ============================================================

reliance1, requests1, time1, col1 = load_study(1)
reliance2, requests2, time2, col2 = load_study(2)

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
ax_a, ax_b, ax_c, ax_d = axes[0]
ax_f, ax_g, ax_h, ax_e = axes[1]

# Row 1: Study 1
plot_adoption_distribution(ax_a, reliance1, col1)
plot_vs_adoption(ax_b, time1, col1, "chat_time", "Chat time (mins)")
plot_vs_adoption(ax_c, time1, col1, "num_requests", "# requests")
plot_request_kde_by_adoption(ax_d, requests1, "High", "High Adoption")

# Row 2: Study 2 (f-h) + Study 1 low adoption (e)
plot_adoption_distribution(ax_f, reliance2, col2)
plot_vs_adoption(ax_g, time2, col2, "chat_time", "Chat time (mins)")
plot_vs_adoption(ax_h, time2, col2, "num_requests", "# requests")
plot_request_kde_by_adoption(ax_e, requests1, "Low", "Low Adoption")

# Axis limits
ax_b.set_ylim(0, 13)
ax_g.set_ylim(0, 6)
ax_c.set_ylim(0, 7)
ax_h.set_ylim(0, 6)
ax_d.set_ylim(0, 0.25)
ax_e.set_ylim(0, 0.25)
ax_f.get_legend().remove()

# Panel labels ((d)/(e) sit slightly further left)
for label, ax in {"(a)": ax_a, "(b)": ax_b, "(c)": ax_c, "(d)": ax_d,
                  "(f)": ax_f, "(g)": ax_g, "(h)": ax_h, "(e)": ax_e}.items():
    x = -0.26 if label in ("(d)", "(e)") else -0.23
    ax.text(x, 1.0, label, transform=ax.transAxes, fontsize=22, fontweight="bold",
            ha="left", va="bottom")

fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.09, wspace=0.32, hspace=0.42)

# Study labels above the first panel of each row
for ax, label in [(ax_a, "Study 1"), (ax_f, "Study 2")]:
    pos = ax.get_position()
    fig.text(pos.x0, pos.y1 + 0.02, label, fontsize=24, fontweight="bold", ha="left", va="bottom")

# Horizontal divider between the rows, across columns (a)-(c) only
y_sep = (ax_a.get_position().y0 + ax_f.get_position().y1) / 2 - 0.015
fig.add_artist(Line2D([ax_a.get_position().x0 - 0.05, ax_c.get_position().x1 + 0.0095], [y_sep, y_sep],
                      transform=fig.transFigure, color="black", linewidth=1.5, alpha=0.5))

# Vertical divider setting off the request-timing column
x_sep = (ax_h.get_position().x1 + ax_e.get_position().x0) / 2 - 0.02
fig.add_artist(Line2D([x_sep, x_sep], [ax_f.get_position().y0 - 0.055, ax_f.get_position().y1 + 0.055],
                      transform=fig.transFigure, color="black", linewidth=1.5, alpha=0.5))

save_fig(fig, "figure_3", dpi=300)
