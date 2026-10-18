"""
Shared configuration and helpers for every figure / table script in scripts/viz/.

Each script is meant to be run on its own from the repository root, e.g.

    python scripts/viz/figure_2.py

and imports what it needs from here:

    * paths        -- where data is read from and outputs are written to
    * study config -- which adoption measure is used in each study
    * style        -- rcParams, palette_map, pretty_map, group labels
    * statistics   -- jackknife inference for pairwise-similarity U-statistics
                      and Benjamini-Hochberg correction
    * plotting     -- small helpers reused across several figures
    * I/O          -- save_fig / save_table

Terminology: in the data, the originality incentive is coded "DIV" and the
technical-quality incentive "CONV". In the paper these are "O" and "T", so
e.g. AI-DIV is displayed as AI-O and SELF-CONV as SELF-T.
"""

import math
import re
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FIGS_DIR = REPO_ROOT / "figs"
TABLES_DIR = REPO_ROOT / "tables"


def raw(study, filename):
    """Path to a raw data file, e.g. raw(1, 'submissions.csv')."""
    return DATA_DIR / f"study{study}" / "raw" / filename


def processed(study, filename):
    """Path to a processed data file, e.g. processed(2, 'Y.csv')."""
    return DATA_DIR / f"study{study}" / "processed" / filename


# ============================================================
# STUDY CONFIGURATION
# ============================================================

# Adoption rate A_i used to split AI-assisted participants into
# High / Low adoption (median split within each study).
ADOPTION_COL = {
    1: "adoption_10",
    2: "adoption_3",
}

# Time window (minutes) shown in similarity-trajectory plots.
TRAJECTORY_MAX_MINUTES = {
    1: 26,
    2: 12,
}


# ============================================================
# STYLE
# ============================================================

def set_style(font_size=24):
    """
    Global matplotlib defaults used by every figure. The base font size is
    24; the dense small-multiple grids (Figs. S12-S14) were laid out for 15.
    """
    # Both the main-text and appendix notebooks rendered in Arial. Mapping the
    # serif family to Arial too keeps elements that explicitly ask for
    # fontfamily="serif" (heatmap tick labels) in Arial.
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial"]
    matplotlib.rcParams["font.serif"] = ["Arial"]
    matplotlib.rcParams["font.size"] = font_size
    # seaborn's strip-plot jitter and regression-band bootstrap use NumPy's
    # global random state; seed it so figures are identical across runs
    np.random.seed(0)


# Muted cool (AI) / warm (SELF) bases
ai_base = sns.color_palette("muted")[0]    # muted blue
self_base = sns.color_palette("muted")[3]  # muted red


def lighten(color, amount=0.35):
    """Blend toward white by `amount` (0 = no change, 1 = white)."""
    r, g, b = color
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def darken(color, amount=0.25):
    """Blend toward black by `amount` (0 = no change, 1 = black)."""
    r, g, b = color
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


palette_map = {
    "AI-DIV": lighten(ai_base, 0.4),       # AI-O   (lighter cool)
    "AI-CONV": darken(ai_base, 0.3),       # AI-T   (darker cool)
    "SELF-DIV": lighten(self_base, 0.3),   # SELF-O (lighter warm)
    "SELF-CONV": darken(self_base, 0.25),  # SELF-T (darker warm)
}

GROUP_ORDER = ["AI-DIV", "AI-CONV", "SELF-DIV", "SELF-CONV"]
GROUP_LABELS = {
    "AI-DIV": "AI-O",
    "AI-CONV": "AI-T",
    "SELF-DIV": "SELF-O",
    "SELF-CONV": "SELF-T",
}

# Colors for AI request types (editing vs drafting)
EDIT_COLOR = "#77A777"
DRAFT_COLOR = "#DAD796"

# Display names of the similarity metrics in figures
pretty_map = {
    "all-MiniLM-L12-v2": "all-MiniLM-L12",
    "all-mpnet-base-v2": "all-mpnet-base",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Embedding",
    "google/embeddinggemma-300m": "embeddinggemma",
    "AnnaWegmann/Style-Embedding": "Style-Embedding",
    "byt5": "byt5",
    "star": "STAR",
    "compression_ratio": "Compression ratio",
    "self_bleu": "BLEU",
    "rougel": "ROUGE-L",
    "n_gram_diversity_score_2": "NGDS (n=2)",
    "n_gram_diversity_score_3": "NGDS (n=3)",
}

# Same names, LaTeX-formatted, for tables
latex_name_map = {
    **pretty_map,
    "n_gram_diversity_score_2": r"NGDS ($n=2$)",
    "n_gram_diversity_score_3": r"NGDS ($n=3$)",
}

# Metric rows are split into two blocks in every figure / table:
# embedding-based metrics come first, lexical metrics start here.
LEXICAL_START = "compression_ratio"

# For NGDS, higher = MORE diverse (opposite of the other similarity
# metrics). Their tests are one-sided in the opposite direction, and in
# heatmaps their effect sign is flipped so colors read the same way.
FLIP_METRICS = ("n_gram_diversity_score_2", "n_gram_diversity_score_3")


def metric_cols(Y):
    """Similarity-metric columns of a Y*.csv file (everything after the IDs)."""
    return [c for c in Y.columns if c not in ("participant_id", "group_key")]


def is_lexical(metric, all_metrics):
    return list(all_metrics).index(metric) >= list(all_metrics).index(LEXICAL_START)


def alternative_for(metric):
    """Direction of the one-sided test: AI should INCREASE similarity."""
    return "less" if "n_gram" in metric else "greater"


# ============================================================
# STATISTICS: jackknife for pairwise-similarity U-statistics
# ============================================================
#
# Y_i is participant i's average pairwise similarity to every other
# participant in the same group. The group U-statistic (mean pairwise
# similarity) is U = mean(Y_i), and the leave-one-participant-out
# U-statistic is (n U - 2 Y_i) / (n - 2).

def u_loo(y):
    """Leave-one-participant-out U-statistics."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    return (n * y.mean() - 2 * y) / (n - 2)


def jackknife_var(values):
    """Jackknife variance from a vector of leave-one-out estimates."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    return (n - 1) / n * np.sum((values - values.mean()) ** 2)


def jackknife_u_stats(y):
    """U-statistic, its jackknife variance / SE, and n for one group."""
    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    n = len(y)
    if n <= 2:
        raise ValueError("Each group must contain more than 2 observations.")
    var_U = jackknife_var(u_loo(y))
    return {"U": float(y.mean()), "var_U": float(var_U), "se": float(np.sqrt(var_U)), "n": n}


def jackknife_u_comparison(Y1, Y2, alternative="greater", alpha=0.05):
    """
    Test Delta = U1 - U2 for two independent groups, using jackknife
    variance estimates and a normal approximation.
    """
    g1 = jackknife_u_stats(Y1)
    g2 = jackknife_u_stats(Y2)

    effect = g1["U"] - g2["U"]
    se_effect = np.sqrt(g1["var_U"] + g2["var_U"])  # independent groups
    z = effect / se_effect

    if alternative == "greater":
        p_one_sided = stats.norm.sf(z)
    elif alternative == "less":
        p_one_sided = stats.norm.cdf(z)
    else:
        raise ValueError("alternative must be 'greater' or 'less'")

    zcrit = stats.norm.ppf(1 - alpha / 2)
    return {
        "effect": float(effect),
        "se": float(se_effect),
        "ci_low": float(effect - zcrit * se_effect),
        "ci_high": float(effect + zcrit * se_effect),
        "p_one_sided": float(p_one_sided),
        "p_two_sided": float(2 * stats.norm.sf(abs(z))),
    }


def bh_adjust(pvals):
    """
    Benjamini-Hochberg adjustment.
    Input: list of p-values (may include None / nan).
    Output: adjusted p-values aligned to the input; None where missing.
    """
    idx_p = []
    for i, p in enumerate(pvals):
        try:
            pf = float(p)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(pf):
            continue
        idx_p.append((i, min(max(pf, 0.0), 1.0)))

    out = [None] * len(pvals)
    m = len(idx_p)
    if m == 0:
        return out

    # q_(r) = p_(r) * m / r, then enforce monotonicity from the top
    idx_p_sorted = sorted(idx_p, key=lambda t: t[1])
    q = [p * m / r for r, (_, p) in enumerate(idx_p_sorted, start=1)]
    for k in range(m - 2, -1, -1):
        q[k] = min(q[k], q[k + 1])

    for qv, (i, _) in zip(q, idx_p_sorted):
        out[i] = min(qv, 1.0)
    return out


def add_bh(res, y_cols, comparisons, p_key="p_one_sided", out_key="p_one_sided_bh"):
    """
    Add BH-adjusted p-values to res[metric][comparison], correcting across
    metrics separately within each comparison.
    """
    for comp in comparisons:
        valid = [m for m in y_cols if np.isfinite(res[m][comp].get(p_key, np.nan))]
        for m, p_bh in zip(valid, bh_adjust([res[m][comp][p_key] for m in valid])):
            res[m][comp][out_key] = p_bh
    return res


def run_comparisons(Y, y_cols, comparisons):
    """
    Pairwise group contrasts for every metric, with BH correction.

    comparisons: {name: (group1, group2)}, effect = U(group1) - U(group2).
    Returns res[metric][name] = jackknife_u_comparison(...) + p_one_sided_bh.
    """
    res = {}
    for metric in y_cols:
        res[metric] = {}
        for name, (g1, g2) in comparisons.items():
            res[metric][name] = jackknife_u_comparison(
                Y.loc[Y["group_key"] == g1, metric],
                Y.loc[Y["group_key"] == g2, metric],
                alternative=alternative_for(metric),
            )
    return add_bh(res, y_cols, comparisons)


# Final-submission contrasts used throughout the paper
FINAL_COMPARISONS = {
    "DIV": ("AI-DIV", "SELF-DIV"),     # (AI - SELF) under originality
    "CONV": ("AI-CONV", "SELF-CONV"),  # (AI - SELF) under technical quality
    "AI": ("AI-CONV", "AI-DIV"),       # AI (T - O)
    "SELF": ("SELF-CONV", "SELF-DIV"), # SELF (T - O)
}


def load_Y_with_first_draft(study):
    """
    Final-submission similarities (Y.csv) stacked with the similarities of
    the first AI-suggested draft (Y_first_draft.csv), whose group keys get
    a "_first" suffix (e.g. "AI-DIV_first").
    """
    Y = pd.read_csv(processed(study, "Y.csv"))
    Y_draft = pd.read_csv(processed(study, "Y_first_draft.csv"))
    y_cols = metric_cols(Y)
    Y_draft["group_key"] = Y_draft["group_key"] + "_first"
    return pd.concat([Y, Y_draft], ignore_index=True), y_cols


# ============================================================
# AI ADOPTION
# ============================================================

def add_reliance_type(reliance, reliance_col):
    """
    Median split of the adoption rate (within the given study):
    'High' if strictly above the median, 'Low' otherwise, NaN if missing.
    """
    reliance = reliance.copy()
    median = reliance[reliance_col].median()
    reliance["reliance_type"] = reliance[reliance_col].apply(
        lambda x: np.nan if pd.isna(x) else ("High" if x > median else "Low")
    )
    return reliance


def load_Y_with_reliance(study, reliance_col=None):
    """Y.csv merged with the adoption rate and High/Low adoption label."""
    reliance_col = reliance_col or ADOPTION_COL[study]
    reliance = add_reliance_type(pd.read_csv(processed(study, "ai_reliance.csv")), reliance_col)
    Y = pd.read_csv(processed(study, "Y.csv"))
    y_cols = metric_cols(Y)
    Y = Y.merge(reliance[["participant_id", reliance_col, "reliance_type"]], on="participant_id", how="left")
    return Y, y_cols


def load_time_spent(study, reliance_col=None):
    """
    Per-participant time use for AI-assisted participants, merged with the
    adoption rate and AI-request counts / prompt lengths.
    """
    import ast

    reliance_col = reliance_col or ADOPTION_COL[study]
    reliance = add_reliance_type(pd.read_csv(processed(study, "ai_reliance.csv")), reliance_col)

    ai_requests = pd.read_csv(processed(study, "ai_requests.csv"))
    ai_requests["ai_requests"] = ai_requests["ai_requests"].apply(ast.literal_eval)
    ai_requests["num_requests"] = ai_requests["ai_requests"].apply(len)
    # number of whitespace-delimited words across all of a participant's prompts
    ai_requests["num_words"] = ai_requests["ai_requests"].apply(
        lambda x: len(re.findall(r"\S+", " ".join(x)))
    )
    ai_requests["avg_num_words"] = ai_requests["num_words"] / ai_requests["num_requests"]

    time_spent = pd.read_csv(processed(study, "time_spent.csv"))
    time_spent = time_spent.merge(
        reliance[["participant_id", reliance_col, "reliance_type"]], on="participant_id", how="left"
    )
    time_spent = time_spent.merge(
        ai_requests[["participant_id", "num_requests", "num_words", "avg_num_words"]],
        on="participant_id", how="left",
    )
    time_spent = time_spent[time_spent["group_key"].str.contains("AI")].copy()
    time_spent["chat_time_per_request"] = time_spent["chat_time"] / time_spent["num_requests"]
    time_spent["num_words_hundreds"] = time_spent["num_words"] / 100
    return time_spent


def compute_effects_by_adoption(Y, y_cols, reliance_level):
    """
    AI-vs-SELF contrasts within one adoption stratum (Y from
    load_Y_with_reliance). Only AI groups are restricted to the stratum.

        DIV  = AI-O[level] - SELF-O
        CONV = AI-T[level] - SELF-T
        AI   = AI-T[level] - AI-O[level]
    """
    def subset(group_key, level):
        df = Y[Y["group_key"] == group_key]
        return df if level is None else df[df["reliance_type"] == level]

    comparisons = {
        "DIV": (("AI-DIV", reliance_level), ("SELF-DIV", None)),
        "CONV": (("AI-CONV", reliance_level), ("SELF-CONV", None)),
        "AI": (("AI-CONV", reliance_level), ("AI-DIV", reliance_level)),
    }
    res = {}
    for metric in y_cols:
        res[metric] = {
            name: jackknife_u_comparison(subset(*a)[metric], subset(*b)[metric],
                                         alternative=alternative_for(metric))
            for name, (a, b) in comparisons.items()
        }
    return res


# ============================================================
# DEMOGRAPHICS (published only as summary statistics)
# ============================================================

def load_demographics_summary(study, table):
    """
    One-variable demographic summaries (processed/demographics_summary.csv),
    for table 'by_group' (Tables S1/S2) or 'by_adoption' (Tables S3/S11).
    Individual-level demographics are not released; the summary is produced
    privately from them.
    """
    summary = pd.read_csv(processed(study, "demographics_summary.csv"), keep_default_na=False,
                          na_values=[""])
    return summary[summary["table"] == table]


def summary_cell(summary, group, variable, level=None):
    """The summary row for one table cell."""
    rows = summary[(summary["group"] == group) & (summary["variable"] == variable)]
    if level is not None:
        rows = rows[rows["level"] == level]
    if len(rows) != 1:
        raise KeyError(f"no unique summary row for {group}/{variable}/{level}")
    return rows.iloc[0]


def fmt_mean_se(row, float_fmt="{:.2f}"):
    """'mean (SE)'; just the mean when n = 1; blank when n = 0."""
    if pd.isna(row["n"]) or row["n"] == 0:
        return ""
    if pd.isna(row["se"]):
        return float_fmt.format(row["mean"])
    return f"{float_fmt.format(row['mean'])} ({float_fmt.format(row['se'])})"


# ============================================================
# SIGNIFICANCE FORMATTING
# ============================================================

def p_to_stars(p):
    """Plain-text stars: * p<0.1, ** p<0.05, *** p<0.01."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def p_to_latex_stars(p):
    """Same thresholds as p_to_stars, as LaTeX math-mode stars."""
    return " ".join([r"\star"] * len(p_to_stars(p)))


def fmt_latex_effect(eff, p, p_bh, decimals=3):
    """
    $effect^{raw stars}_{red BH stars}$ -- black superscript stars use the
    unadjusted p-value, red subscript stars the BH-adjusted p-value.
    """
    return (
        f"${float(eff):.{decimals}f}"
        f"^{{{p_to_latex_stars(p)}}}"
        f"_{{\\textcolor{{red}}{{{p_to_latex_stars(p_bh)}}}}}$"
    )


# ============================================================
# PLOTTING HELPERS
# ============================================================

def wrap_labels(labels, width=14):
    return ["\n".join(textwrap.wrap(l, width=width)) for l in labels]


def make_muted_diverging_cmap(base="RdBu", blend=0.32, n=256, reverse=True, over="#8B4343"):
    """Diverging colormap blended toward white; values above vmax use `over`."""
    cm = plt.get_cmap(base + ("_r" if reverse else ""), n)
    colors = cm(np.linspace(0, 1, n))
    colors[:, :3] = (1 - blend) * colors[:, :3] + blend
    cmap = ListedColormap(colors)
    cmap.set_over(over)
    return cmap


def effect_matrix(res, metric_keys, col_keys, p_key="p_one_sided"):
    """
    Effect sizes M[metric, contrast] and significance stars A[metric, contrast]
    for a heatmap. NGDS effects are sign-flipped (see FLIP_METRICS).
    """
    M = np.full((len(metric_keys), len(col_keys)), np.nan)
    A = np.full((len(metric_keys), len(col_keys)), "", dtype=object)
    for i, mk in enumerate(metric_keys):
        for j, ck in enumerate(col_keys):
            v = res.get(mk, {}).get(ck)
            if v is None:
                continue
            eff = float(v["effect"])
            M[i, j] = -eff if mk in FLIP_METRICS else eff
            A[i, j] = p_to_stars(v[p_key])
    return M, A


def annotate_stars(ax, A, fontsize=20):
    """White, black-outlined significance stars centered in heatmap cells."""
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j]:
                txt = ax.text(j, i, A[i, j], ha="center", va="center",
                              fontsize=fontsize, color="white", fontweight="bold")
                txt.set_path_effects([pe.Stroke(linewidth=1.6, foreground="black"), pe.Normal()])


def style_heatmap_grid(ax, nrow, ncol):
    """White cell borders and full-matrix y-limits for an imshow heatmap."""
    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", top=False, left=False)
    ax.set_ylim(nrow - 0.5, -0.5)


def add_metric_split(ax, metric_keys, x_right=None, color="black", linewidth=2.5):
    """Horizontal line separating embedding metrics from lexical metrics."""
    split_idx = list(metric_keys).index(LEXICAL_START)
    if 0 < split_idx < len(metric_keys):
        ax.hlines(split_idx - 0.5, -0.5, ax.get_xlim()[1] if x_right is None else x_right,
                  colors=color, linewidth=linewidth)
    return split_idx


def add_block_labels(ax, metric_keys, x=-0.32, fontsize=20):
    """Rotated 'Embeddings' / 'Lexical' labels to the left of the row labels."""
    split_idx = list(metric_keys).index(LEXICAL_START)

    def row_center(i0, i1):
        return (i0 + (i1 - 1)) / 2

    for label, (i0, i1) in [("Embeddings", (0, split_idx)),
                            ("Lexical", (split_idx, len(metric_keys)))]:
        if i1 > i0:
            ax.text(x, row_center(i0, i1), label, rotation=90, va="center", ha="right",
                    transform=ax.get_yaxis_transform(), fontsize=fontsize, fontweight="bold")


def plot_similarity_distribution(ax, data, y, x="type", order=None, dodge=True,
                                 width=0.8, point_scale=1.0, show_strip=True):
    """
    Violin (distribution) + jittered points + mean +/- SE for each group.
    Used for Fig. 1(a), Fig. S11 (top), Fig. S12 and Fig. S15(a).
    """
    sns.violinplot(
        data=data, x=x, y=y, hue="group_key", order=order, hue_order=GROUP_ORDER,
        palette=palette_map, inner=None, cut=0, density_norm="width", width=width,
        dodge=dodge, linewidth=0.8, alpha=0.30, legend=False, ax=ax,
    )
    # Force violin transparency and push them to the background
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            collection.set_alpha(0.25)
            collection.set_zorder(1)

    if show_strip:
        sns.stripplot(
            data=data, x=x, y=y, hue="group_key", order=order, hue_order=GROUP_ORDER,
            palette=palette_map, dodge=dodge, jitter=0.18, size=4, alpha=0.18,
            linewidth=0, legend=False, zorder=2, ax=ax,
        )

    sns.pointplot(
        data=data, x=x, y=y, hue="group_key", order=order, hue_order=GROUP_ORDER,
        errorbar="se", dodge=0.6 if dodge else False, linestyles="none",
        scale=point_scale, markers="o", palette=palette_map, capsize=0.08, ax=ax,
    )


def relabel_group_legend(ax, n=4, **kwargs):
    """Replace data group keys in the legend with paper labels (AI-O, ...)."""
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles[:n], [GROUP_LABELS[g] for g in GROUP_ORDER][:n], title=None, **kwargs)


def load_trajectory(study, metric):
    """
    Similarity of in-progress snapshots (every 30 s, texts >= 200 words) in
    long format with a `minutes` column.
    """
    df = pd.read_csv(processed(study, f"Y_30secs_filtered_200words_{metric.replace('/', '_')}.csv"))
    df_long = df.drop(columns=["participant_id"]).melt(
        id_vars="group_key", var_name="time", value_name="value"
    )
    df_long["time"] = df_long["time"].astype(int)
    df_long["minutes"] = (df_long["time"] + 1) * 30 / 60
    return df_long


def plot_request_kde(ax, ai_requests, xmax, bw_method=0.25, normalize_by=1.0):
    """
    KDE of AI request times (minutes from start), scaled to approximate
    counts per one-minute bin, separately for editing and drafting requests.
    """
    x = np.linspace(0, xmax, 500)
    for req_type, color, label in [("editing", EDIT_COLOR, "Editing"),
                                   ("drafting", DRAFT_COLOR, "Drafting")]:
        t = ai_requests.loc[ai_requests["type"] == req_type, "time_from_start_mins"].dropna().to_numpy()
        y = gaussian_kde(t, bw_method=bw_method)(x) * len(t) * 1.0 / normalize_by
        ax.plot(x, y, color=color, label=label)
        ax.fill_between(x, y, color=color, alpha=0.25)


def build_text_corpus(study):
    """
    Final submissions followed by the first (draft) and third (draft2) AI
    drafts, in the same order as the stored PCA coordinates (PCA.pkl).

    Returns (texts, group, stage, first_drafts) where stage is one of
    'final', 'draft', 'draft2'.
    """
    texts = pd.read_csv(raw(study, "submissions.csv"))
    drafts = pd.read_csv(processed(study, "drafts_cleaned.csv"))

    first = drafts[drafts["first_draft"].notna()]
    third = drafts[drafts["third_draft"].notna()]

    texts_list = (texts["final_text"].tolist()
                  + first["first_draft"].tolist()
                  + third["third_draft"].tolist())
    group = np.array(texts["group_key"].tolist()
                     + first["group_key"].tolist()
                     + third["group_key"].tolist())
    stage = np.array(["final"] * len(texts) + ["draft"] * len(first) + ["draft2"] * len(third))
    return texts_list, group, stage, first["first_draft"].tolist()


PCA_MARKERS = {"draft": "x", "final": "o", "draft2": "s"}
PCA_ALPHAS = {"draft": 0.55, "final": 0.85, "draft2": 0.55}


def scatter_pca(ax, X2, group, stage, s=None):
    """2-D PCA scatter: color = group, marker = stage (final / draft 1 / draft 3)."""
    X2 = np.asarray(X2)
    kwargs = {} if s is None else {"s": s}
    for g in np.unique(group):
        for st in ("final", "draft2", "draft"):
            idx = (group == g) & (stage == st)
            if np.any(idx):
                ax.scatter(X2[idx, 0], X2[idx, 1], label=f"{g} ({st})",
                           marker=PCA_MARKERS[st], color=palette_map[g],
                           alpha=PCA_ALPHAS[st], **kwargs)


def pca_stage_handles(markersize=None):
    kw = {} if markersize is None else {"markersize": markersize}
    return [
        Line2D([0], [0], marker="o", linestyle="", label="Final", color="gray", **kw),
        Line2D([0], [0], marker="x", linestyle="", label="Draft 1", color="gray", **kw),
        Line2D([0], [0], marker="s", linestyle="", label="Draft 3", color="gray", **kw),
    ]


def pca_group_handles():
    return [
        Line2D([0], [0], marker="o", linestyle="", label=GROUP_LABELS[g],
               markerfacecolor=palette_map[g], markeredgecolor="white",
               markeredgewidth=0.5, markersize=8)
        for g in GROUP_ORDER
    ]


def embed_and_pca(texts):
    """Embed texts with all-MiniLM-L12-v2 and project to 2-D with PCA."""
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return PCA(n_components=2, random_state=0).fit_transform(embs)


# ============================================================
# OUTPUT
# ============================================================

def save_fig(fig, name, **kwargs):
    """Save to figs/<name>.png (bbox_inches='tight' unless overridden)."""
    FIGS_DIR.mkdir(exist_ok=True)
    kwargs.setdefault("bbox_inches", "tight")
    path = FIGS_DIR / f"{name}.png"
    fig.savefig(path, **kwargs)
    plt.close(fig)
    print(f"Saved {path.relative_to(REPO_ROOT)}")


def save_table(latex, name):
    """Save a LaTeX table to tables/<name>.tex."""
    TABLES_DIR.mkdir(exist_ok=True)
    path = TABLES_DIR / f"{name}.tex"
    path.write_text(latex.strip() + "\n")
    print(f"Saved {path.relative_to(REPO_ROOT)}")
