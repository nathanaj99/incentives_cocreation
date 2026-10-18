"""
Table S6: reliability and human validation of the LLM technical-quality
ratings (Study 1).

  Within-LLM reliability -- agreement among 3 independent LLM runs
  LLM-Human validity     -- LLM consensus (per-dimension median of the 3
                            runs) vs. one human judge on a subsample

Technical quality = organization score + technique score.

Output: tables/table_s6.tex
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from common import processed, save_table

STUDY = 1
COLS = ["participant_id", "organization_score", "technique_score", "technical_quality_score"]
DIMS = ["organization_score", "technique_score", "technical_quality_score"]
RUNS = ["run1", "run2", "run3"]

# ============================================================
# DATA
# ============================================================

q = [pd.read_csv(processed(STUDY, f"technical_quality_{r}.csv"))[COLS] for r in RUNS]
quality = (
    q[0].merge(q[1], on="participant_id", suffixes=("_run1", "_run2"))
    .merge(q[2].rename(columns={d: f"{d}_run3" for d in DIMS}), on="participant_id")
)

# LLM consensus: median of each component, total = sum of component medians
for d in ["organization_score", "technique_score"]:
    quality[f"{d}_llm"] = quality[[f"{d}_{r}" for r in RUNS]].median(axis=1)
quality["technical_quality_score_llm"] = quality["organization_score_llm"] + quality["technique_score_llm"]

human = pd.read_csv(processed(STUDY, "human_judge_1.csv"))[COLS]
human["technical_quality_score"] = human["organization_score"] + human["technique_score"]
human = human.rename(columns={d: f"{d}_human" for d in DIMS})

validation = quality.merge(human, on="participant_id", how="inner")
print("Number of human-rated stories:", len(validation))

# ============================================================
# STATISTICS
# ============================================================

def icc2_1(data):
    """ICC(2,1): two-way random effects, absolute agreement, single rater."""
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data).any(axis=1)]
    n, k = data.shape
    grand = data.mean()
    ss_rows = k * np.sum((data.mean(axis=1) - grand) ** 2)
    ss_cols = n * np.sum((data.mean(axis=0) - grand) ** 2)
    ss_error = np.sum((data - grand) ** 2) - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    return (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)


def run_agreement(df, score):
    """Share of items where all 3 runs agree exactly / lie within 1 point."""
    x = df[[f"{score}_{r}" for r in RUNS]].dropna()
    score_range = x.max(axis=1) - x.min(axis=1)
    return {"exact_agreement": (score_range == 0).mean(), "within_one": (score_range <= 1).mean()}


def mean_pairwise_kappa(df, score):
    """Mean quadratic-weighted kappa over the 3 pairs of runs."""
    kappas = []
    for a, b in combinations([f"{score}_{r}" for r in RUNS], 2):
        tmp = df[[a, b]].dropna()
        kappas.append(cohen_kappa_score(tmp[a], tmp[b], weights="quadratic"))
    return np.mean(kappas)


def llm_vs_human(df, score):
    llm, hum = df[f"{score}_llm"], df[f"{score}_human"]
    return {
        "icc": icc2_1(df[[f"{score}_human", f"{score}_llm"]].to_numpy()),
        "kappa": cohen_kappa_score(hum, llm, weights="quadratic"),
        "pearson": pearsonr(hum, llm)[0],
        "spearman": spearmanr(hum, llm)[0],
        "exact": (hum == llm).mean(),
        "within_one": (np.abs(hum - llm) <= 1).mean(),
        "mae": np.abs(hum - llm).mean(),
        "bias": (llm - hum).mean(),  # positive = LLM scores higher
    }


def fmt_pct(x):
    pct = 100 * x
    return f"{int(round(pct))}\\%" if np.isclose(pct, round(pct)) else f"{pct:.1f}\\%"


icc = {d: icc2_1(quality[[f"{d}_{r}" for r in RUNS]].to_numpy()) for d in DIMS}
kappa = {d: mean_pairwise_kappa(quality, d) for d in DIMS[:2]}
agree = {d: run_agreement(quality, d) for d in DIMS}
valid = {d: llm_vs_human(validation, d) for d in DIMS}

org, tech, tot = DIMS

# ============================================================
# LATEX
# ============================================================

def row(label, values):
    return f"{label}\n    & " + "\n    & ".join(values) + r" \\"


latex = "\n".join([
    r"\begin{table}[ht]",
    r"\centering",
    r"\small",
    r"\caption{Reliability and human validation of technical quality ratings.}",
    r"\label{tab:quality_validation}",
    r"\begin{tabular}{lccc}",
    r"\toprule",
    r"Measure & Organization & Technique & Total quality \\",
    r"\midrule",
    "",
    r"\rowcolor{lightgray}",
    r"\multicolumn{4}{l}{\textbf{Within-LLM Reliability}} \\",
    "",
    row("ICC(2,1)", [f"{icc[d]:.3f}" for d in DIMS]),
    "",
    row(r"Mean quadratic $\kappa$", [f"{kappa[org]:.3f}", f"{kappa[tech]:.3f}", r"\textemdash"]),
    "",
    row("All 3 runs exactly agree", [fmt_pct(agree[d]["exact_agreement"]) for d in DIMS]),
    "",
    row("All 3 within 1 point", [fmt_pct(agree[d]["within_one"]) for d in DIMS]),
    "",
    r"\addlinespace",
    "",
    r"\rowcolor{lightgray}",
    r"\multicolumn{4}{l}{\textbf{LLM--Human Validity}} \\",
    "",
    row("ICC(2,1)", [f"{valid[d]['icc']:.3f}" for d in DIMS]),
    "",
    row(r"Quadratic $\kappa$", [f"{valid[org]['kappa']:.3f}", f"{valid[tech]['kappa']:.3f}", r"\textemdash"]),
    "",
    row(r"Pearson $r$", [f"{valid[d]['pearson']:.3f}" for d in DIMS]),
    "",
    row(r"Spearman $\rho$", [f"{valid[d]['spearman']:.3f}" for d in DIMS]),
    "",
    row("Exact agreement", [fmt_pct(valid[d]["exact"]) for d in DIMS]),
    "",
    row("Within 1 point", [fmt_pct(valid[d]["within_one"]) for d in DIMS]),
    "",
    row("MAE", [f"{valid[d]['mae']:.3f}" for d in DIMS]),
    "",
    row(r"Mean LLM $-$ Human", [f"{valid[d]['bias']:+.3f}" for d in DIMS]),
    "",
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
])

save_table(latex, "table_s6")
