"""
Table S8: scale-free measures of AI homogenization (both studies).

  rho_b = (Delta_T - Delta_O) / Delta_T, with Delta_j = U(AI-j) - U(SELF-j):
          share of the technical-quality AI homogenization effect offset by
          the originality incentive (higher is better)
  R_j   = (U(AI-j final) - U(SELF-j)) / (U(AI-j first draft) - U(SELF-j)):
          share of first-draft homogenization retained in final submissions
          (lower is better)

Ratios are reported only when the denominator passes a minimum magnitude
(MIN_DENOMINATOR) and its 95% CI excludes zero; otherwise N/A. SEs come from
a delete-one jackknife within each independent cell (final and draft AI
samples are treated as separate samples, so R's SE is approximate when
participants overlap).

Output: tables/table_s8.tex
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import (LEXICAL_START, jackknife_var, latex_name_map, metric_cols, processed,
                    save_table, u_loo)

MIN_DENOMINATOR = 0.005
CHECK_DENOMINATOR_CI = True
ZCRIT = stats.norm.ppf(0.975)
NA = (np.nan, np.nan, np.nan)


def values(Y, group, metric):
    return Y.loc[Y["group_key"] == group, metric].dropna().to_numpy(dtype=float)


def denominator_ok(denominator, *samples):
    """Minimum-magnitude and 95%-CI-excludes-zero checks on a denominator."""
    if abs(denominator) < MIN_DENOMINATOR:
        return False
    if CHECK_DENOMINATOR_CI:
        se = np.sqrt(sum(jackknife_var(u_loo(y)) for y in samples))
        if abs(denominator) <= ZCRIT * se:
            return False
    return True


def calculate_R(Y_final, Y_draft, metric, incentive):
    """R_j with its jackknife SE and 95% CI half-width."""
    final = values(Y_final, f"AI-{incentive}", metric)
    draft = values(Y_draft, f"AI-{incentive}", metric)
    self_y = values(Y_final, f"SELF-{incentive}", metric)
    if min(len(final), len(draft), len(self_y)) <= 2:
        return NA

    Uf, Ud, Us = final.mean(), draft.mean(), self_y.mean()
    if not denominator_ok(Ud - Us, draft, self_y):
        return NA

    R = (Uf - Us) / (Ud - Us)
    # Delete-one estimates, perturbing one independent cell at a time
    R_f = (u_loo(final) - Us) / (Ud - Us)
    R_d = (Uf - Us) / (u_loo(draft) - Us)
    R_s = (Uf - u_loo(self_y)) / (Ud - u_loo(self_y))
    se = np.sqrt(jackknife_var(R_f) + jackknife_var(R_d) + jackknife_var(R_s))
    return R, se, ZCRIT * se


def calculate_rho(Y_final, metric):
    """rho_b with its jackknife SE and 95% CI half-width."""
    groups = {
        "AI_T": values(Y_final, "AI-CONV", metric),
        "SELF_T": values(Y_final, "SELF-CONV", metric),
        "AI_O": values(Y_final, "AI-DIV", metric),
        "SELF_O": values(Y_final, "SELF-DIV", metric),
    }
    if min(len(y) for y in groups.values()) <= 2:
        return NA

    U = {k: y.mean() for k, y in groups.items()}

    def rho_of(u):
        dT = u["AI_T"] - u["SELF_T"]
        dO = u["AI_O"] - u["SELF_O"]
        return (dT - dO) / dT

    if not denominator_ok(U["AI_T"] - U["SELF_T"], groups["AI_T"], groups["SELF_T"]):
        return NA

    variance = 0.0
    for name, y in groups.items():
        rho_loo = np.array([rho_of({**U, name: u}) for u in u_loo(y)])
        variance += jackknife_var(rho_loo)
    se = np.sqrt(variance)
    return rho_of(U), se, ZCRIT * se


# ============================================================
# BOTH STUDIES
# ============================================================

results = {}
for study in (1, 2):
    Y_final = pd.read_csv(processed(study, "Y.csv"))
    Y_draft = pd.read_csv(processed(study, "Y_first_draft.csv"))
    y_cols = metric_cols(Y_final)
    results[study] = {}
    for metric in y_cols:
        results[study][metric] = {
            "rho": calculate_rho(Y_final, metric),
            "DIV": calculate_R(Y_final, Y_draft, metric, "DIV"),
            "CONV": calculate_R(Y_final, Y_draft, metric, "CONV"),
        }

# Plain-text summary for a quick look
summary = pd.DataFrame([
    {"metric": m, **{f"S{s}_{k}": results[s][m][k][0] for s in (1, 2) for k in ("rho", "DIV", "CONV")}}
    for m in y_cols
])
print(summary.round(3).to_string(index=False))

# ============================================================
# LATEX
# ============================================================

def fmt_cell(result):
    R, _, margin = result
    if not (np.isfinite(R) and np.isfinite(margin)):
        return "N/A"
    return rf"${100 * R:.1f}\% \pm {100 * margin:.1f}\%$"


lines = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\small",
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r" & \multicolumn{3}{c}{\textbf{Study 1}} & \multicolumn{3}{c}{\textbf{Study 2}} \\",
    r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
    r"Metric & $\rho_b\uparrow$ & $R_O\downarrow$ & $R_T\downarrow$"
    r" & $\rho_b\uparrow$ & $R_O\downarrow$ & $R_T\downarrow$ \\",
    r"\midrule",
]

split = y_cols.index(LEXICAL_START)
for title, metrics in [("Embeddings", y_cols[:split]), ("Lexical based", y_cols[split:])]:
    if title != "Embeddings":
        lines.append(r"\midrule")
    lines += [r"\rowcolor{lightgray}", rf"\textbf{{{title}}} & & & & & & \\"]
    for metric in metrics:
        cells = [latex_name_map[metric]]
        cells += [fmt_cell(results[s][metric][k]) for s in (1, 2) for k in ("rho", "DIV", "CONV")]
        lines.append(" & ".join(cells) + r" \\")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Scale-free measures of AI homogenization."
    r" $\rho_b$ denotes the share of the AI homogenization"
    r" effect under the technical-quality incentive offset"
    r" by the originality incentive (higher is better)."
    r" $R_O$ and $R_T$ denote the share of initial AI-draft"
    r" homogenization retained in final submissions under"
    r" the originality and technical-quality incentives,"
    r" respectively (lower is better)."
    r" Entries report estimates $\pm$ 95\% confidence"
    r" interval half-widths. N/A indicates that the"
    r" denominator fails the minimum-magnitude or"
    r" statistical-precision criterion.}",
    r"\label{tab:homogenization_retention}",
    r"\end{table}",
]

save_table("\n".join(lines), "table_s8")
