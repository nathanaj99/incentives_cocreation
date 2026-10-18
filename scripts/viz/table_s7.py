"""
Table S7: AI x incentive interaction,

    theta = (U_AI-T - U_AI-O) - (U_SELF-T - U_SELF-O),

for both studies. The four randomized cells are independent, so
Var(theta) is the sum of the four jackknife variances. Two-sided tests;
BH correction across metrics within each study.

Output: tables/table_s7.tex
"""

from textwrap import dedent

import numpy as np
import pandas as pd
from scipy import stats

from common import (LEXICAL_START, bh_adjust, fmt_latex_effect, jackknife_u_stats,
                    latex_name_map, metric_cols, processed, save_table)


def jackknife_u_interaction(Y, metric):
    cell = {g: jackknife_u_stats(Y.loc[Y["group_key"] == g, metric])
            for g in ["AI-CONV", "AI-DIV", "SELF-CONV", "SELF-DIV"]}
    ai_effect = cell["AI-CONV"]["U"] - cell["AI-DIV"]["U"]
    self_effect = cell["SELF-CONV"]["U"] - cell["SELF-DIV"]["U"]
    interaction = ai_effect - self_effect
    se = np.sqrt(sum(c["var_U"] for c in cell.values()))
    z = interaction / se
    return {"effect": interaction, "se": se, "z": z, "p": 2 * stats.norm.sf(abs(z)),
            "ai_effect": ai_effect, "self_effect": self_effect}


def run_interaction_analysis(study):
    Y = pd.read_csv(processed(study, "Y.csv"))
    y_cols = metric_cols(Y)
    res = {m: jackknife_u_interaction(Y, m) for m in y_cols}
    for m, p_bh in zip(y_cols, bh_adjust([res[m]["p"] for m in y_cols])):
        res[m]["p_bh"] = p_bh
    return res, y_cols


res1, y_cols = run_interaction_analysis(1)
res2, _ = run_interaction_analysis(2)

rows = []
for metric in y_cols:
    if metric == LEXICAL_START:
        rows += [r"\midrule", r"\rowcolor{lightgray}", r"\textbf{Lexical-based} & & \\"]
    cells = [fmt_latex_effect(r[metric]["effect"], r[metric]["p"], r[metric]["p_bh"]) for r in (res1, res2)]
    rows.append(f"{latex_name_map[metric]} & {cells[0]} & {cells[1]} \\\\")
body = "\n".join(rows)

latex = dedent(rf"""
\begin{{table}}[ht]
\centering
\small
\begin{{tabular}}{{lrr}}
\toprule
Metric &
\textbf{{Study 1}} &
\textbf{{Study 2}} \\
\midrule
\rowcolor{{lightgray}}
\textbf{{Embeddings}} & & \\
{body}
\bottomrule
\end{{tabular}}
\caption{{
AI $\times$ incentive interaction effects, defined as
$(\mathrm{{AI\text{{-}}T}}-\mathrm{{AI\text{{-}}O}})
-(\mathrm{{SELF\text{{-}}T}}-\mathrm{{SELF\text{{-}}O}})$.
Inference uses jackknife variance estimates for the pairwise-similarity
U-statistics with a normal approximation and two-sided tests.
Black significance stars indicate unadjusted p-values, while red stars
indicate Benjamini--Hochberg FDR-adjusted p-values. BH correction is
applied separately within each study across metrics:
$^{{*}}p<0.1$, $^{{**}}p<0.05$, $^{{***}}p<0.01$.
}}
\label{{tab:interaction}}
\end{{table}}
""")

save_table(latex, "table_s7")
