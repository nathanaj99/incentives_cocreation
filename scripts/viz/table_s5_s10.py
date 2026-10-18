"""
Tables S5 / S10: effect sizes on final submissions stratified by Low / High
AI adoption (Study 1 / Study 2). Only AI groups are split; each AI stratum
is compared to the full SELF group with the same incentive (see
common.compute_effects_by_adoption). Stars use unadjusted one-sided p.

Outputs: tables/table_s5.tex, tables/table_s10.tex
"""

from textwrap import dedent

from common import (LEXICAL_START, compute_effects_by_adoption, latex_name_map,
                    load_Y_with_reliance, p_to_stars, save_table)

COLS = ["DIV", "CONV", "AI"]


def arrow(metric):
    """Direction in which values indicate MORE diversity."""
    return r"$\uparrow$" if "n_gram" in metric else r"$\downarrow$"


def make_table(study):
    Y, y_cols = load_Y_with_reliance(study)
    res_low = compute_effects_by_adoption(Y, y_cols, "Low")
    res_high = compute_effects_by_adoption(Y, y_cols, "High")

    def cell(res, metric, c):
        v = res[metric][c]
        return f"{v['effect']:.3f}{p_to_stars(v['p_one_sided'])}"

    rows = []
    for metric in y_cols:
        if metric == y_cols[0]:
            rows += [r"\rowcolor{lightgray}", r"\textbf{Embeddings} & & & & & & \\"]
        if metric == LEXICAL_START:
            rows += [r"\midrule", r"\rowcolor{lightgray}", r"\textbf{Lexical} & & & & & & \\"]
        # Interleave Low / High for each comparison
        cells = [cell(res, metric, c) for c in COLS for res in (res_low, res_high)]
        rows.append(f"{latex_name_map[metric]} {arrow(metric)} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)

    return dedent(rf"""
\begin{{table}}[ht]
\centering
\small
\begin{{tabular}}{{lrr|rr|rr}}
\toprule
Comparison & \multicolumn{{2}}{{c}}{{\textbf{{AI-O - SELF-O}}}} & \multicolumn{{2}}{{c}}{{\textbf{{AI-T - SELF-T}}}} & \multicolumn{{2}}{{c}}{{\textbf{{AI-T - AI-O}}}} \\
\midrule
AI Adoption level & Low & High & Low & High & Low & High \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}}
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Study {study}: effect sizes from differences in mean pairwise similarity using jackknife variance estimates for the pairwise-similarity U-statistics with a normal approximation and one-sided tests: $^{{*}}$ $p<0.1$, $^{{**}}$ $p<0.05$, $^{{***}}$ $p<0.01$, over the final submissions stratified by low and high adoption of AI among the AI-assisted group.}}
\label{{tab:effects_low_high_study{study}}}
\end{{table}}
""")


save_table(make_table(1), "table_s5")
save_table(make_table(2), "table_s10")
