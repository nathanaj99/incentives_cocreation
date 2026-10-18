"""
Tables S4 / S9: effect sizes of the main hypothesis tests (Study 1 / Study 2).

For each metric:
  Final      -- contrasts between final submissions (FINAL_COMPARISONS)
  First (AI) -- the same AI-vs-SELF / T-vs-O contrasts, but with the AI
                groups represented by their first AI-suggested draft:
                  DIV  = AI-O first draft - SELF-O final
                  CONV = AI-T first draft - SELF-T final
                  AI   = AI-T first draft - AI-O first draft
Black superscript stars: unadjusted one-sided p; red subscript stars:
BH-adjusted (across metrics, within each contrast).

Outputs: tables/table_s4.tex, tables/table_s9.tex
"""

from textwrap import dedent

from common import (FINAL_COMPARISONS, LEXICAL_START, fmt_latex_effect, latex_name_map,
                    load_Y_with_first_draft, run_comparisons, save_table)

FIRST_DRAFT_COMPARISONS = {
    "DIV": ("AI-DIV_first", "SELF-DIV"),
    "CONV": ("AI-CONV_first", "SELF-CONV"),
    "AI": ("AI-CONV_first", "AI-DIV_first"),
}


def get_cell(res, metric, comparison):
    v = res.get(metric, {}).get(comparison)
    if v is None:
        return ""
    return fmt_latex_effect(v["effect"], v["p_one_sided"], v["p_one_sided_bh"])


def make_table(study):
    Y, y_cols = load_Y_with_first_draft(study)
    res = run_comparisons(Y, y_cols, FINAL_COMPARISONS)
    res_first = run_comparisons(Y, y_cols, FIRST_DRAFT_COMPARISONS)

    rows = []
    for metric in y_cols:
        if metric == LEXICAL_START:
            rows += [r"\midrule", r"\rowcolor{lightgray}",
                     r"\textbf{Lexical based} &  &  &  &  &  &  &  \\"]
        cells = [
            get_cell(res, metric, "DIV"), get_cell(res_first, metric, "DIV"),
            get_cell(res, metric, "CONV"), get_cell(res_first, metric, "CONV"),
            get_cell(res, metric, "AI"), get_cell(res_first, metric, "AI"),
            get_cell(res, metric, "SELF"),
        ]
        rows.append(f"{latex_name_map[metric]} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)

    return dedent(rf"""
\begin{{table}}[ht]
\centering
\small
\begin{{tabular}}{{lrr|rr|rr|r}}
\toprule
Comparison & \multicolumn{{2}}{{c}}{{\textbf{{AI-O -- SELF-O}}}} & \multicolumn{{2}}{{c}}{{\textbf{{AI-T -- SELF-T}}}} & \multicolumn{{2}}{{c}}{{\textbf{{AI (T -- O)}}}} & \textbf{{SELF (T -- O)}} \\
\midrule
Version & Final & First (AI) & Final & First (AI) & Final & First (AI) & Final \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}}
\midrule
\rowcolor{{lightgray}}
\textbf{{Embeddings}} &  &  &  &  &  &  &  \\
{body}
\bottomrule
\end{{tabular}}
\caption{{Study {study}: effect sizes from differences in mean pairwise similarity over the final submissions (``Final'') and the first valid draft the AI suggested when available (``First (AI)''). Inference uses jackknife variance estimates for the pairwise-similarity U-statistics with a normal approximation and one-sided tests. Black significance stars are standard p-values, while red stars use Benjamini--Hochberg (BH) FDR-adjusted p-values: $^{{*}}$ $p<0.1$, $^{{**}}$ $p<0.05$, $^{{***}}$ $p<0.01$.}}
\label{{tab:effects_combined_study{study}}}
\end{{table}}
""")


save_table(make_table(1), "table_s4")
save_table(make_table(2), "table_s9")
