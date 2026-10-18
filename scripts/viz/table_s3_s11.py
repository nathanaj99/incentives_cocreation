"""
Tables S3 / S11: demographics of High vs. Low AI adopters
(Study 1 / Study 2). High/Low is the within-study median split of the
adoption rate (ADOPTION_COL), so only AI-assisted participants appear.
Built from the published summary statistics
(processed/demographics_summary.csv, table 'by_adoption').

Outputs: tables/table_s3.tex, tables/table_s11.tex
"""

import pandas as pd

from common import fmt_mean_se, load_demographics_summary, save_table, summary_cell

GROUPS = ["High", "Low"]

CTS_VARS = ["total_approvals", "age", "ai_exp_general", "ai_exp_writing"]
BINARY_VARS = ["student", "sex_male", "residence_usa", "primary_language_english_only"]
CATEGORICAL_VARS = ["ethnicity", "employment_status"]

VARIABLE_NAMES = {
    "total_approvals": "Total Approvals",
    "age": "Age",
    "ai_exp_general": "AI Exp. (General)",
    "ai_exp_writing": "AI Exp. (Writing)",
    "student": "Student",
    "sex_male": "Sex (Male)",
    "residence_usa": "Residence in US",
    "primary_language_english_only": "Primary Lang (English)",
}
CATEGORICAL_NAMES = {
    "ethnicity": "Ethnicity",
    "employment_status": "Employment status",
}
# Pretty names for categorical levels (looked up as-is, then lower-cased)
LEVEL_NAMES = {
    "asian": "Asian", "black": "Black", "mixed": "Mixed", "other": "Other", "white": "White",
    "nipw": "NIPW",
    "fulltime": "Full time", "full_time": "Full time", "full time": "Full time",
    "parttime": "Part time", "part_time": "Part time", "part time": "Part time",
    "other/expired": "Other/Expired", "other_expired": "Other/Expired",
    "unemployed": "Unemployed", "unemployed_looking": "Unemployed (looking)",
}


def pretty_level(level):
    s = str(level)
    return LEVEL_NAMES.get(s, LEVEL_NAMES.get(s.lower(), s.replace("_", " ").title()))


def fmt_pct(row, pct_fmt="{:.1f}\\%"):
    """Share of the group's non-missing answers at this level."""
    return "" if pd.isna(row["n"]) or row["n"] == 0 else pct_fmt.format(row["percent"])


def make_2col_summary_table_latex(summary, study):
    n_high, n_low = (int(summary_cell(summary, g, "n_participants")["count"]) for g in GROUPS)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        f"Adoption rate & High ($n = {n_high}$) & Low ($n={n_low}$) \\\\",
        r"\midrule",
    ]

    # Continuous and binary variables: mean (SE)
    for title, variables in [("Continuous variables", CTS_VARS), ("Binary variables", BINARY_VARS)]:
        lines.append(r"\rowcolor{lightgray}\multicolumn{3}{l}{\textit{" + title + r"}} \\")
        for var in variables:
            row = [VARIABLE_NAMES.get(var, var.replace("_", " ").title())]
            row += [fmt_mean_se(summary_cell(summary, g, var)) for g in GROUPS]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")

    # Categorical variables: % of the High / Low group, levels sorted by name
    for var in CATEGORICAL_VARS:
        lines.append(r"\rowcolor{lightgray}\multicolumn{3}{l}{\textit{"
                     + CATEGORICAL_NAMES.get(var, var) + r"}} \\")
        # levels observed in either group
        rows = summary[summary["variable"] == var]
        levels = rows.loc[rows["count"] > 0, "level"].unique()
        for level in sorted(levels, key=pretty_level):
            row = [r"\quad " + pretty_level(level)]
            row += [fmt_pct(summary_cell(summary, g, var, level)) for g in GROUPS]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Participant demographics for high and low adoption groups in Study {study}. "
        r"For continuous and binary variables, we display the mean and standard "
        r"error (in parentheses). ``NIPW'' stands for Not in Paid Work. "
        r"AI Experience is on a Likert scale from 0 to 4, with 0 being not at "
        r"all familiar and 4 being extremely familiar with AI tools.}",
        rf"\label{{tab:summary_2col_study{study}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


for study, table_name in [(1, "table_s3"), (2, "table_s11")]:
    # NB: unlike Tables S1/S2, any student status other than "yes" counts as 0 here
    save_table(make_2col_summary_table_latex(load_demographics_summary(study, "by_adoption"), study),
               table_name)
