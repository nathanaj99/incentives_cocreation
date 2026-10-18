"""
Tables S1 / S2: balance tables of participant demographics by group
(Study 1 / Study 2), built from the published summary statistics
(processed/demographics_summary.csv, table 'by_group').

Outputs: tables/table_s1.tex, tables/table_s2.tex
"""

from common import fmt_mean_se, load_demographics_summary, save_table, summary_cell

GROUP_ORDER = ["AI-DIV", "SELF-DIV", "AI-CONV", "SELF-CONV"]
# LaTeX macros defined in the paper preamble
GROUP_LATEX = [r"\aiorig", r"\selforig", r"\aiqual", r"\selfqual"]

VAR_LABELS = {
    # Continuous
    "total_approvals": "Prolific Approvals",
    "age": "Age",
    # Binary
    "student": "Student",
    "sex_male": "Sex (Male)",
    "residence_usa": "Residence (USA)",
    "primary_language_english_only": "Primary Language (English)",
    # Categorical
    "ethnicity": "Ethnicity",
    "employment_status": "Employment Status",
}

CTS_VARS = ["total_approvals", "age"]
BINARY_VARS = ["student", "sex_male", "residence_usa", "primary_language_english_only"]
CATEGORICAL_VARS = ["ethnicity", "employment_status"]

CATEGORICAL_ORDERS = {
    "ethnicity": ["Asian", "Black", "Mixed", "Other", "White"],
    "employment_status": ["NIPW", "fulltime", "other/expired", "parttime", "unemployed_looking"],
}
CATEGORICAL_LABELS = {
    "employment_status": {
        "NIPW": "NIPW",
        "fulltime": "Full time",
        "other/expired": "Other",
        "parttime": "Part time",
        "unemployed_looking": "Unemployed (looking)",
    }
}


def make_balance_table_latex(summary, study):
    n_groups = len(GROUP_ORDER)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "c" * n_groups + "}",
        r"\toprule",
        " & " + " & ".join(GROUP_LATEX) + r" \\",
        r"\midrule",
    ]

    def section(title):
        lines.append(r"\rowcolor{lightgray}\multicolumn{" + str(n_groups + 1)
                     + r"}{l}{\textit{" + title + r"}} \\")

    # Continuous and binary variables: mean (SE) per group
    for title, variables in [("Continuous variables", CTS_VARS), ("Binary variables", BINARY_VARS)]:
        section(title)
        for var in variables:
            row = [VAR_LABELS.get(var, var)]
            row += [fmt_mean_se(summary_cell(summary, g, var)) for g in GROUP_ORDER]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")

    # Categorical variables: counts per group
    for var in CATEGORICAL_VARS:
        section(VAR_LABELS.get(var, var))
        for level in CATEGORICAL_ORDERS[var]:
            label = CATEGORICAL_LABELS.get(var, {}).get(level, level)
            row = [r"\quad " + str(label)]
            row += [str(int(summary_cell(summary, g, var, level)["count"])) for g in GROUP_ORDER]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")

    if lines[-1] == r"\addlinespace":
        lines.pop()

    n_per_group = [int(summary_cell(summary, g, "n_participants")["count"]) for g in GROUP_ORDER]
    n_text = (f"All groups consist of $n={n_per_group[0]}$ participants. "
              if len(set(n_per_group)) == 1 else "")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Balance table of participant demographics by group for Study {study}. "
        r"For continuous and binary variables, we display the mean and standard error "
        rf"(in parentheses). {n_text}``NIPW'' stands for Not in Paid Work.}}",
        rf"\label{{tab:balance_study{study}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


for study, table_name in [(1, "table_s1"), (2, "table_s2")]:
    save_table(make_balance_table_latex(load_demographics_summary(study, "by_group"), study), table_name)
