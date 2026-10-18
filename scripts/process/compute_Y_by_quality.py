"""
Y_by_quality.csv: similarity of each final submission to the other
submissions in the same (group, technical-quality score) cell -- i.e.
similarity conditional on quality (Fig. 5).

Cells with fewer than 3 submissions get NaN. n_cell is the cell size.

Inputs: raw/submissions.csv and the LLM quality scores
          Study 1 -- processed/technical_quality_median.csv
          Study 2 -- processed/technical_quality.csv
Output: processed/Y_by_quality.csv  (cells in order of first appearance)
"""

import pandas as pd

from common import load_submissions, output, parse_args, should_skip
from similarity import add_metric_args, write_y, y_table

QUALITY_FILE = {1: "technical_quality_median.csv", 2: "technical_quality.csv"}
MIN_CELL_SIZE = 3


def main():
    args = parse_args(__doc__, extra=add_metric_args)
    for study in args.study:
        print(f"Study {study}")
        out = output(study, "Y_by_quality.csv")
        if should_skip(out, args.force):
            continue

        quality = pd.read_csv(output(study, QUALITY_FILE[study]))[["participant_id", "technical_quality_score"]]
        df = load_submissions(study).merge(quality, on="participant_id", how="inner")
        if not df["final_text"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError("Every submission must have nonempty final_text.")
        # Numeric scores so that e.g. 6 and "6" share a cell
        df["technical_quality_score"] = pd.to_numeric(df["technical_quality_score"], errors="raise")

        cells = df.groupby(["group_key", "technical_quality_score"], sort=False, dropna=False)
        groups = {key: cell["final_text"].tolist() for key, cell in cells}
        ids = {key: cell["participant_id"].tolist() for key, cell in cells}
        # Cells with a missing quality score are NaN too
        groups = {k: (v if not any(pd.isna(x) for x in k) else [None] * len(v)) for k, v in groups.items()}

        table = y_table(groups, args.metrics, args.device, min_valid=MIN_CELL_SIZE)
        frame = pd.DataFrame({
            "participant_id": sum((ids[k] for k in groups), []),
            "group_key": sum(([k[0]] * len(ids[k]) for k in groups), []),
            "technical_quality_score": sum(([k[1]] * len(ids[k]) for k in groups), []),
            "n_cell": sum(([len(ids[k])] * len(ids[k]) for k in groups), []),
        })
        for name in args.metrics:
            frame[name] = sum((table[name][k] for k in groups), [])
        write_y(frame, out, args.metrics)


if __name__ == "__main__":
    main()
