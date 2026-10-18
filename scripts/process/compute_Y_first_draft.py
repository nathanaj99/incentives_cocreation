"""
Y_first_draft.csv: similarity of each AI participant's FIRST AI-generated
draft (drafts_cleaned.csv) to the other first drafts in their group.
Participants without a qualifying first draft are excluded.

Input:  processed/drafts_cleaned.csv
Output: processed/Y_first_draft.csv  (rows ordered AI-DIV, AI-CONV)
"""

import pandas as pd

from common import output, parse_args, should_skip
from similarity import add_metric_args, write_y, y_frame

GROUP_ORDER = ["AI-DIV", "AI-CONV"]


def main():
    args = parse_args(__doc__, extra=add_metric_args)
    for study in args.study:
        print(f"Study {study}")
        out = output(study, "Y_first_draft.csv")
        if should_skip(out, args.force):
            continue
        drafts = pd.read_csv(output(study, "drafts_cleaned.csv"))
        drafts = drafts[drafts["first_draft"].notna()]
        frame = y_frame(drafts, "first_draft", GROUP_ORDER, args.metrics, args.device)
        write_y(frame, out, args.metrics)


if __name__ == "__main__":
    main()
