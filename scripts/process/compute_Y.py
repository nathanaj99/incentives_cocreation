"""
Y.csv: similarity of each participant's FINAL submission to the other final
submissions in their group (see similarity.py), for every metric.

Input:  raw/submissions.csv
Output: processed/Y.csv  (rows ordered AI-DIV, SELF-DIV, AI-CONV, SELF-CONV)

Loads every embedding model in turn; a GPU is strongly recommended.
"""

from common import GROUP_ORDER, load_submissions, output, parse_args, should_skip
from similarity import add_metric_args, write_y, y_frame


def main():
    args = parse_args(__doc__, extra=add_metric_args)
    for study in args.study:
        print(f"Study {study}")
        out = output(study, "Y.csv")
        if should_skip(out, args.force):
            continue
        submissions = load_submissions(study)
        frame = y_frame(submissions, "final_text", GROUP_ORDER, args.metrics, args.device)
        write_y(frame, out, args.metrics)


if __name__ == "__main__":
    main()
