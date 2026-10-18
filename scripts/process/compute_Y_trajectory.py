"""
Y_30secs_filtered_200words_<metric>.csv: similarity trajectories while
writing (Fig. 1b, Fig. S13, Fig. S15b).

Every 30 s after the first writing-phase snapshot, take the main-text-box
snapshot closest in time (up to 59 time points); time point 59, and every
point after the participant stopped, is the final submission. Snapshots
shorter than MIN_WORDS are treated as missing. At each time point, Y_i is
computed within each group among the participants with a valid text; if a
group has <= 5 valid texts at that point, the whole group is NaN.

One file per metric, columns: participant_id, group_key, 0, 1, ..., 59
(column t = minute (t + 1) / 2). The "200words" in the filename is kept for
both studies for compatibility, although Study 2 uses a 70-word minimum.

Inputs: raw/submissions.csv, raw/snapshots.csv
Output: processed/Y_30secs_filtered_200words_<metric>.csv (one per metric;
        '/' in model names becomes '_')

The most expensive step: 4 groups x 60 time points per metric.
"""

import numpy as np
import pandas as pd

from common import (GROUP_ORDER, load_snapshots, load_submissions, output, parse_args,
                    save_csv, should_skip)
from similarity import add_metric_args, y_table

MIN_WORDS = {1: 200, 2: 70}  # short story vs. product pitch
INTERVAL = pd.Timedelta(seconds=30)
N_POINTS = 60                # 59 snapshots + the final submission
MIN_VALID_PER_GROUP = 6      # i.e. more than 5 valid texts


def snapshot_grid(study):
    """DataFrame (participants x N_POINTS) of texts, NaN where too short."""
    submissions = load_submissions(study)
    snaps = load_snapshots(study)
    snaps = snaps[snaps["phase"] == "writing"]

    rows = []
    for pid in submissions["participant_id"].unique():
        final = submissions.loc[submissions["participant_id"] == pid, "final_text"].iloc[0]
        subset = snaps[snaps["participant_id"] == pid].sort_values("timestamp")
        main_box = subset[subset["textbox"] == "main"]["timestamp"]

        texts = []
        current = subset["timestamp"].min() + INTERVAL
        while current <= subset["timestamp"].max() and len(texts) < N_POINTS - 1:
            texts.append(subset.loc[(main_box - current).abs().idxmin(), "text"])
            current += INTERVAL
        texts += [final] * (N_POINTS - len(texts))  # final submission onward
        rows.append(texts)

    grid = pd.DataFrame(rows)
    min_words = MIN_WORDS[study]
    grid = grid.apply(lambda col: col.map(
        lambda x: x if isinstance(x, str) and len(x.split()) >= min_words else np.nan))
    grid.insert(0, "group_key", submissions.drop_duplicates("participant_id")["group_key"].tolist())
    grid.insert(0, "participant_id", submissions["participant_id"].unique())
    return grid


def main():
    args = parse_args(__doc__, extra=add_metric_args)
    for study in args.study:
        print(f"Study {study}")
        def path(metric):
            return output(study, f"Y_30secs_filtered_200words_{metric.replace('/', '_')}.csv")

        metrics = [m for m in args.metrics if not should_skip(path(m), args.force)]
        if not metrics:
            continue

        grid = snapshot_grid(study)
        groups = {(g, t): grid.loc[grid["group_key"] == g, t].tolist()
                  for g in GROUP_ORDER for t in range(N_POINTS)}
        base = pd.concat([grid.loc[grid["group_key"] == g, ["participant_id", "group_key"]] for g in GROUP_ORDER],
                         ignore_index=True)

        for metric in metrics:  # save each metric as soon as it is done
            table = y_table(groups, [metric], args.device, min_valid=MIN_VALID_PER_GROUP)
            frame = base.copy()
            for t in range(N_POINTS):
                frame[str(t)] = sum((table[metric][(g, t)] for g in GROUP_ORDER), [])
            save_csv(frame, path(metric))


if __name__ == "__main__":
    main()
