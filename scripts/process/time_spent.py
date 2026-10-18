"""
time_spent.csv: how each participant spent their time (minutes).

  total_time          first to last snapshot of the session
  brainstorm_time     first to last brainstorm-phase snapshot
  freetext_time       time in the main text box: summed duration of
                      consecutive 'main' snapshot blocks for AI participants;
                      the whole writing phase for SELF participants
  chat_time           summed duration of consecutive chat blocks (AI only)
  total_writing_time  first to last writing-phase snapshot
  num_turns_chat      number of chat blocks (AI only)
  idle_time           sum of the 10 longest gaps between snapshots

Inputs: raw/submissions.csv, raw/snapshots.csv
Output: processed/time_spent.csv
"""

import numpy as np
import pandas as pd

from common import load_snapshots, load_submissions, output, parse_args, save_csv

N_IDLE_GAPS = 10


def snapshot_type(row):
    if row["phase"] == "brainstorm":
        return "brainstorm"
    if row["phase"] == "writing":
        return "writing" if row["textbox"] == "main" else "chat"
    return None


def span_minutes(timestamps):
    return (timestamps.max() - timestamps.min()).total_seconds() / 60


def block_durations(writing):
    """Duration (min) of each run of consecutive same-type writing-phase snapshots."""
    s = writing.sort_values("timestamp").copy()
    s["block_id"] = (s["type"] != s["type"].shift()).cumsum()
    return s.groupby("block_id").agg(
        type=("type", "first"),
        minutes=("timestamp", span_minutes),
    )


def participant_times(subset):
    group_key = subset["group_key"].iloc[0]
    writing = subset[subset["phase"] == "writing"]

    total_writing_time = span_minutes(writing["timestamp"])
    freetext_time = total_writing_time
    chat_time = num_turns_chat = np.nan

    if "AI" in group_key:
        blocks = block_durations(writing)
        chat_time = blocks.loc[blocks["type"] == "chat", "minutes"].sum()
        num_turns_chat = int((blocks["type"] == "chat").sum())
        freetext_time = blocks.loc[blocks["type"] == "writing", "minutes"].sum()

    # Idle time: the largest gaps between consecutive snapshots
    gaps = subset.sort_values("timestamp")["timestamp"].diff().dt.total_seconds().iloc[1:]
    idle_time = gaps.nlargest(N_IDLE_GAPS).sum() / 60

    return {
        "group_key": group_key,
        "total_time": span_minutes(subset["timestamp"]),
        "brainstorm_time": span_minutes(subset.loc[subset["type"] == "brainstorm", "timestamp"]),
        "freetext_time": freetext_time,
        "chat_time": chat_time,
        "total_writing_time": total_writing_time,
        "num_turns_chat": num_turns_chat,
        "idle_time": idle_time,
    }


def main():
    args = parse_args(__doc__)
    for study in args.study:
        print(f"Study {study}")
        submissions = load_submissions(study)
        snaps = load_snapshots(study)
        snaps["type"] = snaps.apply(snapshot_type, axis=1)

        rows = [{"participant_id": pid, **participant_times(snaps[snaps["participant_id"] == pid])}
                for pid in submissions["participant_id"].unique()]
        save_csv(pd.DataFrame(rows), output(study, "time_spent.csv"))


if __name__ == "__main__":
    main()
