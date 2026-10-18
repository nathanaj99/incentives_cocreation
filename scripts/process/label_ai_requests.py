"""
ai_requests_labeled.csv: each AI request labeled with the type of assistance
requested (drafting / editing / creative consulting / miscellaneous) and the
time it was made.

  1. LLM labeling (OpenAI; needs OPENAI_API_KEY). Each request/reply pair is
     labeled independently. Progress is checkpointed after every call in
     ai_requests_labeled.checkpoint.csv, so an interrupted run resumes where
     it stopped; failed calls stay unlabeled and are retried on the next run.
  2. Timing (no API). A request's time is that of the chat-box snapshot whose
     text best matches the request (word-set Jaccard), measured from the
     participant's first writing-phase snapshot.

Inputs: processed/ai_requests_explode.csv, raw/snapshots.csv
Output: processed/ai_requests_labeled.csv

Use --timing-only to redo step 2 on an existing ai_requests_labeled.csv.
"""

import json
import re
import time
from typing import Literal

import pandas as pd
from tqdm.auto import tqdm

from common import (OPENAI_MODEL, atomic_write_csv, load_snapshots, openai_client, output,
                    parse_args, save_csv, should_skip)
from prompts import LABEL_SYSTEM_PROMPT, LABEL_TYPES

MAX_RETRIES = 3


# ============================================================
# 1. LLM LABELING
# ============================================================

def label_interaction(client, ai_request, ai_output):
    """Return a valid label, or None if every attempt fails."""
    from pydantic import BaseModel

    class InteractionLabel(BaseModel):
        type: Literal["drafting", "editing", "creative consulting", "miscellaneous"]

    payload = json.dumps({"ai_requests": ai_request, "ai_outputs": ai_output}, ensure_ascii=False)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.parse(
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": LABEL_SYSTEM_PROMPT},
                    {"role": "user", "content": "Classify this interaction:\n" + payload},
                ],
                text_format=InteractionLabel,
            )
            if response.output_parsed is None:
                raise ValueError("The API returned no parsed classification.")
            return response.output_parsed.type
        except Exception as error:
            tqdm.write(f"Error on attempt {attempt + 1}/{MAX_RETRIES}: {error}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    # An API failure is NOT evidence for the miscellaneous category
    return None


def read_str_csv(path):
    # Keep IDs, empty strings, and literal text such as "NA" exactly as written
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def label_requests(study):
    """Label every request, resuming from the checkpoint. Returns the labeled frame or None."""
    df = read_str_csv(output(study, "ai_requests_explode.csv"))
    columns = list(df.columns)
    checkpoint = output(study, "ai_requests_labeled.checkpoint.csv")

    if checkpoint.exists():
        results = read_str_csv(checkpoint)
        # The checkpoint must belong to exactly this input (same rows, same order)
        if list(results.columns) != columns + ["type"] or not results[columns].equals(df):
            raise ValueError(f"{checkpoint} does not match the current input; delete it to relabel.")
    else:
        results = df.copy()
        results["type"] = ""

    # Track completion by row position (duplicate requests stay separate rows)
    pending = results.index[~results["type"].isin(LABEL_TYPES)].tolist()
    print(f"  {len(df) - len(pending)} / {len(df)} rows already labeled")
    atomic_write_csv(results, checkpoint)

    if pending:
        client = openai_client()
        for i in tqdm(pending, desc="Labeling interactions"):
            label = label_interaction(client, df.at[i, "ai_requests"], df.at[i, "ai_outputs"])
            if label is None:
                tqdm.write(f"Row {i} failed; leaving it pending.")
            else:
                results.at[i, "type"] = label
            atomic_write_csv(results, checkpoint)

    remaining = int((~results["type"].isin(LABEL_TYPES)).sum())
    if remaining:
        print(f"  {remaining} rows are still unlabeled; rerun to retry them.")
        return None
    print(results["type"].value_counts().to_string())
    return results


# ============================================================
# 2. REQUEST TIMING
# ============================================================

def jaccard_similarity(text1, text2):
    tokens1 = set(re.findall(r"\b\w+\b", str(text1).lower()))
    tokens2 = set(re.findall(r"\b\w+\b", str(text2).lower()))
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def add_request_times(requests, snapshots):
    """Seconds / minutes from the start of writing at which each request was typed."""
    writing = snapshots[snapshots["phase"] == "writing"].sort_values("timestamp", kind="stable")
    times = []
    for _, row in requests.iterrows():
        subset = writing[writing["participant_id"] == row["participant_id"]]
        from_start = (subset["timestamp"] - subset["timestamp"].iloc[0]).dt.total_seconds()
        chat = subset["textbox"] == "chat"
        similarity = subset.loc[chat, "text"].apply(lambda x: jaccard_similarity(x, row["ai_requests"]))
        times.append(from_start[similarity.idxmax()])
    requests = requests.copy()
    requests["time_from_start"] = times
    requests["time_from_start_mins"] = requests["time_from_start"] / 60
    return requests


def main():
    def extra(parser):
        parser.add_argument("--timing-only", action="store_true",
                            help="recompute request times on an existing ai_requests_labeled.csv (no API)")

    args = parse_args(__doc__, extra=extra)
    for study in args.study:
        print(f"Study {study}")
        out = output(study, "ai_requests_labeled.csv")

        if args.timing_only:
            labeled = pd.read_csv(out).drop(columns=["time_from_start", "time_from_start_mins"], errors="ignore")
        else:
            if should_skip(out, args.force):
                continue
            labeled = label_requests(study)
            if labeled is None:
                continue

        save_csv(add_request_times(labeled, load_snapshots(study)), out)


if __name__ == "__main__":
    main()
