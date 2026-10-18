"""
LLM-as-judge technical quality of the final submissions (OpenAI; needs
OPENAI_API_KEY). Each submission gets two 1-4 rubric scores plus short
explanations; technical_quality_score is their sum (2-8).

  Study 1 (short story):   organization + technique. Graded in 3 independent
                           runs; technical_quality_median.csv holds the median
                           total score over the runs.
  Study 2 (product pitch): information + communication. One run.

Results are saved after every API call, and a rerun skips submissions that
already have a score (so an interrupted run resumes). Failed calls are
saved with blank scores and retried on the next run.

Input:   raw/submissions.csv
Outputs: Study 1 -- processed/technical_quality_run{1,2,3}.csv,
                    processed/technical_quality_median.csv
         Study 2 -- processed/technical_quality.csv
"""

import time
from typing import Literal

import pandas as pd
from tqdm.auto import tqdm

from common import OPENAI_MODEL, load_submissions, openai_client, output, parse_args, save_csv
from prompts import PITCH_QUALITY_PROMPT, STORY_QUALITY_PROMPT

MAX_RETRIES = 3
N_RUNS_STUDY1 = 3

# Rubric dimensions and prompt framing per study
RUBRICS = {
    1: dict(system_prompt=STORY_QUALITY_PROMPT, dims=("organization", "technique"),
            noun="story", tag="story"),
    2: dict(system_prompt=PITCH_QUALITY_PROMPT, dims=("information", "communication"),
            noun="product pitch", tag="product_pitch"),
}


def output_columns(dims):
    cols = ["participant_id"]
    for d in dims:
        cols += [f"{d}_score", f"{d}_explanation"]
    return cols + ["technical_quality_score"]


def evaluation_schema(dims):
    """Pydantic model with a 1-4 score and an explanation per rubric dimension."""
    from pydantic import create_model
    fields = {}
    for d in dims:
        fields[f"{d}_score"] = (Literal[1, 2, 3, 4], ...)
        fields[f"{d}_explanation"] = (str, ...)
    return create_model("Evaluation", **fields)


def evaluate(client, text, rubric, schema):
    """Scores for one submission; all-None if every attempt fails."""
    dims = rubric["dims"]
    user_prompt = f"""
Evaluate the following submitted {rubric['noun']} according to the rubric.

<{rubric['tag']}>
{text}
</{rubric['tag']}>
"""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.parse(
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": rubric["system_prompt"]},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=schema,
            )
            e = response.output_parsed.model_dump()
            e["technical_quality_score"] = sum(e[f"{d}_score"] for d in dims)
            return e
        except Exception as error:
            print(f"\nError on attempt {attempt + 1}/{MAX_RETRIES}: {error}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return {c: None for c in output_columns(dims)[1:]}


def grade_all(study, out_path, client_holder):
    """Grade every submission not yet scored in out_path, saving after each call."""
    rubric = RUBRICS[study]
    columns = output_columns(rubric["dims"])
    schema = evaluation_schema(rubric["dims"])
    submissions = load_submissions(study)

    if out_path.exists():
        results = pd.read_csv(out_path)
        # Keep only successful rows; failed ones are retried below
        results = results[results["technical_quality_score"].notna()]
    else:
        results = pd.DataFrame(columns=columns)
    completed = set(results["participant_id"].astype(str))
    print(f"  {out_path.name}: {len(completed)}/{len(submissions)} already graded")

    for _, row in tqdm(submissions[["participant_id", "final_text"]].iterrows(),
                       total=len(submissions), desc="Grading"):
        pid = str(row["participant_id"])
        if pid in completed:
            continue
        text = row["final_text"]
        if pd.isna(text) or not str(text).strip():
            evaluation = {c: None for c in columns[1:]}  # nothing to grade
        else:
            client_holder["client"] = client_holder.get("client") or openai_client()
            evaluation = evaluate(client_holder["client"], str(text), rubric, schema)
        results = pd.concat([results, pd.DataFrame([{"participant_id": pid, **evaluation}])],
                            ignore_index=True)
        results.to_csv(out_path, index=False)  # save after every API call

    print(f"  Saved {out_path.name}")
    return results


def main():
    args = parse_args(__doc__)
    client_holder = {}  # create the OpenAI client only if a call is needed
    for study in args.study:
        print(f"Study {study}")
        if args.force:
            print("  --force: existing scores are kept; delete the files to regrade from scratch")

        if study == 1:
            runs = [grade_all(1, output(1, f"technical_quality_run{r}.csv"), client_holder)
                    for r in range(1, N_RUNS_STUDY1 + 1)]
            median = (pd.concat(runs, ignore_index=True)
                      .groupby("participant_id", as_index=False)["technical_quality_score"].median())
            save_csv(median, output(1, "technical_quality_median.csv"))
        else:
            grade_all(2, output(2, "technical_quality.csv"), client_holder)


if __name__ == "__main__":
    main()
