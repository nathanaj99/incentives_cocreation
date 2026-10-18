# Data processing

Scripts that turn `data/study{1,2}/raw/` into `data/study{1,2}/processed/`.
Run from the repository root; every script processes both studies by default
(`--study 1` or `--study 2` to restrict, `--force` to overwrite). `run_all.sh`
at the repository root runs them in order, with toggles for the expensive steps.

| Step | Script | Output(s) | Cost |
|---|---|---|---|
| 1 | `time_spent.py` | `time_spent.csv` | cheap |
| 1 | `ai_requests.py` | `ai_requests.csv`, `ai_requests_explode.csv` | cheap |
| 1 | `ai_reliance.py` | `ai_reliance.csv` | cheap (all-MiniLM-L12-v2) |
| 2 | `label_ai_requests.py` | `ai_requests_labeled.csv` | OpenAI (`--timing-only`: cheap) |
| 2 | `extract_drafts.py` | `drafts_cleaned.csv` | OpenAI |
| 2 | `grade_quality.py` | S1: `technical_quality_run{1,2,3}.csv`, `technical_quality_median.csv`; S2: `technical_quality.csv` | OpenAI |
| 3 | `compute_Y.py` | `Y.csv` | embedding models |
| 3 | `compute_Y_first_draft.py` | `Y_first_draft.csv` | embedding models |
| 3 | `compute_Y_brainstorm.py` | `Y_brainstorm.csv` (Study 1 only) | embedding models |
| 3 | `compute_Y_by_quality.py` | `Y_by_quality.csv` | embedding models |
| 3 | `compute_Y_trajectory.py` | `Y_30secs_filtered_200words_<metric>.csv` (12 files) | embedding models, heaviest |
| 3 | `pca.py` | `PCA.pkl` | embedding models |

Dependencies between steps: `ai_requests.py` → `label_ai_requests.py` and
`extract_drafts.py` → `compute_Y_first_draft.py` and `pca.py`;
`grade_quality.py` → `compute_Y_by_quality.py`.

`human_judge_1.csv` (Study 1) is a manual annotation and is not generated.

## Published data and privacy

- Participants are identified by a random study ID, `participant_id`
  (`P001`-`P200` for Study 1, `P201`-`P600` for Study 2), never by their
  Prolific ID. Only participants in the analyzed sample are included, and
  Prolific IDs or study/session IDs that participants pasted into the text
  boxes have been replaced (by the participant's study ID or `[REDACTED]`).
- Snapshot timestamps are relative: each participant's first snapshot is set
  to 2000-01-01 00:00:00 UTC and later snapshots keep their exact offsets, so
  durations are preserved but real dates and times of day are not released.
  Session IDs are not released.
- Individual-level demographics are not released. Tables S1-S3 and S11 are
  built from `processed/demographics_summary.csv`, which holds only
  one-variable summary statistics (group sizes, means and standard errors,
  category counts and percentages) and was produced from the private data.

Shared code: `common.py` (paths, CLI, raw-data loading, OpenAI client),
`similarity.py` (similarity metrics and the participant-level outcome Y_i),
`prompts.py` (LLM system prompts).

## OpenAI steps

Set your key first: `export OPENAI_API_KEY=sk-...` (optionally `OPENAI_MODEL`,
default `gpt-6-luna`). The LLM steps checkpoint after every call and resume
where they stopped. A finished output is never recomputed unless you pass
`--force`.

## Embedding steps

These steps load `all-MiniLM-L12-v2`, `all-mpnet-base-v2`, `Qwen/Qwen3-Embedding-0.6B`,
`google/embeddinggemma-300m`, `AnnaWegmann/Style-Embedding`, `google/byt5-small`
and `AIDA-UPM/star` one at a time. A GPU is strongly recommended (`--device`,
default cuda > mps > cpu). To recompute only some metrics, use e.g.
`--metrics rougel self_bleu --force`; other columns of an existing file are kept.

## Python packages

pandas, numpy, scipy, scikit-learn, tqdm, nltk, sacrebleu, torch, transformers,
sentence-transformers, openai, pydantic.
