"""
Shared helpers for the data-processing scripts in scripts/process/.

Every script reads raw data from data/study{1,2}/raw/ (and earlier outputs
from data/study{1,2}/processed/) and writes to data/study{1,2}/processed/.
Run from the repository root, e.g.

    python scripts/process/time_spent.py                 # both studies
    python scripts/process/time_spent.py --study 2       # Study 2 only
    python scripts/process/compute_Y.py --force          # overwrite outputs

Expensive steps (LLM calls, embedding models) skip outputs that already exist
unless --force is passed, so re-running the pipeline never re-bills the API
by accident.

Set PROCESS_OUTPUT_DIR to write outputs somewhere other than data/ (e.g. to
compare a re-run against the committed files): outputs then go to
$PROCESS_OUTPUT_DIR/study{n}/.
"""

import argparse
import ast
import json
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
STUDIES = (1, 2)

# Study 1 = short story; Study 2 = product pitch
GROUP_ORDER = ["AI-DIV", "SELF-DIV", "AI-CONV", "SELF-CONV"]

# LLM used for labeling, draft extraction, and quality grading
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-luna")


# ============================================================
# PATHS
# ============================================================

def raw(study, filename):
    return DATA_DIR / f"study{study}" / "raw" / filename


def processed(study, filename):
    """Input path for a previously processed file."""
    return DATA_DIR / f"study{study}" / "processed" / filename


def output(study, filename):
    """Output path (data/study{n}/processed/ unless PROCESS_OUTPUT_DIR is set)."""
    root = os.environ.get("PROCESS_OUTPUT_DIR")
    base = Path(root) / f"study{study}" if root else DATA_DIR / f"study{study}" / "processed"
    base.mkdir(parents=True, exist_ok=True)
    return base / filename


def should_skip(path, force):
    """True (and a message) if `path` exists and we are not forcing a rerun."""
    if Path(path).exists() and not force:
        print(f"  {display(path)} exists; skipping (use --force to regenerate)")
        return True
    return False


def display(path):
    try:
        return Path(path).resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"  Saved {display(path)} ({len(df)} rows)")


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args(description, studies=STUDIES, extra=None):
    """
    Standard CLI: --study {1,2} (repeatable; default = all applicable
    studies) and --force. `extra(parser)` can add script-specific options.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--study", type=int, nargs="+", choices=list(studies),
                        default=list(studies), help="which studies to process (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="regenerate outputs even if they already exist")
    if extra is not None:
        extra(parser)
    return parser.parse_args()


# ============================================================
# LOADING RAW DATA
# ============================================================

def parse_json_cell(x):
    """Parse a JSON (or Python-literal) string cell; pass through anything else."""
    if not isinstance(x, str):
        return x
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return ast.literal_eval(x)


def to_utc(series):
    """
    Parse snapshot timestamps such as '2000-01-01 00:03:12.361+00' (works on
    pandas 1.x and 2.x). Published timestamps are participant-relative: each
    participant's first snapshot is 2000-01-01 00:00:00, so only differences
    within a participant are meaningful.
    """
    try:
        return pd.to_datetime(series, format="ISO8601", utc=True)
    except (TypeError, ValueError):
        return pd.to_datetime(series, utc=True)


def load_submissions(study):
    return pd.read_csv(raw(study, "submissions.csv"))


def load_snapshots(study):
    snaps = pd.read_csv(raw(study, "snapshots.csv"))
    snaps["timestamp"] = to_utc(snaps["timestamp"])
    return snaps


def split_transcript(transcript):
    """
    Split an AI chat transcript (alternating participant / AI turns, starting
    with the participant) into (requests, outputs).
    """
    turns = parse_json_cell(transcript)
    requests = [t["content"] for i, t in enumerate(turns) if i % 2 == 0]
    outputs = [t["content"] for i, t in enumerate(turns) if i % 2 == 1]
    return requests, outputs


def ai_submissions(study):
    """Submissions from the AI-assisted groups only."""
    sub = load_submissions(study)
    return sub[sub["group_key"].str.contains("AI")].copy()


# ============================================================
# OPENAI
# ============================================================

def openai_client():
    """
    OpenAI client; the API key is read from the OPENAI_API_KEY environment
    variable. Retries are handled by the calling code, not the SDK.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export your key first, e.g.\n"
            "    export OPENAI_API_KEY=sk-..."
        )
    from openai import OpenAI
    return OpenAI(max_retries=0)


def atomic_write_csv(df, path):
    """Write a CSV via a temporary file so an interrupted run never leaves a partial file."""
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_json(value, path):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
