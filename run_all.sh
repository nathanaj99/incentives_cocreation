#!/usr/bin/env bash
# Reproduce the processed data, figures, and tables.
#
#   ./run_all.sh                                   # cheap steps + figures/tables
#   STUDIES=2 RUN_LLM=1 ./run_all.sh               # Study 2 only, with LLM steps
#   RUN_LLM=1 RUN_EMBED=1 FORCE=1 ./run_all.sh     # regenerate everything
#
# Toggles (environment variables):
#   STUDIES    which studies to process            (default: "1 2")
#   RUN_LLM    1 = OpenAI steps: request labels, draft extraction, quality
#              grading. Needs OPENAI_API_KEY. Costs money.  (default: 0)
#   RUN_EMBED  1 = similarity outcomes (Y*.csv) and PCA.pkl with all
#              embedding models. GPU strongly recommended.  (default: 0)
#   RUN_VIZ    1 = figures and tables                  (default: 1)
#   FORCE      1 = overwrite existing processed files  (default: 0)
#
# Steps that are switched off use the processed files already in data/.

set -euo pipefail
cd "$(dirname "$0")"

STUDIES=${STUDIES:-"1 2"}
RUN_LLM=${RUN_LLM:-0}
RUN_EMBED=${RUN_EMBED:-0}
RUN_VIZ=${RUN_VIZ:-1}
FORCE=${FORCE:-0}

FORCE_FLAG=""
[[ "$FORCE" == 1 ]] && FORCE_FLAG="--force"

process() {  # process <script> [extra args]: run a processing step for $STUDIES
    local script=$1; shift
    echo "== process/$script"
    python "scripts/process/$script" --study $STUDIES $FORCE_FLAG "$@"
}

# ---- 1. Cheap, deterministic steps (raw data -> processed) ----
process time_spent.py
process ai_requests.py
process ai_reliance.py

# ---- 2. LLM steps (OpenAI API) ----
if [[ "$RUN_LLM" == 1 ]]; then
    process label_ai_requests.py     # -> ai_requests_labeled.csv
    process extract_drafts.py        # -> drafts_cleaned.csv
    process grade_quality.py         # -> technical_quality*.csv
else
    # Request times need no API; refresh them on the existing labels
    process label_ai_requests.py --timing-only
fi

# ---- 3. Similarity outcomes and PCA (embedding models) ----
if [[ "$RUN_EMBED" == 1 ]]; then
    process compute_Y.py
    process compute_Y_first_draft.py
    if [[ " $STUDIES " == *" 1 "* ]]; then   # brainstorms exist in Study 1 only
        echo "== process/compute_Y_brainstorm.py"
        python scripts/process/compute_Y_brainstorm.py --study 1 $FORCE_FLAG
    fi
    process compute_Y_by_quality.py
    process compute_Y_trajectory.py
    process pca.py
fi

# ---- 4. Figures and tables (always use both studies) ----
if [[ "$RUN_VIZ" == 1 ]]; then
    for script in scripts/viz/figure_*.py scripts/viz/table_*.py; do
        echo "== ${script#scripts/}"
        python "$script"
    done
fi
