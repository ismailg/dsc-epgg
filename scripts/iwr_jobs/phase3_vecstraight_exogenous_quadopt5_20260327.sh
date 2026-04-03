#!/usr/bin/env bash
set -euo pipefail

cd "${IWR_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

export RUN_DATE="${RUN_DATE:-20260327}"
export RUN_KIND="${RUN_KIND:-iwr}"
export TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-15}"
export EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-15}"
export PYTHON_BIN="${PYTHON_BIN:-/export/scratch/iguennou/runs/dsc-epgg/phase3-15seed-main-story-iwr-20260317/venv-py310/bin/python}"

./scripts/run_phase3_vecstraight_exogenous_family.sh uniform iwr
./scripts/run_phase3_vecstraight_exogenous_family.sh fixed1 iwr
