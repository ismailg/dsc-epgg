#!/usr/bin/env bash
set -euo pipefail

cd "${IWR_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

export RUN_DATE="${RUN_DATE:-20260325}"
export RUN_KIND="${RUN_KIND:-iwr}"
export TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-10}"
export EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-15}"
export PYTHON_BIN="${PYTHON_BIN:-/export/scratch/iguennou/runs/dsc-epgg/phase3-15seed-main-story-iwr-20260317/venv-py310/bin/python}"

./scripts/run_phase3_vecstraight_continuation_queue_iwr.sh \
  quadopt4 \
  50000:public_random \
  100000:sender_shuffle \
  100000:fixed0
