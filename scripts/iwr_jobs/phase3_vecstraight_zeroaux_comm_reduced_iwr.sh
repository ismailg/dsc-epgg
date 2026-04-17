#!/usr/bin/env bash
set -euo pipefail

cd "${IWR_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

export RUN_DATE="${RUN_DATE:-20260415}"
export RUN_KIND="${RUN_KIND:-iwr}"
export TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-15}"
export PYTHON_BIN="${PYTHON_BIN:-${IWR_PROJECT_DIR:-$(pwd)}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "missing python interpreter: ${PYTHON_BIN}" >&2
  exit 2
fi

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

./scripts/run_phase3_vecstraight_comm_history_zeroaux_reduced.sh "${RUN_KIND}"
