#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 OUT_ROOT" >&2
  exit 2
fi

OUT_ROOT="$1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

wait_for_file() {
  local path="$1"
  while [[ ! -f "${path}" ]]; do
    sleep 20
  done
}

wait_for_file "${OUT_ROOT}/fixed0/suite/checkpoint_suite_main.csv"
wait_for_file "${OUT_ROOT}/uniform/suite/checkpoint_suite_main.csv"
wait_for_file "${OUT_ROOT}/public_random/suite/checkpoint_suite_main.csv"

mkdir -p "${OUT_ROOT}/report"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m src.analysis.summarize_phase3_channel_controls \
  --mode_suite learned "outputs/eval/phase3_annealed_ext150k_15seeds/suite/checkpoint_suite_main.csv" \
  --mode_suite always_zero "${OUT_ROOT}/fixed0/suite/checkpoint_suite_main.csv" \
  --mode_suite indep_random "${OUT_ROOT}/uniform/suite/checkpoint_suite_main.csv" \
  --mode_suite public_random "${OUT_ROOT}/public_random/suite/checkpoint_suite_main.csv" \
  --out_dir "${OUT_ROOT}/report"
