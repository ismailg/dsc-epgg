#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 CHECKPOINT_DIR OUT_ROOT [MAX_WORKERS]" >&2
  exit 2
fi

CHECKPOINT_DIR="$1"
OUT_ROOT="$2"
MAX_WORKERS="${3:-2}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST_PATH="${CHECKPOINT_DIR}/phase3_seed_expansion_manifest.json"

wait_for_file() {
  local path="$1"
  while [[ ! -f "${path}" ]]; do
    sleep 30
  done
}

mkdir -p "${OUT_ROOT}/report"

cd "${REPO_ROOT}"

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

wait_for_file "${MANIFEST_PATH}"

bash scripts/run_phase3_sameckpt_eval_mode_local.sh sender_shuffle "${CHECKPOINT_DIR}" "${OUT_ROOT}" "${MAX_WORKERS}"

"${PYTHON_BIN}" -m src.analysis.summarize_phase3_channel_controls \
  --mode_suite learned "outputs/eval/phase3_annealed_ext150k_15seeds/suite/checkpoint_suite_main.csv" \
  --mode_suite sender_shuffle "${OUT_ROOT}/sender_shuffle/suite/checkpoint_suite_main.csv" \
  --out_dir "${OUT_ROOT}/report/sender_shuffle_summary"
