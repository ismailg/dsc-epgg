#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-local}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONUNBUFFERED=1
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY="${OBJC_DISABLE_INITIALIZE_FORK_SAFETY:-YES}"

printf '[recommended-smoke] step=1 noise-sweep subset\n'
OBS_MODES_STR="private public_noisy public_exact" \
SIGMA_VALUES_STR="0.5" \
SEEDS_STR="101" \
N_EVAL_EPISODES="${N_EVAL_EPISODES:-2}" \
MAX_WORKERS="${MAX_WORKERS:-1}" \
RUN_DATE="${RUN_DATE}" \
PYTHON_BIN="${PYTHON_BIN}" \
"${REPO_ROOT}/scripts/run_phase3_vecstraight_noise_sweep_eval.sh" "${RUN_KIND}"

printf '[recommended-smoke] step=2 comm-history factorial smoke\n'
SEEDS_STR="101 202" \
N_EPISODES="${N_EPISODES:-16}" \
T_STEPS="${T_STEPS:-100}" \
LOG_INTERVAL="${LOG_INTERVAL:-8}" \
REGIME_LOG_INTERVAL="${REGIME_LOG_INTERVAL:-8}" \
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-8}" \
TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-1}" \
RUN_DATE="${RUN_DATE}" \
PYTHON_BIN="${PYTHON_BIN}" \
"${REPO_ROOT}/scripts/run_phase3_vecstraight_comm_history_factorial.sh" all "${RUN_KIND}"

printf '[recommended-smoke] done run_kind=%s run_date=%s\n' "${RUN_KIND}" "${RUN_DATE}"
