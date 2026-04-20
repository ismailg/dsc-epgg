#!/usr/bin/env bash
set -euo pipefail

: "${IWR_RUN_DIR:?IWR_RUN_DIR must be set by the IWR launcher}"
: "${IWR_PROJECT_DIR:?IWR_PROJECT_DIR must be set by the IWR launcher}"

cd "${IWR_PROJECT_DIR}"

RUN_DATE="${RUN_DATE:-20260417}"
PYTHON_BIN="${PYTHON_BIN:-${IWR_PROJECT_DIR}/.venv/bin/python}"
EXISTING_ROOT_BASE="${EXISTING_ROOT_BASE:-${IWR_PROJECT_DIR}/inputs/clean_msgsource}"
OUT_BASE="${OUT_BASE:-${IWR_RUN_DIR}/outputs/eval}"
MAX_WORKERS="${MAX_WORKERS:-8}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-300}"
SAMPLE_EVAL_EPISODES="${SAMPLE_EVAL_EPISODES:-1}"
LOG_PATH="${IWR_RUN_DIR}/zeroaux_crossover_eval.log"

export PYTHONUNBUFFERED=1
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY="${OBJC_DISABLE_INITIALIZE_FORK_SAFETY:-YES}"

mkdir -p "${IWR_RUN_DIR}"
: > "${LOG_PATH}"

{
  echo "[iwr crossover] project=${IWR_PROJECT_DIR}"
  echo "[iwr crossover] existing_root_base=${EXISTING_ROOT_BASE}"
  echo "[iwr crossover] out_base=${OUT_BASE}"
  RUN_DATE="${RUN_DATE}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  EXISTING_ROOT_BASE="${EXISTING_ROOT_BASE}" \
  OUT_BASE="${OUT_BASE}" \
  MAX_WORKERS="${MAX_WORKERS}" \
  N_EVAL_EPISODES="${N_EVAL_EPISODES}" \
  SAMPLE_EVAL_EPISODES="${SAMPLE_EVAL_EPISODES}" \
  "${IWR_PROJECT_DIR}/scripts/run_phase3_vecstraight_zeroaux_crossover_eval.sh" iwr
} 2>&1 | tee -a "${LOG_PATH}"
