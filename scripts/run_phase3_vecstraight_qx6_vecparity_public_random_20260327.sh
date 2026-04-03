#!/usr/bin/env bash
set -euo pipefail

RUN_DATE="${RUN_DATE:-20260327}"
MODE="public_random"
IWR_PROJECT_DIR="${IWR_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
IWR_RUN_DIR="${IWR_RUN_DIR:-/export/scratch/iguennou/runs/dsc-epgg-vectorized/phase3-vecstraight-qx6-public-random-vecparity-20260327}"
NEXT_ROOT="${NEXT_ROOT:-${IWR_PROJECT_DIR}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}}"
MANIFEST_DIR="${MANIFEST_DIR:-${NEXT_ROOT}/manifests_iwr_${MODE}_vecparity}"
COMM_MANIFEST="${COMM_MANIFEST:-${MANIFEST_DIR}/cond1_all_15seeds_iwr.txt}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${MANIFEST_DIR}/cond2_all_15seeds_iwr.txt}"
STATUS_DIR="${STATUS_DIR:-${IWR_RUN_DIR}/status}"
PROGRESS_LOG="${PROGRESS_LOG:-${STATUS_DIR}/progress.log}"

mkdir -p "${STATUS_DIR}"

log() {
  printf '[%s] [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${MODE}" "$*" | tee -a "${PROGRESS_LOG}"
}

export PYTHON_BIN="${PYTHON_BIN:-/export/scratch/iguennou/runs/dsc-epgg/phase3-15seed-main-story-iwr-20260317/venv-py310/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "missing IWR python interpreter: ${PYTHON_BIN}" >&2
  exit 2
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

log "preparing IWR-local manifests for vectorized parity relaunch"
NEXT_ROOT="${NEXT_ROOT}" \
MANIFEST_DIR="${MANIFEST_DIR}" \
COMM_MANIFEST="${COMM_MANIFEST}" \
BASELINE_MANIFEST="${BASELINE_MANIFEST}" \
./scripts/run_phase3_vecstraight_prepare_iwr_manifests.sh "${RUN_DATE}" | tee -a "${PROGRESS_LOG}"

log "starting vectorized continuation branch=50000 mode=${MODE}"
NEXT_ROOT="${NEXT_ROOT}" \
COMM_MANIFEST="${COMM_MANIFEST}" \
BASELINE_MANIFEST="${BASELINE_MANIFEST}" \
RUN_DATE="${RUN_DATE}" \
TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-15}" \
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-15}" \
NUM_ENVS="${NUM_ENVS:-8}" \
ENV_BACKEND="${ENV_BACKEND:-subproc}" \
ENV_START_METHOD="${ENV_START_METHOD:-spawn}" \
REGIME_LOG_INTERVAL="${REGIME_LOG_INTERVAL:-400}" \
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25000}" \
MSG_ENTROPY_COEFF="${MSG_ENTROPY_COEFF:-0.01}" \
MSG_ENTROPY_COEFF_FINAL="${MSG_ENTROPY_COEFF_FINAL:-0.0}" \
./scripts/run_phase3_vecstraight_continuation_mode.sh 50000 "${MODE}" iwr | tee -a "${PROGRESS_LOG}"

log "completed vectorized continuation branch=50000 mode=${MODE}"
