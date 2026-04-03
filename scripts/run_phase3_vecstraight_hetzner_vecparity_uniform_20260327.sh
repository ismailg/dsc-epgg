#!/usr/bin/env bash
set -euo pipefail

RUN_DATE="${RUN_DATE:-20260327}"
MODE="uniform"
HETZNER_PROJECT_DIR="${HETZNER_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
HETZNER_RUN_DIR="${HETZNER_RUN_DIR:-/root/compute-work/runs/dsc-epgg-vectorized/phase3-vecstraight-hetzner-uniform-vecparity-20260327}"
NEXT_ROOT="${NEXT_ROOT:-${HETZNER_PROJECT_DIR}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}}"
MANIFEST_DIR="${MANIFEST_DIR:-${NEXT_ROOT}/manifests_hetzner_${MODE}_vecparity}"
COMM_MANIFEST="${COMM_MANIFEST:-${MANIFEST_DIR}/cond1_all_15seeds_hetzner.txt}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${MANIFEST_DIR}/cond2_all_15seeds_hetzner.txt}"
STATUS_DIR="${STATUS_DIR:-${HETZNER_RUN_DIR}/status}"
PROGRESS_LOG="${PROGRESS_LOG:-${STATUS_DIR}/progress.log}"

COND1_BATCH_ROOT="${COND1_BATCH_ROOT:-${HETZNER_PROJECT_DIR}/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323/outputs/phase3_cond1_15seeds_train_only/train}"
COND1_STANDALONE_ROOT="${COND1_STANDALONE_ROOT:-${HETZNER_PROJECT_DIR}/iwr-results/phase3-150k-straight-c1-s101-subproc-20260323/outputs/phase3_straight_c1_s101_subproc/checkpoints}"
COND2_ROOT="${COND2_ROOT:-${HETZNER_PROJECT_DIR}/iwr-results/phase3-150k-cond2-15seed-trainonly-20260324/code/outputs/train/phase3-150k-cond2-15seed-trainonly-20260324/outputs/phase3_cond2_15seeds_train_only/train}"

mkdir -p "${STATUS_DIR}"

log() {
  printf '[%s] [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${MODE}" "$*" | tee -a "${PROGRESS_LOG}"
}

export PYTHON_BIN="${PYTHON_BIN:-${HETZNER_PROJECT_DIR}/.venv/bin/python}"
export PATH="${HETZNER_PROJECT_DIR}/.venv/bin:${PATH}"

log "preparing Hetzner-local manifests for vectorized parity relaunch"
NEXT_ROOT="${NEXT_ROOT}" \
MANIFEST_DIR="${MANIFEST_DIR}" \
COMM_MANIFEST="${COMM_MANIFEST}" \
BASELINE_MANIFEST="${BASELINE_MANIFEST}" \
COND1_BATCH_ROOT="${COND1_BATCH_ROOT}" \
COND1_STANDALONE_ROOT="${COND1_STANDALONE_ROOT}" \
COND2_ROOT="${COND2_ROOT}" \
./scripts/run_phase3_vecstraight_prepare_iwr_manifests.sh "${RUN_DATE}" | tee -a "${PROGRESS_LOG}"

log "starting vectorized continuation branch=50000 mode=${MODE}"
NEXT_ROOT="${NEXT_ROOT}" \
COMM_MANIFEST="${COMM_MANIFEST}" \
BASELINE_MANIFEST="${BASELINE_MANIFEST}" \
RUN_DATE="${RUN_DATE}" \
TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-5}" \
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-15}" \
NUM_ENVS="${NUM_ENVS:-8}" \
ENV_BACKEND="${ENV_BACKEND:-subproc}" \
ENV_START_METHOD="${ENV_START_METHOD:-spawn}" \
REGIME_LOG_INTERVAL="${REGIME_LOG_INTERVAL:-400}" \
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25000}" \
MSG_ENTROPY_COEFF="${MSG_ENTROPY_COEFF:-0.01}" \
MSG_ENTROPY_COEFF_FINAL="${MSG_ENTROPY_COEFF_FINAL:-0.0}" \
./scripts/run_phase3_vecstraight_continuation_mode.sh 50000 "${MODE}" hetzner | tee -a "${PROGRESS_LOG}"

log "completed vectorized continuation branch=50000 mode=${MODE}"
