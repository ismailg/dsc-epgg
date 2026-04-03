#!/usr/bin/env bash
set -euo pipefail

RUN_DATE="${RUN_DATE:-20260328}"
HETZNER_PROJECT_DIR="${HETZNER_PROJECT_DIR:-/root/compute-work/projects/dsc-epgg-vectorized}"
HETZNER_RUN_DIR="${HETZNER_RUN_DIR:-/root/compute-work/runs/dsc-epgg-vectorized/phase3-vecstraight-exogenous-hetzner-20260328}"
NEXT_ROOT="${NEXT_ROOT:-${HETZNER_PROJECT_DIR}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}}"
MANIFEST_DIR="${MANIFEST_DIR:-${NEXT_ROOT}/manifests_hetzner_exogenous}"
COMM_MANIFEST="${COMM_MANIFEST:-${MANIFEST_DIR}/cond1_all_15seeds_hetzner.txt}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${MANIFEST_DIR}/cond2_all_15seeds_hetzner.txt}"
STATUS_DIR="${STATUS_DIR:-${HETZNER_RUN_DIR}/status}"
PROGRESS_LOG="${PROGRESS_LOG:-${STATUS_DIR}/progress.log}"

COND1_BATCH_ROOT="${COND1_BATCH_ROOT:-${HETZNER_PROJECT_DIR}/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323/outputs/phase3_cond1_15seeds_train_only/train}"
COND1_STANDALONE_ROOT="${COND1_STANDALONE_ROOT:-${HETZNER_PROJECT_DIR}/iwr-results/phase3-150k-straight-c1-s101-subproc-20260323/outputs/phase3_straight_c1_s101_subproc/checkpoints}"
COND2_ROOT="${COND2_ROOT:-${HETZNER_PROJECT_DIR}/iwr-results/phase3-150k-cond2-15seed-trainonly-20260324/code/outputs/train/phase3-150k-cond2-15seed-trainonly-20260324/outputs/phase3_cond2_15seeds_train_only/train}"

mkdir -p "${STATUS_DIR}"

log() {
  printf '[%s] [exogenous] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${PROGRESS_LOG}"
}

export PYTHON_BIN="${PYTHON_BIN:-${HETZNER_PROJECT_DIR}/.venv/bin/python}"
export PATH="${HETZNER_PROJECT_DIR}/.venv/bin:${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "${HETZNER_PROJECT_DIR}"

log "preparing Hetzner-local cond1/cond2 manifests for exogenous families"
NEXT_ROOT="${NEXT_ROOT}" \
MANIFEST_DIR="${MANIFEST_DIR}" \
COMM_MANIFEST="${COMM_MANIFEST}" \
BASELINE_MANIFEST="${BASELINE_MANIFEST}" \
COND1_BATCH_ROOT="${COND1_BATCH_ROOT}" \
COND1_STANDALONE_ROOT="${COND1_STANDALONE_ROOT}" \
COND2_ROOT="${COND2_ROOT}" \
./scripts/run_phase3_vecstraight_prepare_iwr_manifests.sh "${RUN_DATE}" | tee -a "${PROGRESS_LOG}"

for mode in public_random fixed0 uniform fixed1; do
  log "starting mode=${mode} train_max_workers=${TRAIN_MAX_WORKERS:-15} eval_max_workers=${EVAL_MAX_WORKERS:-15}"
  RUN_DATE="${RUN_DATE}" \
  RUN_KIND="hetzner" \
  NEXT_ROOT="${NEXT_ROOT}" \
  BASELINE_MANIFEST="${BASELINE_MANIFEST}" \
  TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-15}" \
  EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-15}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  ./scripts/run_phase3_vecstraight_exogenous_family.sh "${mode}" hetzner | tee -a "${PROGRESS_LOG}"
  log "completed mode=${mode}"
done

log "building cross-family summary"
RUN_DATE="${RUN_DATE}" \
RUN_KIND="hetzner" \
PYTHON_BIN="${PYTHON_BIN}" \
./scripts/run_phase3_vecstraight_exogenous_summary.sh | tee -a "${PROGRESS_LOG}"

log "all exogenous families complete"
