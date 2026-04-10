#!/usr/bin/env bash
set -euo pipefail

RUN_DATE="${RUN_DATE:-20260410}"
HETZNER_PROJECT_DIR="${HETZNER_PROJECT_DIR:-/root/compute-work/projects/dsc-epgg-vectorized}"
HETZNER_RUN_DIR="${HETZNER_RUN_DIR:-/root/compute-work/runs/dsc-epgg-vectorized/phase3-vecstraight-clean-msgsource-20260410}"
NEXT_ROOT="${NEXT_ROOT:-${HETZNER_PROJECT_DIR}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${NEXT_ROOT}/manifests/cond2_all_15seeds.txt}"
STATUS_DIR="${STATUS_DIR:-${HETZNER_RUN_DIR}/status}"
PROGRESS_LOG="${PROGRESS_LOG:-${STATUS_DIR}/progress.log}"
MODES=(
  learned
  public_random
  uniform
  fixed0
  fixed1
)

mkdir -p "${STATUS_DIR}"

log() {
  printf '[%s] [clean-msgsource] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${PROGRESS_LOG}"
}

export PYTHON_BIN="${PYTHON_BIN:-${HETZNER_PROJECT_DIR}/.venv/bin/python}"
export PATH="${HETZNER_PROJECT_DIR}/.venv/bin:${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY="${OBJC_DISABLE_INITIALIZE_FORK_SAFETY:-YES}"

cd "${HETZNER_PROJECT_DIR}"

if [[ ! -f "${BASELINE_MANIFEST}" ]]; then
  log "baseline manifest missing; running vecstraight setup for RUN_DATE=${RUN_DATE}"
  RUN_DATE="${RUN_DATE}" "${HETZNER_PROJECT_DIR}/scripts/run_phase3_vecstraight_setup.sh" | tee -a "${PROGRESS_LOG}"
fi

for mode in "${MODES[@]}"; do
  log "starting mode=${mode} train_max_workers=${TRAIN_MAX_WORKERS:-auto} eval_max_workers=${EVAL_MAX_WORKERS:-15}"
  RUN_DATE="${RUN_DATE}" \
  RUN_KIND="hetzner" \
  NEXT_ROOT="${NEXT_ROOT}" \
  BASELINE_MANIFEST="${BASELINE_MANIFEST}" \
  TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-auto}" \
  EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-15}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  ./scripts/run_phase3_vecstraight_clean_msgsource_family.sh "${mode}" hetzner | tee -a "${PROGRESS_LOG}"
  log "completed mode=${mode}"
done

log "all clean msg_source families complete"
