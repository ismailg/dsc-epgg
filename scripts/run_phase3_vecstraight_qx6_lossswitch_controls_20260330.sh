#!/usr/bin/env bash
set -euo pipefail

RUN_DATE="${RUN_DATE:-20260330}"
BRANCH_EP="${BRANCH_EP:-50000}"
IWR_PROJECT_DIR="${IWR_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
IWR_RUN_DIR="${IWR_RUN_DIR:-/export/scratch/iguennou/runs/dsc-epgg-vectorized/phase3-vecstraight-qx6-lossswitch-controls-20260330}"
NEXT_ROOT="${NEXT_ROOT:-${IWR_PROJECT_DIR}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}}"
MANIFEST_DIR="${MANIFEST_DIR:-${NEXT_ROOT}/manifests_iwr_lossswitch_controls}"
COMM_MANIFEST="${COMM_MANIFEST:-${MANIFEST_DIR}/cond1_all_15seeds_iwr.txt}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${MANIFEST_DIR}/cond2_all_15seeds_iwr.txt}"
STATUS_DIR="${STATUS_DIR:-${IWR_RUN_DIR}/status}"
PROGRESS_LOG="${PROGRESS_LOG:-${STATUS_DIR}/progress.log}"
MANIFEST_PATH="${MANIFEST_PATH:-${STATUS_DIR}/manifest.txt}"
ARM_LOG_DIR="${ARM_LOG_DIR:-${STATUS_DIR}/arm_logs}"
ARMS=(none_base none_zeroaux uniform_zeroaux)
PARALLEL_ARMS="${PARALLEL_ARMS:-1}"

mkdir -p "${STATUS_DIR}"
mkdir -p "${ARM_LOG_DIR}"
: > "${PROGRESS_LOG}"

log() {
  printf '[%s] [loss-switch-batch] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${PROGRESS_LOG}"
}

export PYTHON_BIN="${PYTHON_BIN:-/export/scratch/iguennou/runs/dsc-epgg/phase3-15seed-main-story-iwr-20260317/venv-py310/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "missing IWR python interpreter: ${PYTHON_BIN}" >&2
  exit 2
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

{
  printf 'run_date=%s\n' "${RUN_DATE}"
  printf 'branch_ep=%s\n' "${BRANCH_EP}"
  printf 'iwr_project_dir=%s\n' "${IWR_PROJECT_DIR}"
  printf 'iwr_run_dir=%s\n' "${IWR_RUN_DIR}"
  printf 'next_root=%s\n' "${NEXT_ROOT}"
  printf 'manifest_dir=%s\n' "${MANIFEST_DIR}"
  printf 'comm_manifest=%s\n' "${COMM_MANIFEST}"
  printf 'baseline_manifest=%s\n' "${BASELINE_MANIFEST}"
  printf 'train_max_workers=%s\n' "${TRAIN_MAX_WORKERS:-15}"
  printf 'eval_max_workers=%s\n' "${EVAL_MAX_WORKERS:-15}"
  printf 'num_envs=%s\n' "${NUM_ENVS:-8}"
  printf 'env_backend=%s\n' "${ENV_BACKEND:-subproc}"
  printf 'env_start_method=%s\n' "${ENV_START_METHOD:-spawn}"
  printf 'entropy_schedule=%s\n' "${ENTROPY_SCHEDULE:-linear}"
  printf 'lr_schedule=%s\n' "${LR_SCHEDULE:-cosine}"
  printf 'regime_log_interval=%s\n' "${REGIME_LOG_INTERVAL:-400}"
  printf 'checkpoint_interval=%s\n' "${CHECKPOINT_INTERVAL:-25000}"
  printf 'msg_entropy_coeff=%s\n' "${MSG_ENTROPY_COEFF:-0.01}"
  printf 'msg_entropy_coeff_final=%s\n' "${MSG_ENTROPY_COEFF_FINAL:-0.0}"
  printf 'parallel_arms=%s\n' "${PARALLEL_ARMS}"
  printf 'arm_log_dir=%s\n' "${ARM_LOG_DIR}"
  printf 'arms=%s\n' "${ARMS[*]}"
} > "${MANIFEST_PATH}"

log "preparing IWR-local cond1/cond2 manifests for loss-switch controls"
NEXT_ROOT="${NEXT_ROOT}" \
MANIFEST_DIR="${MANIFEST_DIR}" \
COMM_MANIFEST="${COMM_MANIFEST}" \
BASELINE_MANIFEST="${BASELINE_MANIFEST}" \
./scripts/run_phase3_vecstraight_prepare_iwr_manifests.sh "${RUN_DATE}" | tee -a "${PROGRESS_LOG}"

launch_arm() {
  local arm="$1"
  local arm_log="${ARM_LOG_DIR}/${arm}.log"
  log "starting arm=${arm} branch=${BRANCH_EP} train_max_workers=${TRAIN_MAX_WORKERS:-15} eval_max_workers=${EVAL_MAX_WORKERS:-15} arm_log=${arm_log}"
  (
    NEXT_ROOT="${NEXT_ROOT}" \
    COMM_MANIFEST="${COMM_MANIFEST}" \
    BASELINE_MANIFEST="${BASELINE_MANIFEST}" \
    RUN_DATE="${RUN_DATE}" \
    BRANCH_EP="${BRANCH_EP}" \
    TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-15}" \
    EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-15}" \
    NUM_ENVS="${NUM_ENVS:-8}" \
    ENV_BACKEND="${ENV_BACKEND:-subproc}" \
    ENV_START_METHOD="${ENV_START_METHOD:-spawn}" \
    REGIME_LOG_INTERVAL="${REGIME_LOG_INTERVAL:-400}" \
    CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25000}" \
    MSG_ENTROPY_COEFF="${MSG_ENTROPY_COEFF:-0.01}" \
    MSG_ENTROPY_COEFF_FINAL="${MSG_ENTROPY_COEFF_FINAL:-0.0}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    ./scripts/run_phase3_vecstraight_lossswitch_control_arm.sh "${arm}" iwr >"${arm_log}" 2>&1
  ) &
  local pid="$!"
  printf '%s\t%s\t%s\n' "${arm}" "${pid}" "${arm_log}" >> "${STATUS_DIR}/arm_pids.tsv"
  log "launched arm=${arm} pid=${pid}"
  ARM_PIDS+=("${pid}")
  ARM_NAMES+=("${arm}")
}

declare -a ARM_PIDS=()
declare -a ARM_NAMES=()
: > "${STATUS_DIR}/arm_pids.tsv"

if [[ "${PARALLEL_ARMS}" == "1" ]]; then
  for arm in "${ARMS[@]}"; do
    launch_arm "${arm}"
  done

  batch_status=0
  for idx in "${!ARM_PIDS[@]}"; do
    pid="${ARM_PIDS[$idx]}"
    arm="${ARM_NAMES[$idx]}"
    if wait "${pid}"; then
      log "completed arm=${arm}"
    else
      code="$?"
      log "arm failed arm=${arm} exit=${code}"
      batch_status=1
    fi
  done
else
  batch_status=0
  for arm in "${ARMS[@]}"; do
    launch_arm "${arm}"
    pid="${ARM_PIDS[-1]}"
    if wait "${pid}"; then
      log "completed arm=${arm}"
    else
      code="$?"
      log "arm failed arm=${arm} exit=${code}"
      batch_status=1
      break
    fi
  done
fi

if [[ "${batch_status}" -ne 0 ]]; then
  log "loss-switch control batch failed"
  exit "${batch_status}"
fi

log "all loss-switch control arms complete"
