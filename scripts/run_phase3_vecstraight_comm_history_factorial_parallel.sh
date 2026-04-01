#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-local}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
TOTAL_WORKERS="${TOTAL_WORKERS:-auto}"
CELLS=(
  with_comm_full_history
  with_comm_reduced_history
  without_comm_full_history
  without_comm_reduced_history
)

STATUS_ROOT="${REPO_ROOT}/outputs/train/phase3_vecstraight_comm_history_factorial_parallel_${RUN_KIND}_${RUN_DATE}/status"
PROGRESS_LOG="${STATUS_ROOT}/progress.log"
PIDMAP_PATH="${STATUS_ROOT}/cell_pids.tsv"
MANIFEST_PATH="${STATUS_ROOT}/manifest.txt"
CELL_LOG_DIR="${STATUS_ROOT}/cell_logs"

mkdir -p "${STATUS_ROOT}" "${CELL_LOG_DIR}"
: > "${PROGRESS_LOG}"
: > "${PIDMAP_PATH}"

detect_default_total_workers() {
  local nproc_val suggested
  nproc_val="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
  if ! [[ "${nproc_val}" =~ ^[0-9]+$ ]] || (( nproc_val < 1 )); then
    nproc_val=1
  fi
  suggested=$(( nproc_val / 2 ))
  if (( suggested < ${#CELLS[@]} )); then
    suggested="${#CELLS[@]}"
  fi
  printf '%s\n' "${suggested}"
}

if [[ "${TOTAL_WORKERS}" == "auto" ]]; then
  TOTAL_WORKERS="$(detect_default_total_workers)"
fi
if ! [[ "${TOTAL_WORKERS}" =~ ^[0-9]+$ ]] || (( TOTAL_WORKERS < ${#CELLS[@]} )); then
  echo "TOTAL_WORKERS=${TOTAL_WORKERS} must resolve to an integer >= ${#CELLS[@]} to run all cells in parallel" >&2
  exit 2
fi

log_progress() {
  printf '%s %s\n' "[$(date '+%Y-%m-%d %H:%M:%S')]" "$*" | tee -a "${PROGRESS_LOG}"
}

{
  printf 'run_kind=%s\n' "${RUN_KIND}"
  printf 'run_date=%s\n' "${RUN_DATE}"
  printf 'repo_root=%s\n' "${REPO_ROOT}"
  printf 'python_bin=%s\n' "${PYTHON_BIN}"
  printf 'total_workers=%s\n' "${TOTAL_WORKERS}"
  printf 'cells=%s\n' "${CELLS[*]}"
} > "${MANIFEST_PATH}"

base_workers=$(( TOTAL_WORKERS / ${#CELLS[@]} ))
remainder=$(( TOTAL_WORKERS % ${#CELLS[@]} ))

declare -a CELL_PIDS=()

for idx in "${!CELLS[@]}"; do
  cell="${CELLS[idx]}"
  cell_workers="${base_workers}"
  if (( idx < remainder )); then
    cell_workers=$(( cell_workers + 1 ))
  fi
  cell_log="${CELL_LOG_DIR}/${cell}.log"
  log_progress "[launch] cell=${cell} workers=${cell_workers} log=${cell_log}"
  (
    export PYTHON_BIN="${PYTHON_BIN}"
    export RUN_DATE="${RUN_DATE}"
    export TRAIN_MAX_WORKERS="${cell_workers}"
    ./scripts/run_phase3_vecstraight_comm_history_factorial.sh "${cell}" "${RUN_KIND}"
  ) >"${cell_log}" 2>&1 &
  pid="$!"
  printf '%s\t%s\t%s\n' "${cell}" "${cell_workers}" "${pid}" >> "${PIDMAP_PATH}"
  CELL_PIDS+=("${pid}")
done

status=0
for idx in "${!CELLS[@]}"; do
  cell="${CELLS[idx]}"
  pid="${CELL_PIDS[idx]}"
  if wait "${pid}"; then
    log_progress "[cell done] cell=${cell} status=ok"
  else
    code="$?"
    log_progress "[cell done] cell=${cell} status=failed exit=${code}"
    status=1
  fi
done

if (( status != 0 )); then
  log_progress "[parallel factorial done] status=failed"
  exit "${status}"
fi

log_progress "[parallel factorial done] status=ok"
