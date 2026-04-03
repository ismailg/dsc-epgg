#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 QUEUE_NAME BRANCH:MODE [BRANCH:MODE ...]" >&2
  exit 2
fi

QUEUE_NAME="$1"
shift

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_KIND="${RUN_KIND:-iwr}"
NEXT_ROOT="${NEXT_ROOT:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}}"
MANIFEST_DIR="${MANIFEST_DIR:-${NEXT_ROOT}/manifests_iwr}"
COMM_MANIFEST="${COMM_MANIFEST:-${MANIFEST_DIR}/cond1_all_15seeds_iwr.txt}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${MANIFEST_DIR}/cond2_all_15seeds_iwr.txt}"
STATUS_DIR="${STATUS_DIR:-${IWR_RUN_DIR:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}/jobs}/status}"
PROGRESS_LOG="${PROGRESS_LOG:-${STATUS_DIR}/progress.log}"
TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-10}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-15}"

mkdir -p "${STATUS_DIR}"

log() {
  printf '[%s] [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${QUEUE_NAME}" "$*" | tee -a "${PROGRESS_LOG}"
}

log "preparing IWR manifests"
NEXT_ROOT="${NEXT_ROOT}" \
MANIFEST_DIR="${MANIFEST_DIR}" \
COMM_MANIFEST="${COMM_MANIFEST}" \
BASELINE_MANIFEST="${BASELINE_MANIFEST}" \
./scripts/run_phase3_vecstraight_prepare_iwr_manifests.sh "${RUN_DATE}" | tee -a "${PROGRESS_LOG}"

TOTAL_ITEMS="$#"
CURRENT=0
for SPEC in "$@"; do
  BRANCH_EP="${SPEC%%:*}"
  MODE="${SPEC#*:}"
  CURRENT="$((CURRENT + 1))"
  log "starting item ${CURRENT}/${TOTAL_ITEMS}: branch=${BRANCH_EP} mode=${MODE}"
  NEXT_ROOT="${NEXT_ROOT}" \
  COMM_MANIFEST="${COMM_MANIFEST}" \
  BASELINE_MANIFEST="${BASELINE_MANIFEST}" \
  TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS}" \
  EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS}" \
  RUN_DATE="${RUN_DATE}" \
  ./scripts/run_phase3_vecstraight_continuation_mode.sh "${BRANCH_EP}" "${MODE}" "${RUN_KIND}" | tee -a "${PROGRESS_LOG}"
  log "completed item ${CURRENT}/${TOTAL_ITEMS}: branch=${BRANCH_EP} mode=${MODE}"
done

log "queue complete"
