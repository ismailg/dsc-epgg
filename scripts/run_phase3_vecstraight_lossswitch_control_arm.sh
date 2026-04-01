#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 ARM [RUN_KIND]" >&2
  exit 2
fi

ARM="$1"
RUN_KIND="${2:-local}"
BRANCH_EP="${BRANCH_EP:-50000}"

case "${BRANCH_EP}" in
  50000|100000) ;;
  *)
    echo "unsupported BRANCH_EP=${BRANCH_EP}; expected 50000 or 100000" >&2
    exit 2
    ;;
esac

case "${ARM}" in
  none_base)
    MODE="none"
    SIGN_LAMBDA="0.1"
    LIST_LAMBDA="0.1"
    ;;
  none_zeroaux)
    MODE="none"
    SIGN_LAMBDA="0.0"
    LIST_LAMBDA="0.0"
    ;;
  uniform_zeroaux)
    MODE="uniform"
    SIGN_LAMBDA="0.0"
    LIST_LAMBDA="0.0"
    ;;
  *)
    echo "unsupported ARM=${ARM}; expected one of: none_base none_zeroaux uniform_zeroaux" >&2
    exit 2
    ;;
esac

REPO_ROOT="${PROJECT_ROOT:-${IWR_PROJECT_DIR:-${HETZNER_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
NEXT_ROOT="${NEXT_ROOT:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}}"
COMM_MANIFEST="${COMM_MANIFEST:-${NEXT_ROOT}/manifests_iwr/cond1_all_15seeds_iwr.txt}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${NEXT_ROOT}/manifests_iwr/cond2_all_15seeds_iwr.txt}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_sameckpt_continuation_${BRANCH_EP}_${ARM}_15seeds_${RUN_KIND}_${RUN_DATE}}"
STATUS_ROOT="${STATUS_ROOT:-${OUT_ROOT}/status}"
PROGRESS_LOG="${PROGRESS_LOG:-${STATUS_ROOT}/progress.log}"
MANIFEST_PATH="${MANIFEST_PATH:-${STATUS_ROOT}/manifest.txt}"

mkdir -p "${STATUS_ROOT}"
: > "${PROGRESS_LOG}"

log_progress() {
  printf '[%s] [loss-switch:%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${ARM}" "$*" | tee -a "${PROGRESS_LOG}"
}

{
  printf 'arm=%s\n' "${ARM}"
  printf 'mode=%s\n' "${MODE}"
  printf 'branch_ep=%s\n' "${BRANCH_EP}"
  printf 'run_kind=%s\n' "${RUN_KIND}"
  printf 'run_date=%s\n' "${RUN_DATE}"
  printf 'repo_root=%s\n' "${REPO_ROOT}"
  printf 'python_bin=%s\n' "${PYTHON_BIN}"
  printf 'next_root=%s\n' "${NEXT_ROOT}"
  printf 'comm_manifest=%s\n' "${COMM_MANIFEST}"
  printf 'baseline_manifest=%s\n' "${BASELINE_MANIFEST}"
  printf 'out_root=%s\n' "${OUT_ROOT}"
  printf 'sign_lambda=%s\n' "${SIGN_LAMBDA}"
  printf 'list_lambda=%s\n' "${LIST_LAMBDA}"
} > "${MANIFEST_PATH}"

if [[ ! -f "${COMM_MANIFEST}" || ! -f "${BASELINE_MANIFEST}" ]]; then
  log_progress "missing cond1/cond2 manifests; rebuilding local manifests"
  RUN_DATE="${RUN_DATE}" "${REPO_ROOT}/scripts/run_phase3_vecstraight_prepare_iwr_manifests.sh" | tee -a "${PROGRESS_LOG}"
fi

log_progress "starting branch=${BRANCH_EP} mode=${MODE} sign_lambda=${SIGN_LAMBDA} list_lambda=${LIST_LAMBDA}"
OUT_ROOT="${OUT_ROOT}" \
NEXT_ROOT="${NEXT_ROOT}" \
COMM_MANIFEST="${COMM_MANIFEST}" \
BASELINE_MANIFEST="${BASELINE_MANIFEST}" \
RUN_DATE="${RUN_DATE}" \
PYTHON_BIN="${PYTHON_BIN}" \
SIGN_LAMBDA="${SIGN_LAMBDA}" \
LIST_LAMBDA="${LIST_LAMBDA}" \
./scripts/run_phase3_vecstraight_continuation_mode.sh "${BRANCH_EP}" "${MODE}" "${RUN_KIND}" | tee -a "${PROGRESS_LOG}"
log_progress "completed branch=${BRANCH_EP} mode=${MODE}"

printf '[phase3-loss-switch] arm=%s mode=%s out_root=%s\n' "${ARM}" "${MODE}" "${OUT_ROOT}"
