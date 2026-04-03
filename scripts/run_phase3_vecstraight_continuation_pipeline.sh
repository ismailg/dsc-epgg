#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 BRANCH_EP [RUN_KIND]" >&2
  exit 2
fi

BRANCH_EP="$1"
RUN_KIND="${2:-local}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"

case "${BRANCH_EP}" in
  50000)
    STAGE_NAME="Stage 5"
    NEXT_ACTION="Stage 6 same-checkpoint continuations from 100k"
    MODES=(fixed0 uniform public_random sender_shuffle)
    ;;
  100000)
    STAGE_NAME="Stage 6"
    NEXT_ACTION="Stage 7 history audit"
    MODES=(sender_shuffle fixed0)
    ;;
  *)
    echo "unsupported branch episode: ${BRANCH_EP}" >&2
    exit 2
    ;;
esac

tracker_update() {
  "${PYTHON_BIN}" -m src.analysis.update_phase3_vecstraight_tracker --run_date "${RUN_DATE}" --stage "${STAGE_NAME}" "${@:1}"
}

tracker_update \
  --field "status=in_progress" \
  --field "command=RUN_DATE=${RUN_DATE} ./scripts/run_phase3_vecstraight_continuation_pipeline.sh ${BRANCH_EP} ${RUN_KIND}" \
  --field "output_root=${REPO_ROOT}/outputs/eval/phase3_vecstraight_sameckpt_continuation_${BRANCH_EP}_*_${RUN_KIND}_${RUN_DATE}" \
  --field "validation=running continuation modes sequentially" \
  --field "next_action=${NEXT_ACTION}"

TOTAL_MODES="${#MODES[@]}"
DONE_MODES=()
CURRENT=0
for MODE in "${MODES[@]}"; do
  if ! RUN_DATE="${RUN_DATE}" ./scripts/run_phase3_vecstraight_continuation_mode.sh "${BRANCH_EP}" "${MODE}" "${RUN_KIND}"; then
    tracker_update \
      --field "status=failed" \
      --field "validation=failed during mode ${MODE}; completed modes: ${DONE_MODES[*]:-none}" \
      --field "next_action=fix only this branch, rerun smoke if needed, then rerun this pipeline"
    exit 1
  fi
  DONE_MODES+=("${MODE}")
  CURRENT="$((CURRENT + 1))"
  tracker_update \
    --field "status=in_progress" \
    --field "validation=completed modes: ${DONE_MODES[*]}" \
    --field "next_action=${NEXT_ACTION}"
  echo "[continuation-pipeline] progress current=${CURRENT} total=${TOTAL_MODES} stage=${STAGE_NAME} mode=${MODE}"
done

tracker_update \
  --field "status=done" \
  --field "validation=validated suite outputs for modes: ${DONE_MODES[*]}" \
  --field "next_action=${NEXT_ACTION}"
