#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-local}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"

tracker_update() {
  "${PYTHON_BIN}" -m src.analysis.update_phase3_vecstraight_tracker --run_date "${RUN_DATE}" --stage "$1" "${@:2}"
}

run_stage() {
  local stage_name="$1"
  local stage_cmd="$2"
  local out_root="$3"
  local validation_text="$4"
  local next_action="$5"
  local progress_current="$6"

  tracker_update "${stage_name}" \
    --field "status=in_progress" \
    --field "command=${stage_cmd}" \
    --field "output_root=${out_root}" \
    --field "validation=${validation_text}" \
    --field "next_action=${next_action}"

  if ! /bin/zsh -lc "${stage_cmd}"; then
    tracker_update "${stage_name}" \
      --field "status=failed" \
      --field "command=${stage_cmd}" \
      --field "output_root=${out_root}" \
      --field "validation=failed during execution; inspect stage output root logs" \
      --field "next_action=fix only this stage, rerun smoke if applicable, then rerun full stage"
    exit 1
  fi

  tracker_update "${stage_name}" \
    --field "status=done" \
    --field "command=${stage_cmd}" \
    --field "output_root=${out_root}" \
    --field "validation=${validation_text}" \
    --field "next_action=${next_action}"

  echo "[post-eval-pipeline] progress current=${progress_current} total=4 stage=${stage_name}"
}

STAGE4_OUT="${REPO_ROOT}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}/smoke/continuation_sender_shuffle"
STAGE5_OUT="${REPO_ROOT}/outputs/eval/phase3_vecstraight_sameckpt_continuation_50000_*_${RUN_KIND}_${RUN_DATE}"
STAGE6_OUT="${REPO_ROOT}/outputs/eval/phase3_vecstraight_sameckpt_continuation_100000_*_${RUN_KIND}_${RUN_DATE}"
STAGE7_OUT="${REPO_ROOT}/outputs/eval/phase3_vecstraight_history_audit_150000_15seeds_${RUN_KIND}_${RUN_DATE}"

run_stage \
  "Stage 4" \
  "cd \"${REPO_ROOT}\" && RUN_DATE=${RUN_DATE} ./scripts/run_phase3_vecstraight_stage4_trainer_parity.sh" \
  "${STAGE4_OUT}" \
  "trainer tests plus sender_shuffle continuation smoke" \
  "Stage 5 same-checkpoint continuations from 50k" \
  "1"

run_stage \
  "Stage 5" \
  "cd \"${REPO_ROOT}\" && RUN_DATE=${RUN_DATE} ./scripts/run_phase3_vecstraight_continuation_pipeline.sh 50000 ${RUN_KIND}" \
  "${STAGE5_OUT}" \
  "validated suite outputs for fixed0,uniform,public_random,sender_shuffle" \
  "Stage 6 same-checkpoint continuations from 100k" \
  "2"

run_stage \
  "Stage 6" \
  "cd \"${REPO_ROOT}\" && RUN_DATE=${RUN_DATE} ./scripts/run_phase3_vecstraight_continuation_pipeline.sh 100000 ${RUN_KIND}" \
  "${STAGE6_OUT}" \
  "validated suite outputs for sender_shuffle and fixed0" \
  "Stage 7 history audit" \
  "3"

run_stage \
  "Stage 7" \
  "cd \"${REPO_ROOT}\" && RUN_DATE=${RUN_DATE} MAX_WORKERS=15 ./scripts/run_phase3_vecstraight_history_audit.sh 150000 ${RUN_KIND}" \
  "${STAGE7_OUT}" \
  "history audit summary" \
  "done" \
  "4"

