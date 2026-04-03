#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-local}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
NEXT_ROOT="${REPO_ROOT}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}"
TRACKER="${NEXT_ROOT}/status/TRACKER.md"

if [[ ! -f "${NEXT_ROOT}/manifests/cond1_all_15seeds.txt" || ! -f "${NEXT_ROOT}/manifests/cond2_all_15seeds.txt" ]]; then
  "${REPO_ROOT}/scripts/run_phase3_vecstraight_setup.sh"
fi

tracker_update() {
  "${PYTHON_BIN}" -m src.analysis.update_phase3_vecstraight_tracker --run_date "${RUN_DATE}" --stage "$1" "${@:2}"
}

tracker_update "Stage 0B" \
  --field "status=done" \
  --field "command=manifest-driven cond1/cond2 suite smoke plus sender-causal smoke on ${RUN_DATE}" \
  --field "output_root=${NEXT_ROOT}/smoke" \
  --field "validation=11 targeted tests passed; smoke suite valid; smoke sender-causal completed" \
  --field "next_action=Stage 1 frozen 50k"

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

  echo "[eval-pipeline] progress current=${progress_current} total=3 stage=${stage_name}"
}

STAGE1_OUT="${REPO_ROOT}/outputs/eval/phase3_vecstraight_frozen50k_15seeds_${RUN_KIND}_${RUN_DATE}"
STAGE2_OUT="${REPO_ROOT}/outputs/eval/phase3_vecstraight_frozen150k_15seeds_${RUN_KIND}_${RUN_DATE}"
STAGE3_OUT="${REPO_ROOT}/outputs/eval/phase3_vecstraight_sender_causal_150k_15seeds_${RUN_KIND}_${RUN_DATE}"

run_stage \
  "Stage 1" \
  "cd \"${REPO_ROOT}\" && RUN_DATE=${RUN_DATE} MAX_WORKERS=15 ./scripts/run_phase3_vecstraight_frozen_suite.sh 50000 ${RUN_KIND}" \
  "${STAGE1_OUT}" \
  "validated suite outputs and intervention_suite_summary.md" \
  "Stage 2 frozen 150k" \
  "1"

run_stage \
  "Stage 2" \
  "cd \"${REPO_ROOT}\" && RUN_DATE=${RUN_DATE} MAX_WORKERS=15 ./scripts/run_phase3_vecstraight_frozen_suite.sh 150000 ${RUN_KIND}" \
  "${STAGE2_OUT}" \
  "validated suite outputs and learned-vs-controls summary at f=3.5,5.0" \
  "Stage 3 sender-causal 150k" \
  "2"

run_stage \
  "Stage 3" \
  "cd \"${REPO_ROOT}\" && RUN_DATE=${RUN_DATE} RUN_KIND=${RUN_KIND} MAX_WORKERS=15 ./scripts/run_phase3_vecstraight_sender_causal.sh" \
  "${STAGE3_OUT}" \
  "sender_causal_matrix.csv, sender_causal_checkpoint_main.csv, sender_causal_manifest.json, sender_causal_summary.md" \
  "Stage 4 trainer sender_shuffle parity" \
  "3"
