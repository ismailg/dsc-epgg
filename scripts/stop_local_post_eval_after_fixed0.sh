#!/usr/bin/env bash
set -euo pipefail

RUN_DATE="${1:-${RUN_DATE:-$(date +%Y%m%d)}}"
POLL_SEC="${POLL_SEC:-5}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

JOB_DIR="${REPO_ROOT}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}/jobs/post_eval_pipeline_local"
SUMMARY_PATH="${REPO_ROOT}/outputs/eval/phase3_vecstraight_sameckpt_continuation_50000_fixed0_15seeds_local_${RUN_DATE}/report/intervention_suite_summary.md"
LOG_PATH="${JOB_DIR}/stop_after_fixed0.log"

mkdir -p "${JOB_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_PATH}"
}

kill_if_alive() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    log "sent TERM to pid=${pid}"
  fi
}

find_and_kill_patterns() {
  local pattern
  for pattern in \
    "run_phase3_vecstraight_continuation_mode.sh 50000 uniform local" \
    "run_phase3_vecstraight_continuation_mode.sh 50000 public_random local" \
    "run_phase3_vecstraight_continuation_mode.sh 50000 sender_shuffle local" \
    "run_phase3_vecstraight_continuation_pipeline.sh 100000 local" \
    "run_phase3_vecstraight_history_audit.sh 150000 local" \
    "phase3_vecstraight_sameckpt_continuation_50000_uniform_15seeds_local_${RUN_DATE}" \
    "phase3_vecstraight_sameckpt_continuation_50000_public_random_15seeds_local_${RUN_DATE}" \
    "phase3_vecstraight_sameckpt_continuation_50000_sender_shuffle_15seeds_local_${RUN_DATE}" \
    "phase3_vecstraight_sameckpt_continuation_100000_sender_shuffle_15seeds_local_${RUN_DATE}" \
    "phase3_vecstraight_sameckpt_continuation_100000_fixed0_15seeds_local_${RUN_DATE}" \
    "phase3_vecstraight_history_audit_150000_15seeds_local_${RUN_DATE}"
  do
    while IFS= read -r pid; do
      [[ -z "${pid}" ]] && continue
      kill_if_alive "${pid}"
    done < <(pgrep -f "${pattern}" || true)
  done
}

log "waiting for fixed0 completion summary at ${SUMMARY_PATH}"
while [[ ! -f "${SUMMARY_PATH}" ]]; do
  sleep "${POLL_SEC}"
done

log "fixed0 summary detected"

POST_PID="$(cat "${JOB_DIR}/pipeline.pid" 2>/dev/null || true)"
CONT_PID="$(pgrep -f "run_phase3_vecstraight_continuation_pipeline.sh 50000 local" | head -n 1 || true)"

kill_if_alive "${CONT_PID}"
kill_if_alive "${POST_PID}"

for _ in $(seq 1 24); do
  find_and_kill_patterns
  sleep 5
done

log "local post-eval stop guard complete"
