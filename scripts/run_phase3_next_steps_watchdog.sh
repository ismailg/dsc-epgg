#!/bin/zsh
set -euo pipefail

ROOT="/Users/mbp17/POSTDOC/NPS26/dsc-epgg"
cd "$ROOT"

PIPELINE_SCRIPT="./scripts/run_phase3_next_steps_suite.sh"
PIPELINE_LOG="outputs/train/phase3_next_steps/pipeline.log"
WATCHDOG_LOG="outputs/train/phase3_next_steps/watchdog.log"
DONE_SENTINEL="outputs/eval/phase3_annealed_ext150k_5seeds/report/welfare_weighted_mean.csv"
PID_FILE="outputs/train/phase3_next_steps/pipeline.pid"
SLEEP_SECONDS=300

mkdir -p "outputs/train/phase3_next_steps"

timestamp() {
  /bin/date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[phase3-next-watchdog] %s %s\n' "$(timestamp)" "$1" | tee -a "$WATCHDOG_LOG"
}

while true; do
  if [[ -f "$DONE_SENTINEL" ]]; then
    log "done sentinel present: $DONE_SENTINEL"
    exit 0
  fi

  if [[ -f "$PID_FILE" ]]; then
    PIPE_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  else
    PIPE_PID=""
  fi

  if [[ -n "$PIPE_PID" ]] && kill -0 "$PIPE_PID" >/dev/null 2>&1; then
    log "pipeline already running"
  else
    if [[ -n "$PIPE_PID" ]]; then
      rm -f "$PID_FILE"
    fi
    log "pipeline not running; restarting"
    /bin/zsh -lc "cd '$ROOT' && ./scripts/run_phase3_next_steps_suite.sh >> '$PIPELINE_LOG' 2>&1" &
  fi

  sleep "$SLEEP_SECONDS"
done
