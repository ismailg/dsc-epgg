#!/bin/zsh
set -euo pipefail

ROOT="/Users/mbp17/POSTDOC/NPS26/dsc-epgg"
cd "$ROOT"

PIPELINE_SCRIPT="./scripts/run_phase3_high_info_suite.sh"
PIPELINE_LOG="outputs/train/phase3_high_info/pipeline.log"
WATCHDOG_LOG="outputs/train/phase3_high_info/watchdog.log"
DONE_SENTINEL="outputs/eval/phase3_compare/annealed_vs_unannealed_trajectory_mean.csv"
PID_PATTERN="scripts/run_phase3_high_info_suite.sh"
SLEEP_SECONDS=300

mkdir -p "outputs/train/phase3_high_info"

timestamp() {
  /bin/date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[phase3-high-info-watchdog] %s %s\n' "$(timestamp)" "$1" | tee -a "$WATCHDOG_LOG"
}

while true; do
  if [[ -f "$DONE_SENTINEL" ]]; then
    log "done sentinel present: $DONE_SENTINEL"
    exit 0
  fi

  if pgrep -fl "$PID_PATTERN" >/dev/null 2>&1; then
    log "pipeline already running"
  else
    log "pipeline not running; restarting"
    /bin/zsh -lc "cd '$ROOT' && ./scripts/run_phase3_high_info_suite.sh >> '$PIPELINE_LOG' 2>&1" &
  fi

  sleep "$SLEEP_SECONDS"
done
