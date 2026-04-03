#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_ROOT="${REPO_ROOT}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}"
JOB_DIR="${RUN_ROOT}/jobs/eval_first_pipeline_local"
mkdir -p "${JOB_DIR}"

PIPE_STDOUT="${JOB_DIR}/pipeline.stdout.log"
PIPE_STDERR="${JOB_DIR}/pipeline.stderr.log"
PIPE_PID="${JOB_DIR}/pipeline.pid"
PIPE_META="${JOB_DIR}/pipeline.meta.json"
WATCH_STDOUT="${JOB_DIR}/watch.stdout.log"
WATCH_STDERR="${JOB_DIR}/watch.stderr.log"
WATCH_PID="${JOB_DIR}/watch.pid"
WATCH_META="${JOB_DIR}/watch.meta.json"
WATCH_STATE="${JOB_DIR}/watch.state.json"
WATCH_EVENTS="${JOB_DIR}/watch.events.jsonl"

"${PYTHON_BIN}" scripts/launch_detached_local.py \
  --cwd "${REPO_ROOT}" \
  --stdout "${PIPE_STDOUT}" \
  --stderr "${PIPE_STDERR}" \
  --pidfile "${PIPE_PID}" \
  --metadata-json "${PIPE_META}" \
  --env "RUN_DATE=${RUN_DATE}" \
  --env "RUN_KIND=local" \
  -- /bin/bash "${REPO_ROOT}/scripts/run_phase3_vecstraight_eval_pipeline.sh" local

"${PYTHON_BIN}" scripts/launch_detached_local.py \
  --cwd "${REPO_ROOT}" \
  --stdout "${WATCH_STDOUT}" \
  --stderr "${WATCH_STDERR}" \
  --pidfile "${WATCH_PID}" \
  --metadata-json "${WATCH_META}" \
  -- "${PYTHON_BIN}" "${REPO_ROOT}/scripts/progress_watch.py" \
    --pidfile "${PIPE_PID}" \
    --mode regex-file \
    --source "${PIPE_STDOUT}" \
    --pattern "\\[eval-pipeline\\] progress current=(?P<current>[0-9]+) total=(?P<total>[0-9]+)" \
    --state-json "${WATCH_STATE}" \
    --events-jsonl "${WATCH_EVENTS}" \
    --label "phase3_vecstraight_eval_pipeline_local_${RUN_DATE}" \
    --poll-seconds 20

echo "[launch] job_dir=${JOB_DIR}"
echo "[launch] pipeline_pid=$(cat "${PIPE_PID}")"
echo "[launch] watcher_pid=$(cat "${WATCH_PID}")"
echo "[launch] pipeline_stdout=${PIPE_STDOUT}"
echo "[launch] watch_state=${WATCH_STATE}"
