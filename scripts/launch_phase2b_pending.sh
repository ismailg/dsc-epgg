#!/bin/zsh
set -uo pipefail

ROOT="/Users/mbp17/POSTDOC/NPS26/dsc-epgg"
cd "$ROOT"

OUT_DIR="outputs/eval/phase2b_suite"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

MAX_COMPUTE_JOBS=9

active_compute_jobs() {
  ps aux | grep -E "src/analysis/evaluate_regime_conditional.py|src.experiments_pgg_v0.train_ppo" | grep -v grep | wc -l | tr -d ' '
}

wait_for_capacity() {
  local count
  while true; do
    count="$(active_compute_jobs)"
    if [ "${count:-0}" -lt "$MAX_COMPUTE_JOBS" ]; then
      break
    fi
    sleep 10
  done
}

launch_eval() {
  local name="$1"
  local ckpt="$2"
  local out_csv="$3"
  shift 3
  if [ -f "$out_csv" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] skip $name (exists)" | tee -a "$LOG_DIR/pending_launcher.log"
    return
  fi
  wait_for_capacity
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] launch $name" | tee -a "$LOG_DIR/pending_launcher.log"
  nohup env OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 src/analysis/evaluate_regime_conditional.py \
    --checkpoints_glob "$ckpt" \
    --n_eval_episodes 300 \
    --eval_seed 9001 \
    --greedy \
    "$@" \
    --out_csv "$out_csv" >"$LOG_DIR/${name}.log" 2>&1 &
}

for MODE in marginal zeros fixed0 fixed1 flip uniform; do
  launch_eval "cond1_s202_greedy_${MODE}" \
    "outputs/train/phase2b/cond1_seed202.pt" \
    "$OUT_DIR/cond1_s202_greedy_${MODE}.csv" \
    --msg_intervention "$MODE"
done

launch_eval "crossplay_s101_sender50k_recv200k" \
  "outputs/train/phase2b/cond1_seed101.pt" \
  "$OUT_DIR/crossplay_s101_sender50k_recv200k.csv" \
  --msg_intervention none \
  --cross_play_checkpoint outputs/train/phase2b/cond1_seed101_ep50000.pt

launch_eval "crossplay_s101_sender100k_recv200k" \
  "outputs/train/phase2b/cond1_seed101.pt" \
  "$OUT_DIR/crossplay_s101_sender100k_recv200k.csv" \
  --msg_intervention none \
  --cross_play_checkpoint outputs/train/phase2b/cond1_seed101_ep100000.pt

launch_eval "crossplay_s202_sender50k_recv200k" \
  "outputs/train/phase2b/cond1_seed202.pt" \
  "$OUT_DIR/crossplay_s202_sender50k_recv200k.csv" \
  --msg_intervention none \
  --cross_play_checkpoint outputs/train/phase2b/cond1_seed202_ep50000.pt

launch_eval "crossplay_s202_sender100k_recv200k" \
  "outputs/train/phase2b/cond1_seed202.pt" \
  "$OUT_DIR/crossplay_s202_sender100k_recv200k.csv" \
  --msg_intervention none \
  --cross_play_checkpoint outputs/train/phase2b/cond1_seed202_ep100000.pt

echo "[$(date '+%Y-%m-%d %H:%M:%S')] pending jobs queued" | tee -a "$LOG_DIR/pending_launcher.log"
