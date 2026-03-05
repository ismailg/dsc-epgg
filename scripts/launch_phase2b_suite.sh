#!/bin/zsh
set -uo pipefail

ROOT="/Users/mbp17/POSTDOC/NPS26/dsc-epgg"
cd "$ROOT"

OUT_DIR="outputs/eval/phase2b_suite"
LOG_DIR="$OUT_DIR/logs"
TRAIN_DIR="outputs/train/phase2b/anneal_diag"
TRAIN_LOG_DIR="$TRAIN_DIR/logs"
TRAIN_METRICS_DIR="$TRAIN_DIR/metrics"
mkdir -p "$OUT_DIR" "$LOG_DIR" "$TRAIN_LOG_DIR" "$TRAIN_METRICS_DIR"

MAX_JOBS=9

wait_for_slot() {
  local count
  while true; do
    count="$(jobs -pr | wc -l | tr -d ' ')"
    if [ "${count:-0}" -lt "$MAX_JOBS" ]; then
      break
    fi
    wait -n || true
  done
}

launch_job() {
  local name="$1"
  shift
  wait_for_slot
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] launch $name" | tee -a "$LOG_DIR/launcher.log"
  nohup "$@" >"$LOG_DIR/${name}.log" 2>&1 &
}

launch_eval() {
  local name="$1"
  local ckpt="$2"
  local out_csv="$3"
  shift 3
  if [ -f "$out_csv" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] skip $name (exists)" | tee -a "$LOG_DIR/launcher.log"
    return
  fi
  launch_job "$name" env OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 src/analysis/evaluate_regime_conditional.py \
    --checkpoints_glob "$ckpt" \
    --n_eval_episodes 300 \
    --eval_seed 9001 \
    --greedy \
    "$@" \
    --out_csv "$out_csv"
}

TRAIN_OUT="$TRAIN_DIR/ent_msg0_seed101_from100k_20k.pt"

if [ ! -f "$TRAIN_OUT" ]; then
  launch_job "train_ent_msg0_seed101_from100k_20k" env OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 -m src.experiments_pgg_v0.train_ppo \
    --n_agents 4 \
    --T 20 \
    --F 0.5 1.5 2.5 3.5 5.0 \
    --sigmas 0.5 0.5 0.5 0.5 \
    --rho 0.05 \
    --epsilon_tremble 0.05 \
    --n_episodes 20000 \
    --lr 3e-4 \
    --gamma 0.99 \
    --lam 0.95 \
    --clip_ratio 0.2 \
    --entropy_coeff 0.01 \
    --msg_entropy_coeff 0.0 \
    --reward_scale 20.0 \
    --hidden_size 64 \
    --comm_enabled \
    --n_senders 4 \
    --vocab_size 2 \
    --msg_dropout 0.1 \
    --regime_log_interval 1000 \
    --checkpoint_interval 10000 \
    --init_ckpt outputs/train/phase2b/cond1_seed101_ep100000.pt \
    --save_path "$TRAIN_OUT" \
    --seed 101 \
    --condition_name ent_msg0_seed101_from100k_20k \
    --metrics_jsonl_path "$TRAIN_METRICS_DIR/ent_msg0_seed101_from100k_20k.jsonl"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] skip D training (checkpoint exists)" | tee -a "$LOG_DIR/launcher.log"
fi

# Workstream A
for MODE in none marginal zeros fixed0 fixed1 flip uniform; do
  EXTRA=()
  if [ "$MODE" = "none" ]; then
    EXTRA+=(--posterior_strat)
  fi
  launch_eval "cond1_s101_greedy_${MODE}" \
    "outputs/train/phase2b/cond1_seed101.pt" \
    "$OUT_DIR/cond1_s101_greedy_${MODE}.csv" \
    --msg_intervention "$MODE" "${EXTRA[@]}"
done

for MODE in none marginal zeros fixed0 fixed1 flip uniform; do
  EXTRA=()
  if [ "$MODE" = "none" ]; then
    EXTRA+=(--posterior_strat)
  fi
  launch_eval "cond1_s202_greedy_${MODE}" \
    "outputs/train/phase2b/cond1_seed202.pt" \
    "$OUT_DIR/cond1_s202_greedy_${MODE}.csv" \
    --msg_intervention "$MODE" "${EXTRA[@]}"
done

# Workstream B
launch_eval "crossplay_s101_sender50k_recv200k" \
  "outputs/train/phase2b/cond1_seed101.pt" \
  "$OUT_DIR/crossplay_s101_sender50k_recv200k.csv" \
  --msg_intervention none \
  --cross_play_checkpoint "outputs/train/phase2b/cond1_seed101_ep50000.pt"

launch_eval "crossplay_s101_sender100k_recv200k" \
  "outputs/train/phase2b/cond1_seed101.pt" \
  "$OUT_DIR/crossplay_s101_sender100k_recv200k.csv" \
  --msg_intervention none \
  --cross_play_checkpoint "outputs/train/phase2b/cond1_seed101_ep100000.pt"

launch_eval "crossplay_s202_sender50k_recv200k" \
  "outputs/train/phase2b/cond1_seed202.pt" \
  "$OUT_DIR/crossplay_s202_sender50k_recv200k.csv" \
  --msg_intervention none \
  --cross_play_checkpoint "outputs/train/phase2b/cond1_seed202_ep50000.pt"

launch_eval "crossplay_s202_sender100k_recv200k" \
  "outputs/train/phase2b/cond1_seed202.pt" \
  "$OUT_DIR/crossplay_s202_sender100k_recv200k.csv" \
  --msg_intervention none \
  --cross_play_checkpoint "outputs/train/phase2b/cond1_seed202_ep100000.pt"

# Workstream C
launch_eval "cond2_s101_greedy_none" \
  "outputs/train/phase2b/cond2_seed101.pt" \
  "$OUT_DIR/cond2_s101_greedy_none.csv" \
  --msg_intervention none \
  --posterior_strat

launch_eval "cond2_s202_greedy_none" \
  "outputs/train/phase2b/cond2_seed202.pt" \
  "$OUT_DIR/cond2_s202_greedy_none.csv" \
  --msg_intervention none \
  --posterior_strat

# Wait for A+B+C and optional training to finish before D evals.
wait || true

# Workstream D evals
launch_eval "ent_msg0_s101_greedy_none" \
  "$TRAIN_OUT" \
  "$OUT_DIR/ent_msg0_s101_greedy_none.csv" \
  --msg_intervention none \
  --posterior_strat

launch_eval "ent_msg0_s101_greedy_marginal" \
  "$TRAIN_OUT" \
  "$OUT_DIR/ent_msg0_s101_greedy_marginal.csv" \
  --msg_intervention marginal

wait || true
echo "[$(date '+%Y-%m-%d %H:%M:%S')] phase2b suite complete" | tee -a "$LOG_DIR/launcher.log"
