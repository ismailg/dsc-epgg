#!/bin/zsh
set -euo pipefail

ROOT="/Users/mbp17/POSTDOC/NPS26/dsc-epgg"
cd "$ROOT"

if [[ -x "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3" ]]; then
  PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

export PATH="/Library/Frameworks/Python.framework/Versions/3.10/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mpl_phase4_sameckpt
export XDG_CACHE_HOME=/tmp/xdg_phase4_sameckpt

FIVE_SEEDS=(101 202 303 404 505)
MODES=(fixed0 uniform public_random)
BASE_50K_DIR="outputs/train/phase3_annealed_trimmed"
LEARNED_150K_SUITE="outputs/eval/phase3_annealed_ext150k_5seeds/suite/checkpoint_suite_main.csv"
SAMECKPT_ROOT="outputs/train/phase3_sameckpt_continuation_5seeds"
SAMECKPT_EVAL_ROOT="outputs/eval/phase3_sameckpt_continuation_5seeds"
STATS_ROOT="outputs/eval/phase3_stats"
INTEROP_ROOT="outputs/eval/phase3_interoperability"
RUN_ROOT="outputs/train/phase3_sameckpt_continuation_5seeds/run_logs"
mkdir -p "$SAMECKPT_ROOT" "$SAMECKPT_EVAL_ROOT" "$STATS_ROOT" "$INTEROP_ROOT" "$RUN_ROOT"

log() {
  printf '[phase4-sameckpt] %s\n' "$1"
}

ensure_idle_training() {
  if ps aux | grep '[t]rain_ppo' | grep 'src.experiments_pgg_v0.train_ppo' >/dev/null 2>&1; then
    echo "refusing to start same-checkpoint suite while other train_ppo workers are active" >&2
    exit 2
  fi
}

copy_if_missing() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "missing required checkpoint: $src" >&2
    exit 1
  fi
  if [[ ! -f "$dst" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
  fi
}

hydrate_learned_50k() {
  local out_dir="$1"
  mkdir -p "$out_dir"
  for seed in "${FIVE_SEEDS[@]}"; do
    copy_if_missing "$BASE_50K_DIR/cond1_seed${seed}.pt" "$out_dir/cond1_seed${seed}_ep50000.pt"
  done
}

train_sameckpt_continuation_mode() {
  local mode="$1"
  local out_dir="$SAMECKPT_ROOT/$mode"
  log "train mode=$mode"
  hydrate_learned_50k "$out_dir"
  "$PYTHON_BIN" -m src.experiments_pgg_v0.run_phase3_seed_expansion \
    --out_dir "$out_dir" \
    --init_checkpoint_dir "$BASE_50K_DIR" \
    --init_episode 50000 \
    --episode_offset 50000 \
    --schedule_total_episodes 150000 \
    --conditions cond1 \
    --seeds "${FIVE_SEEDS[@]}" \
    --n_episodes 100000 \
    --T 100 \
    --rho 0.05 \
    --epsilon_tremble 0.05 \
    --sigmas 0.5 0.5 0.5 0.5 \
    --gamma 0.99 \
    --reward_scale 20.0 \
    --lr 3e-4 \
    --min_lr 1e-5 \
    --lr_schedule cosine \
    --entropy_coeff 0.01 \
    --entropy_schedule linear \
    --entropy_coeff_final 0.001 \
    --msg_entropy_coeff 0.01 \
    --msg_entropy_coeff_final 0.0 \
    --checkpoint_interval 50000 \
    --regime_log_interval 5000 \
    --log_interval 1000 \
    --msg_training_intervention "$mode" \
    --max_workers 3 \
    --skip_existing
}

eval_sameckpt_continuation_mode() {
  local mode="$1"
  local ckpt_dir="$SAMECKPT_ROOT/$mode"
  local out_dir="$SAMECKPT_EVAL_ROOT/$mode"
  log "eval mode=$mode"
  /bin/rm -rf "$out_dir"
  "$PYTHON_BIN" -m src.analysis.run_phase3_trimmed_eval \
    --checkpoint_dir "$ckpt_dir" \
    --suite_out_dir "$out_dir/suite" \
    --crossplay_out_dir "$out_dir/crossplay" \
    --comm_condition cond1 \
    --baseline_condition "" \
    --seeds "${FIVE_SEEDS[@]}" \
    --milestones 50000 100000 150000 \
    --interventions none zeros fixed0 fixed1 permute_slots \
    --crossplay_sender_milestones 50000 100000 150000 \
    --crossplay_receiver_milestones 150000 \
    --n_eval_episodes 300 \
    --eval_seed 9001 \
    --max_workers 4

  "$PYTHON_BIN" -m src.analysis.aggregate_phase3_report \
    --suite_main_csv "$out_dir/suite/checkpoint_suite_main.csv" \
    --suite_comm_csv "$out_dir/suite/checkpoint_suite_comm.csv" \
    --suite_trace_csv "$out_dir/suite/checkpoint_suite_trace.csv" \
    --suite_sender_csv "$out_dir/suite/checkpoint_suite_sender_semantics.csv" \
    --suite_receiver_csv "$out_dir/suite/checkpoint_suite_receiver_semantics.csv" \
    --suite_posterior_csv "$out_dir/suite/checkpoint_suite_posterior_strat.csv" \
    --crossplay_main_csv "$out_dir/crossplay/crossplay_matrix_main.csv" \
    --out_dir "$out_dir/report" \
    --out_md "$out_dir/report/PHASE3_SAMECKPT_${mode:u}.md"

  "$PYTHON_BIN" -m src.analysis.summarize_phase3_welfare \
    --suite_main_csv "$out_dir/suite/checkpoint_suite_main.csv" \
    --bundle_label "sameckpt_$mode" \
    --out_csv "$out_dir/report/welfare_weighted_raw.csv" \
    --out_mean_csv "$out_dir/report/welfare_weighted_mean.csv"
}

summarize_sameckpt_controls() {
  log "summarize same-checkpoint channel controls"
  local out_dir="$SAMECKPT_EVAL_ROOT/report"
  mkdir -p "$out_dir"
  "$PYTHON_BIN" -m src.analysis.summarize_phase3_channel_controls \
    --mode_suite learned "$LEARNED_150K_SUITE" \
    --mode_suite always_zero "$SAMECKPT_EVAL_ROOT/fixed0/suite/checkpoint_suite_main.csv" \
    --mode_suite indep_random "$SAMECKPT_EVAL_ROOT/uniform/suite/checkpoint_suite_main.csv" \
    --mode_suite public_random "$SAMECKPT_EVAL_ROOT/public_random/suite/checkpoint_suite_main.csv" \
    --out_dir "$out_dir"
}

compute_stats() {
  log "compute bootstrap CIs and paired deltas"
  "$PYTHON_BIN" -m src.analysis.compute_bootstrap_cis \
    --main_suite_csv "$LEARNED_150K_SUITE" \
    --sameckpt_suite always_zero "$SAMECKPT_EVAL_ROOT/fixed0/suite/checkpoint_suite_main.csv" \
    --sameckpt_suite indep_random "$SAMECKPT_EVAL_ROOT/uniform/suite/checkpoint_suite_main.csv" \
    --sameckpt_suite public_random "$SAMECKPT_EVAL_ROOT/public_random/suite/checkpoint_suite_main.csv" \
    --out_dir "$STATS_ROOT"
}

summarize_permutation() {
  log "summarize sender-slot permutation effect"
  "$PYTHON_BIN" -m src.analysis.summarize_sender_slot_permutation \
    --bundle learned "$LEARNED_150K_SUITE" \
    --bundle sameckpt_fixed0 "$SAMECKPT_EVAL_ROOT/fixed0/suite/checkpoint_suite_main.csv" \
    --bundle sameckpt_uniform "$SAMECKPT_EVAL_ROOT/uniform/suite/checkpoint_suite_main.csv" \
    --bundle sameckpt_public_random "$SAMECKPT_EVAL_ROOT/public_random/suite/checkpoint_suite_main.csv" \
    --out_dir "$INTEROP_ROOT"
}

ensure_idle_training

for mode in "${MODES[@]}"; do
  train_sameckpt_continuation_mode "$mode" > "$RUN_ROOT/${mode}.train.log" 2>&1
  eval_sameckpt_continuation_mode "$mode" > "$RUN_ROOT/${mode}.eval.log" 2>&1
  log "completed mode=$mode"
done

summarize_sameckpt_controls > "$RUN_ROOT/sameckpt_controls.summary.log" 2>&1
compute_stats > "$RUN_ROOT/stats.log" 2>&1
summarize_permutation > "$RUN_ROOT/permute_slots.log" 2>&1
log "phase4 same-checkpoint suite complete"
