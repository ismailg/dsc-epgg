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

SEEDS=(101 202 303)
CONTROL_ROOT="outputs/train/phase3_channel_controls_50k"
CONTROL_EVAL_ROOT="outputs/eval/phase3_channel_controls_50k"
EXT_ROOT="outputs/train/phase3_annealed_ext150k_3seeds"
EXT_EVAL_ROOT="outputs/eval/phase3_annealed_ext150k_3seeds"
COMPARE_ROOT="outputs/eval/phase3_compare"

mkdir -p "$CONTROL_ROOT" "$CONTROL_EVAL_ROOT" "$EXT_ROOT" "$EXT_EVAL_ROOT" "$COMPARE_ROOT"

log() {
  printf '[phase3-high-info] %s\n' "$1"
}

run_control_mode() {
  local mode="$1"
  local out_dir="$CONTROL_ROOT/$mode"
  log "train control mode=$mode"
  "$PYTHON_BIN" -m src.experiments_pgg_v0.run_phase3_seed_expansion \
    --fixed_f_dir outputs/train/fixed_f_grid \
    --out_dir "$out_dir" \
    --conditions cond1 \
    --seeds "${SEEDS[@]}" \
    --n_episodes 50000 \
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
    --checkpoint_interval 25000 \
    --regime_log_interval 5000 \
    --log_interval 1000 \
    --msg_training_intervention "$mode" \
    --max_workers 2 \
    --skip_existing
}

eval_control_mode() {
  local mode="$1"
  local ckpt_dir="$CONTROL_ROOT/$mode"
  local out_dir="$CONTROL_EVAL_ROOT/$mode"
  log "eval control mode=$mode"
  "$PYTHON_BIN" -m src.analysis.run_phase3_trimmed_eval \
    --checkpoint_dir "$ckpt_dir" \
    --suite_out_dir "$out_dir/suite" \
    --crossplay_out_dir "$out_dir/crossplay" \
    --comm_condition cond1 \
    --baseline_condition "" \
    --seeds "${SEEDS[@]}" \
    --milestones 25000 50000 \
    --interventions none zeros fixed0 fixed1 permute_slots \
    --crossplay_sender_milestones 25000 50000 \
    --crossplay_receiver_milestones 50000 \
    --n_eval_episodes 300 \
    --eval_seed 9001 \
    --max_workers 4 \
    --skip_existing
}

train_controls_parallel() {
  log "launch control training"
  run_control_mode fixed0 > "$CONTROL_ROOT/fixed0.train.log" 2>&1 &
  local pid_fixed0=$!
  run_control_mode uniform > "$CONTROL_ROOT/uniform.train.log" 2>&1 &
  local pid_uniform=$!
  run_control_mode public_random > "$CONTROL_ROOT/public_random.train.log" 2>&1 &
  local pid_public=$!
  wait "$pid_fixed0" "$pid_uniform" "$pid_public"
}

eval_controls_serial() {
  eval_control_mode fixed0 > "$CONTROL_EVAL_ROOT/fixed0.eval.log" 2>&1
  eval_control_mode uniform > "$CONTROL_EVAL_ROOT/uniform.eval.log" 2>&1
  eval_control_mode public_random > "$CONTROL_EVAL_ROOT/public_random.eval.log" 2>&1
  "$PYTHON_BIN" -m src.analysis.summarize_phase3_channel_controls \
    --learned_suite_csv outputs/eval/phase3_annealed_trimmed_all/suite/checkpoint_suite_main.csv \
    --zero_suite_csv "$CONTROL_EVAL_ROOT/fixed0/suite/checkpoint_suite_main.csv" \
    --uniform_suite_csv "$CONTROL_EVAL_ROOT/uniform/suite/checkpoint_suite_main.csv" \
    --public_suite_csv "$CONTROL_EVAL_ROOT/public_random/suite/checkpoint_suite_main.csv" \
    --out_dir "$CONTROL_EVAL_ROOT/report"
}

extend_annealed_horizon() {
  log "train annealed continuation to 150k"
  "$PYTHON_BIN" -m src.experiments_pgg_v0.run_phase3_seed_expansion \
    --out_dir "$EXT_ROOT" \
    --init_checkpoint_dir outputs/train/phase3_annealed_trimmed \
    --init_episode 50000 \
    --episode_offset 50000 \
    --schedule_total_episodes 150000 \
    --conditions cond1 cond2 \
    --seeds "${SEEDS[@]}" \
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
    --max_workers 4 \
    --skip_existing
}

hydrate_50k_into_extension_dir() {
  log "copy 50k checkpoints into extension dir"
  for seed in "${SEEDS[@]}"; do
    for cond in cond1 cond2; do
      local src="outputs/train/phase3_annealed_trimmed/${cond}_seed${seed}.pt"
      local dst="$EXT_ROOT/${cond}_seed${seed}_ep50000.pt"
      if [[ -f "$src" && ! -f "$dst" ]]; then
        cp "$src" "$dst"
      fi
    done
  done
}

eval_extension() {
  log "eval annealed extension"
  "$PYTHON_BIN" -m src.analysis.run_phase3_trimmed_eval \
    --checkpoint_dir "$EXT_ROOT" \
    --suite_out_dir "$EXT_EVAL_ROOT/suite" \
    --crossplay_out_dir "$EXT_EVAL_ROOT/crossplay" \
    --comm_condition cond1 \
    --baseline_condition cond2 \
    --seeds "${SEEDS[@]}" \
    --milestones 50000 100000 150000 \
    --interventions none zeros fixed0 fixed1 permute_slots \
    --crossplay_sender_milestones 50000 100000 150000 \
    --crossplay_receiver_milestones 150000 \
    --n_eval_episodes 300 \
    --eval_seed 9001 \
    --max_workers 4 \
    --skip_existing

  "$PYTHON_BIN" -m src.analysis.run_phase3_common_polarity_rescue \
    --checkpoint_dir "$EXT_ROOT" \
    --sender_summary_csv "$EXT_EVAL_ROOT/suite/checkpoint_suite_sender_semantics.csv" \
    --out_dir "$EXT_EVAL_ROOT/common_polarity_rescue" \
    --condition cond1 \
    --basis regime \
    --seeds "${SEEDS[@]}" \
    --milestones 50000 100000 150000 \
    --n_eval_episodes 300 \
    --max_workers 2

  "$PYTHON_BIN" -m src.analysis.aggregate_phase3_report \
    --suite_main_csv "$EXT_EVAL_ROOT/suite/checkpoint_suite_main.csv" \
    --suite_comm_csv "$EXT_EVAL_ROOT/suite/checkpoint_suite_comm.csv" \
    --suite_trace_csv "$EXT_EVAL_ROOT/suite/checkpoint_suite_trace.csv" \
    --suite_sender_csv "$EXT_EVAL_ROOT/suite/checkpoint_suite_sender_semantics.csv" \
    --suite_receiver_csv "$EXT_EVAL_ROOT/suite/checkpoint_suite_receiver_semantics.csv" \
    --suite_posterior_csv "$EXT_EVAL_ROOT/suite/checkpoint_suite_posterior_strat.csv" \
    --crossplay_main_csv "$EXT_EVAL_ROOT/crossplay/crossplay_matrix_main.csv" \
    --out_dir "$EXT_EVAL_ROOT/report" \
    --out_md "$EXT_EVAL_ROOT/report/PHASE3_ANNEALED_EXT150K_3SEEDS.md"

  "$PYTHON_BIN" -m src.analysis.plot_phase3_fragmentation \
    --sender_alignment_csv "$EXT_EVAL_ROOT/report/sender_alignment_summary.csv" \
    --receiver_summary_csv "$EXT_EVAL_ROOT/report/receiver_semantics_summary.csv" \
    --receiver_by_sender_csv "$EXT_EVAL_ROOT/report/receiver_by_sender_summary.csv" \
    --sender_summary_csv "$EXT_EVAL_ROOT/report/sender_semantics_summary.csv" \
    --out_dir "$EXT_EVAL_ROOT/fragmentation_figures"
}

plot_compare() {
  log "plot annealed vs unannealed trajectory"
  "$PYTHON_BIN" -m src.analysis.plot_phase3_annealed_vs_unannealed \
    --unannealed_suite_csv outputs/eval/phase3/checkpoint_suite/checkpoint_suite_main.csv \
    --unannealed_fragment_csv outputs/eval/phase3/fragmentation_figures/fragmentation_over_time.csv \
    --unannealed_rescue_csv outputs/eval/phase3/common_polarity_rescue/common_polarity_rescue_summary.csv \
    --annealed_suite_csvs \
      outputs/eval/phase3_annealed_trimmed_all/suite/checkpoint_suite_main.csv \
      "$EXT_EVAL_ROOT/suite/checkpoint_suite_main.csv" \
    --annealed_fragment_csvs \
      outputs/eval/phase3_annealed_trimmed_all/fragmentation_figures/fragmentation_over_time.csv \
      "$EXT_EVAL_ROOT/fragmentation_figures/fragmentation_over_time.csv" \
    --annealed_rescue_csvs \
      outputs/eval/phase3_annealed_trimmed_all/common_polarity_rescue/common_polarity_rescue_summary.csv \
      "$EXT_EVAL_ROOT/common_polarity_rescue/common_polarity_rescue_summary.csv" \
    --seeds 101 202 \
    --out_dir "$COMPARE_ROOT"
}

log "start"
train_controls_parallel
eval_controls_serial
extend_annealed_horizon
hydrate_50k_into_extension_dir
eval_extension
plot_compare
log "done"
