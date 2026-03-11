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
export MPLCONFIGDIR=/tmp/mpl_phase3_next
export XDG_CACHE_HOME=/tmp/xdg_phase3_next

CONTROL_SEEDS=(101 202 303)
FIVE_SEEDS=(101 202 303 404 505)
BASE_50K_DIR="outputs/train/phase3_annealed_trimmed"
BASE_3SEED_150K_DIR="outputs/train/phase3_annealed_ext150k_3seeds"
MUTE_ROOT="outputs/train/phase3_mute_after50k_ext150k_3seeds"
MUTE_EVAL_ROOT="outputs/eval/phase3_mute_after50k_ext150k_3seeds"
CONTROL_50K_ROOT="outputs/train/phase3_channel_controls_50k"
CONTROL_150K_ROOT="outputs/train/phase3_channel_controls_ext150k_3seeds"
CONTROL_150K_EVAL_ROOT="outputs/eval/phase3_channel_controls_ext150k_3seeds"
FIVESEED_150K_ROOT="outputs/train/phase3_annealed_ext150k_5seeds"
FIVESEED_150K_EVAL_ROOT="outputs/eval/phase3_annealed_ext150k_5seeds"
COMPARE_ROOT="outputs/eval/phase3_compare"
PIPELINE_ROOT="outputs/train/phase3_next_steps"
PID_FILE="$PIPELINE_ROOT/pipeline.pid"

mkdir -p "$PIPELINE_ROOT" "$MUTE_ROOT" "$MUTE_EVAL_ROOT" "$CONTROL_150K_ROOT" "$CONTROL_150K_EVAL_ROOT" "$FIVESEED_150K_ROOT" "$FIVESEED_150K_EVAL_ROOT" "$COMPARE_ROOT"

echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

log() {
  printf '[phase3-next] %s\n' "$1"
}

copy_if_missing() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" && ! -f "$dst" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
  fi
}

hydrate_learned_50k() {
  local target_root="$1"
  local seeds=("${@:2}")
  for seed in "${seeds[@]}"; do
    copy_if_missing "$BASE_50K_DIR/cond1_seed${seed}.pt" "$target_root/cond1_seed${seed}_ep50000.pt"
    copy_if_missing "$BASE_50K_DIR/cond2_seed${seed}.pt" "$target_root/cond2_seed${seed}_ep50000.pt"
  done
}

hydrate_3seed_150k() {
  local target_root="$1"
  for seed in "${CONTROL_SEEDS[@]}"; do
    for cond in cond1 cond2; do
      copy_if_missing "$BASE_3SEED_150K_DIR/${cond}_seed${seed}_ep100000.pt" "$target_root/${cond}_seed${seed}_ep100000.pt"
      copy_if_missing "$BASE_3SEED_150K_DIR/${cond}_seed${seed}_ep50000.pt" "$target_root/${cond}_seed${seed}_ep50000.pt"
      copy_if_missing "$BASE_3SEED_150K_DIR/${cond}_seed${seed}.pt" "$target_root/${cond}_seed${seed}.pt"
    done
  done
}

train_mute_after_50k() {
  log "train mute-after-50k continuation"
  hydrate_learned_50k "$MUTE_ROOT" "${CONTROL_SEEDS[@]}"
  "$PYTHON_BIN" -m src.experiments_pgg_v0.run_phase3_seed_expansion \
    --out_dir "$MUTE_ROOT" \
    --init_checkpoint_dir "$BASE_50K_DIR" \
    --init_episode 50000 \
    --episode_offset 50000 \
    --schedule_total_episodes 150000 \
    --conditions cond1 \
    --seeds "${CONTROL_SEEDS[@]}" \
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
    --msg_training_intervention fixed0 \
    --max_workers 3 \
    --skip_existing
}

eval_mute_after_50k() {
  log "eval mute-after-50k continuation"
  rm -rf "$MUTE_EVAL_ROOT"
  "$PYTHON_BIN" -m src.analysis.run_phase3_trimmed_eval \
    --checkpoint_dir "$MUTE_ROOT" \
    --suite_out_dir "$MUTE_EVAL_ROOT/suite" \
    --crossplay_out_dir "$MUTE_EVAL_ROOT/crossplay" \
    --comm_condition cond1 \
    --baseline_condition "" \
    --seeds "${CONTROL_SEEDS[@]}" \
    --milestones 50000 100000 150000 \
    --interventions none zeros fixed0 fixed1 permute_slots \
    --crossplay_sender_milestones 50000 100000 150000 \
    --crossplay_receiver_milestones 150000 \
    --n_eval_episodes 300 \
    --eval_seed 9001 \
    --max_workers 4

  "$PYTHON_BIN" -m src.analysis.aggregate_phase3_report \
    --suite_main_csv "$MUTE_EVAL_ROOT/suite/checkpoint_suite_main.csv" \
    --suite_comm_csv "$MUTE_EVAL_ROOT/suite/checkpoint_suite_comm.csv" \
    --suite_trace_csv "$MUTE_EVAL_ROOT/suite/checkpoint_suite_trace.csv" \
    --suite_sender_csv "$MUTE_EVAL_ROOT/suite/checkpoint_suite_sender_semantics.csv" \
    --suite_receiver_csv "$MUTE_EVAL_ROOT/suite/checkpoint_suite_receiver_semantics.csv" \
    --suite_posterior_csv "$MUTE_EVAL_ROOT/suite/checkpoint_suite_posterior_strat.csv" \
    --crossplay_main_csv "$MUTE_EVAL_ROOT/crossplay/crossplay_matrix_main.csv" \
    --out_dir "$MUTE_EVAL_ROOT/report" \
    --out_md "$MUTE_EVAL_ROOT/report/PHASE3_MUTE_AFTER50K_EXT150K_3SEEDS.md"

  "$PYTHON_BIN" -m src.analysis.summarize_phase3_welfare \
    --suite_main_csv "$MUTE_EVAL_ROOT/suite/checkpoint_suite_main.csv" \
    --bundle_label mute_after50k \
    --out_csv "$MUTE_EVAL_ROOT/report/welfare_weighted_raw.csv" \
    --out_mean_csv "$MUTE_EVAL_ROOT/report/welfare_weighted_mean.csv"
}

train_controls_to_150k_mode() {
  local mode="$1"
  local out_dir="$CONTROL_150K_ROOT/$mode"
  log "train control continuation mode=$mode"
  hydrate_learned_50k "$out_dir" "${CONTROL_SEEDS[@]}"
  "$PYTHON_BIN" -m src.experiments_pgg_v0.run_phase3_seed_expansion \
    --out_dir "$out_dir" \
    --init_checkpoint_dir "$CONTROL_50K_ROOT/$mode" \
    --init_episode 50000 \
    --episode_offset 50000 \
    --schedule_total_episodes 150000 \
    --conditions cond1 \
    --seeds "${CONTROL_SEEDS[@]}" \
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
    --max_workers 2 \
    --skip_existing
}

train_controls_to_150k_parallel() {
  log "train 150k control continuations"
  train_controls_to_150k_mode fixed0 > "$PIPELINE_ROOT/fixed0_150k.train.log" 2>&1 &
  local pid_fixed0=$!
  train_controls_to_150k_mode uniform > "$PIPELINE_ROOT/uniform_150k.train.log" 2>&1 &
  local pid_uniform=$!
  train_controls_to_150k_mode public_random > "$PIPELINE_ROOT/public_random_150k.train.log" 2>&1 &
  local pid_public=$!
  wait "$pid_fixed0" "$pid_uniform" "$pid_public"
}

eval_controls_to_150k_mode() {
  local mode="$1"
  local ckpt_dir="$CONTROL_150K_ROOT/$mode"
  local out_dir="$CONTROL_150K_EVAL_ROOT/$mode"
  log "eval 150k control mode=$mode"
  rm -rf "$out_dir"
  "$PYTHON_BIN" -m src.analysis.run_phase3_trimmed_eval \
    --checkpoint_dir "$ckpt_dir" \
    --suite_out_dir "$out_dir/suite" \
    --crossplay_out_dir "$out_dir/crossplay" \
    --comm_condition cond1 \
    --baseline_condition "" \
    --seeds "${CONTROL_SEEDS[@]}" \
    --milestones 50000 100000 150000 \
    --interventions none zeros fixed0 fixed1 permute_slots \
    --crossplay_sender_milestones 50000 100000 150000 \
    --crossplay_receiver_milestones 150000 \
    --n_eval_episodes 300 \
    --eval_seed 9001 \
    --max_workers 4

  "$PYTHON_BIN" -m src.analysis.summarize_phase3_welfare \
    --suite_main_csv "$out_dir/suite/checkpoint_suite_main.csv" \
    --bundle_label "$mode" \
    --out_csv "$out_dir/report/welfare_weighted_raw.csv" \
    --out_mean_csv "$out_dir/report/welfare_weighted_mean.csv"
}

eval_controls_to_150k() {
  log "eval 150k control continuations"
  eval_controls_to_150k_mode fixed0 > "$PIPELINE_ROOT/fixed0_150k.eval.log" 2>&1
  eval_controls_to_150k_mode uniform > "$PIPELINE_ROOT/uniform_150k.eval.log" 2>&1
  eval_controls_to_150k_mode public_random > "$PIPELINE_ROOT/public_random_150k.eval.log" 2>&1

  "$PYTHON_BIN" -m src.analysis.summarize_phase3_channel_controls \
    --learned_suite_csv outputs/eval/phase3_annealed_ext150k_3seeds/suite/checkpoint_suite_main.csv \
    --zero_suite_csv "$CONTROL_150K_EVAL_ROOT/fixed0/suite/checkpoint_suite_main.csv" \
    --uniform_suite_csv "$CONTROL_150K_EVAL_ROOT/uniform/suite/checkpoint_suite_main.csv" \
    --public_suite_csv "$CONTROL_150K_EVAL_ROOT/public_random/suite/checkpoint_suite_main.csv" \
    --out_dir "$CONTROL_150K_EVAL_ROOT/report"
}

prepare_5seed_extension_dir() {
  log "prepare 5-seed 150k directory"
  hydrate_3seed_150k "$FIVESEED_150K_ROOT"
  hydrate_learned_50k "$FIVESEED_150K_ROOT" 404 505
}

extend_annealed_to_150k_5seeds() {
  log "extend learned/no-comm to 150k for seeds 404/505"
  prepare_5seed_extension_dir
  "$PYTHON_BIN" -m src.experiments_pgg_v0.run_phase3_seed_expansion \
    --out_dir "$FIVESEED_150K_ROOT" \
    --init_checkpoint_dir "$BASE_50K_DIR" \
    --init_episode 50000 \
    --episode_offset 50000 \
    --schedule_total_episodes 150000 \
    --conditions cond1 cond2 \
    --seeds 404 505 \
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

eval_5seed_extension() {
  log "eval 5-seed 150k learned/no-comm bundle"
  rm -rf "$FIVESEED_150K_EVAL_ROOT"
  "$PYTHON_BIN" -m src.analysis.run_phase3_trimmed_eval \
    --checkpoint_dir "$FIVESEED_150K_ROOT" \
    --suite_out_dir "$FIVESEED_150K_EVAL_ROOT/suite" \
    --crossplay_out_dir "$FIVESEED_150K_EVAL_ROOT/crossplay" \
    --comm_condition cond1 \
    --baseline_condition cond2 \
    --seeds "${FIVE_SEEDS[@]}" \
    --milestones 50000 100000 150000 \
    --interventions none zeros fixed0 fixed1 permute_slots \
    --crossplay_sender_milestones 50000 100000 150000 \
    --crossplay_receiver_milestones 150000 \
    --n_eval_episodes 300 \
    --eval_seed 9001 \
    --max_workers 4

  "$PYTHON_BIN" -m src.analysis.run_phase3_common_polarity_rescue \
    --checkpoint_dir "$FIVESEED_150K_ROOT" \
    --sender_summary_csv "$FIVESEED_150K_EVAL_ROOT/suite/checkpoint_suite_sender_semantics.csv" \
    --out_dir "$FIVESEED_150K_EVAL_ROOT/common_polarity_rescue" \
    --condition cond1 \
    --basis regime \
    --seeds "${FIVE_SEEDS[@]}" \
    --milestones 50000 100000 150000 \
    --n_eval_episodes 300 \
    --max_workers 2

  "$PYTHON_BIN" -m src.analysis.aggregate_phase3_report \
    --suite_main_csv "$FIVESEED_150K_EVAL_ROOT/suite/checkpoint_suite_main.csv" \
    --suite_comm_csv "$FIVESEED_150K_EVAL_ROOT/suite/checkpoint_suite_comm.csv" \
    --suite_trace_csv "$FIVESEED_150K_EVAL_ROOT/suite/checkpoint_suite_trace.csv" \
    --suite_sender_csv "$FIVESEED_150K_EVAL_ROOT/suite/checkpoint_suite_sender_semantics.csv" \
    --suite_receiver_csv "$FIVESEED_150K_EVAL_ROOT/suite/checkpoint_suite_receiver_semantics.csv" \
    --suite_posterior_csv "$FIVESEED_150K_EVAL_ROOT/suite/checkpoint_suite_posterior_strat.csv" \
    --crossplay_main_csv "$FIVESEED_150K_EVAL_ROOT/crossplay/crossplay_matrix_main.csv" \
    --out_dir "$FIVESEED_150K_EVAL_ROOT/report" \
    --out_md "$FIVESEED_150K_EVAL_ROOT/report/PHASE3_ANNEALED_EXT150K_5SEEDS.md"

  "$PYTHON_BIN" -m src.analysis.plot_phase3_fragmentation \
    --sender_alignment_csv "$FIVESEED_150K_EVAL_ROOT/report/sender_alignment_summary.csv" \
    --receiver_summary_csv "$FIVESEED_150K_EVAL_ROOT/report/receiver_semantics_summary.csv" \
    --receiver_by_sender_csv "$FIVESEED_150K_EVAL_ROOT/report/receiver_by_sender_summary.csv" \
    --sender_summary_csv "$FIVESEED_150K_EVAL_ROOT/report/sender_semantics_summary.csv" \
    --out_dir "$FIVESEED_150K_EVAL_ROOT/fragmentation_figures"

  "$PYTHON_BIN" -m src.analysis.summarize_phase3_welfare \
    --suite_main_csv "$FIVESEED_150K_EVAL_ROOT/suite/checkpoint_suite_main.csv" \
    --bundle_label annealed_ext150k_5seeds \
    --out_csv "$FIVESEED_150K_EVAL_ROOT/report/welfare_weighted_raw.csv" \
    --out_mean_csv "$FIVESEED_150K_EVAL_ROOT/report/welfare_weighted_mean.csv"
}

plot_compare_updated() {
  log "update annealed-vs-unannealed trajectory figure"
  "$PYTHON_BIN" -m src.analysis.plot_phase3_annealed_vs_unannealed \
    --unannealed_suite_csv outputs/eval/phase3/checkpoint_suite/checkpoint_suite_main.csv \
    --unannealed_fragment_csv outputs/eval/phase3/fragmentation_figures/fragmentation_over_time.csv \
    --unannealed_rescue_csv outputs/eval/phase3/common_polarity_rescue/common_polarity_rescue_summary.csv \
    --annealed_suite_csvs \
      outputs/eval/phase3_annealed_trimmed_all/suite/checkpoint_suite_main.csv \
      outputs/eval/phase3_annealed_ext150k_3seeds/suite/checkpoint_suite_main.csv \
      "$FIVESEED_150K_EVAL_ROOT/suite/checkpoint_suite_main.csv" \
    --annealed_fragment_csvs \
      outputs/eval/phase3_annealed_trimmed_all/fragmentation_figures/fragmentation_over_time.csv \
      outputs/eval/phase3_annealed_ext150k_3seeds/fragmentation_figures/fragmentation_over_time.csv \
      "$FIVESEED_150K_EVAL_ROOT/fragmentation_figures/fragmentation_over_time.csv" \
    --annealed_rescue_csvs \
      outputs/eval/phase3_annealed_trimmed_all/common_polarity_rescue/common_polarity_rescue_summary.csv \
      outputs/eval/phase3_annealed_ext150k_3seeds/common_polarity_rescue/common_polarity_rescue_summary.csv \
      "$FIVESEED_150K_EVAL_ROOT/common_polarity_rescue/common_polarity_rescue_summary.csv" \
    --seeds 101 202 \
    --out_dir "$COMPARE_ROOT"
}

log "start"
train_mute_after_50k
eval_mute_after_50k
train_controls_to_150k_parallel
eval_controls_to_150k
extend_annealed_to_150k_5seeds
eval_5seed_extension
plot_compare_updated
log "done"
