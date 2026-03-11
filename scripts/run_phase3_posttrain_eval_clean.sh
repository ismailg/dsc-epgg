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
export MPLCONFIGDIR=/tmp/mpl_phase3_post
export XDG_CACHE_HOME=/tmp/xdg_phase3_post

MUTE_TRAIN="outputs/train/phase3_mute_after50k_ext150k_3seeds"
MUTE_EVAL="outputs/eval/phase3_mute_after50k_ext150k_3seeds"
CTRL_TRAIN="outputs/train/phase3_channel_controls_ext150k_3seeds"
CTRL_EVAL="outputs/eval/phase3_channel_controls_ext150k_3seeds"
FIVE_TRAIN="outputs/train/phase3_annealed_ext150k_5seeds"
FIVE_EVAL="outputs/eval/phase3_annealed_ext150k_5seeds"
COMPARE_ROOT="outputs/eval/phase3_compare"

ensure_idle_training() {
  if ps aux | grep '[t]rain_ppo' | grep 'src.experiments_pgg_v0.train_ppo' >/dev/null 2>&1; then
    echo "refusing to run post-training eval while train_ppo workers are still active" >&2
    echo "wait for training to finish, then rerun this script" >&2
    exit 2
  fi
}

ensure_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
}

ensure_bundle_complete() {
  local root="$1"; shift
  for path in "$@"; do
    ensure_file "$root/$path"
  done
}

clean_dir() {
  local path="$1"
  /bin/rm -rf "$path"
  /bin/mkdir -p "$path"
}

run_mute_eval() {
  ensure_bundle_complete "$MUTE_TRAIN" \
    "cond1_seed101_fixed0.pt" \
    "cond1_seed202_fixed0.pt" \
    "cond1_seed303_fixed0.pt"
  clean_dir "$MUTE_EVAL"

  "$PYTHON_BIN" -m src.analysis.run_phase3_trimmed_eval \
    --checkpoint_dir "$MUTE_TRAIN" \
    --suite_out_dir "$MUTE_EVAL/suite" \
    --crossplay_out_dir "$MUTE_EVAL/crossplay" \
    --comm_condition cond1 \
    --baseline_condition "" \
    --seeds 101 202 303 \
    --milestones 50000 100000 150000 \
    --interventions none zeros fixed0 fixed1 permute_slots \
    --crossplay_sender_milestones 50000 100000 150000 \
    --crossplay_receiver_milestones 150000 \
    --n_eval_episodes 300 \
    --eval_seed 9001 \
    --max_workers 4

  "$PYTHON_BIN" -m src.analysis.aggregate_phase3_report \
    --suite_main_csv "$MUTE_EVAL/suite/checkpoint_suite_main.csv" \
    --suite_comm_csv "$MUTE_EVAL/suite/checkpoint_suite_comm.csv" \
    --suite_trace_csv "$MUTE_EVAL/suite/checkpoint_suite_trace.csv" \
    --suite_sender_csv "$MUTE_EVAL/suite/checkpoint_suite_sender_semantics.csv" \
    --suite_receiver_csv "$MUTE_EVAL/suite/checkpoint_suite_receiver_semantics.csv" \
    --suite_posterior_csv "$MUTE_EVAL/suite/checkpoint_suite_posterior_strat.csv" \
    --crossplay_main_csv "$MUTE_EVAL/crossplay/crossplay_matrix_main.csv" \
    --out_dir "$MUTE_EVAL/report" \
    --out_md "$MUTE_EVAL/report/PHASE3_MUTE_AFTER50K_EXT150K_3SEEDS.md"

  "$PYTHON_BIN" -m src.analysis.summarize_phase3_welfare \
    --suite_main_csv "$MUTE_EVAL/suite/checkpoint_suite_main.csv" \
    --bundle_label mute_after50k \
    --out_csv "$MUTE_EVAL/report/welfare_weighted_raw.csv" \
    --out_mean_csv "$MUTE_EVAL/report/welfare_weighted_mean.csv"
}

run_controls_eval() {
  for mode in fixed0 uniform public_random; do
    ensure_bundle_complete "$CTRL_TRAIN/$mode" \
      "cond1_seed101_${mode}.pt" \
      "cond1_seed202_${mode}.pt" \
      "cond1_seed303_${mode}.pt"
    clean_dir "$CTRL_EVAL/$mode"

    "$PYTHON_BIN" -m src.analysis.run_phase3_trimmed_eval \
      --checkpoint_dir "$CTRL_TRAIN/$mode" \
      --suite_out_dir "$CTRL_EVAL/$mode/suite" \
      --crossplay_out_dir "$CTRL_EVAL/$mode/crossplay" \
      --comm_condition cond1 \
      --baseline_condition "" \
      --seeds 101 202 303 \
      --milestones 50000 100000 150000 \
      --interventions none zeros fixed0 fixed1 permute_slots \
      --crossplay_sender_milestones 50000 100000 150000 \
      --crossplay_receiver_milestones 150000 \
      --n_eval_episodes 300 \
      --eval_seed 9001 \
      --max_workers 4

    "$PYTHON_BIN" -m src.analysis.summarize_phase3_welfare \
      --suite_main_csv "$CTRL_EVAL/$mode/suite/checkpoint_suite_main.csv" \
      --bundle_label "$mode" \
      --out_csv "$CTRL_EVAL/$mode/report/welfare_weighted_raw.csv" \
      --out_mean_csv "$CTRL_EVAL/$mode/report/welfare_weighted_mean.csv"
  done

  "$PYTHON_BIN" -m src.analysis.summarize_phase3_channel_controls \
    --learned_suite_csv outputs/eval/phase3_annealed_ext150k_3seeds/suite/checkpoint_suite_main.csv \
    --zero_suite_csv "$CTRL_EVAL/fixed0/suite/checkpoint_suite_main.csv" \
    --uniform_suite_csv "$CTRL_EVAL/uniform/suite/checkpoint_suite_main.csv" \
    --public_suite_csv "$CTRL_EVAL/public_random/suite/checkpoint_suite_main.csv" \
    --out_dir "$CTRL_EVAL/report"
}

run_fiveseed_eval() {
  ensure_bundle_complete "$FIVE_TRAIN" \
    "cond1_seed101.pt" "cond1_seed202.pt" "cond1_seed303.pt" "cond1_seed404.pt" "cond1_seed505.pt" \
    "cond2_seed101.pt" "cond2_seed202.pt" "cond2_seed303.pt" "cond2_seed404.pt" "cond2_seed505.pt"
  clean_dir "$FIVE_EVAL"

  "$PYTHON_BIN" -m src.analysis.run_phase3_trimmed_eval \
    --checkpoint_dir "$FIVE_TRAIN" \
    --suite_out_dir "$FIVE_EVAL/suite" \
    --crossplay_out_dir "$FIVE_EVAL/crossplay" \
    --comm_condition cond1 \
    --baseline_condition cond2 \
    --seeds 101 202 303 404 505 \
    --milestones 50000 100000 150000 \
    --interventions none zeros fixed0 fixed1 permute_slots \
    --crossplay_sender_milestones 50000 100000 150000 \
    --crossplay_receiver_milestones 150000 \
    --n_eval_episodes 300 \
    --eval_seed 9001 \
    --max_workers 4

  "$PYTHON_BIN" -m src.analysis.run_phase3_common_polarity_rescue \
    --checkpoint_dir "$FIVE_TRAIN" \
    --sender_summary_csv "$FIVE_EVAL/suite/checkpoint_suite_sender_semantics.csv" \
    --out_dir "$FIVE_EVAL/common_polarity_rescue" \
    --condition cond1 \
    --basis regime \
    --seeds 101 202 303 404 505 \
    --milestones 50000 100000 150000 \
    --n_eval_episodes 300 \
    --max_workers 2

  "$PYTHON_BIN" -m src.analysis.aggregate_phase3_report \
    --suite_main_csv "$FIVE_EVAL/suite/checkpoint_suite_main.csv" \
    --suite_comm_csv "$FIVE_EVAL/suite/checkpoint_suite_comm.csv" \
    --suite_trace_csv "$FIVE_EVAL/suite/checkpoint_suite_trace.csv" \
    --suite_sender_csv "$FIVE_EVAL/suite/checkpoint_suite_sender_semantics.csv" \
    --suite_receiver_csv "$FIVE_EVAL/suite/checkpoint_suite_receiver_semantics.csv" \
    --suite_posterior_csv "$FIVE_EVAL/suite/checkpoint_suite_posterior_strat.csv" \
    --crossplay_main_csv "$FIVE_EVAL/crossplay/crossplay_matrix_main.csv" \
    --out_dir "$FIVE_EVAL/report" \
    --out_md "$FIVE_EVAL/report/PHASE3_ANNEALED_EXT150K_5SEEDS.md"

  "$PYTHON_BIN" -m src.analysis.plot_phase3_fragmentation \
    --sender_alignment_csv "$FIVE_EVAL/report/sender_alignment_summary.csv" \
    --receiver_summary_csv "$FIVE_EVAL/report/receiver_semantics_summary.csv" \
    --receiver_by_sender_csv "$FIVE_EVAL/report/receiver_by_sender_summary.csv" \
    --sender_summary_csv "$FIVE_EVAL/report/sender_semantics_summary.csv" \
    --out_dir "$FIVE_EVAL/fragmentation_figures"

  "$PYTHON_BIN" -m src.analysis.summarize_phase3_welfare \
    --suite_main_csv "$FIVE_EVAL/suite/checkpoint_suite_main.csv" \
    --bundle_label annealed_ext150k_5seeds \
    --out_csv "$FIVE_EVAL/report/welfare_weighted_raw.csv" \
    --out_mean_csv "$FIVE_EVAL/report/welfare_weighted_mean.csv"
}

update_compare() {
  "$PYTHON_BIN" -m src.analysis.plot_phase3_annealed_vs_unannealed \
    --unannealed_suite_csv outputs/eval/phase3/checkpoint_suite/checkpoint_suite_main.csv \
    --unannealed_fragment_csv outputs/eval/phase3/fragmentation_figures/fragmentation_over_time.csv \
    --unannealed_rescue_csv outputs/eval/phase3/common_polarity_rescue/common_polarity_rescue_summary.csv \
    --annealed_suite_csvs \
      outputs/eval/phase3_annealed_trimmed_all/suite/checkpoint_suite_main.csv \
      outputs/eval/phase3_annealed_ext150k_3seeds/suite/checkpoint_suite_main.csv \
      "$FIVE_EVAL/suite/checkpoint_suite_main.csv" \
    --annealed_fragment_csvs \
      outputs/eval/phase3_annealed_trimmed_all/fragmentation_figures/fragmentation_over_time.csv \
      outputs/eval/phase3_annealed_ext150k_3seeds/fragmentation_figures/fragmentation_over_time.csv \
      "$FIVE_EVAL/fragmentation_figures/fragmentation_over_time.csv" \
    --annealed_rescue_csvs \
      outputs/eval/phase3_annealed_trimmed_all/common_polarity_rescue/common_polarity_rescue_summary.csv \
      outputs/eval/phase3_annealed_ext150k_3seeds/common_polarity_rescue/common_polarity_rescue_summary.csv \
      "$FIVE_EVAL/common_polarity_rescue/common_polarity_rescue_summary.csv" \
    --seeds 101 202 \
    --out_dir "$COMPARE_ROOT"
}

ensure_idle_training
run_mute_eval
run_controls_eval
run_fiveseed_eval
update_compare
