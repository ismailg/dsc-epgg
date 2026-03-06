#!/bin/zsh
set -euo pipefail

REPO_ROOT="/Users/mbp17/POSTDOC/NPS26/dsc-epgg"

TRAIN_ROOT="outputs/train/phase3_annealed_trimmed"
PILOT_EVAL_ROOT="outputs/eval/phase3_annealed_trimmed_seed101"
FULL_EVAL_ROOT="outputs/eval/phase3_annealed_trimmed_all"

PILOT_SUITE_OUT="${PILOT_EVAL_ROOT}/suite"
PILOT_CROSSPLAY_OUT="${PILOT_EVAL_ROOT}/crossplay"
PILOT_REPORT_OUT="${PILOT_EVAL_ROOT}/report"
PILOT_RESCUE_OUT="${PILOT_EVAL_ROOT}/common_polarity_rescue"
PILOT_FRAGMENT_OUT="${PILOT_EVAL_ROOT}/fragmentation_figures"

FULL_SUITE_OUT="${FULL_EVAL_ROOT}/suite"
FULL_CROSSPLAY_OUT="${FULL_EVAL_ROOT}/crossplay"
FULL_REPORT_OUT="${FULL_EVAL_ROOT}/report"
FULL_RESCUE_OUT="${FULL_EVAL_ROOT}/common_polarity_rescue"
FULL_FRAGMENT_OUT="${FULL_EVAL_ROOT}/fragmentation_figures"

cd "${REPO_ROOT}"

mkdir -p \
  "${TRAIN_ROOT}" \
  "${PILOT_SUITE_OUT}" "${PILOT_CROSSPLAY_OUT}" "${PILOT_REPORT_OUT}" "${PILOT_RESCUE_OUT}" "${PILOT_FRAGMENT_OUT}" \
  "${FULL_SUITE_OUT}" "${FULL_CROSSPLAY_OUT}" "${FULL_REPORT_OUT}" "${FULL_RESCUE_OUT}" "${FULL_FRAGMENT_OUT}"

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[phase3-pipeline] step=pilot-train start=$(date -u +%FT%TZ)"
python3 -u -m src.experiments_pgg_v0.run_phase3_seed_expansion \
  --out_dir "${TRAIN_ROOT}" \
  --fixed_f_dir outputs/train/fixed_f_grid \
  --conditions cond1 cond2 \
  --seeds 101 \
  --n_episodes 50000 \
  --checkpoint_interval 25000 \
  --log_interval 1000 \
  --regime_log_interval 1000 \
  --max_workers 2 \
  --lr_schedule cosine \
  --min_lr 1e-5 \
  --entropy_schedule linear \
  --entropy_coeff 0.01 \
  --entropy_coeff_final 0.001 \
  --msg_entropy_coeff 0.01 \
  --msg_entropy_coeff_final 0.0 \
  --skip_existing

echo "[phase3-pipeline] step=pilot-eval start=$(date -u +%FT%TZ)"
python3 -u -m src.analysis.run_phase3_trimmed_eval \
  --checkpoint_dir "${TRAIN_ROOT}" \
  --suite_out_dir "${PILOT_SUITE_OUT}" \
  --crossplay_out_dir "${PILOT_CROSSPLAY_OUT}" \
  --comm_condition cond1 \
  --baseline_condition cond2 \
  --seeds 101 \
  --milestones 25000 50000 \
  --interventions none zeros fixed0 fixed1 \
  --crossplay_sender_milestones 25000 50000 \
  --crossplay_receiver_milestones 50000 \
  --n_eval_episodes 300 \
  --max_workers 4 \
  --skip_existing

python3 -u -m src.analysis.aggregate_phase3_report \
  --suite_main_csv "${PILOT_SUITE_OUT}/checkpoint_suite_main.csv" \
  --suite_comm_csv "${PILOT_SUITE_OUT}/checkpoint_suite_comm.csv" \
  --suite_trace_csv "${PILOT_SUITE_OUT}/checkpoint_suite_trace.csv" \
  --suite_sender_csv "${PILOT_SUITE_OUT}/checkpoint_suite_sender_semantics.csv" \
  --suite_receiver_csv "${PILOT_SUITE_OUT}/checkpoint_suite_receiver_semantics.csv" \
  --suite_posterior_csv "${PILOT_SUITE_OUT}/checkpoint_suite_posterior_strat.csv" \
  --crossplay_main_csv "${PILOT_CROSSPLAY_OUT}/crossplay_matrix_main.csv" \
  --out_dir "${PILOT_REPORT_OUT}" \
  --out_md "${PILOT_REPORT_OUT}/PHASE3_ANNEALED_SEED101.md"

echo "[phase3-pipeline] step=pilot-rescue-fragmentation start=$(date -u +%FT%TZ)"
python3 -u -m src.analysis.run_phase3_common_polarity_rescue \
  --checkpoint_dir "${TRAIN_ROOT}" \
  --sender_summary_csv "${PILOT_REPORT_OUT}/sender_semantics_summary.csv" \
  --out_dir "${PILOT_RESCUE_OUT}" \
  --condition cond1 \
  --basis regime \
  --seeds 101 \
  --milestones 25000 50000 \
  --n_eval_episodes 300 \
  --max_workers 2 \
  --skip_existing

python3 -u -m src.analysis.summarize_common_polarity_rescue \
  --base_main_csv "${PILOT_SUITE_OUT}/checkpoint_suite_main.csv" \
  --rescue_main_csv "${PILOT_RESCUE_OUT}/common_polarity_rescue_main.csv" \
  --out_csv "${PILOT_RESCUE_OUT}/common_polarity_rescue_summary.csv"

python3 -u -m src.analysis.plot_phase3_fragmentation \
  --sender_alignment_csv "${PILOT_REPORT_OUT}/sender_alignment_summary.csv" \
  --receiver_summary_csv "${PILOT_REPORT_OUT}/receiver_semantics_summary.csv" \
  --receiver_by_sender_csv "${PILOT_REPORT_OUT}/receiver_by_sender_summary.csv" \
  --sender_summary_csv "${PILOT_REPORT_OUT}/sender_semantics_summary.csv" \
  --out_dir "${PILOT_FRAGMENT_OUT}"

echo "[phase3-pipeline] step=expand-train start=$(date -u +%FT%TZ)"
python3 -u -m src.experiments_pgg_v0.run_phase3_seed_expansion \
  --out_dir "${TRAIN_ROOT}" \
  --fixed_f_dir outputs/train/fixed_f_grid \
  --conditions cond1 cond2 \
  --seeds 202 303 404 505 \
  --n_episodes 50000 \
  --checkpoint_interval 25000 \
  --log_interval 1000 \
  --regime_log_interval 1000 \
  --max_workers 4 \
  --lr_schedule cosine \
  --min_lr 1e-5 \
  --entropy_schedule linear \
  --entropy_coeff 0.01 \
  --entropy_coeff_final 0.001 \
  --msg_entropy_coeff 0.01 \
  --msg_entropy_coeff_final 0.0 \
  --skip_existing

echo "[phase3-pipeline] step=expand-eval-report start=$(date -u +%FT%TZ)"
python3 -u -m src.analysis.run_phase3_trimmed_eval \
  --checkpoint_dir "${TRAIN_ROOT}" \
  --suite_out_dir "${FULL_SUITE_OUT}" \
  --crossplay_out_dir "${FULL_CROSSPLAY_OUT}" \
  --comm_condition cond1 \
  --baseline_condition cond2 \
  --seeds 101 202 303 404 505 \
  --milestones 25000 50000 \
  --interventions none zeros fixed0 fixed1 \
  --crossplay_sender_milestones 25000 50000 \
  --crossplay_receiver_milestones 50000 \
  --n_eval_episodes 300 \
  --max_workers 6 \
  --skip_existing

python3 -u -m src.analysis.aggregate_phase3_report \
  --suite_main_csv "${FULL_SUITE_OUT}/checkpoint_suite_main.csv" \
  --suite_comm_csv "${FULL_SUITE_OUT}/checkpoint_suite_comm.csv" \
  --suite_trace_csv "${FULL_SUITE_OUT}/checkpoint_suite_trace.csv" \
  --suite_sender_csv "${FULL_SUITE_OUT}/checkpoint_suite_sender_semantics.csv" \
  --suite_receiver_csv "${FULL_SUITE_OUT}/checkpoint_suite_receiver_semantics.csv" \
  --suite_posterior_csv "${FULL_SUITE_OUT}/checkpoint_suite_posterior_strat.csv" \
  --crossplay_main_csv "${FULL_CROSSPLAY_OUT}/crossplay_matrix_main.csv" \
  --out_dir "${FULL_REPORT_OUT}" \
  --out_md "${FULL_REPORT_OUT}/PHASE3_ANNEALED_ALL_SEEDS.md"

echo "[phase3-pipeline] step=expand-rescue-fragmentation start=$(date -u +%FT%TZ)"
python3 -u -m src.analysis.run_phase3_common_polarity_rescue \
  --checkpoint_dir "${TRAIN_ROOT}" \
  --sender_summary_csv "${FULL_REPORT_OUT}/sender_semantics_summary.csv" \
  --out_dir "${FULL_RESCUE_OUT}" \
  --condition cond1 \
  --basis regime \
  --seeds 101 202 303 404 505 \
  --milestones 25000 50000 \
  --n_eval_episodes 300 \
  --max_workers 4 \
  --skip_existing

python3 -u -m src.analysis.summarize_common_polarity_rescue \
  --base_main_csv "${FULL_SUITE_OUT}/checkpoint_suite_main.csv" \
  --rescue_main_csv "${FULL_RESCUE_OUT}/common_polarity_rescue_main.csv" \
  --out_csv "${FULL_RESCUE_OUT}/common_polarity_rescue_summary.csv"

python3 -u -m src.analysis.plot_phase3_fragmentation \
  --sender_alignment_csv "${FULL_REPORT_OUT}/sender_alignment_summary.csv" \
  --receiver_summary_csv "${FULL_REPORT_OUT}/receiver_semantics_summary.csv" \
  --receiver_by_sender_csv "${FULL_REPORT_OUT}/receiver_by_sender_summary.csv" \
  --sender_summary_csv "${FULL_REPORT_OUT}/sender_semantics_summary.csv" \
  --out_dir "${FULL_FRAGMENT_OUT}"

echo "[phase3-pipeline] done=$(date -u +%FT%TZ)"
