#!/bin/zsh
set -euo pipefail

REPO_ROOT="/Users/mbp17/POSTDOC/NPS26/dsc-epgg"
TRAIN_OUT="outputs/train/phase3_annealed_seed101"
EVAL_ROOT="outputs/eval/phase3_annealed_seed101"
SUITE_OUT="${EVAL_ROOT}/suite"
CROSSPLAY_OUT="${EVAL_ROOT}/crossplay"
REPORT_OUT="${EVAL_ROOT}/report"

cd "${REPO_ROOT}"

mkdir -p "${TRAIN_OUT}" "${EVAL_ROOT}" "${SUITE_OUT}" "${CROSSPLAY_OUT}" "${REPORT_OUT}"

python3 -m src.experiments_pgg_v0.run_phase3_seed_expansion \
  --out_dir "${TRAIN_OUT}" \
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

python3 -m src.analysis.run_phase3_trimmed_eval \
  --checkpoint_dir "${TRAIN_OUT}" \
  --suite_out_dir "${SUITE_OUT}" \
  --crossplay_out_dir "${CROSSPLAY_OUT}" \
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

python3 -m src.analysis.aggregate_phase3_report \
  --suite_main_csv "${SUITE_OUT}/checkpoint_suite_main.csv" \
  --suite_comm_csv "${SUITE_OUT}/checkpoint_suite_comm.csv" \
  --suite_trace_csv "${SUITE_OUT}/checkpoint_suite_trace.csv" \
  --suite_sender_csv "${SUITE_OUT}/checkpoint_suite_sender_semantics.csv" \
  --suite_receiver_csv "${SUITE_OUT}/checkpoint_suite_receiver_semantics.csv" \
  --suite_posterior_csv "${SUITE_OUT}/checkpoint_suite_posterior_strat.csv" \
  --crossplay_main_csv "${CROSSPLAY_OUT}/crossplay_matrix_main.csv" \
  --out_dir "${REPORT_OUT}" \
  --out_md "${REPORT_OUT}/PHASE3_ANNEALED_SEED101.md"
