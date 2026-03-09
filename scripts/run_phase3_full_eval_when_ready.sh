#!/bin/zsh
set -euo pipefail

REPO_ROOT="/Users/mbp17/POSTDOC/NPS26/dsc-epgg"
PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
export PATH="/Library/Frameworks/Python.framework/Versions/3.10/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
TRAIN_ROOT="outputs/train/phase3_annealed_trimmed"
FULL_EVAL_ROOT="outputs/eval/phase3_annealed_trimmed_all"
FULL_SUITE_OUT="${FULL_EVAL_ROOT}/suite"
FULL_CROSSPLAY_OUT="${FULL_EVAL_ROOT}/crossplay"
FULL_REPORT_OUT="${FULL_EVAL_ROOT}/report"
FULL_RESCUE_OUT="${FULL_EVAL_ROOT}/common_polarity_rescue"
FULL_FRAGMENT_OUT="${FULL_EVAL_ROOT}/fragmentation_figures"

cd "${REPO_ROOT}"

mkdir -p "${FULL_SUITE_OUT}" "${FULL_CROSSPLAY_OUT}" "${FULL_REPORT_OUT}" "${FULL_RESCUE_OUT}" "${FULL_FRAGMENT_OUT}"

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

required=(
  "outputs/train/phase3_annealed_trimmed/cond1_seed101.pt"
  "outputs/train/phase3_annealed_trimmed/cond2_seed101.pt"
  "outputs/train/phase3_annealed_trimmed/cond1_seed101_ep25000.pt"
  "outputs/train/phase3_annealed_trimmed/cond2_seed101_ep25000.pt"
  "outputs/train/phase3_annealed_trimmed/cond1_seed202.pt"
  "outputs/train/phase3_annealed_trimmed/cond2_seed202.pt"
  "outputs/train/phase3_annealed_trimmed/cond1_seed202_ep25000.pt"
  "outputs/train/phase3_annealed_trimmed/cond2_seed202_ep25000.pt"
  "outputs/train/phase3_annealed_trimmed/cond1_seed303.pt"
  "outputs/train/phase3_annealed_trimmed/cond2_seed303.pt"
  "outputs/train/phase3_annealed_trimmed/cond1_seed303_ep25000.pt"
  "outputs/train/phase3_annealed_trimmed/cond2_seed303_ep25000.pt"
  "outputs/train/phase3_annealed_trimmed/cond1_seed404.pt"
  "outputs/train/phase3_annealed_trimmed/cond2_seed404.pt"
  "outputs/train/phase3_annealed_trimmed/cond1_seed404_ep25000.pt"
  "outputs/train/phase3_annealed_trimmed/cond2_seed404_ep25000.pt"
  "outputs/train/phase3_annealed_trimmed/cond1_seed505.pt"
  "outputs/train/phase3_annealed_trimmed/cond2_seed505.pt"
  "outputs/train/phase3_annealed_trimmed/cond1_seed505_ep25000.pt"
  "outputs/train/phase3_annealed_trimmed/cond2_seed505_ep25000.pt"
)

echo "[phase3-full-eval] wait-start=$(/bin/date -u +%FT%TZ)"
while true; do
  missing=()
  for path in "${required[@]}"; do
    if [[ ! -f "${path}" ]]; then
      missing+=("${path}")
    fi
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    break
  fi
  echo "[phase3-full-eval] waiting missing=${#missing[@]} first=${missing[1]}"
  /bin/sleep 300
done

echo "[phase3-full-eval] eval-start=$(/bin/date -u +%FT%TZ)"
"${PYTHON_BIN}" -u -m src.analysis.run_phase3_trimmed_eval \
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

"${PYTHON_BIN}" -u -m src.analysis.aggregate_phase3_report \
  --suite_main_csv "${FULL_SUITE_OUT}/checkpoint_suite_main.csv" \
  --suite_comm_csv "${FULL_SUITE_OUT}/checkpoint_suite_comm.csv" \
  --suite_trace_csv "${FULL_SUITE_OUT}/checkpoint_suite_trace.csv" \
  --suite_sender_csv "${FULL_SUITE_OUT}/checkpoint_suite_sender_semantics.csv" \
  --suite_receiver_csv "${FULL_SUITE_OUT}/checkpoint_suite_receiver_semantics.csv" \
  --suite_posterior_csv "${FULL_SUITE_OUT}/checkpoint_suite_posterior_strat.csv" \
  --crossplay_main_csv "${FULL_CROSSPLAY_OUT}/crossplay_matrix_main.csv" \
  --out_dir "${FULL_REPORT_OUT}" \
  --out_md "${FULL_REPORT_OUT}/PHASE3_ANNEALED_ALL_SEEDS.md"

"${PYTHON_BIN}" -u -m src.analysis.run_phase3_common_polarity_rescue \
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

"${PYTHON_BIN}" -u -m src.analysis.summarize_common_polarity_rescue \
  --base_main_csv "${FULL_SUITE_OUT}/checkpoint_suite_main.csv" \
  --rescue_main_csv "${FULL_RESCUE_OUT}/common_polarity_rescue_main.csv" \
  --out_csv "${FULL_RESCUE_OUT}/common_polarity_rescue_summary.csv"

"${PYTHON_BIN}" -u -m src.analysis.plot_phase3_fragmentation \
  --sender_alignment_csv "${FULL_REPORT_OUT}/sender_alignment_summary.csv" \
  --receiver_summary_csv "${FULL_REPORT_OUT}/receiver_semantics_summary.csv" \
  --receiver_by_sender_csv "${FULL_REPORT_OUT}/receiver_by_sender_summary.csv" \
  --sender_summary_csv "${FULL_REPORT_OUT}/sender_semantics_summary.csv" \
  --out_dir "${FULL_FRAGMENT_OUT}"

echo "[phase3-full-eval] done=$(/bin/date -u +%FT%TZ)"
