#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-local}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
NEXT_ROOT="${REPO_ROOT}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}"
COMM_MANIFEST="${NEXT_ROOT}/manifests/cond1_all_15seeds.txt"
BASELINE_MANIFEST="${NEXT_ROOT}/manifests/cond2_all_15seeds.txt"
MAX_WORKERS="${MAX_WORKERS:-4}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-300}"
EVAL_SEED="${EVAL_SEED:-9001}"
SEEDS=(101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616)
INTERVENTIONS=(none zeros marginal fixed0 fixed1 indep_random public_random public_marginal sender_shuffle permute_slots)
MILESTONES=(50000 150000)

if [[ ! -f "${COMM_MANIFEST}" || ! -f "${BASELINE_MANIFEST}" ]]; then
  RUN_DATE="${RUN_DATE}" "${REPO_ROOT}/scripts/run_phase3_vecstraight_setup.sh"
fi

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

for idx in "${!MILESTONES[@]}"; do
  milestone="${MILESTONES[idx]}"
  case "${milestone}" in
    50000) label="frozen50k_expanded" ;;
    150000) label="frozen150k_expanded" ;;
    *) echo "unsupported milestone: ${milestone}" >&2; exit 2 ;;
  esac

  OUT_ROOT="${REPO_ROOT}/outputs/eval/phase3_vecstraight_${label}_15seeds_${RUN_KIND}_${RUN_DATE}"
  SUITE_OUT="${OUT_ROOT}/suite"
  REPORT_OUT="${OUT_ROOT}/report"

  mkdir -p "${SUITE_OUT}" "${REPORT_OUT}"

  "${PYTHON_BIN}" -m src.analysis.run_phase3_checkpoint_suite \
    --comm_checkpoint_manifest "${COMM_MANIFEST}" \
    --baseline_checkpoint_manifest "${BASELINE_MANIFEST}" \
    --out_dir "${SUITE_OUT}" \
    --comm_condition cond1 \
    --baseline_condition cond2 \
    --seeds "${SEEDS[@]}" \
    --milestones "${milestone}" \
    --interventions "${INTERVENTIONS[@]}" \
    --n_eval_episodes "${N_EVAL_EPISODES}" \
    --eval_seed "${EVAL_SEED}" \
    --max_workers "${MAX_WORKERS}" \
    --skip_existing

  "${PYTHON_BIN}" -m src.analysis.validate_checkpoint_suite_outputs \
    --manifest "${SUITE_OUT}/checkpoint_suite_manifest.json" \
    --suite_dir "${SUITE_OUT}" \
    --expected-seeds "${SEEDS[@]}" \
    --expected-episodes "${milestone}" \
    --expected-interventions none zeros marginal fixed0 fixed1 indep_random public_random public_marginal sender_shuffle permute_slots

  "${PYTHON_BIN}" -m src.analysis.summarize_phase3_intervention_suite \
    --suite_main_csv "${SUITE_OUT}/checkpoint_suite_main.csv" \
    --out_dir "${REPORT_OUT}" \
    --checkpoint_episode "${milestone}" \
    --source_train_root "${COMM_MANIFEST};${BASELINE_MANIFEST}" \
    --source_eval_root "${OUT_ROOT}" \
    --seed_count 15

  printf '[frozen-expanded] progress current=%d total=%d milestone=%s\n' "$((idx + 1))" "${#MILESTONES[@]}" "${milestone}"
done
