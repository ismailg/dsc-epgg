#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-local}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_zeroaux_manifests_${RUN_DATE}}"
COMM_MANIFEST="${COMM_MANIFEST:-${MANIFEST_ROOT}/manifests/cond1_zeroaux_clean_msgsource_learned_full_history_all_15seeds_25k_50k_100k_150k.txt}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${MANIFEST_ROOT}/manifests/cond2_zeroaux_nocomm_full_history_all_15seeds_25k_50k_100k_150k.txt}"
MILESTONE="${MILESTONE:-150000}"
MAX_WORKERS="${MAX_WORKERS:-4}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-300}"
EVAL_SEED="${EVAL_SEED:-9001}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUT_LABEL_PREFIX="${OUT_LABEL_PREFIX:-phase3_vecstraight_zeroaux}"
RUN_KIND_LABEL="${RUN_KIND//\//_}"
INTERVENTIONS=(none zeros marginal fixed0 fixed1 indep_random public_random sender_shuffle permute_slots)
HISTORYS=(none zero_temporal clamp_temporal_high clamp_temporal_low zero_last_coop zero_last_action zero_ewma clamp_ewma_high)
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/eval/${OUT_LABEL_PREFIX}_message_history_grid_${MILESTONE}_15seeds_${RUN_KIND_LABEL}_${RUN_DATE}}"
SUITE_OUT="${OUT_ROOT}/suite"
REPORT_OUT="${OUT_ROOT}/report"

if [[ ! -f "${COMM_MANIFEST}" || ! -f "${BASELINE_MANIFEST}" ]]; then
  echo "missing zero-aux manifests" >&2
  echo "  COMM_MANIFEST=${COMM_MANIFEST}" >&2
  echo "  BASELINE_MANIFEST=${BASELINE_MANIFEST}" >&2
  echo "  RUN_DATE=${RUN_DATE} ./scripts/run_phase3_vecstraight_prepare_zeroaux_manifests.sh" >&2
  RUN_DATE="${RUN_DATE}" ./scripts/run_phase3_vecstraight_prepare_zeroaux_manifests.sh
fi

if [[ ! -f "${COMM_MANIFEST}" || ! -f "${BASELINE_MANIFEST}" ]]; then
  echo "zero-aux manifests still missing after preparation" >&2
  exit 1
fi

mkdir -p "${SUITE_OUT}" "${REPORT_OUT}"

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

"${PYTHON_BIN}" -m src.analysis.run_phase3_checkpoint_suite \
  --comm_checkpoint_manifest "${COMM_MANIFEST}" \
  --baseline_checkpoint_manifest "${BASELINE_MANIFEST}" \
  --out_dir "${SUITE_OUT}" \
  --comm_condition cond1 \
  --baseline_condition cond2 \
  --seeds 101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616 \
  --milestones "${MILESTONE}" \
  --interventions "${INTERVENTIONS[@]}" \
  --history_interventions "${HISTORYS[@]}" \
  --n_eval_episodes "${N_EVAL_EPISODES}" \
  --eval_seed "${EVAL_SEED}" \
  --max_workers "${MAX_WORKERS}" \
  --skip_semantics \
  $([[ "${SKIP_EXISTING}" == "1" ]] && printf '%s' "--skip_existing")

"${PYTHON_BIN}" -m src.analysis.validate_checkpoint_suite_outputs \
  --manifest "${SUITE_OUT}/checkpoint_suite_manifest.json" \
  --suite_dir "${SUITE_OUT}" \
  --expected-seeds 101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616 \
  --expected-episodes "${MILESTONE}" \
  --expected-interventions none zeros marginal fixed0 fixed1 indep_random public_random sender_shuffle permute_slots

for ablation in "${INTERVENTIONS[@]}"; do
  "${PYTHON_BIN}" -m src.analysis.summarize_phase3_history_audit \
    --suite_main_csv "${SUITE_OUT}/checkpoint_suite_main.csv" \
    --out_dir "${REPORT_OUT}/${ablation}" \
    --checkpoint_episode "${MILESTONE}" \
    --ablation "${ablation}" \
    --source_train_root "${COMM_MANIFEST};${BASELINE_MANIFEST}" \
    --source_eval_root "${OUT_ROOT}" \
    --seed_count 15
done

"${PYTHON_BIN}" -m src.analysis.summarize_phase3_history_audit \
  --suite_main_csv "${SUITE_OUT}/checkpoint_suite_main.csv" \
  --out_dir "${REPORT_OUT}/baseline_none" \
  --checkpoint_episode "${MILESTONE}" \
  --ablation baseline_none \
  --source_train_root "${COMM_MANIFEST};${BASELINE_MANIFEST}" \
  --source_eval_root "${OUT_ROOT}" \
  --seed_count 15

printf '%s\n' \
  "milestone=${MILESTONE}" \
  "comm_manifest=${COMM_MANIFEST}" \
  "baseline_manifest=${BASELINE_MANIFEST}" \
  "interventions=${INTERVENTIONS[*]}" \
  "histories=${HISTORYS[*]}" \
  > "${REPORT_OUT}/message_history_grid_meta.txt"

printf '[zeroaux-message-history-grid] complete run_date=%s run_kind=%s out_root=%s\n' \
  "${RUN_DATE}" "${RUN_KIND}" "${OUT_ROOT}"
