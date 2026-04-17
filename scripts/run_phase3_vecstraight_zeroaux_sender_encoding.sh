#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-local}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_KIND_LABEL="${RUN_KIND_LABEL:-${RUN_KIND}}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_zeroaux_manifests_${RUN_DATE}}"
COMM_MANIFEST="${COMM_MANIFEST:-${MANIFEST_ROOT}/manifests/cond1_zeroaux_clean_msgsource_learned_full_history_all_15seeds_25k_50k_100k_150k.txt}"
OUT_LABEL_PREFIX="${OUT_LABEL_PREFIX:-phase3_vecstraight_zeroaux}"
OUT_ROOT="${REPO_ROOT}/outputs/eval/${OUT_LABEL_PREFIX}_frozen150k_natural_intended_15seeds_${RUN_KIND_LABEL}_${RUN_DATE}"
SUITE_OUT="${OUT_ROOT}/suite"
REPORT_OUT="${OUT_ROOT}/report/sender_encoding_decomposition"
MAX_WORKERS="${MAX_WORKERS:-15}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-300}"
EVAL_SEED="${EVAL_SEED:-9001}"
CONDITION="${CONDITION:-cond1}"
EPISODE="${EPISODE:-150000}"
SEEDS=(101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616)

if [[ ! -f "${COMM_MANIFEST}" ]]; then
  echo "missing zero-aux cond1 manifest: ${COMM_MANIFEST}" >&2
  echo "build it with:" >&2
  echo "  RUN_DATE=${RUN_DATE} ./scripts/run_phase3_vecstraight_prepare_zeroaux_manifests.sh" >&2
  exit 2
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
  --baseline_condition "" \
  --out_dir "${SUITE_OUT}" \
  --comm_condition "${CONDITION}" \
  --seeds "${SEEDS[@]}" \
  --milestones "${EPISODE}" \
  --interventions none \
  --n_eval_episodes "${N_EVAL_EPISODES}" \
  --eval_seed "${EVAL_SEED}" \
  --max_workers "${MAX_WORKERS}" \
  --skip_existing

"${PYTHON_BIN}" -m src.analysis.validate_checkpoint_suite_outputs \
  --manifest "${SUITE_OUT}/checkpoint_suite_manifest.json" \
  --suite_dir "${SUITE_OUT}" \
  --expected-seeds "${SEEDS[@]}" \
  --expected-episodes "${EPISODE}" \
  --expected-interventions none

"${PYTHON_BIN}" -m src.analysis.summarize_phase3_sender_encoding \
  --trace_csv "${SUITE_OUT}/checkpoint_suite_trace.csv" \
  --out_dir "${REPORT_OUT}" \
  --condition "${CONDITION}" \
  --checkpoint_episode "${EPISODE}" \
  --ablation none \
  --history_intervention none \
  --sender_remap none \
  --cross_play none \
  --eval_policy greedy \
  --suite_kind comm

printf '[zeroaux-sender-encoding] complete run_date=%s run_kind=%s out_root=%s\n' \
  "${RUN_DATE}" "${RUN_KIND_LABEL}" "${OUT_ROOT}"
