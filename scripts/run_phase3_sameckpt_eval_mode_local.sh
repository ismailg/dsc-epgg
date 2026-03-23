#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 MODE OUT_ROOT [MAX_WORKERS]" >&2
  echo "   or: $0 MODE CHECKPOINT_DIR OUT_ROOT [MAX_WORKERS]" >&2
  exit 2
fi

MODE="$1"
if [[ $# -ge 3 && -d "$2" ]]; then
  CHECKPOINT_DIR="$2"
  OUT_ROOT="$3"
  MAX_WORKERS="${4:-4}"
else
  CHECKPOINT_DIR="outputs/train/phase3_sameckpt_continuation_15seeds/${MODE}"
  OUT_ROOT="$2"
  MAX_WORKERS="${3:-4}"
fi

case "${MODE}" in
  fixed0|uniform|public_random|sender_shuffle) ;;
  *)
    echo "unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SEEDS=(101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616)
MODE_OUT="${OUT_ROOT}/${MODE}"
MODE_UPPER="$(printf '%s' "${MODE}" | tr '[:lower:]' '[:upper:]')"

mkdir -p "${MODE_OUT}/suite" "${MODE_OUT}/report"

cd "${REPO_ROOT}"

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

MILESTONES_STR="${MILESTONES:-$("${PYTHON_BIN}" -m src.analysis.checkpoint_artifacts --checkpoint_dir "${CHECKPOINT_DIR}" --condition cond1 --print milestones)}"
if [[ -z "${MILESTONES_STR// }" ]]; then
  echo "could not infer milestones from ${CHECKPOINT_DIR}" >&2
  exit 1
fi
read -r -a MILESTONES_ARR <<< "${MILESTONES_STR}"

"${PYTHON_BIN}" -m src.analysis.run_phase3_checkpoint_suite \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --out_dir "${MODE_OUT}/suite" \
  --comm_condition cond1 \
  --baseline_condition "" \
  --seeds "${SEEDS[@]}" \
  --milestones "${MILESTONES_ARR[@]}" \
  --interventions none zeros fixed0 fixed1 permute_slots \
  --n_eval_episodes 300 \
  --eval_seed 9001 \
  --max_workers "${MAX_WORKERS}" \
  --skip_existing

"${PYTHON_BIN}" -m src.analysis.validate_checkpoint_suite_outputs \
  --manifest "${MODE_OUT}/suite/checkpoint_suite_manifest.json" \
  --suite_dir "${MODE_OUT}/suite" \
  --expected-seeds "${SEEDS[@]}" \
  --expected-episodes "${MILESTONES_ARR[@]}" \
  --expected-interventions none zeros fixed0 fixed1 permute_slots

"${PYTHON_BIN}" -m src.analysis.aggregate_phase3_report \
  --suite_main_csv "${MODE_OUT}/suite/checkpoint_suite_main.csv" \
  --suite_comm_csv "${MODE_OUT}/suite/checkpoint_suite_comm.csv" \
  --suite_trace_csv "${MODE_OUT}/suite/checkpoint_suite_trace.csv" \
  --suite_sender_csv "${MODE_OUT}/suite/checkpoint_suite_sender_semantics.csv" \
  --suite_receiver_csv "${MODE_OUT}/suite/checkpoint_suite_receiver_semantics.csv" \
  --suite_posterior_csv "${MODE_OUT}/suite/checkpoint_suite_posterior_strat.csv" \
  --crossplay_main_csv "" \
  --control_main_csv "" \
  --control_comm_csv "" \
  --out_dir "${MODE_OUT}/report" \
  --out_md "${MODE_OUT}/report/PHASE3_SAMECKPT_${MODE_UPPER}_15SEEDS_LOCAL_20260319.md"
