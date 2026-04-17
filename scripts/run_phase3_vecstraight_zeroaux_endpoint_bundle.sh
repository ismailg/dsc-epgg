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
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${MANIFEST_ROOT}/manifests/cond2_zeroaux_nocomm_full_history_all_15seeds_25k_50k_100k_150k.txt}"
OUT_LABEL_PREFIX="${OUT_LABEL_PREFIX:-phase3_vecstraight_zeroaux}"
MAX_WORKERS="${MAX_WORKERS:-4}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-300}"
EVAL_SEED="${EVAL_SEED:-9001}"

if [[ ! -f "${COMM_MANIFEST}" || ! -f "${BASELINE_MANIFEST}" ]]; then
  echo "missing zero-aux manifests" >&2
  echo "expected:" >&2
  echo "  ${COMM_MANIFEST}" >&2
  echo "  ${BASELINE_MANIFEST}" >&2
  echo "build them with:" >&2
  echo "  RUN_DATE=${RUN_DATE} ./scripts/run_phase3_vecstraight_prepare_zeroaux_manifests.sh" >&2
  exit 2
fi

COMM_MANIFEST="${COMM_MANIFEST}" \
BASELINE_MANIFEST="${BASELINE_MANIFEST}" \
OUT_LABEL_PREFIX="${OUT_LABEL_PREFIX}" \
RUN_KIND_LABEL="${RUN_KIND_LABEL}" \
RUN_DATE="${RUN_DATE}" \
MAX_WORKERS="${MAX_WORKERS}" \
N_EVAL_EPISODES="${N_EVAL_EPISODES}" \
EVAL_SEED="${EVAL_SEED}" \
./scripts/run_phase3_vecstraight_frozen_suite_expanded.sh "${RUN_KIND}"

COMM_MANIFEST="${COMM_MANIFEST}" \
OUT_LABEL_PREFIX="${OUT_LABEL_PREFIX}" \
RUN_KIND_LABEL="${RUN_KIND_LABEL}" \
RUN_DATE="${RUN_DATE}" \
RUN_KIND="${RUN_KIND}" \
MAX_WORKERS="${MAX_WORKERS}" \
N_EVAL_EPISODES="${N_EVAL_EPISODES}" \
EVAL_SEED="${EVAL_SEED}" \
./scripts/run_phase3_vecstraight_sender_causal.sh

FROZEN_150_ROOT="${REPO_ROOT}/outputs/eval/${OUT_LABEL_PREFIX}_frozen150k_expanded_15seeds_${RUN_KIND_LABEL}_${RUN_DATE}"
TRACE_CSV="${FROZEN_150_ROOT}/suite/checkpoint_suite_trace.csv"
LOWDIM_OUT="${FROZEN_150_ROOT}/report/lowdim_mechanism"

if [[ ! -f "${TRACE_CSV}" ]]; then
  echo "missing frozen 150k trace csv: ${TRACE_CSV}" >&2
  exit 2
fi

"${PYTHON_BIN}" -m src.analysis.summarize_phase3_lowdim_mechanism \
  --trace_csv "${TRACE_CSV}" \
  --out_dir "${LOWDIM_OUT}" \
  --condition cond1 \
  --checkpoint_episode 150000 \
  --ablation none \
  --history_intervention none \
  --eval_policy greedy \
  --suite_kind comm

printf '[zeroaux-endpoint] complete run_date=%s run_kind=%s out_label_prefix=%s\n' \
  "${RUN_DATE}" "${RUN_KIND_LABEL}" "${OUT_LABEL_PREFIX}"
