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
REFERENCE_MAIN_CSV="${REFERENCE_MAIN_CSV:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_zeroaux_frozen150k_expanded_15seeds_local_${RUN_DATE}/suite/checkpoint_suite_main.csv}"
SENDER_SEMANTICS_CSV="${SENDER_SEMANTICS_CSV:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_zeroaux_frozen150k_expanded_15seeds_local_${RUN_DATE}/suite/checkpoint_suite_sender_semantics.csv}"
OUT_LABEL_PREFIX="${OUT_LABEL_PREFIX:-phase3_vecstraight_zeroaux}"
OUT_ROOT="${REPO_ROOT}/outputs/eval/${OUT_LABEL_PREFIX}_cross_seed_transfer_flip15seeds_matched_${RUN_KIND_LABEL}_${RUN_DATE}"
RUN_OUT="${OUT_ROOT}/run"
SUMMARY_OUT="${OUT_ROOT}/summary"
INPUTS_OUT="${OUT_ROOT}/inputs"
INPUT_CHECKPOINTS="${INPUTS_OUT}/checkpoints.txt"
MAX_WORKERS="${MAX_WORKERS:-4}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-300}"
EVAL_SEED="${EVAL_SEED:-9001}"
ALIGNMENT_MODE="${ALIGNMENT_MODE:-flip}"
CONDITION="${CONDITION:-cond1}"
EPISODE="${EPISODE:-150000}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SEEDS=(101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616)

if [[ ! -f "${COMM_MANIFEST}" ]]; then
  echo "missing zero-aux cond1 manifest: ${COMM_MANIFEST}" >&2
  echo "build it with:" >&2
  echo "  RUN_DATE=${RUN_DATE} ./scripts/run_phase3_vecstraight_prepare_zeroaux_manifests.sh" >&2
  exit 2
fi

if [[ ! -f "${REFERENCE_MAIN_CSV}" ]]; then
  echo "missing frozen reference main csv: ${REFERENCE_MAIN_CSV}" >&2
  exit 2
fi

if [[ ! -f "${SENDER_SEMANTICS_CSV}" ]]; then
  echo "missing frozen sender semantics csv: ${SENDER_SEMANTICS_CSV}" >&2
  exit 2
fi

mkdir -p "${RUN_OUT}" "${SUMMARY_OUT}" "${INPUTS_OUT}"

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

COMM_MANIFEST="${COMM_MANIFEST}" \
CONDITION="${CONDITION}" \
EPISODE="${EPISODE}" \
INPUTS_OUT="${INPUTS_OUT}" \
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from src.analysis.checkpoint_artifacts import resolve_manifest_checkpoint_path

manifest = os.environ["COMM_MANIFEST"]
condition = os.environ["CONDITION"]
episode = int(os.environ["EPISODE"])
out_path = Path(os.environ["INPUTS_OUT"]) / "checkpoints.txt"
seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1111, 1212, 1313, 1414, 1515, 1616]
paths = [
    resolve_manifest_checkpoint_path(manifest, condition=condition, seed=seed, episode=episode)
    for seed in seeds
]
out_path.write_text("\n".join(paths) + "\n", encoding="utf-8")
PY

"${PYTHON_BIN}" -m src.analysis.run_phase3_cross_seed_transfer_suite \
  --checkpoint_manifest "${INPUT_CHECKPOINTS}" \
  --condition "${CONDITION}" \
  --episode "${EPISODE}" \
  --out_dir "${RUN_OUT}" \
  --alignment_mode "${ALIGNMENT_MODE}" \
  --n_eval_episodes "${N_EVAL_EPISODES}" \
  --eval_seed "${EVAL_SEED}" \
  --max_workers "${MAX_WORKERS}" \
  $( [[ "${SKIP_EXISTING}" == "1" ]] && printf '%s' "--skip_existing" )

"${PYTHON_BIN}" -m src.analysis.summarize_phase3_cross_seed_transfer \
  --transfer_main_csv "${RUN_OUT}/cross_seed_transfer_main.csv" \
  --reference_main_csv "${REFERENCE_MAIN_CSV}" \
  --sender_semantics_csv "${SENDER_SEMANTICS_CSV}" \
  --out_dir "${SUMMARY_OUT}"

printf '[zeroaux-xseed] complete run_date=%s run_kind=%s out_root=%s\n' \
  "${RUN_DATE}" "${RUN_KIND_LABEL}" "${OUT_ROOT}"
