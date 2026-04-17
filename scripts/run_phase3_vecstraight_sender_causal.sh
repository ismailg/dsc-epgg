#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_KIND="${RUN_KIND:-local}"
NEXT_ROOT="${REPO_ROOT}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}"
COMM_MANIFEST="${COMM_MANIFEST:-${NEXT_ROOT}/manifests/cond1_all_15seeds.txt}"
OUT_LABEL_PREFIX="${OUT_LABEL_PREFIX:-phase3_vecstraight}"
RUN_KIND_LABEL="${RUN_KIND_LABEL:-${RUN_KIND}}"
OUT_ROOT="${REPO_ROOT}/outputs/eval/${OUT_LABEL_PREFIX}_sender_causal_150k_15seeds_${RUN_KIND_LABEL}_${RUN_DATE}"
MAX_WORKERS="${MAX_WORKERS:-15}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-300}"
EVAL_SEED="${EVAL_SEED:-9001}"
SEEDS=(101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616)

mkdir -p "${OUT_ROOT}"

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

"${PYTHON_BIN}" -m src.analysis.run_phase3_sender_causal_suite \
  --checkpoint_manifest "${COMM_MANIFEST}" \
  --out_dir "${OUT_ROOT}" \
  --condition cond1 \
  --seeds "${SEEDS[@]}" \
  --milestones 150000 \
  --n_eval_episodes "${N_EVAL_EPISODES}" \
  --eval_seed "${EVAL_SEED}" \
  --max_workers "${MAX_WORKERS}" \
  --skip_existing

"${PYTHON_BIN}" -m src.analysis.validate_sender_causal_outputs \
  --manifest "${OUT_ROOT}/sender_causal_manifest.json" \
  --suite_dir "${OUT_ROOT}" \
  --expected-seeds "${SEEDS[@]}" \
  --expected-episodes 150000

"${PYTHON_BIN}" -m src.analysis.summarize_phase3_sender_causal \
  --sender_causal_csv "${OUT_ROOT}/sender_causal_matrix.csv" \
  --out_dir "${OUT_ROOT}/report" \
  --condition cond1 \
  --ablation none \
  --checkpoint_episode 150000 \
  --cross_play none \
  --sender_remap none \
  --source_train_root "${COMM_MANIFEST}" \
  --source_eval_root "${OUT_ROOT}" \
  --seed_count 15
