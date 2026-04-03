#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
NEXT_ROOT="${REPO_ROOT}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}"
COMM_MANIFEST="${NEXT_ROOT}/manifests/cond1_all_15seeds.txt"
OUT_ROOT="${NEXT_ROOT}/smoke/continuation_sender_shuffle"
TRAIN_OUT="${OUT_ROOT}/train"

mkdir -p "${TRAIN_OUT}"

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

"${PYTHON_BIN}" -m src.experiments_pgg_v0.run_phase3_seed_expansion \
  --out_dir "${TRAIN_OUT}" \
  --init_checkpoint_manifest "${COMM_MANIFEST}" \
  --init_episode 100000 \
  --conditions cond1 \
  --seeds 101 \
  --n_episodes 1 \
  --episode_offset 100000 \
  --schedule_total_episodes 100001 \
  --checkpoint_interval 1 \
  --msg_training_intervention sender_shuffle \
  --max_workers 1

test -f "${TRAIN_OUT}/cond1_seed101.pt"
echo "[continuation-smoke] out_root=${OUT_ROOT}"
