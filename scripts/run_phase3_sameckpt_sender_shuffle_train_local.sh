#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 INIT_EPISODE OUT_DIR [MAX_WORKERS]" >&2
  echo "supported INIT_EPISODE values: 50000 100000" >&2
  exit 2
fi

INIT_EPISODE="$1"
OUT_DIR="$2"
MAX_WORKERS="${3:-2}"

case "${INIT_EPISODE}" in
  50000)
    N_EPISODES=100000
    ;;
  100000)
    N_EPISODES=50000
    ;;
  *)
    echo "unsupported INIT_EPISODE: ${INIT_EPISODE}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEEDS=(101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616)

mkdir -p "${OUT_DIR}"

cd "${REPO_ROOT}"

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

"${PYTHON_BIN}" -m src.experiments_pgg_v0.run_phase3_seed_expansion \
  --out_dir "${OUT_DIR}" \
  --init_checkpoint_dir "outputs/train/phase3_annealed_ext150k_15seeds" \
  --init_episode "${INIT_EPISODE}" \
  --episode_offset "${INIT_EPISODE}" \
  --schedule_total_episodes 150000 \
  --conditions cond1 \
  --seeds "${SEEDS[@]}" \
  --n_episodes "${N_EPISODES}" \
  --T 100 \
  --log_interval 1000 \
  --regime_log_interval 5000 \
  --checkpoint_interval 50000 \
  --msg_training_intervention sender_shuffle \
  --msg_training_history_len 4096 \
  --max_workers "${MAX_WORKERS}" \
  --skip_existing
