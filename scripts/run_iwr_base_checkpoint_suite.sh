#!/usr/bin/env bash
set -euo pipefail

cd "${IWR_PROJECT_DIR:-$(pwd)}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PY="${PY:-python3}"
RUN_DIR="${IWR_RUN_DIR:-$(pwd)}"
OUT_ROOT="${OUT_ROOT:-${RUN_DIR}/outputs/base_checkpoint_suite}"
SUITE_OUT="${SUITE_OUT:-${OUT_ROOT}/suite}"
MANIFEST="${MANIFEST:-${OUT_ROOT}/checkpoints.txt}"
MAX_WORKERS="${MAX_WORKERS:-2}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-300}"
EVAL_SEED="${EVAL_SEED:-9001}"
CONDITION="${CONDITION:-cond1}"
USE_STANDALONE_SEED101="${USE_STANDALONE_SEED101:-1}"
SEEDS="${SEEDS:-101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616}"
EPISODES="${EPISODES:-25000 50000 100000 150000}"

export STANDALONE_ROOT="${STANDALONE_ROOT:-/export/scratch/iguennou/runs/dsc-epgg-vectorized/phase3-150k-straight-c1-s101-subproc-20260323/outputs/phase3_straight_c1_s101_subproc/checkpoints}"
export BATCH_ROOT="${BATCH_ROOT:-/export/scratch/iguennou/runs/dsc-epgg-vectorized/phase3-150k-cond1-15seed-trainonly-20260323/outputs/phase3_cond1_15seeds_train_only/train}"
export CONDITION
export USE_STANDALONE_SEED101
export SEEDS
export EPISODES

mkdir -p "${OUT_ROOT}"

"${PY}" - <<'PYEOF' > "${MANIFEST}"
import os
from pathlib import Path

standalone_root = Path(os.environ.get("STANDALONE_ROOT", ""))
batch_root = Path(os.environ["BATCH_ROOT"])
condition = str(os.environ.get("CONDITION", "cond1"))
use_standalone_seed101 = str(os.environ.get("USE_STANDALONE_SEED101", "1")).strip() not in {"0", "false", "False"}
seeds = [int(x) for x in str(os.environ.get("SEEDS", "")).split() if str(x).strip()]
episodes = [int(x) for x in str(os.environ.get("EPISODES", "")).split() if str(x).strip()]

paths = []
for seed in seeds:
    root = standalone_root if (use_standalone_seed101 and seed == 101 and str(standalone_root)) else batch_root
    for ep in episodes:
        path = root / f"{condition}_seed{seed}.pt" if ep == 150000 else root / f"{condition}_seed{seed}_ep{ep}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(str(path))

print("\n".join(paths))
PYEOF

"${PY}" -m src.analysis.run_base_checkpoint_suite \
  --checkpoint_manifest "${MANIFEST}" \
  --out_dir "${SUITE_OUT}" \
  --n_eval_episodes "${N_EVAL_EPISODES}" \
  --eval_seed "${EVAL_SEED}" \
  --max_workers "${MAX_WORKERS}"
