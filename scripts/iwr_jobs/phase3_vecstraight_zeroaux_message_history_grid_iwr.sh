#!/usr/bin/env bash
set -euo pipefail

: "${IWR_RUN_DIR:?IWR_RUN_DIR must be set by the IWR launcher}"
: "${IWR_PROJECT_DIR:?IWR_PROJECT_DIR must be set by the IWR launcher}"

cd "${IWR_PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

RUN_DATE="${RUN_DATE:-20260417}"
INPUT_ROOT="${INPUT_ROOT:-${IWR_RUN_DIR}/inputs/zeroaux_message_history_grid}"
COMM_CKPT_ROOT="${COMM_CKPT_ROOT:-${INPUT_ROOT}/cond1}"
BASELINE_CKPT_ROOT="${BASELINE_CKPT_ROOT:-${INPUT_ROOT}/cond2}"
MANIFEST_DIR="${MANIFEST_DIR:-${INPUT_ROOT}/manifests}"
COMM_MANIFEST="${COMM_MANIFEST:-${MANIFEST_DIR}/cond1_zeroaux_clean_msgsource_learned_150000.txt}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${MANIFEST_DIR}/cond2_zeroaux_nocomm_full_history_150000.txt}"
OUT_ROOT="${OUT_ROOT:-${IWR_RUN_DIR}/outputs/eval/phase3_vecstraight_zeroaux_message_history_grid_150000_15seeds_iwr_${RUN_DATE}}"
MAX_WORKERS="${MAX_WORKERS:-24}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-300}"
EVAL_SEED="${EVAL_SEED:-9001}"

mkdir -p "${MANIFEST_DIR}" "${OUT_ROOT}"

export COMM_CKPT_ROOT
export BASELINE_CKPT_ROOT
export COMM_MANIFEST
export BASELINE_MANIFEST

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1111, 1212, 1313, 1414, 1515, 1616]
specs = [
    ("cond1", Path(os.environ["COMM_CKPT_ROOT"]), Path(os.environ["COMM_MANIFEST"])),
    ("cond2", Path(os.environ["BASELINE_CKPT_ROOT"]), Path(os.environ["BASELINE_MANIFEST"])),
]

for condition, root, manifest in specs:
    paths = []
    for seed in seeds:
        checkpoint = root / f"{condition}_seed{seed}.pt"
        run_json = checkpoint.with_suffix(".run.json")
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if not run_json.exists():
            raise FileNotFoundError(run_json)
        paths.append(str(checkpoint.resolve()))
    manifest.write_text("\n".join(paths) + "\n", encoding="utf-8")
    print(f"[zeroaux-message-history-grid-iwr] wrote {manifest} rows={len(paths)}")
PY

export PYTHON_BIN
export RUN_DATE
export COMM_MANIFEST
export BASELINE_MANIFEST
export OUT_ROOT
export MAX_WORKERS
export N_EVAL_EPISODES
export EVAL_SEED
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

./scripts/run_phase3_vecstraight_zeroaux_message_history_grid.sh iwr
