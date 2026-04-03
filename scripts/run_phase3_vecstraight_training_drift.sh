#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-local}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_training_drift_15seeds_${RUN_KIND}_${RUN_DATE}}"

if [[ "${RUN_KIND}" == "iwr" ]]; then
  COND1_BATCH_METRICS="${COND1_BATCH_METRICS:-/export/scratch/iguennou/runs/dsc-epgg-vectorized/phase3-150k-cond1-15seed-trainonly-20260323/outputs/phase3_cond1_15seeds_train_only/metrics}"
  COND1_STANDALONE_METRICS="${COND1_STANDALONE_METRICS:-/export/scratch/iguennou/runs/dsc-epgg-vectorized/phase3-150k-straight-c1-s101-subproc-20260323/outputs/phase3_straight_c1_s101_subproc/metrics}"
  COND2_METRICS="${COND2_METRICS:-/export/scratch/iguennou/runs/dsc-epgg-vectorized/phase3-150k-cond2-15seed-trainonly-20260324/code/outputs/train/phase3-150k-cond2-15seed-trainonly-20260324/outputs/phase3_cond2_15seeds_train_only/metrics}"
else
  COND1_BATCH_METRICS="${COND1_BATCH_METRICS:-${REPO_ROOT}/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323/outputs/phase3_cond1_15seeds_train_only/metrics}"
  COND1_STANDALONE_METRICS="${COND1_STANDALONE_METRICS:-${REPO_ROOT}/iwr-results/phase3-150k-straight-c1-s101-subproc-20260323/outputs/phase3_straight_c1_s101_subproc/metrics}"
  COND2_METRICS="${COND2_METRICS:-}"
  if [[ -z "${COND2_METRICS}" ]]; then
    echo "local mode requires COND2_METRICS to be set to a local cond2 metrics directory" >&2
    exit 2
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

"${PYTHON_BIN}" -m src.analysis.plot_phase3_training_drift \
  --metrics_dirs "${COND1_BATCH_METRICS}" "${COND1_STANDALONE_METRICS}" "${COND2_METRICS}" \
  --out_dir "${OUT_ROOT}" \
  --training_family phase3_vecstraight \
  --source_repo dsc-epgg-vectorized \
  --figure_title "Phase-3 training trajectories: communication vs no communication (vecstraight)" \
  --figure_note "Training-window JSONL metrics aggregated from supplied straight-family metrics dirs. Shaded bands show SEM across available seeds; cond1 seed 101 comes from the standalone subproc run."
