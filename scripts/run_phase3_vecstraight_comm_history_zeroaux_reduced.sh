#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-local}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
if [[ -n "${IWR_RUN_DIR:-}" ]]; then
  OUTPUT_BASE="${IWR_RUN_DIR}/outputs/train"
else
  OUTPUT_BASE="${REPO_ROOT}/outputs/train"
fi

export SIGN_LAMBDA="${SIGN_LAMBDA:-0.0}"
export LIST_LAMBDA="${LIST_LAMBDA:-0.0}"
export OUT_ROOT="${OUT_ROOT:-${OUTPUT_BASE}/phase3_vecstraight_comm_history_factorial_with_comm_reduced_history_zeroaux_15seeds_${RUN_KIND}_${RUN_DATE}}"

exec ./scripts/run_phase3_vecstraight_comm_history_factorial.sh with_comm_reduced_history "${RUN_KIND}"
