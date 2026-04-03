#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_KIND="${RUN_KIND:-local}"

LEARNED_SUITE_CSV="${LEARNED_SUITE_CSV:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_frozen150k_expanded_15seeds_local_20260327/suite/checkpoint_suite_main.csv}"
FIXED0_SUITE_CSV="${FIXED0_SUITE_CSV:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_exogenous_fixed0_15seeds_${RUN_KIND}_${RUN_DATE}/suite/checkpoint_suite_main.csv}"
FIXED1_SUITE_CSV="${FIXED1_SUITE_CSV:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_exogenous_fixed1_15seeds_${RUN_KIND}_${RUN_DATE}/suite/checkpoint_suite_main.csv}"
PUBLIC_SUITE_CSV="${PUBLIC_SUITE_CSV:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_exogenous_public_random_15seeds_${RUN_KIND}_${RUN_DATE}/suite/checkpoint_suite_main.csv}"
UNIFORM_SUITE_CSV="${UNIFORM_SUITE_CSV:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_exogenous_uniform_15seeds_${RUN_KIND}_${RUN_DATE}/suite/checkpoint_suite_main.csv}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_exogenous_channel_controls_${RUN_KIND}_${RUN_DATE}/report}"

"${PYTHON_BIN}" -m src.analysis.summarize_phase3_channel_controls \
  --mode_suite_cond learned "${LEARNED_SUITE_CSV}" cond1 \
  --mode_suite_cond no_comm "${LEARNED_SUITE_CSV}" cond2 \
  --mode_suite_cond fixed0 "${FIXED0_SUITE_CSV}" cond1 \
  --mode_suite_cond fixed1 "${FIXED1_SUITE_CSV}" cond1 \
  --mode_suite_cond public_random "${PUBLIC_SUITE_CSV}" cond1 \
  --mode_suite_cond uniform "${UNIFORM_SUITE_CSV}" cond1 \
  --out_dir "${OUT_DIR}"

printf '[phase3-exogenous-summary] out_dir=%s\n' "${OUT_DIR}"
