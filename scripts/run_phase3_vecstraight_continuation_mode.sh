#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 BRANCH_EP MODE [RUN_KIND]" >&2
  exit 2
fi

BRANCH_EP="$1"
MODE="$2"
RUN_KIND="${3:-local}"

case "${BRANCH_EP}" in
  50000|100000) ;;
  *)
    echo "unsupported branch episode: ${BRANCH_EP}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
NEXT_ROOT="${NEXT_ROOT:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}}"
COMM_MANIFEST="${COMM_MANIFEST:-${NEXT_ROOT}/manifests/cond1_all_15seeds.txt}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${NEXT_ROOT}/manifests/cond2_all_15seeds.txt}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_sameckpt_continuation_${BRANCH_EP}_${MODE}_15seeds_${RUN_KIND}_${RUN_DATE}}"
TRAIN_OUT="${OUT_ROOT}/train"
SUITE_OUT="${OUT_ROOT}/suite"
REPORT_OUT="${OUT_ROOT}/report"
TOTAL_EPISODES="${TOTAL_EPISODES:-150000}"
CONT_N_EPISODES="$((TOTAL_EPISODES - BRANCH_EP))"
TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-10}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-15}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-300}"
EVAL_SEED="${EVAL_SEED:-9001}"
SEEDS=(101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616)
INTERVENTIONS=(none zeros fixed0 fixed1 indep_random public_random sender_shuffle permute_slots)
EVAL_MILESTONES=(150000)
if [[ "${BRANCH_EP}" == "50000" ]]; then
  EVAL_MILESTONES=(100000 150000)
fi

mkdir -p "${TRAIN_OUT}" "${SUITE_OUT}" "${REPORT_OUT}"

# Preserve the base vecstraight training regime for continuation runs.
NUM_ENVS="${NUM_ENVS:-8}"
ENV_BACKEND="${ENV_BACKEND:-subproc}"
ENV_START_METHOD="${ENV_START_METHOD:-spawn}"
REGIME_LOG_INTERVAL="${REGIME_LOG_INTERVAL:-400}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25000}"
MSG_ENTROPY_COEFF="${MSG_ENTROPY_COEFF:-0.01}"
MSG_ENTROPY_COEFF_FINAL="${MSG_ENTROPY_COEFF_FINAL:-0.0}"
if [[ "${MODE}" == "none" ]]; then
  DEFAULT_SIGN_LAMBDA="0.1"
  DEFAULT_LIST_LAMBDA="0.1"
else
  DEFAULT_SIGN_LAMBDA="0.0"
  DEFAULT_LIST_LAMBDA="0.0"
fi
SIGN_LAMBDA="${SIGN_LAMBDA:-${DEFAULT_SIGN_LAMBDA}}"
LIST_LAMBDA="${LIST_LAMBDA:-${DEFAULT_LIST_LAMBDA}}"

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
  --init_episode "${BRANCH_EP}" \
  --conditions cond1 \
  --seeds "${SEEDS[@]}" \
  --n_episodes "${CONT_N_EPISODES}" \
  --num_envs "${NUM_ENVS}" \
  --count_env_episodes \
  --env_backend "${ENV_BACKEND}" \
  --env_start_method "${ENV_START_METHOD}" \
  --episode_offset "${BRANCH_EP}" \
  --schedule_total_episodes "${TOTAL_EPISODES}" \
  --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
  --regime_log_interval "${REGIME_LOG_INTERVAL}" \
  --msg_entropy_coeff "${MSG_ENTROPY_COEFF}" \
  --msg_entropy_coeff_final "${MSG_ENTROPY_COEFF_FINAL}" \
  --sign_lambda "${SIGN_LAMBDA}" \
  --list_lambda "${LIST_LAMBDA}" \
  --disable_comm_fallback \
  --msg_training_intervention "${MODE}" \
  --max_workers "${TRAIN_MAX_WORKERS}" \
  --skip_existing

"${PYTHON_BIN}" -m src.analysis.run_phase3_checkpoint_suite \
  --checkpoint_dir "${TRAIN_OUT}" \
  --baseline_checkpoint_manifest "${BASELINE_MANIFEST}" \
  --out_dir "${SUITE_OUT}" \
  --comm_condition cond1 \
  --baseline_condition cond2 \
  --seeds "${SEEDS[@]}" \
  --milestones "${EVAL_MILESTONES[@]}" \
  --interventions "${INTERVENTIONS[@]}" \
  --n_eval_episodes "${N_EVAL_EPISODES}" \
  --eval_seed "${EVAL_SEED}" \
  --max_workers "${EVAL_MAX_WORKERS}" \
  --skip_existing

"${PYTHON_BIN}" -m src.analysis.validate_checkpoint_suite_outputs \
  --manifest "${SUITE_OUT}/checkpoint_suite_manifest.json" \
  --suite_dir "${SUITE_OUT}" \
  --expected-seeds "${SEEDS[@]}" \
  --expected-episodes "${EVAL_MILESTONES[@]}" \
  --expected-interventions "${INTERVENTIONS[@]}"

"${PYTHON_BIN}" -m src.analysis.summarize_phase3_intervention_suite \
  --suite_main_csv "${SUITE_OUT}/checkpoint_suite_main.csv" \
  --out_dir "${REPORT_OUT}" \
  --checkpoint_episode 150000 \
  --source_train_root "${COMM_MANIFEST};${TRAIN_OUT}" \
  --source_eval_root "${OUT_ROOT}" \
  --seed_count 15
