#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODE [RUN_KIND]" >&2
  exit 2
fi

MODE="$1"
RUN_KIND="${2:-local}"

case "${MODE}" in
  learned|fixed0|fixed1|public_random|uniform) ;;
  *)
    echo "unsupported MODE=${MODE}; expected one of: learned fixed0 fixed1 public_random uniform" >&2
    exit 2
    ;;
esac

REPO_ROOT="${PROJECT_ROOT:-${IWR_PROJECT_DIR:-${HETZNER_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
NEXT_ROOT="${NEXT_ROOT:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${NEXT_ROOT}/manifests/cond2_all_15seeds.txt}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_clean_msgsource_${MODE}_15seeds_${RUN_KIND}_${RUN_DATE}}"
TRAIN_OUT="${OUT_ROOT}/train"
SUITE_OUT="${OUT_ROOT}/suite"
REPORT_OUT="${OUT_ROOT}/report"
STATUS_ROOT="${OUT_ROOT}/status"
PROGRESS_LOG="${STATUS_ROOT}/progress.log"
MANIFEST_PATH="${STATUS_ROOT}/manifest.txt"
PIDMAP_PATH="${STATUS_ROOT}/seed_pids.tsv"
TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-auto}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-15}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-300}"
EVAL_SEED="${EVAL_SEED:-9001}"
NUM_ENVS="${NUM_ENVS:-8}"
ENV_BACKEND="${ENV_BACKEND:-subproc}"
ENV_START_METHOD="${ENV_START_METHOD:-spawn}"
MSG_DROPOUT="${MSG_DROPOUT:-0.1}"
SEEDS=(101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616)
EVAL_MILESTONES=(50000 150000)
EVAL_INTERVENTIONS=(none)

if [[ ! -f "${BASELINE_MANIFEST}" ]]; then
  RUN_DATE="${RUN_DATE}" "${REPO_ROOT}/scripts/run_phase3_vecstraight_setup.sh"
fi

mkdir -p "${TRAIN_OUT}/metrics" "${TRAIN_OUT}/logs" "${SUITE_OUT}" "${REPORT_OUT}" "${STATUS_ROOT}"

export PYTHONUNBUFFERED=1
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY="${OBJC_DISABLE_INITIALIZE_FORK_SAFETY:-YES}"

detect_default_train_workers() {
  local nproc_val suggested seed_count
  nproc_val="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
  if ! [[ "${nproc_val}" =~ ^[0-9]+$ ]] || (( nproc_val < 1 )); then
    nproc_val=1
  fi
  seed_count="${#SEEDS[@]}"
  suggested=$(( nproc_val / 2 ))
  if (( suggested < 1 )); then
    suggested=1
  fi
  if (( suggested > seed_count )); then
    suggested="${seed_count}"
  fi
  printf '%s\n' "${suggested}"
}

if [[ "${TRAIN_MAX_WORKERS}" == "auto" ]]; then
  TRAIN_MAX_WORKERS="$(detect_default_train_workers)"
fi
if ! [[ "${TRAIN_MAX_WORKERS}" =~ ^[0-9]+$ ]] || (( TRAIN_MAX_WORKERS < 1 )); then
  echo "TRAIN_MAX_WORKERS=${TRAIN_MAX_WORKERS} must resolve to a positive integer" >&2
  exit 2
fi

{
  printf 'mode=%s\n' "${MODE}"
  printf 'run_kind=%s\n' "${RUN_KIND}"
  printf 'run_date=%s\n' "${RUN_DATE}"
  printf 'repo_root=%s\n' "${REPO_ROOT}"
  printf 'python_bin=%s\n' "${PYTHON_BIN}"
  printf 'train_out=%s\n' "${TRAIN_OUT}"
  printf 'suite_out=%s\n' "${SUITE_OUT}"
  printf 'report_out=%s\n' "${REPORT_OUT}"
  printf 'baseline_manifest=%s\n' "${BASELINE_MANIFEST}"
  printf 'train_max_workers=%s\n' "${TRAIN_MAX_WORKERS}"
  printf 'eval_max_workers=%s\n' "${EVAL_MAX_WORKERS}"
  printf 'n_eval_episodes=%s\n' "${N_EVAL_EPISODES}"
  printf 'eval_seed=%s\n' "${EVAL_SEED}"
  printf 'num_envs=%s\n' "${NUM_ENVS}"
  printf 'env_backend=%s\n' "${ENV_BACKEND}"
  printf 'env_start_method=%s\n' "${ENV_START_METHOD}"
  printf 'msg_dropout=%s\n' "${MSG_DROPOUT}"
  printf 'sign_lambda=0.0\n'
  printf 'list_lambda=0.0\n'
  printf 'msg_source_mode=%s\n' "${MODE}"
  printf 'msg_training_intervention=none\n'
  printf 'seeds=%s\n' "${SEEDS[*]}"
} > "${MANIFEST_PATH}"

: > "${PROGRESS_LOG}"
: > "${PIDMAP_PATH}"

log_progress() {
  printf '%s %s\n' "[$(date '+%Y-%m-%d %H:%M:%S')]" "$*" | tee -a "${PROGRESS_LOG}"
}

active_pids=()
all_pids=()

prune_active() {
  local -a kept=()
  local pid
  if (( ${#active_pids[@]} == 0 )); then
    active_pids=()
    return
  fi
  for pid in "${active_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kept+=("${pid}")
    fi
  done
  if (( ${#kept[@]} == 0 )); then
    active_pids=()
  else
    active_pids=("${kept[@]}")
  fi
}

launch_seed() {
  local seed="$1"
  local log_path="${TRAIN_OUT}/logs/cond1_seed${seed}.log"
  local save_path="${TRAIN_OUT}/cond1_seed${seed}.pt"
  local metrics_path="${TRAIN_OUT}/metrics/cond1_seed${seed}.jsonl"
  local -a cmd=(
    "${PYTHON_BIN}" "src/experiments_pgg_v0/train_ppo.py"
    --n_agents 4
    --T 100
    --n_episodes 150000
    --num_envs "${NUM_ENVS}"
    --count_env_episodes
    --env_backend "${ENV_BACKEND}"
    --env_start_method "${ENV_START_METHOD}"
    --endowment 4.0
    --F 0.5 1.5 2.5 3.5 5.0
    --sigmas 0.5 0.5 0.5 0.5
    --rho 0.05
    --epsilon_tremble 0.05
    --episode_offset 0
    --schedule_total_episodes 0
    --hidden_size 64
    --lr 0.0003
    --gamma 0.99
    --lam 0.95
    --clip_ratio 0.2
    --value_coeff 0.5
    --entropy_coeff 0.01
    --entropy_schedule linear
    --entropy_coeff_final 0.001
    --msg_entropy_coeff 0.01
    --msg_entropy_coeff_final 0.0
    --max_grad_norm 0.5
    --ppo_epochs 4
    --mini_batch_size 32
    --sign_lambda 0.0
    --list_lambda 0.0
    --seed "${seed}"
    --log_interval 1000
    --save_path "${save_path}"
    --reward_scale 20.0
    --lr_schedule cosine
    --min_lr 1e-05
    --condition_name cond1
    --regime_log_interval 400
    --metrics_jsonl_path "${metrics_path}"
    --checkpoint_interval 25000
    --comm_enabled
    --n_senders 4
    --vocab_size 2
    --msg_dropout "${MSG_DROPOUT}"
    --disable_comm_fallback
    --msg_source_mode "${MODE}"
  )

  (
    log_progress "[seed start] mode=${MODE} seed=${seed} log=${log_path}"
    if "${cmd[@]}" >"${log_path}" 2>&1; then
      log_progress "[seed done] mode=${MODE} seed=${seed}"
    else
      code="$?"
      log_progress "[seed fail] mode=${MODE} seed=${seed} exit=${code}"
      exit "${code}"
    fi
  ) &
  local pid="$!"
  printf '%s\t%s\n' "${seed}" "${pid}" >> "${PIDMAP_PATH}"
  active_pids+=("${pid}")
  all_pids+=("${pid}")
  log_progress "[launch] mode=${MODE} seed=${seed} pid=${pid}"
}

log_progress "[train batch start] mode=${MODE} max_parallel=${TRAIN_MAX_WORKERS}"

for seed in "${SEEDS[@]}"; do
  while true; do
    prune_active
    if [[ "${#active_pids[@]}" -lt "${TRAIN_MAX_WORKERS}" ]]; then
      break
    fi
    sleep 5
  done
  launch_seed "${seed}"
done

status=0
for pid in "${all_pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  log_progress "[train batch done] mode=${MODE} status=failed"
  exit "${status}"
fi
log_progress "[train batch done] mode=${MODE} status=ok"

log_progress "[suite start] mode=${MODE}"
"${PYTHON_BIN}" -m src.analysis.run_phase3_checkpoint_suite \
  --checkpoint_dir "${TRAIN_OUT}" \
  --baseline_checkpoint_manifest "${BASELINE_MANIFEST}" \
  --out_dir "${SUITE_OUT}" \
  --comm_condition cond1 \
  --baseline_condition cond2 \
  --seeds "${SEEDS[@]}" \
  --milestones "${EVAL_MILESTONES[@]}" \
  --interventions "${EVAL_INTERVENTIONS[@]}" \
  --n_eval_episodes "${N_EVAL_EPISODES}" \
  --eval_seed "${EVAL_SEED}" \
  --max_workers "${EVAL_MAX_WORKERS}" \
  --skip_existing

"${PYTHON_BIN}" -m src.analysis.validate_checkpoint_suite_outputs \
  --manifest "${SUITE_OUT}/checkpoint_suite_manifest.json" \
  --suite_dir "${SUITE_OUT}" \
  --expected-seeds "${SEEDS[@]}" \
  --expected-episodes "${EVAL_MILESTONES[@]}" \
  --expected-interventions "${EVAL_INTERVENTIONS[@]}"

"${PYTHON_BIN}" -m src.analysis.summarize_phase3_intervention_suite \
  --suite_main_csv "${SUITE_OUT}/checkpoint_suite_main.csv" \
  --out_dir "${REPORT_OUT}" \
  --checkpoint_episode 150000 \
  --source_train_root "${TRAIN_OUT}" \
  --source_eval_root "${OUT_ROOT}" \
  --seed_count 15

log_progress "[suite done] mode=${MODE}"
printf '[phase3-clean-msgsource] mode=%s out_root=%s\n' "${MODE}" "${OUT_ROOT}"
