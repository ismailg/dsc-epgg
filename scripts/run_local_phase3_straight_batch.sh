#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

CONDITION="${CONDITION:-cond2}"
RUN_LABEL="${RUN_LABEL:-phase3-150k-${CONDITION}-15seed-trainonly-local-$(date +%Y%m%d-%H%M%S)}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SEEDS=(101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616)

RUN_DIR="${PROJECT_ROOT}/outputs/train/${RUN_LABEL}"
OUT_ROOT="${RUN_DIR}/outputs/phase3_${CONDITION}_15seeds_train_only"
TRAIN_ROOT="${OUT_ROOT}/train"
METRICS_ROOT="${OUT_ROOT}/metrics"
LOGS_ROOT="${OUT_ROOT}/logs"
STATUS_ROOT="${OUT_ROOT}/status"
PROGRESS_LOG="${STATUS_ROOT}/progress.log"
MANIFEST_PATH="${STATUS_ROOT}/manifest.txt"
PIDMAP_PATH="${STATUS_ROOT}/seed_pids.tsv"

mkdir -p "${TRAIN_ROOT}" "${METRICS_ROOT}" "${LOGS_ROOT}" "${STATUS_ROOT}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY="${OBJC_DISABLE_INITIALIZE_FORK_SAFETY:-YES}"

{
  printf 'run_label=%s\n' "${RUN_LABEL}"
  printf 'run_dir=%s\n' "${RUN_DIR}"
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'condition=%s\n' "${CONDITION}"
  printf 'max_parallel=%s\n' "${MAX_PARALLEL}"
  printf 'python_bin=%s\n' "${PYTHON_BIN}"
  printf 'seeds_batch=%s\n' "${SEEDS[*]}"
  printf 'train_root=%s\n' "${TRAIN_ROOT}"
  printf 'metrics_root=%s\n' "${METRICS_ROOT}"
  printf 'logs_root=%s\n' "${LOGS_ROOT}"
} > "${MANIFEST_PATH}"

: > "${PROGRESS_LOG}"
: > "${PIDMAP_PATH}"

log_progress() {
  printf '%s %s\n' "[$(date '+%Y-%m-%d %H:%M:%S')]" "$*" | tee -a "${PROGRESS_LOG}"
}

cond_args=()
case "${CONDITION}" in
  cond1)
    cond_args+=(--comm_enabled --n_senders 4 --vocab_size 2 --msg_dropout 0.1 --disable_comm_fallback)
    ;;
  cond2)
    cond_args+=(--n_senders 0)
    ;;
  *)
    echo "Unsupported CONDITION=${CONDITION}" >&2
    exit 2
    ;;
esac

active_pids=()
all_pids=()

prune_active() {
  local kept=()
  local pid
  for pid in "${active_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kept+=("${pid}")
    fi
  done
  active_pids=("${kept[@]:-}")
}

launch_seed() {
  local seed="$1"
  local log_path="${LOGS_ROOT}/${CONDITION}_seed${seed}.log"
  local save_path="${TRAIN_ROOT}/${CONDITION}_seed${seed}.pt"
  local metrics_path="${METRICS_ROOT}/${CONDITION}_seed${seed}.jsonl"
  local -a cmd=(
    "${PYTHON_BIN}" "src/experiments_pgg_v0/train_ppo.py"
    --n_agents 4
    --T 100
    --n_episodes 150000
    --num_envs 8
    --count_env_episodes
    --env_backend subproc
    --env_start_method spawn
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
    --sign_lambda 0.1
    --list_lambda 0.1
    --seed "${seed}"
    --log_interval 1000
    --save_path "${save_path}"
    --reward_scale 20.0
    --lr_schedule cosine
    --min_lr 1e-05
    --condition_name "${CONDITION}"
    --regime_log_interval 400
    --metrics_jsonl_path "${metrics_path}"
    --checkpoint_interval 25000
  )
  cmd+=("${cond_args[@]}")

  (
    log_progress "[seed start] seed=${seed} log=${log_path}"
    if "${cmd[@]}" >"${log_path}" 2>&1; then
      log_progress "[seed done] seed=${seed}"
    else
      code="$?"
      log_progress "[seed fail] seed=${seed} exit=${code}"
      exit "${code}"
    fi
  ) &
  local pid="$!"
  printf '%s\t%s\n' "${seed}" "${pid}" >> "${PIDMAP_PATH}"
  active_pids+=("${pid}")
  all_pids+=("${pid}")
  log_progress "[launch] seed=${seed} pid=${pid}"
}

log_progress "[batch start] condition=${CONDITION} max_parallel=${MAX_PARALLEL}"

for seed in "${SEEDS[@]}"; do
  while true; do
    prune_active
    if [ "${#active_pids[@]}" -lt "${MAX_PARALLEL}" ]; then
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

if [ "${status}" -eq 0 ]; then
  log_progress "[batch done] status=ok"
else
  log_progress "[batch done] status=failed"
fi

exit "${status}"
