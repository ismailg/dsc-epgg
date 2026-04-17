#!/usr/bin/env bash
set -euo pipefail

TARGET_CELL="${1:-all}"
RUN_KIND="${2:-local}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-auto}"
SIGN_LAMBDA="${SIGN_LAMBDA:-0.1}"
LIST_LAMBDA="${LIST_LAMBDA:-0.1}"
SEEDS_STR="${SEEDS_STR:-101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616}"
read -r -a SEEDS <<< "${SEEDS_STR}"
N_EPISODES="${N_EPISODES:-150000}"
T_STEPS="${T_STEPS:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
REGIME_LOG_INTERVAL="${REGIME_LOG_INTERVAL:-400}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25000}"
CELLS=(
  with_comm_full_history
  with_comm_reduced_history
  without_comm_full_history
  without_comm_reduced_history
)

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

for value_name in N_EPISODES LOG_INTERVAL REGIME_LOG_INTERVAL CHECKPOINT_INTERVAL; do
  value="${!value_name}"
  if (( value % 8 != 0 )); then
    echo "${value_name}=${value} must be divisible by 8 to preserve the base vectorized count_env_episodes contract" >&2
    exit 2
  fi
done

run_cell() {
  local cell="$1"
  local condition history_mode comm_flag
  local -a cond_args=()

  case "${cell}" in
    with_comm_full_history)
      condition="cond1"
      history_mode="full"
      comm_flag="with_comm"
      cond_args+=(--comm_enabled --n_senders 4 --vocab_size 2 --msg_dropout 0.1 --disable_comm_fallback)
      ;;
    with_comm_reduced_history)
      condition="cond1"
      history_mode="reduced"
      comm_flag="with_comm"
      cond_args+=(--comm_enabled --n_senders 4 --vocab_size 2 --msg_dropout 0.1 --disable_comm_fallback)
      ;;
    without_comm_full_history)
      condition="cond2"
      history_mode="full"
      comm_flag="without_comm"
      cond_args+=(--n_senders 0)
      ;;
    without_comm_reduced_history)
      condition="cond2"
      history_mode="reduced"
      comm_flag="without_comm"
      cond_args+=(--n_senders 0)
      ;;
    *)
      echo "unsupported cell=${cell}" >&2
      exit 2
      ;;
  esac

  local out_root="${OUT_ROOT:-${REPO_ROOT}/outputs/train/phase3_vecstraight_comm_history_factorial_${cell}_15seeds_${RUN_KIND}_${RUN_DATE}}"
  local train_out="${out_root}/train"
  local metrics_root="${train_out}/metrics"
  local logs_root="${train_out}/logs"
  local status_root="${out_root}/status"
  local progress_log="${status_root}/progress.log"
  local manifest_path="${status_root}/manifest.txt"
  local pidmap_path="${status_root}/seed_pids.tsv"

  mkdir -p "${metrics_root}" "${logs_root}" "${status_root}"

  {
    printf 'cell=%s\n' "${cell}"
    printf 'comm_mode=%s\n' "${comm_flag}"
    printf 'history_mode=%s\n' "${history_mode}"
    printf 'condition=%s\n' "${condition}"
    printf 'run_kind=%s\n' "${RUN_KIND}"
    printf 'run_date=%s\n' "${RUN_DATE}"
    printf 'repo_root=%s\n' "${REPO_ROOT}"
    printf 'python_bin=%s\n' "${PYTHON_BIN}"
    printf 'out_root=%s\n' "${out_root}"
    printf 'train_out=%s\n' "${train_out}"
    printf 'train_max_workers=%s\n' "${TRAIN_MAX_WORKERS}"
    printf 'n_episodes=%s\n' "${N_EPISODES}"
    printf 'T_steps=%s\n' "${T_STEPS}"
    printf 'log_interval=%s\n' "${LOG_INTERVAL}"
    printf 'regime_log_interval=%s\n' "${REGIME_LOG_INTERVAL}"
    printf 'checkpoint_interval=%s\n' "${CHECKPOINT_INTERVAL}"
    printf 'sign_lambda=%s\n' "${SIGN_LAMBDA}"
    printf 'list_lambda=%s\n' "${LIST_LAMBDA}"
    printf 'seeds=%s\n' "${SEEDS[*]}"
  } > "${manifest_path}"

  : > "${progress_log}"
  : > "${pidmap_path}"

  log_progress() {
    printf '%s %s\n' "[$(date '+%Y-%m-%d %H:%M:%S')]" "$*" | tee -a "${progress_log}"
  }

  local -a active_pids=()
  local -a all_pids=()

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
    local log_path="${logs_root}/${condition}_seed${seed}.log"
    local save_path="${train_out}/${condition}_seed${seed}.pt"
    local metrics_path="${metrics_root}/${condition}_seed${seed}.jsonl"
    local -a cmd=(
      "${PYTHON_BIN}" "src/experiments_pgg_v0/train_ppo.py"
      --n_agents 4
      --T "${T_STEPS}"
      --n_episodes "${N_EPISODES}"
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
      --sign_lambda "${SIGN_LAMBDA}"
      --list_lambda "${LIST_LAMBDA}"
      --history_mode "${history_mode}"
      --seed "${seed}"
      --log_interval "${LOG_INTERVAL}"
      --save_path "${save_path}"
      --reward_scale 20.0
      --lr_schedule cosine
      --min_lr 1e-05
      --condition_name "${condition}"
      --regime_log_interval "${REGIME_LOG_INTERVAL}"
      --metrics_jsonl_path "${metrics_path}"
      --checkpoint_interval "${CHECKPOINT_INTERVAL}"
    )
    cmd+=("${cond_args[@]}")

    (
      log_progress "[seed start] cell=${cell} seed=${seed} log=${log_path}"
      if "${cmd[@]}" >"${log_path}" 2>&1; then
        log_progress "[seed done] cell=${cell} seed=${seed}"
      else
        code="$?"
        log_progress "[seed fail] cell=${cell} seed=${seed} exit=${code}"
        exit "${code}"
      fi
    ) &
    local pid="$!"
    printf '%s\t%s\n' "${seed}" "${pid}" >> "${pidmap_path}"
    active_pids+=("${pid}")
    all_pids+=("${pid}")
    log_progress "[launch] cell=${cell} seed=${seed} pid=${pid}"
  }

  log_progress "[train batch start] cell=${cell} condition=${condition} history_mode=${history_mode} max_parallel=${TRAIN_MAX_WORKERS}"

  local seed
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

  local status=0
  local pid
  for pid in "${all_pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    log_progress "[train batch done] cell=${cell} status=failed"
    exit "${status}"
  fi

  log_progress "[train batch done] cell=${cell} status=ok"
  printf '[phase3-comm-history-factorial] cell=%s out_root=%s\n' "${cell}" "${out_root}"
}

if [[ "${TARGET_CELL}" == "all" ]]; then
  for cell in "${CELLS[@]}"; do
    run_cell "${cell}"
  done
else
  run_cell "${TARGET_CELL}"
fi
