#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-local}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
NEXT_ROOT="${REPO_ROOT}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}"
COMM_MANIFEST="${COMM_MANIFEST:-${NEXT_ROOT}/manifests/cond1_all_15seeds.txt}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${NEXT_ROOT}/manifests/cond2_all_15seeds.txt}"
OUT_LABEL_PREFIX="${OUT_LABEL_PREFIX:-phase3_vecstraight}"
RUN_KIND_LABEL="${RUN_KIND_LABEL:-${RUN_KIND}}"
MILESTONE="${MILESTONE:-150000}"
MAX_WORKERS="${MAX_WORKERS:-4}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-300}"
EVAL_SEED="${EVAL_SEED:-9001}"
SEEDS_STR="${SEEDS_STR:-101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616}"
read -r -a SEEDS <<< "${SEEDS_STR}"
INTERVENTIONS=(none public_random)
OBS_MODES_STR="${OBS_MODES_STR:-private public_noisy public_exact}"
read -r -a OBS_MODES <<< "${OBS_MODES_STR}"
SIGMA_VALUES_STR="${SIGMA_VALUES_STR:-0.0 0.25 0.5 1.0}"
read -r -a SIGMA_VALUES <<< "${SIGMA_VALUES_STR}"

sigma_label_from_value() {
  local value="$1"
  awk -v val="${value}" 'BEGIN { printf("sigma%03d\n", int((val * 100.0) + 0.5)) }'
}

TOTAL_SWEEPS=0
for obs_mode in "${OBS_MODES[@]}"; do
  if [[ "${obs_mode}" == "public_exact" ]]; then
    TOTAL_SWEEPS="$((TOTAL_SWEEPS + 1))"
  else
    TOTAL_SWEEPS="$((TOTAL_SWEEPS + ${#SIGMA_VALUES[@]}))"
  fi
done

if [[ ! -f "${COMM_MANIFEST}" || ! -f "${BASELINE_MANIFEST}" ]]; then
  RUN_DATE="${RUN_DATE}" "${REPO_ROOT}/scripts/run_phase3_vecstraight_setup.sh"
fi

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

run_suite() {
  local obs_mode="$1"
  local sigma_label="$2"
  local sigma_value="$3"
  local progress_idx="$4"
  local out_root="${REPO_ROOT}/outputs/eval/${OUT_LABEL_PREFIX}_noise_sweep_${obs_mode}_${sigma_label}_${MILESTONE}_15seeds_${RUN_KIND_LABEL}_${RUN_DATE}"
  local suite_out="${out_root}/suite"
  local report_out="${out_root}/report"
  local -a suite_args=()
  local eval_sigma_meta=""
  local public_sigma_meta=""

  if [[ "${obs_mode}" == "private" ]]; then
    suite_args+=(--eval_sigmas "${sigma_value}")
    eval_sigma_meta="${sigma_value}"
  elif [[ "${obs_mode}" == "public_noisy" ]]; then
    suite_args+=(--observability_mode public_noisy --public_signal_sigma "${sigma_value}")
    public_sigma_meta="${sigma_value}"
  elif [[ "${obs_mode}" == "public_exact" ]]; then
    suite_args+=(--observability_mode public_exact)
  else
    echo "unsupported obs_mode=${obs_mode}" >&2
    exit 2
  fi

  mkdir -p "${suite_out}" "${report_out}"

  "${PYTHON_BIN}" -m src.analysis.run_phase3_checkpoint_suite \
    --comm_checkpoint_manifest "${COMM_MANIFEST}" \
    --baseline_checkpoint_manifest "${BASELINE_MANIFEST}" \
    --out_dir "${suite_out}" \
    --comm_condition cond1 \
    --baseline_condition cond2 \
    --seeds "${SEEDS[@]}" \
    --milestones "${MILESTONE}" \
    --interventions "${INTERVENTIONS[@]}" \
    --n_eval_episodes "${N_EVAL_EPISODES}" \
    --eval_seed "${EVAL_SEED}" \
    --max_workers "${MAX_WORKERS}" \
    "${suite_args[@]}" \
    --skip_semantics \
    --skip_existing

  "${PYTHON_BIN}" -m src.analysis.validate_checkpoint_suite_outputs \
    --manifest "${suite_out}/checkpoint_suite_manifest.json" \
    --suite_dir "${suite_out}" \
    --expected-seeds "${SEEDS[@]}" \
    --expected-episodes "${MILESTONE}" \
    --expected-interventions none public_random

  "${PYTHON_BIN}" -m src.analysis.summarize_phase3_intervention_suite \
    --suite_main_csv "${suite_out}/checkpoint_suite_main.csv" \
    --out_dir "${report_out}" \
    --checkpoint_episode "${MILESTONE}" \
    --source_train_root "${COMM_MANIFEST};${BASELINE_MANIFEST}" \
    --source_eval_root "${out_root}" \
    --seed_count 15

  printf '%s\n' \
    "observability_mode=${obs_mode}" \
    "sigma_label=${sigma_label}" \
    "eval_sigma=${eval_sigma_meta}" \
    "public_signal_sigma=${public_sigma_meta}" \
    "milestone=${MILESTONE}" > "${report_out}/noise_sweep_meta.txt"
  printf '[noise-sweep] progress current=%d total=%d obs_mode=%s sigma=%s label=%s\n' \
    "${progress_idx}" "${TOTAL_SWEEPS}" "${obs_mode}" "${sigma_value}" "${sigma_label}"
}

progress_idx=0
for obs_mode in "${OBS_MODES[@]}"; do
  if [[ "${obs_mode}" == "public_exact" ]]; then
    progress_idx="$((progress_idx + 1))"
    run_suite public_exact exact public_exact "${progress_idx}"
    continue
  fi
  for sigma in "${SIGMA_VALUES[@]}"; do
    sigma_label="$(sigma_label_from_value "${sigma}")"
    progress_idx="$((progress_idx + 1))"
    run_suite "${obs_mode}" "${sigma_label}" "${sigma}" "${progress_idx}"
  done
done
