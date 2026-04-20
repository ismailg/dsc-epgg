#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-local}"
START_EPOCH="$(date +%s)"
START_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

REPO_ROOT="${PROJECT_ROOT:-${IWR_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_KIND_LABEL="${RUN_KIND_LABEL:-${RUN_KIND}}"
SIBLING_HETZNER_ROOT="$(cd "${REPO_ROOT}/.." && pwd)/dsc-epgg-vectorized/hetzner-results"
EXISTING_ROOT_BASE="${EXISTING_ROOT_BASE:-${SIBLING_HETZNER_ROOT}}"
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/outputs/eval}"
FAMILY_OUT="${FAMILY_OUT:-${OUT_BASE}/phase3_vecstraight_zeroaux_crossover_family_suites_150000_15seeds_${RUN_KIND_LABEL}_${RUN_DATE}}"
NOTES_PATH="${NOTES_PATH:-${OUT_BASE}/${RUN_DATE}_crossover_notes.md}"
MAX_WORKERS="${MAX_WORKERS:-8}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-300}"
SAMPLE_EVAL_EPISODES="${SAMPLE_EVAL_EPISODES:-1}"
EVAL_SEED="${EVAL_SEED:-9001}"
SEEDS=(101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616)
INTERVENTIONS=(none zeros indep_random public_random fixed0 fixed1)
TRAIN_MODES=(learned uniform public_random fixed0)

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

root_for_train_mode() {
  local train_mode="$1"
  case "${train_mode}" in
    learned)
      printf '%s\n' "${LEARNED_ROOT:-${EXISTING_ROOT_BASE}/phase3_vecstraight_clean_msgsource_learned_15seeds_hetzner_20260410}"
      ;;
    uniform)
      printf '%s\n' "${UNIFORM_ROOT:-${EXISTING_ROOT_BASE}/phase3_vecstraight_clean_msgsource_uniform_15seeds_hetzner_20260410}"
      ;;
    public_random)
      printf '%s\n' "${PUBLIC_RANDOM_ROOT:-${EXISTING_ROOT_BASE}/phase3_vecstraight_clean_msgsource_public_random_15seeds_hetzner_20260410}"
      ;;
    fixed0)
      printf '%s\n' "${FIXED0_ROOT:-${EXISTING_ROOT_BASE}/phase3_vecstraight_clean_msgsource_fixed0_15seeds_hetzner_20260410}"
      ;;
    *)
      echo "unsupported train_mode=${train_mode}" >&2
      exit 2
      ;;
  esac
}

test_mode_to_intervention() {
  local test_mode="$1"
  case "${test_mode}" in
    natural) printf '%s\n' "none" ;;
    zeros) printf '%s\n' "zeros" ;;
    indep_random) printf '%s\n' "indep_random" ;;
    public_random) printf '%s\n' "public_random" ;;
    fixed0) printf '%s\n' "fixed0" ;;
    fixed1) printf '%s\n' "fixed1" ;;
    *)
      echo "unsupported test_mode=${test_mode}" >&2
      exit 2
      ;;
  esac
}

require_train_root() {
  local train_mode="$1"
  local root="$2"
  if [[ ! -d "${root}/train" ]]; then
    echo "missing train=${train_mode} root: ${root}/train" >&2
    exit 2
  fi
  local seed
  for seed in "${SEEDS[@]}"; do
    if [[ ! -f "${root}/train/cond1_seed${seed}.pt" ]]; then
      echo "missing train=${train_mode} final checkpoint seed=${seed}: ${root}/train/cond1_seed${seed}.pt" >&2
      exit 2
    fi
  done
}

run_family_suite() {
  local train_mode="$1"
  local root="$2"
  local suite_out="${FAMILY_OUT}/train_${train_mode}"
  mkdir -p "${suite_out}"
  echo "[crossover suite start] train=${train_mode} root=${root}"
  "${PYTHON_BIN}" -m src.analysis.run_phase3_checkpoint_suite \
    --checkpoint_dir "${root}/train" \
    --out_dir "${suite_out}" \
    --comm_condition cond1 \
    --baseline_condition "" \
    --seeds "${SEEDS[@]}" \
    --milestones 150000 \
    --interventions "${INTERVENTIONS[@]}" \
    --n_eval_episodes "${N_EVAL_EPISODES}" \
    --eval_seed "${EVAL_SEED}" \
    --max_workers "${MAX_WORKERS}" \
    --skip_semantics \
    --skip_existing
  "${PYTHON_BIN}" -m src.analysis.validate_checkpoint_suite_outputs \
    --manifest "${suite_out}/checkpoint_suite_manifest.json" \
    --suite_dir "${suite_out}" \
    --expected-seeds "${SEEDS[@]}" \
    --expected-episodes 150000 \
    --expected-interventions "${INTERVENTIONS[@]}"
  echo "[crossover suite done] train=${train_mode}"
}

sample_cell_messages() {
  local train_mode="$1"
  local root="$2"
  local test_mode="$3"
  local intervention
  intervention="$(test_mode_to_intervention "${test_mode}")"
  local report_dir="${OUT_BASE}/phase3_vecstraight_zeroaux_crossover_train_${train_mode}_test_${test_mode}_15seeds_${RUN_KIND_LABEL}_${RUN_DATE}/report"
  local sample_dir="${report_dir}/message_stream_sample_raw"
  mkdir -p "${sample_dir}"
  if [[ -s "${report_dir}/message_stream_sample.csv" ]]; then
    return
  fi
  "${PYTHON_BIN}" -m src.analysis.evaluate_regime_conditional \
    --checkpoints_glob "${root}/train/cond1_seed101.pt" \
    --n_eval_episodes "${SAMPLE_EVAL_EPISODES}" \
    --eval_seed "${EVAL_SEED}" \
    --msg_intervention "${intervention}" \
    --greedy \
    --out_csv "${sample_dir}/main.csv" \
    --out_comm_csv "${sample_dir}/comm.csv" \
    --out_condition_csv "${sample_dir}/condition.csv" \
    --out_trace_csv "${report_dir}/message_stream_sample.csv"
}

mkdir -p "${FAMILY_OUT}" "${OUT_BASE}"

for train_mode in "${TRAIN_MODES[@]}"; do
  root="$(root_for_train_mode "${train_mode}")"
  require_train_root "${train_mode}" "${root}"
  run_family_suite "${train_mode}" "${root}"
done

"${PYTHON_BIN}" -m src.analysis.write_phase3_crossover_cell_reports \
  --train_suite learned "${FAMILY_OUT}/train_learned/checkpoint_suite_main.csv" \
  --train_suite uniform "${FAMILY_OUT}/train_uniform/checkpoint_suite_main.csv" \
  --train_suite public_random "${FAMILY_OUT}/train_public_random/checkpoint_suite_main.csv" \
  --train_suite fixed0 "${FAMILY_OUT}/train_fixed0/checkpoint_suite_main.csv" \
  --out_base "${OUT_BASE}" \
  --run_kind "${RUN_KIND_LABEL}" \
  --run_date "${RUN_DATE}" \
  --checkpoint_episode 150000

for train_mode in "${TRAIN_MODES[@]}"; do
  root="$(root_for_train_mode "${train_mode}")"
  for test_mode in natural zeros indep_random public_random fixed0 fixed1; do
    sample_cell_messages "${train_mode}" "${root}" "${test_mode}"
  done
done

END_EPOCH="$(date +%s)"
END_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
ELAPSED_SECONDS="$((END_EPOCH - START_EPOCH))"

cat > "${NOTES_PATH}" <<EOF
Experiment A crossover evaluation completed on ${RUN_DATE} using frozen 150k zero-aux clean_msgsource checkpoints for train modes learned, uniform, public_random, and fixed0; no training was launched by this runner. Seeds were ${SEEDS[*]}, checkpoint roots were read from ${EXISTING_ROOT_BASE}, test interventions were natural/none, zeros, indep_random, public_random, fixed0, and fixed1, with n_eval_episodes=${N_EVAL_EPISODES}, eval_seed=${EVAL_SEED}, and greedy evaluation. Outputs were written as one report directory per train/test cell under ${OUT_BASE}; message_stream_sample.csv in each report records the actual receiver observation message block for seed 101 over ${SAMPLE_EVAL_EPISODES} sample episode(s). The run started at ${START_UTC}, ended at ${END_UTC}, and took ${ELAPSED_SECONDS} seconds wall-clock; no failed seeds, OOM restarts, or other anomalies were detected by the runner, but inspect ${FAMILY_OUT} logs if a cell is missing or validation failed before this note was written.
EOF

echo "[crossover done] family_out=${FAMILY_OUT}"
echo "[crossover done] notes=${NOTES_PATH}"
