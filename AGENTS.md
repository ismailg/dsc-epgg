# AGENTS.md — DSC-EPGG Week 1–2 Implementation

## Mission
Implement the Week 1–2 scope for DSC-EPGG on top of `marl-emecom` with strict scientific constraints.

## Scope (in)
- Stage 0: bootstrap + interface discovery.
- Stage 1: environment correctness + observation wrapper + tests.
- Stage 2: PPO + communication integration + smoke validation.
- Stage 3: session logging + regime identifiability audit.

## Scope (out)
- PLRNN training and all Week 5+ analyses.

## Scientific constraints (must not violate)
1. Do not leak rewards/welfare/true `f` into agent observations.
2. Keep information-set separation:
   - Set A (agent obs): noisy `f_hat`, endowment, lagged social features, messages.
   - Set B (learning): own reward for return/advantage only.
   - Set C (logging): full ground truth including intended/executed actions and flips.
3. Environment remains communication-agnostic; wrapper handles message features/dropout.
4. Preserve pinned legacy environment API expected by upstream codebase.

## Required implementation order
1. Environment fixes first (multi-step `step()`, Sticky-f, tremble, Box obs space, unclamped `f_hat`).
2. Wrapper integration second (history + EWMA + message marginals/dropout + tensor adapter).
3. PPO third (GAE trajectory buffer, clipped objective, value + entropy, joint action+message log-probs).
4. Logging/audit fourth.

## Communication fallback control
- Attempt up to 2 focused debug cycles for joint comm PPO.
- If unstable after 2 cycles, ship no-comm PPO baseline first.
- Re-enable comm in follow-up patch.

## Validation gates
### Gate 1 (before PPO)
- Environment and wrapper unit tests pass.
- Short smoke run passes without NaNs/crashes.

### Gate 2 (before merge)
- PPO losses finite.
- Entropy non-collapsed.
- Intended/executed action logging integrity verified.
- Stage 2 smoke run passes.

## Testing expectations
- Add/maintain tests for env dynamics, tremble rate, payoff correctness, unclamped observations, wrapper dims/EWMA/lag/dropout, and GAE golden case.
- Prefer deterministic seeds and reproducible smoke commands.

## Working style
- Keep commits focused by stage.
- Document any contract mismatches discovered in upstream code.
- Do not change scientific assumptions silently; if change is needed, explain in commit/PR notes.

## Remote Cluster Job Monitoring
- For detached cluster jobs, launch with a simple `status/progress.log` and per-seed log layout so progress checks do not depend on reconstructing shell commands later.
- From Codex, prefer simple read-only SSH probes: `tail` the batch progress log, sample a few seed logs, and count matching `train_ppo` processes. Avoid fragile nested quoting when a smaller probe answers the question.
- If sandboxed SSH/DNS blocks a check, reuse the same probe with escalated host execution rather than rewriting it into a more complicated command.
- Preferred helper for this repo: `python3 -m src.experiments_pgg_v0.check_iwr_progress --host <host> --run-dir <remote_batch_run_dir>`. The helper auto-discovers the batch progress log, samples representative seed logs, and includes any linked standalone run recorded in the batch `status/manifest.txt`.

## Canonical Vectorized Phase-3 Result Paths
- Before reusing any phase-3 result across repos, read [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md). The current manuscript family is `phase3_staged`; the newer straight vectorized family is `phase3_vecstraight`.
- For the required execution order for new-family manipulations, read [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/PHASE3_VECSTRAIGHT_NEXT_STEPS.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/PHASE3_VECSTRAIGHT_NEXT_STEPS.md) before launching new phase-3 reruns.
- Fetched IWR vectorized 15-seed `cond1` checkpoints: [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323)
- Local greedy checkpoint suite: [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324)
- Fetched IWR vectorized 15-seed `cond2` suite: [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325)
- Canonical aggregated outputs:
  - [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324/suite/checkpoint_suite_main.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324/suite/checkpoint_suite_main.csv)
  - [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324/suite/checkpoint_suite_condition.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324/suite/checkpoint_suite_condition.csv)
  - [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324/report/old_vs_new_base_suite.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324/report/old_vs_new_base_suite.md)
- Straight-run `cond2` comparison outputs:
  - [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325/suite/checkpoint_suite_main.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325/suite/checkpoint_suite_main.csv)
  - [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325/suite/checkpoint_suite_condition.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325/suite/checkpoint_suite_condition.csv)
  - [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325/RESULTS_SUMMARY.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325/RESULTS_SUMMARY.md)
  - [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325/report/old_vs_new_base_suite_cond2.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325/report/old_vs_new_base_suite_cond2.md)
- Straight-run communication-gap recheck:
  - [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_comm_gap_15seeds_local_20260325/RESULTS_SUMMARY.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_comm_gap_15seeds_local_20260325/RESULTS_SUMMARY.md)
  - [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_comm_gap_15seeds_local_20260325/exact_f_gap_table.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_comm_gap_15seeds_local_20260325/exact_f_gap_table.csv)
- Use [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324/RESULTS_SUMMARY.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324/RESULTS_SUMMARY.md) as the first stop before updating manuscript-facing claims.
- Current caution:
  - Under the new straight vectorized design, the late communication gap is still positive at `f=3.5` and becomes positive at `f=5.0` by `150k` (`+16.4 pp` and `+18.8 pp` respectively in the recheck summary), so old manuscript-facing phase-3 claims are not portable across training designs.
