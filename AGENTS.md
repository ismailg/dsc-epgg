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

## Manuscript rule
- For paper work in `paper/neurips2026_comm_vecstraight`, manuscript-facing result numbers must be generated dynamically from Quarto/Python chunks that read the authoritative analysis artifacts. Do not hand-type or manually maintain quantitative results in prose, tables, or captions when they can be sourced from CSV/report outputs.
- If a manuscript number changes, update the generating Quarto code or its source artifact path, not just the rendered text.
- Keep manuscript percentage and percentage-point reporting rounded consistently to one decimal place unless the section explicitly requires a different precision.
- When multiple result families exist, make the source family explicit in the Quarto code and keep all related manuscript references aligned to that same artifact family.

## Hetzner CPU Bootstrap
- For `dsc-epgg-vectorized` on the Hetzner CCX CPU server, use `Python 3.10.x`.
- The current `requirements_locked.txt` effectively requires `>=3.10` while `PettingZoo==1.18.1` still requires `<3.11`, so `3.10` is the compatible runtime band.
- Preferred bootstrap on that server: [`scripts/setup_hetzner_cpu.sh`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/scripts/setup_hetzner_cpu.sh).
- If you use [`codex_setup.sh`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/codex_setup.sh), set `PYTHON_BIN=python3.10`; the script now fails fast on incompatible runtimes.

## Remote Cluster Job Monitoring
- For detached cluster jobs, launch with a simple `status/progress.log` and per-seed log layout so progress checks do not depend on reconstructing shell commands later.
- When a dedicated host has spare CPU and RAM, prefer capacity-filling parallel launches over conservative fixed worker caps. Default launchers should size `TRAIN_MAX_WORKERS` to available host capacity unless a scientific or stability constraint requires serialization.
- If a batch controller already owns queued seeds, do not start overlapping helper launches for those same seeds unless you first disable or replace the controller. Otherwise the first finishing worker will trigger duplicate launches.
- From Codex, prefer simple read-only SSH probes: `tail` the batch progress log, sample a few seed logs, and count matching `train_ppo` processes. Avoid fragile nested quoting when a smaller probe answers the question.
- If sandboxed SSH/DNS blocks a check, reuse the same probe with escalated host execution rather than rewriting it into a more complicated command.
- Preferred helper for this repo: `python3 -m src.experiments_pgg_v0.check_iwr_progress --host <host> --run-dir <remote_batch_run_dir>`. The helper auto-discovers the batch progress log, samples representative seed logs, and includes any linked standalone run recorded in the batch `status/manifest.txt`.
- Do not use `quadopt4` or `quadopt5` for `phase3_vecstraight` training or evaluation reruns. On March 27, 2026, both hosts failed immediately on from-scratch exogenous-family launches (`public_random` on `quadopt4`, `uniform` on `quadopt5`) with `exit=136` / `Floating point exception`, zero live trainers, and zero checkpoints written.
- Both hosts report `AMD Opteron(tm) Processor 6176 SE` CPUs with no AVX-class flags. Treat them as unsupported for this repo's current Python / NumPy / PyTorch stack and do not spend more launch time on them.
- Use [`DATA_MAP.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/DATA_MAP.md) as the canonical registry for current remote run roots, fetched mirrors, and fetch conventions instead of copying dated run inventories into this file.
- Use [`PHASE3_VECSTRAIGHT_NEXT_STEPS.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/PHASE3_VECSTRAIGHT_NEXT_STEPS.md) for the active gating run and live handoff context.

## Canonical Vectorized Phase-3 Result Paths
- Before reusing any phase-3 result across repos, read [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md). The current manuscript family is `phase3_staged`; the newer straight vectorized family is `phase3_vecstraight`.
- For the required execution order for new-family manipulations, read [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/PHASE3_VECSTRAIGHT_NEXT_STEPS.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/PHASE3_VECSTRAIGHT_NEXT_STEPS.md) before launching new phase-3 reruns.
- For the current paper-facing remaining-work checklist, read [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/PHASE3_VECSTRAIGHT_PAPER_TODO.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/PHASE3_VECSTRAIGHT_PAPER_TODO.md) before planning new training, evaluation, or analysis work.
- Use [`DATA_MAP.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/DATA_MAP.md) as the artifact registry for canonical local/remote paths and fetched mirrors.
- Under the new straight vectorized design, the late communication gap is positive at both `f=3.5` and `f=5.0` by `150k` (`+16.4 pp` and `+18.8 pp` in the current recheck), so old manuscript-facing phase-3 claims are not portable across training designs.

## Phase-3 Continuation Parity Contract
- Same-checkpoint continuation runs in `phase3_vecstraight` must preserve the base training regime unless the scientific question explicitly changes it.
- For the current vecstraight manuscript family, the continuation training contract is:
  - `num_envs=8`
  - `count_env_episodes=true`
  - `env_backend=subproc`
  - `env_start_method=spawn`
  - `entropy_schedule=linear`
  - `entropy_coeff=0.01`
  - `entropy_coeff_final=0.001`
  - `msg_entropy_coeff=0.01`
  - `msg_entropy_coeff_final=0.0`
  - `lr_schedule=cosine`
  - `regime_log_interval=400`
  - `checkpoint_interval=25000`
  - `disable_comm_fallback=true` for `comm` continuations
- The only intended differences between base vecstraight training and a same-checkpoint continuation should be:
  - the branch checkpoint (`init_ckpt`)
  - `episode_offset`
  - `schedule_total_episodes`
  - continuation-specific message intervention settings such as `uniform`, `public_random`, `fixed0`, or `sender_shuffle`
- Before trusting any continuation result, verify at least one saved `.run.json` from that run against this contract.
- If a continuation launch path cannot guarantee this parity, do not launch the experiment until the launcher and tests are fixed.
