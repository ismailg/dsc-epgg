# Repo Notes For Future Agents

These notes are additive to the parent [AGENTS.md](/Users/mbp17/POSTDOC/NPS26/AGENTS.md). They capture operational lessons from the Phase 3 annealed training/eval work so future agents do not repeat the same failures.

## Long-running jobs

- Do not rely on `nohup ... &` launched through a one-shot shell tool call. In this environment those background jobs can disappear when the tool invocation returns.
- For long training or chained evaluation pipelines, use a persistent PTY session:
  - start with `exec_command(..., tty=true)`
  - keep the returned `session_id`
  - poll with `write_stdin(session_id, chars=\"\")`
- If the laptop may sleep, run the PTY command under `caffeinate`, for example:
  - `caffeinate -dimsu /bin/zsh scripts/run_phase3_annealed_pipeline.sh`

## CPU / BLAS thread caps

- Cap BLAS/OpenMP threads to `1` for each training worker, otherwise a small number of PPO processes can oversubscribe all cores:
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `VECLIB_MAXIMUM_THREADS=1`
  - `NUMEXPR_NUM_THREADS=1`
- Current practical target on this 10-core machine:
  - `6` concurrent PPO workers is safe
  - keep headroom for eval/cross-play/rescue stages
  - avoid saturating all `10` cores with training alone

## Host monitoring

- Sandbox process inspection can fail for `ps`/`pgrep`. If you need the real host process table, use escalated host commands.
- The minimal health check is:
  - active `train_ppo` worker count
  - metrics JSONL growth
  - checkpoint creation at expected intervals

## Phase 3 canonical paths

- Use these as the authoritative annealed run roots:
  - training: `outputs/train/phase3_annealed_trimmed`
  - seed-101 eval/report: `outputs/eval/phase3_annealed_trimmed_seed101`
  - full multi-seed eval/report: `outputs/eval/phase3_annealed_trimmed_all`
- Treat `outputs/train/phase3_annealed_seed101` as stale. That was from an earlier failed launcher attempt and should not be used as the canonical run root.

## Phase 3 orchestration

- Canonical unattended script:
  - [scripts/run_phase3_annealed_pipeline.sh](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/scripts/run_phase3_annealed_pipeline.sh)
- High-information follow-up script:
  - [scripts/run_phase3_high_info_suite.sh](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/scripts/run_phase3_high_info_suite.sh)
- High-information watchdog:
  - [scripts/run_phase3_high_info_watchdog.sh](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/scripts/run_phase3_high_info_watchdog.sh)
- It runs, in order:
  1. seed-101 annealed pilot train
  2. seed-101 trimmed eval/report
  3. seed-101 rescue/fragmentation analysis
  4. remaining seed training
  5. full trimmed eval/report
  6. full rescue/fragmentation analysis

## Continuation runs

- `train_ppo.py` now supports continuation-aware scheduling via:
  - `--episode_offset`
  - `--schedule_total_episodes`
- Use them whenever extending a prior checkpoint horizon. Otherwise:
  - entropy/LR schedules restart from zero
  - checkpoint filenames are misaligned with effective training episode
  - downstream `50k -> 150k` analysis becomes scientifically invalid
- For the current high-information extension, the correct pattern is:
  - init from `50k`
  - `episode_offset=50000`
  - `schedule_total_episodes=150000`
  - `checkpoint_interval=50000`
  - resulting checkpoints:
    - `_ep100000.pt`
    - final base `.pt` representing `150000`

## Channel controls

- Training-time communication controls now include:
  - `fixed0` for always-zero received channel
  - `uniform` for independent random bits per sender slot
  - `public_random` for a shared random bit broadcast into all sender slots
- Evaluation-time ablations now include:
  - `permute_slots` to randomize sender-slot identity while preserving the delivered bits
- `permute_slots` is the direct test for sender-conditioned decoding:
  - if performance drops, receivers are using sender identity
  - if aggregate token effects stay small while sender-slot effects remain large, prefer the fragmentation interpretation over “messages died”

## Unattended completion

- For multi-stage local runs that must survive intermediate script failures, prefer:
  - `run_phase3_high_info_suite.sh` as the worker
  - `run_phase3_high_info_watchdog.sh` as the supervisor
- The watchdog exits only when the final comparison artifact exists:
  - `outputs/eval/phase3_compare/annealed_vs_unannealed_trajectory_mean.csv`
- This setup is intentionally idempotent:
  - training runners use `--skip_existing`
  - eval/report stages can be safely rerun

## Overlapping seed expansion

- [run_phase3_seed_expansion.py](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/src/experiments_pgg_v0/run_phase3_seed_expansion.py) now uses per-job lock files:
  - lock path is `SAVE_PATH + ".lock"`
- This allows a second expansion runner to be launched safely against the same output root while another pipeline is active.
- Combined with `--skip_existing`, this prevents duplicate seed training if:
  - one runner is already training a checkpoint
  - another runner later reaches the same seed/condition

## Git and running jobs

- `git commit` is safe while training is running.
- `git push` is safe for running jobs, but may still be blocked by host approval/UI.
- Do not:
  - switch branches
  - reset/checkout files
  - edit the same training script mid-run unless absolutely necessary

## What to inspect first after handoff

- Pipeline live log:
  - `outputs/train/phase3_annealed_trimmed/pipeline.log`
- Current metrics:
  - `outputs/train/phase3_annealed_trimmed/metrics/*.jsonl`
- Active sessions/processes:
  - persistent PTY session running the pipeline
  - any extra `run_phase3_seed_expansion` session launched for more parallel seeds

## Scientific interpretation notes

- Late aggregate token effects can be near zero even when sender-specific effects remain large.
- Prefer sender-preserved diagnostics over aggregate token summaries:
  - sender polarity over time
  - alignment index over time
  - receiver-by-sender effects
- Common-polarity rescue is a key mechanistic test for fragmentation/private-code conflict.
