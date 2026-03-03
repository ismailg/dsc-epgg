# Implementation Notes (Stage 0 + Stage 1)

## Repository and contract discovery

Date: 2026-03-03  
Repo: `ismailg/dsc-epgg` (local path `/Users/mbp17/POSTDOC/NPS26/dsc-epgg`)

### Actual upstream contracts found

1. Environment entrypoint:
   - `src/environments/pgg_parallel_v0.py` re-exports `src/environments/pgg/pgg_parallel_v0.py`.
   - All changes were applied in `src/environments/pgg/pgg_parallel_v0.py`.

2. Environment constructor contract:
   - Upstream expects a config-like object with `.items()` yielding key/value pairs.
   - Current implementation supports dict configs used by new `train_ppo.py`.

3. Raw observation contract (upstream):
   - Upstream returned a 1D tensor with only noisy multiplier.
   - New contract returns 2D tensor `[f_hat, endowment]` per agent.

4. Step contract:
   - Preserved legacy 4-tuple: `(observations, rewards, done_bool, infos)`.
   - `done` is a scalar bool (not per-agent dict), matching original training loops.

5. Agent observation handling:
   - Upstream `Agent.set_observation` assumed scalar normalization.
   - Updated to support vector observations and optional normalization (`normalize_obs=False` default).

### Main deviations from pseudocode and why

1. Kept the old 4-tuple done signature:
   - Required to remain compatible with current training/eval code in this repo.

2. Wrapper adapts tensor-first observations:
   - Upstream env uses tensor outputs, so wrapper canonicalizes either tensor or dict inputs.

3. Communication implementation:
   - Joint message/action PPO objective implemented in new training path (`train_ppo.py`) without rewriting legacy files.
   - Legacy classes remain intact for backward compatibility.

### Stage 1 status

- Multi-step env bug fixed (`observations` always defined and returned).
- Sticky-`f` Markov switching implemented with hazard `rho`.
- Trembling-hand flips implemented env-side before reward computation.
- Reward computation uses executed actions and current round `f_t`.
- Observation clamping removed from `observe()`.
- Observation space changed to continuous `Box(shape=(2,), dtype=float32)`.
- Wrapper implemented with lagged features, EWMA, message marginals, and dropout.
- Added mixed-action payoff regression test (2 cooperate / 2 defect) to verify
  both cooperator and defector reward formulas in addition to all-cooperate case.
- Gate 1 and expanded tests pass.
