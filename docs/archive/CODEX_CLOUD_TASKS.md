# Codex Cloud Task Prompts (DSC-EPGG Week 1–2)

Use these as separate cloud tasks (parallelizable by stage).

## Task A — Stage 0 + Stage 1 (Gate 1)
Implement Stage 0 and Stage 1 from `README_IMPLEMENTATION.md` and `AGENTS.md` only.

Requirements:
- Bootstrap and document actual code contracts (obs shape/type, step/infos contract, comm hooks).
- Patch env for multi-step correctness, Sticky-f, tremble, unclamped `f_hat`, and `Box` observation space.
- Ensure `gmm_ = False` and `normalize_obs = False` where needed.
- Add trainer-side observation wrapper with tensor adapter, history features, EWMA, message dropout and per-sender marginals.
- Add/update unit tests for env and wrapper behavior.
- Run a short smoke check and report deterministic command + result.

Deliverables:
- Code changes + tests.
- A brief `IMPLEMENTATION_NOTES_STAGE1.md` documenting discovered contracts and deviations from pseudocode.

## Task B — Stage 2 PPO + Communication (Gate 2)
Implement Stage 2 from `README_IMPLEMENTATION.md` and `AGENTS.md`.

Requirements:
- Add trajectory buffer with intended/executed split and GAE.
- Add PPO trainer (clip objective + value + entropy).
- Include joint action+message log-probs in the surrogate objective for sender agents.
- Reconnect signaling/listening auxiliary losses with config-controlled weights.
- Add end-to-end `train_ppo.py`.
- Apply fallback rule: if joint comm PPO unstable after 2 focused debug cycles, ship no-comm PPO baseline first and document the fallback.
- Add GAE/unit/integration smoke tests.

Deliverables:
- Code + tests.
- `IMPLEMENTATION_NOTES_STAGE2.md` with objective equations used and fallback decision outcome.

## Task C — Stage 3 Logging + Regime Audit
Implement Stage 3 from `README_IMPLEMENTATION.md` and `AGENTS.md`.

Requirements:
- Add session logger for Set C with per-session `.npz` and consolidation utility.
- Add regime identifiability audit script with convergence stats (`mean`, `median`, `p90`) and recommendation when `median <= 2`.
- Add tests or deterministic validation snippets for schema/shape correctness.

Deliverables:
- Code changes.
- `IMPLEMENTATION_NOTES_STAGE3.md` with sample outputs and command lines.
