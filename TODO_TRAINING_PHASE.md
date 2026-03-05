# Training-First TODO (Pre-PLRNN)

## Completed milestones

### Infrastructure & shakedown
- [x] Condition 6 shakedown: no comm, no uncertainty, no tremble (seed 101, 300 episodes)
- [x] Condition 2 shakedown: no comm, symmetric uncertainty, tremble=0.05 (seed 101, 300 episodes)
- [x] Condition 1 shakedown: comm on, symmetric uncertainty, tremble=0.05 (seed 101, 300 episodes)
- [x] All three runs completed without NaN/crash.
- [x] Gamma stability fix validated: fixed `f=5.0`, `gamma=0.99` reaches cooperation ~1.0 in sanity run.
- [x] Regime-conditional evaluator implemented (`src/analysis/evaluate_regime_conditional.py`).
- [x] Streaming regime/f-value metrics integrated into trainer (`train_ppo.py`, JSONL + periodic summaries).
- [x] Baseline 1 scaffold implemented (`src/baselines/bayes_filter.py`, `src/analysis/run_bayesian_baseline1.py`).
- [x] Fixed-`f` sweep launcher added (`src/experiments_pgg_v0/run_fixed_f_sweep.py`).
- [x] Early stopping support added to `run_fixed_f_sweep.py` and overnight orchestrator.

### Phase 1: Fixed-f validation (completed 2026-03-04)
- [x] 25/25 fixed-f runs completed: `f ∈ {0.5, 1.5, 2.5, 3.5, 5.0}` × seeds `{101, 202, 303, 404, 505}`, 50k episodes each.
- [x] Results match game-theoretic predictions:
  - `f=0.5`: cooperation ≈ 0.001 (defection dominant, Δ = -3.5). ✓
  - `f=1.5`: cooperation ≈ 0.003 (defection dominant, Δ = -2.5). ✓
  - `f=2.5`: cooperation ≈ 0.005 (defection dominant, Δ = -1.5). ✓
  - `f=3.5`: cooperation ≈ 0.028 (defection dominant, Δ = -0.5). ✓
  - `f=5.0`: cooperation ≈ 0.977 (cooperation dominant, Δ = +1.0). ✓
- [x] Interpretation: agents correctly converge to Nash equilibrium in all regimes. Cooperation in mixed regimes (1 < f < N) requires coordination mechanisms not present in independent PPO.

### Phase 2: Switching run (completed 2026-03-04, results inconclusive)
- [x] 200k-episode switching run completed (seed 101, warm start from f=5 checkpoint).
- [x] Post-hoc per-f evaluation (500 eval episodes) shows regime ordering:
  - `f=0.5`: coop = 0.072
  - `f=1.5`: coop = 0.103
  - `f=2.5`: coop = 0.304
  - `f=3.5`: coop = 0.468
  - `f=5.0`: coop = 0.530
- [x] Regime ordering present (competitive < mixed < cooperative) but cooperative regime is far below expected (~53% vs ~95-100%).
- [x] Average cooperation showed no learning trend across 200k episodes (oscillates around 0.25).

### Key diagnostic finding
The f=5.0 underperformance (53% cooperation when cooperation is strictly dominant) likely indicates the switching environment degrades the originally strong f=5 policy. With 80% of rounds in defection-optimal regimes, gradient pressure pushes toward defection overall, partially overwriting the f=5 cooperation. The agent learns a "compromised threshold" rather than a sharp regime-conditioned policy.

---

## Current phase: Comm-on vs Comm-off head-to-head (Phase 2b)

### Rationale
Phase 1 established that independent PPO agents converge to Nash equilibrium without coordination mechanisms. This is the scientifically expected baseline. The core research question is whether **communication reshapes the behavioral landscape** — enabling cooperation in mixed regimes, creating signaling conventions, and producing the dynamic richness needed for PLRNN analysis.

### Step 2b.1: Diagnostic tooling (BLOCKING — do before training)
- [ ] Add per-f cooperation tracking (`P(C|f)` for each `f ∈ F`) to the regime-stream JSONL, logged every `regime_log_interval` episodes.
- [ ] Add `P(C|f_hat)` binned diagnostic: bin `f_hat` into intervals and compute cooperation rate per bin. Log periodically or compute post-hoc from session data.
- [ ] For comm-on runs: add MI(`m`; `f`) and MI(`m`; `a`) computation to training metrics or post-hoc evaluation.
- [ ] Add greedy (argmax) evaluation mode to `evaluate_regime_conditional.py` alongside stochastic sampling, to separate policy quality from entropy/tremble effects.

### Step 2b.2: Train Condition 1 vs Condition 2 side-by-side
- [ ] **Condition 2** (comm off, symmetric uncertainty, tremble): 5 seeds × 200k episodes.
  - `--sigmas 0.5 0.5 0.5 0.5 --epsilon_tremble 0.05 --rho 0.05 --F 0.5 1.5 2.5 3.5 5.0`
  - Warm start from Phase 1 f=5 checkpoints (per seed), or train from scratch.
- [ ] **Condition 1** (comm on, symmetric uncertainty, tremble): 5 seeds × 200k episodes.
  - Same as Condition 2 plus `--comm_enabled --vocab_size 2`
  - Warm start from Condition 2 checkpoints (spec Phase 3 curriculum), or train from scratch.
- [ ] Use matched seeds `{101, 202, 303, 404, 505}` across conditions for paired comparison.
- [ ] All runs must use `--metrics_jsonl_path` for streaming per-f metrics.
- [ ] Run `max_workers=2` to stay within CPU budget.

### Step 2b.3: Per-f evaluation at milestones
- [ ] At 50k/100k/200k episodes, run `evaluate_regime_conditional.py` on saved checkpoints.
- [ ] Compare `P(C|f)` curves between Condition 1 and 2.
- [ ] For Condition 1: check MI(`m`; `f`) > 0 and MI(`m`; `a`) > 0.
- [ ] Key success criteria:
  - Condition 1 `P(C|f=5)` > Condition 2 `P(C|f=5)` (comm helps at dominant-strategy threshold).
  - Condition 1 `P(C|f=3.5)` > Condition 2 `P(C|f=3.5)` (comm enables coordination in near-threshold mixed regime).
  - Messages are informative: MI > 0 for at least one of (`m`; `f`) or (`m`; `a`).

### Step 2b.4: Gate decision
- [ ] If comm shows measurable effect → proceed to Dynamic Richness Checklist and data generation.
- [ ] If comm shows NO effect → escalate per spec §2.5 in this order:
  1. Rebalance `F` toward threshold: try `F = {2.0, 3.0, 3.5, 4.0, 5.0}` (concentrate near `f=N`).
  2. Increase `rho` to 0.1 or 0.2 (more regime switches, more need for fast adaptation).
  3. Increase `sigma` to 1.0.
  4. Add lag features (`k_{t-2}`, `k_{t-3}`).

---

## Later phases (after 2b gate passes)

### 3. Scale baseline grid to full multi-seed
- [ ] Run Conditions `{6, 2, 1}` × seeds `{101, 202, 303, 404, 505}` at appropriate episode budget.
- [ ] Keep per-run logs/checkpoints and aggregate summaries.

### 4. Prepare data products for baseline and PLRNN comparisons
- [ ] For each `(condition, seed)`, generate 500 evaluation sessions with session logging.
- [ ] Consolidate to one `.npz` per `(condition, seed)`.
- [ ] Run Baseline 1 on these datasets and store JSON summaries.
- [ ] Extend Baseline 1 evaluation: add multi-step horizons `H ∈ {5, 10, 20}`; report both teacher-forced and open-loop modes.

### 5. Dynamic Richness Checklist gate (must pass before PLRNN)
- [ ] Regime-separated cooperation differences present.
- [ ] History terms significant (`k_{t-1:t-3}`).
- [ ] Non-trivial within-session transitions.
- [ ] MI(message; f) and MI(message; action) > 0 for comm condition.
- [ ] Non-negligible agent heterogeneity.

### 6. Escalation only if checklist fails
- [ ] Increase `rho`.
- [ ] Increase `sigma`.
- [ ] Add lag features (`k_{t-2}`, `k_{t-3}`).
- [ ] Increase tremble/dropout.
- [ ] Rebalance `F` toward threshold (e.g., `F = {2.0, 3.0, 3.5, 4.0, 5.0}`).

## Baseline run templates

Condition 2 (Phase 2b, comm off):
```bash
python3 src/experiments_pgg_v0/train_ppo.py \
  --n_episodes 200000 --T 100 --n_agents 4 \
  --F 0.5 1.5 2.5 3.5 5.0 --rho 0.05 \
  --sigmas 0.5 0.5 0.5 0.5 --epsilon_tremble 0.05 \
  --gamma 0.99 --reward_scale 20.0 \
  --seed 101 --save_path outputs/train/cond2_seed101.pt \
  --log_interval 500 --regime_log_interval 500 \
  --metrics_jsonl_path outputs/train/phase2b/metrics/cond2_seed101.jsonl \
  --condition_name cond2
```

Condition 1 (Phase 2b, comm on):
```bash
python3 src/experiments_pgg_v0/train_ppo.py \
  --n_episodes 200000 --T 100 --n_agents 4 \
  --F 0.5 1.5 2.5 3.5 5.0 --rho 0.05 \
  --sigmas 0.5 0.5 0.5 0.5 --epsilon_tremble 0.05 \
  --gamma 0.99 --reward_scale 20.0 \
  --comm_enabled --vocab_size 2 \
  --seed 101 --save_path outputs/train/cond1_seed101.pt \
  --log_interval 500 --regime_log_interval 500 \
  --metrics_jsonl_path outputs/train/phase2b/metrics/cond1_seed101.jsonl \
  --condition_name cond1
```
