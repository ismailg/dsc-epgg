# Training-First TODO (Pre-PLRNN)

## Status (completed)
- [x] Condition 6 shakedown: no comm, no uncertainty, no tremble (seed 101, 300 episodes)
- [x] Condition 2 shakedown: no comm, symmetric uncertainty, tremble=0.05 (seed 101, 300 episodes)
- [x] Condition 1 shakedown: comm on, symmetric uncertainty, tremble=0.05 (seed 101, 300 episodes)
- [x] All three runs completed without NaN/crash.
- [x] Gamma stability fix validated: fixed `f=5.0`, `gamma=0.99` reaches cooperation ~1.0 in sanity run.
- [x] Regime-conditional evaluator implemented (`src/analysis/evaluate_regime_conditional.py`).
- [x] Streaming regime/f-value metrics integrated into trainer (`train_ppo.py`, JSONL + periodic summaries).
- [x] Baseline 1 scaffold implemented (`src/baselines/bayes_filter.py`, `src/analysis/run_bayesian_baseline1.py`).
- [x] Fixed-`f` sweep launcher added (`src/experiments_pgg_v0/run_fixed_f_sweep.py`).

## Next (required before PLRNN)
1. Complete Stage-1 fixed-`f` validation (Phase 1)
- [ ] Finish/verify fixed `f=5.0` long run checkpoint.
- [ ] Run fixed `f ∈ {0.5, 1.5, 2.5, 3.5}` with seed `101` (50k episodes).
- [ ] Check expected behavior by regime:
  - `f=0.5`: near-zero cooperation.
  - `f=5.0`: near-one cooperation.
  - `f=1.5` and `f=2.5/3.5`: intermediate or transitioning behavior.

2. Complete Stage-2 switching training diagnostics (Phase 2)
- [ ] Continue 200k switching run from warm start.
- [ ] Monitor regime-separated cooperation at 50k/100k/150k/200k from metrics JSONL.
- [ ] Verify no collapse to unconditional policy (regime curves should separate).

3. Scale baseline grid from pilot to full multi-seed
- [ ] Run Conditions `{6, 2, 1}` × seeds `{101, 202, 303, 404, 505}` at 2k episodes if not already complete.
- [ ] Increase episode budget once learning curves indicate undertraining.
- [ ] Keep per-run logs/checkpoints and aggregate summaries.

4. Prepare data products for baseline and PLRNN comparisons
- [ ] For each `(condition, seed)`, generate 500 evaluation sessions with session logging.
- [ ] Consolidate to one `.npz` per `(condition, seed)`.
- [ ] Run Baseline 1 on these datasets and store JSON summaries.
- [ ] Extend Baseline 1 evaluation beyond scaffold:
  - add multi-step horizons `H ∈ {5, 10, 20}`;
  - report both teacher-forced and open-loop modes.

5. Dynamic Richness Checklist gate (must pass before PLRNN)
- [ ] Regime-separated cooperation differences present.
- [ ] History terms significant (`k_{t-1:t-3}`).
- [ ] Non-trivial within-session transitions.
- [ ] MI(message; f) and MI(message; action) > 0 for comm condition.
- [ ] Non-negligible agent heterogeneity.

6. Escalation only if checklist fails
- [ ] Increase `rho`.
- [ ] Increase `sigma`.
- [ ] Add lag features (`k_{t-2}`, `k_{t-3}`).
- [ ] Increase tremble/dropout.

## Baseline run templates

Condition 6:
```bash
python3 src/experiments_pgg_v0/train_ppo.py \
  --n_episodes 2000 --T 100 --n_agents 4 \
  --sigmas 0 0 0 0 --epsilon_tremble 0.0 \
  --seed 101 --save_path outputs/train/cond6_seed101.pt --log_interval 50
```

Condition 2:
```bash
python3 src/experiments_pgg_v0/train_ppo.py \
  --n_episodes 2000 --T 100 --n_agents 4 \
  --sigmas 0.5 0.5 0.5 0.5 --epsilon_tremble 0.05 \
  --seed 101 --save_path outputs/train/cond2_seed101.pt --log_interval 50
```

Condition 1:
```bash
python3 src/experiments_pgg_v0/train_ppo.py \
  --n_episodes 2000 --T 100 --n_agents 4 \
  --sigmas 0.5 0.5 0.5 0.5 --epsilon_tremble 0.05 \
  --comm_enabled --vocab_size 2 \
  --seed 101 --save_path outputs/train/cond1_seed101.pt --log_interval 50
```
