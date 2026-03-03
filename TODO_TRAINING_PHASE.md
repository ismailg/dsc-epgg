# Training-First TODO (Pre-PLRNN)

## Status (completed)
- [x] Condition 6 shakedown: no comm, no uncertainty, no tremble (seed 101, 300 episodes)
- [x] Condition 2 shakedown: no comm, symmetric uncertainty, tremble=0.05 (seed 101, 300 episodes)
- [x] Condition 1 shakedown: comm on, symmetric uncertainty, tremble=0.05 (seed 101, 300 episodes)
- [x] All three runs completed without NaN/crash.

## Next (required before PLRNN)
1. Build evaluation script for regime-conditional behavior checks
- [ ] Compute cooperation rate by true `f_t` and regime.
- [ ] Compute per-agent cooperation and reward stats.
- [ ] Confirm Condition 6 shows policy separation across regimes.

2. Launch multi-seed baseline training
- [ ] Run seeds `{101, 202, 303, 404, 505}` for Condition 6.
- [ ] Run seeds `{101, 202, 303, 404, 505}` for Condition 2.
- [ ] Run seeds `{101, 202, 303, 404, 505}` for Condition 1.
- [ ] Start with 2,000 episodes per run (`T=100`), then scale if needed.

3. Generate evaluation datasets for model-comparison stage
- [ ] For each (condition, seed), collect 500 logged sessions.
- [ ] Consolidate to one file per (condition, seed).

4. Run Dynamic Richness Checklist gate
- [ ] Regime-separated cooperation differences present.
- [ ] History terms significant (`k_{t-1:t-3}`).
- [ ] Non-trivial within-session transitions.
- [ ] MI(message; f) and MI(message; action) > 0 for comm condition.
- [ ] Non-negligible agent heterogeneity.

5. Decide escalation only if checklist fails
- [ ] Increase `rho` first.
- [ ] Increase `sigma`.
- [ ] Add lag features (`k_{t-2}`, `k_{t-3}`) if needed.
- [ ] Increase tremble/dropout if still too trivial.

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
