# Implementation Notes (Stage 2: PPO + Communication)

## New modules

- `src/algos/trajectory_buffer.py`
  - Stores per-step intended actions, executed actions, flips, rewards, values, log-probs.
  - Includes optional message actions/log-probs and listening auxiliary signal.
  - Implements GAE (`gamma=0.99`, `lam=0.95` configurable).

- `src/algos/PPO.py` (extended)
  - Added `PPOAgentV2`: separate action actor, optional message actor, shared value net.
  - Added `PPOTrainer`: clipped surrogate objective with value and entropy terms.
  - Joint comm objective:
    - Action PPO objective always active.
    - Message PPO objective active for sender agents when message samples/logprobs exist.
  - Auxiliary terms:
    - Signaling entropy target penalty (`sign_lambda`).
    - Listening sensitivity penalty (`list_lambda`) using rollout-time policy shift.

- `src/experiments_pgg_v0/train_ppo.py`
  - End-to-end multi-step training loop with wrapper integration.
  - Includes communication path, message dropout, and trajectory collection.
  - Implements time-boxed comm fallback:
    - Up to 2 debug attempts by default.
    - Falls back to no-comm PPO if repeated non-finite metrics occur.

## Objective details

For action policy:
- Ratio: `r = exp(new_logp_action - old_logp_action)`
- Clipped surrogate: `min(r*A, clip(r, 1-eps, 1+eps)*A)`
- Loss: `L = -surrogate + c_v * MSE(V, R) - c_e * entropy`

For message policy (sender-only):
- Same clipped surrogate construction on message log-probs.
- Uses same per-step advantage as action branch.
- Added to total loss with optional signaling regularizer.

## Smoke validation commands

No-comm smoke:
```bash
python3 src/experiments_pgg_v0/train_ppo.py \
  --n_episodes 3 --T 6 --n_agents 4 \
  --sigmas 0.5 0.5 0.5 0.5 \
  --save_path outputs/smoke_agents.pt --log_interval 1
```

Comm-enabled smoke:
```bash
python3 src/experiments_pgg_v0/train_ppo.py \
  --n_episodes 4 --T 8 --n_agents 4 \
  --sigmas 0.5 0.5 0.5 0.5 \
  --comm_enabled --n_senders 2 --vocab_size 2 \
  --save_path outputs/smoke_agents_comm.pt --log_interval 1
```

Observed:
- Finite losses.
- Non-collapsed action entropy.
- Non-zero message-loss and message-entropy in comm run.

