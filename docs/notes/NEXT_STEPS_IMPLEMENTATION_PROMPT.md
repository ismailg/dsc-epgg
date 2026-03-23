# Implementation Prompt: Final Experiments + Paper Writing

## Context

You are working on the DSC-EPGG project: a multi-agent reinforcement learning study of emergent communication in an Extended Public Goods Game (4 agents, 5 hidden regime levels, binary messages, PPO with entropy annealing).

Read `RESEARCH_LOG.md` for the full chronological history. Read `DIDACTIC_OVERVIEW.md` for a didactic walkthrough.

### What we know so far

1. **Communication robustly helps at f=3.5 at 50k** (+30pp, 5/5 seeds positive) and still helps at 150k (+19pp, 4/5 seeds positive). At f=5.0, the benefit has disappeared by 150k (−7pp).

2. **Communication benefit is NOT internalized.** Muting the channel after 50k produces *worse* cooperation than never having communicated (0.367 vs 0.427 at f=3.5). Agents that learn with communication become structurally dependent on the channel.

3. **Learned messages ≈ noise by 150k in continuation experiments.** When separately-trained 50k-checkpoint channel controls are continued to 150k, all channel modes converge: indep_random (0.557) ≥ learned (0.548) ≥ public_random (0.545) ≥ always_zero (0.536).

4. **But semantics matter during initial learning (0–50k).** Separately trained from scratch: learned (0.809) >> indep_random (0.669) > always_zero (0.641) > public_random (0.620) at f=3.5.

5. **Aggregate endpoint metrics are misleading.** Per-sender receiver effects are ~9× larger than aggregate token effects (0.157 vs 0.018 at 150k, 5 seeds).

6. **Alignment is non-monotonic.** Convention alignment can emerge (seed 303: 2:2→4:0) or collapse (seed 505: 4:0→2:2) with continued training.

### What the supervisor flagged as still missing

The existing 150k channel controls (finding #3) load from **separately-trained** 50k checkpoints — each mode was trained from scratch under that mode. The supervisor identified that the cleanest missing experiment is a **same-checkpoint** continuation: from the **identical** learned 50k checkpoints, continue under different channel modes. This directly tests whether later training needs meaningful semantics, any exogenous variation, or just a channel-shaped policy interface.

### Approved priority order for next actions

1. Same-checkpoint 50k→150k continuation controls (KEY EXPERIMENT)
2. Seed-level stats / bootstrap CIs / significance reporting
3. One lightweight interoperability or generalization test
4. Write the communication paper (9-page NeurIPS format)
5. PLRNN to appendix or follow-up

---

## Experiment 1: Same-Checkpoint 50k→150k Continuation Controls (HIGHEST PRIORITY)

### Design

From the **identical** learned 50k checkpoints in `outputs/train/phase3_annealed_trimmed/`, continue training to 150k under 4 different channel modes:

| Mode | `--msg_training_intervention` | What receivers see |
|---|---|---|
| `none` (learned) | `none` | Sender's actual message (already done — this is the 5-seed 150k main result) |
| `fixed0` (always_zero) | `fixed0` | Always token 0 regardless of sender output |
| `uniform` (indep_random) | `uniform` | Independent random token per receiver per timestep |
| `public_random` | `public_random` | Same random token for all receivers each timestep |

**Seeds:** 101, 202, 303, 404, 505 (all 5 seeds)

**Crucial difference from existing channel controls:** The existing `train_controls_to_150k_mode()` function in `scripts/run_phase3_next_steps_suite.sh` uses `--init_checkpoint_dir "$CONTROL_50K_ROOT/$mode"` — loading from separately-trained 50k models. The new experiment must use `--init_checkpoint_dir "$BASE_50K_DIR"` (i.e., `outputs/train/phase3_annealed_trimmed`) for ALL modes. This means every continuation starts from the exact same learned-communication policy, and only the channel content differs during continued training.

**Note on the `none` (learned) condition:** This experiment already exists — it is the 5-seed 150k main result at `outputs/train/phase3_annealed_ext150k_5seeds/`. You do NOT need to retrain it. Just reference those outputs as the `none` condition. Only 3 new modes need training: `fixed0`, `uniform`, `public_random`.

### Infrastructure

**Training script:** `src/experiments_pgg_v0/run_phase3_seed_expansion.py`

**Output directories:**
```
outputs/train/phase3_sameckpt_continuation_5seeds/fixed0/
outputs/train/phase3_sameckpt_continuation_5seeds/uniform/
outputs/train/phase3_sameckpt_continuation_5seeds/public_random/
```

### Shell Function Template

Model this on the existing `train_mute_after_50k()` function (which correctly uses `$BASE_50K_DIR` as init):

```bash
FIVE_SEEDS=(101 202 303 404 505)
BASE_50K_DIR="outputs/train/phase3_annealed_trimmed"
SAMECKPT_ROOT="outputs/train/phase3_sameckpt_continuation_5seeds"
SAMECKPT_EVAL_ROOT="outputs/eval/phase3_sameckpt_continuation_5seeds"

train_sameckpt_continuation_mode() {
  local mode="$1"
  local out_dir="$SAMECKPT_ROOT/$mode"
  log "train same-checkpoint continuation mode=$mode"
  mkdir -p "$out_dir"
  # Hydrate the learned 50k checkpoints into the output dir
  for seed in "${FIVE_SEEDS[@]}"; do
    copy_if_missing "$BASE_50K_DIR/cond1_seed${seed}.pt" "$out_dir/cond1_seed${seed}_ep50000.pt"
  done
  "$PYTHON_BIN" -m src.experiments_pgg_v0.run_phase3_seed_expansion \
    --out_dir "$out_dir" \
    --init_checkpoint_dir "$BASE_50K_DIR" \
    --init_episode 50000 \
    --episode_offset 50000 \
    --schedule_total_episodes 150000 \
    --conditions cond1 \
    --seeds "${FIVE_SEEDS[@]}" \
    --n_episodes 100000 \
    --T 100 \
    --rho 0.05 \
    --epsilon_tremble 0.05 \
    --sigmas 0.5 0.5 0.5 0.5 \
    --gamma 0.99 \
    --reward_scale 20.0 \
    --lr 3e-4 \
    --min_lr 1e-5 \
    --lr_schedule cosine \
    --entropy_coeff 0.01 \
    --entropy_schedule linear \
    --entropy_coeff_final 0.001 \
    --msg_entropy_coeff 0.01 \
    --msg_entropy_coeff_final 0.0 \
    --checkpoint_interval 50000 \
    --regime_log_interval 5000 \
    --log_interval 1000 \
    --msg_training_intervention "$mode" \
    --max_workers 3 \
    --skip_existing
}
```

**Training order:** Run `fixed0`, `uniform`, `public_random` sequentially or in parallel (max 3 concurrent per mode × 5 seeds = 15 jobs; limit to 6-8 concurrent processes total). Each run takes ~6h on M1 MacBook for 100k episodes.

### Evaluation

After training, evaluate each mode with the standard pipeline:

```bash
eval_sameckpt_continuation_mode() {
  local mode="$1"
  local ckpt_dir="$SAMECKPT_ROOT/$mode"
  local out_dir="$SAMECKPT_EVAL_ROOT/$mode"
  log "eval same-checkpoint continuation mode=$mode"
  rm -rf "$out_dir"
  "$PYTHON_BIN" -m src.analysis.run_phase3_trimmed_eval \
    --checkpoint_dir "$ckpt_dir" \
    --suite_out_dir "$out_dir/suite" \
    --crossplay_out_dir "$out_dir/crossplay" \
    --comm_condition cond1 \
    --baseline_condition "" \
    --seeds "${FIVE_SEEDS[@]}" \
    --milestones 50000 100000 150000 \
    --interventions none zeros fixed0 fixed1 permute_slots \
    --crossplay_sender_milestones 50000 100000 150000 \
    --crossplay_receiver_milestones 150000 \
    --n_eval_episodes 300 \
    --eval_seed 9001 \
    --max_workers 4

  "$PYTHON_BIN" -m src.analysis.aggregate_phase3_report \
    --suite_main_csv "$out_dir/suite/checkpoint_suite_main.csv" \
    --suite_comm_csv "$out_dir/suite/checkpoint_suite_comm.csv" \
    --suite_trace_csv "$out_dir/suite/checkpoint_suite_trace.csv" \
    --suite_sender_csv "$out_dir/suite/checkpoint_suite_sender_semantics.csv" \
    --suite_receiver_csv "$out_dir/suite/checkpoint_suite_receiver_semantics.csv" \
    --suite_posterior_csv "$out_dir/suite/checkpoint_suite_posterior_strat.csv" \
    --crossplay_main_csv "$out_dir/crossplay/crossplay_matrix_main.csv" \
    --out_dir "$out_dir/report" \
    --out_md "$out_dir/report/PHASE3_SAMECKPT_${mode^^}.md"

  "$PYTHON_BIN" -m src.analysis.summarize_phase3_welfare \
    --suite_main_csv "$out_dir/suite/checkpoint_suite_main.csv" \
    --bundle_label "sameckpt_$mode" \
    --out_csv "$out_dir/report/welfare_weighted_raw.csv" \
    --out_mean_csv "$out_dir/report/welfare_weighted_mean.csv"
}
```

### Aggregation: Channel Control Summary Table

After all 3 modes are evaluated, produce the same-checkpoint version of the channel control summary. You will need to create or extend `summarize_phase3_channel_controls.py` (or write a new `summarize_sameckpt_channel_controls.py`) to produce a table like:

```
mode, checkpoint_episode, f_regime, mean_p_cooperate, std_p_cooperate, n_seeds
none (learned),  50000, 3.5, <from 5-seed main>, ...
none (learned), 100000, 3.5, <from 5-seed main>, ...
none (learned), 150000, 3.5, <from 5-seed main>, ...
fixed0,          50000, 3.5, <all start same>, ...
fixed0,         100000, 3.5, ..., ...
fixed0,         150000, 3.5, ..., ...
...
```

Key: at episode 50000, ALL modes should show identical values (since they start from the same checkpoint). Any divergence at 100k/150k is caused purely by the channel manipulation during continued training.

**Use the learned condition from:** `outputs/eval/phase3_annealed_ext150k_5seeds/suite/checkpoint_suite_main.csv`

### What This Experiment Tells Us

- If `none` > `fixed0/uniform/public_random` at 150k: meaningful messages still provide advantage even late in training
- If `none` ≈ `uniform` ≈ `public_random` > `fixed0`: any exogenous variation helps, but content doesn't matter
- If all modes converge: the policy has become channel-independent; only the presence of the architecture matters
- If `none` ≈ all modes: identical to finding #3 above, but now definitively from the same starting point

This is the cleanest test of the "semantic window" hypothesis.

---

## Experiment 2: Statistical Analysis Infrastructure

### 2A: Bootstrap Confidence Intervals

Create `src/analysis/compute_bootstrap_cis.py` with:

**Inputs:**
- The 5-seed suite main CSV (`checkpoint_suite_main.csv` from `phase3_annealed_ext150k_5seeds`)
- Optionally: same-checkpoint continuation CSVs

**Computation:** For each (checkpoint, condition, f_regime, metric):
1. Extract per-seed values (seed as unit of inference)
2. Bootstrap resample (B=10000) with replacement from seed-level values
3. Report: mean, 95% CI (percentile method), p-value for difference tests

**Output:** `outputs/eval/phase3_stats/bootstrap_ci_table.csv` with columns:
```
checkpoint_episode, condition, f_regime, metric, mean, ci_lower, ci_upper, n_seeds
```

And for paired comparisons (comm vs no-comm):
```
checkpoint_episode, f_regime, metric, delta_mean, delta_ci_lower, delta_ci_upper, p_value, n_seed_pairs
```

### 2B: Seed-Level Paired Delta Table

From the existing 5-seed data, compute per-seed paired deltas for the key comparisons:

| Comparison | Metric | How |
|---|---|---|
| comm vs no-comm at each checkpoint | P(C\|f=3.5) | per-seed: cond1 − cond2 |
| comm vs no-comm at each checkpoint | P(C\|f=5.0) | per-seed: cond1 − cond2 |
| same-ckpt modes vs learned at 150k | P(C\|f=3.5) | per-seed: mode − learned |

Output as a compact table that can go directly in the paper appendix.

### 2C: Significance Reporting Convention

All tables in the paper should include:
- Mean ± sem (or 95% CI) where n=seeds
- Stars for significance: * p<0.05, ** p<0.01, *** p<0.001 (bootstrap test)
- Bold for the best value in each comparison group

---

## Experiment 3: Lightweight Interoperability Test (Lower Priority)

### Design: Sender-Slot Permutation Test

For a given 150k checkpoint (say seed 303, which achieved alignment):
1. Evaluate normally (baseline)
2. Permute sender identities: agent 0's messages are relabeled as coming from agent 1's slot, etc. (circular permutation of `delivered_msg_agent_*` columns)
3. Compare P(C) between baseline and permuted

This tests whether receivers have learned sender-specific decoding or just attend to aggregate message content.

### Implementation

This can be done as a new intervention mode in `evaluate_regime_conditional.py`:
- Add `--ablate_messages permute_sender_slots` (or similar)
- In the evaluation loop, after messages are computed, circularly permute the sender-to-receiver mapping
- Report as a new intervention row in the suite CSV

Alternatively, write a standalone script `src/analysis/sender_slot_permutation_test.py` that:
1. Loads a checkpoint
2. Runs N episodes with natural messages (baseline)
3. Runs N episodes with circularly permuted sender slots
4. Reports delta P(C) and per-agent deltas

**Expected outcome:** If per-sender decoding is real, permutation should hurt cooperation. If agents only care about "how many 1s did I receive" (aggregate), permutation should have no effect.

### Alternative: Ad-Hoc Partner Test

Less clean but informative: take sender agents from seed 303 (aligned) and pair them with receiver agents from seed 101 (different convention). If communication is truly protocol-based, cross-seed pairing should fail. This is already available via the cross-play infrastructure — just need to run it cross-seed at 150k.

---

## Paper Writing Plan

### Target Venue

NeurIPS 2026 (or AAMAS 2026 if NeurIPS assessment is pessimistic). 9 content pages + unlimited appendix.

### Proposed Title

Choose one:
- **"Communication Helps Learning More Than It Produces a Stable Shared Code"**
- "Aggregate Endpoint Metrics Miss Fragile Communication Protocols in a Hidden-Regime Social Dilemma"

### Structure

#### §1 Introduction (~1.5 pages)

Lead with the endpoint evaluation problem: most emergent communication studies evaluate final checkpoints with aggregate metrics (MI, token accuracy, task success). We show this can be misleading.

Key setup:
- 4-agent EPGG with hidden regime, binary messages
- Communication helps coordination in the mixed regime (+19pp at 150k)
- But the communication benefit has a temporal structure: a "semantic window" during which content matters, after which the learned protocol degrades

End with contributions (4 items):
1. Communication creates a transient semantic advantage (0–50k), not a durable shared code
2. Same-checkpoint continuation controls show content stops mattering after initial learning
3. Per-sender disaggregation reveals effects ~9× larger than aggregate metrics suggest
4. Muting shows communication is not internalized — removing the channel is worse than never having it

#### §2 Setup (~1.5 pages)

- EPGG game mechanics (payoff, sticky regime, hidden multiplier, tremble, noise)
- Agent architecture (PPO, message head, observation wrapper)
- Training: entropy annealing, cosine LR, 5 seeds × 2 conditions
- Evaluation suite: greedy policy, per-regime cooperation, sender/receiver semantics, cross-play, channel controls

Include a clear diagram of the agent architecture + communication channel.

#### §3 Communication Improves Coordination in the Mixed Regime (~1.5 pages)

Present the main comm vs no-comm result:
- Table: P(C|f) for comm vs no-comm at 50k, 100k, 150k (5 seeds, with CIs)
- Communication helps robustly at f=3.5 (4/5 seeds positive at 150k)
- No durable benefit at f=5.0 by 150k
- Cooperation advantage peaks at 50k and declines

Key figure: **Cooperation trajectory over training** — comm vs no-comm for f=3.5 and f=5.0, with seed-level variation shown (fan plot or individual seed lines).

#### §4 The Semantic Window (FLAGSHIP — ~2 pages)

This is the paper's core contribution. Present as a sequence of evidence:

**4a. Separate-training channel controls (50k):**
- Learned >> random >> zero at 50k
- Content matters during initial learning
- Table: P(C|f=3.5) by channel mode at 50k

**4b. Same-checkpoint continuation controls (50k→150k):**
- From identical learned 50k policies, different channels lead to similar 150k performance
- Content stops mattering after the semantic window closes
- Table: P(C|f=3.5) by channel mode at 50k (identical), 100k, 150k (converged)

**4c. Mute experiment:**
- Muting after 50k → worse than never communicating
- Communication creates structural dependence, not internalized knowledge
- Table or bar chart: comm (0.548), no-comm (0.427), muted (0.367)

Key figure: **Semantic window diagram** — a timeline showing (top) separate-training controls diverging by 50k, (bottom) same-checkpoint continuations converging by 150k, with the "semantic window" shaded in between.

#### §5 Why Aggregate Endpoint Metrics Are Misleading (~1.5 pages)

**5a. Per-sender disaggregation:**
- Aggregate token effect near zero (−0.018) but per-sender effects are 0.157
- Table: aggregate vs per-sender effects at 50k, 100k, 150k
- Explanation: private codes with opposite polarities cancel in aggregate

**5b. Alignment dynamics:**
- Alignment can emerge (seed 303: 2:2→4:0) or collapse (seed 505: 4:0→2:2)
- Per-sender causal effects persist even without group alignment

Key figure: **Sender polarity heatmap** — 5 seeds × 3 checkpoints, showing each agent's polarity (color-coded ±). Visualizes alignment emergence, persistence, and collapse.

#### §6 Discussion (~1 page)

- What we showed: communication creates a transient learning scaffold, not a permanent protocol
- Implications for emergent communication research: endpoint evaluation is necessary but not sufficient
- Per-sender analysis as a general methodological contribution
- Limitations: single game, binary vocabulary, 4 agents, PPO-specific training coupling
- Future work: larger vocabulary, heterogeneous agents, other social dilemmas

### Appendix

- **A.** Full experimental parameters (table of all hyperparameters)
- **B.** Per-seed breakdown tables for all main results
- **C.** Annealed vs unannealed trajectory comparison
- **D.** Cross-play analysis
- **E.** PLRNN attractor analysis (if relevant)
- **F.** Welfare-weighted metrics

### Key Figures (6–7 for main paper)

1. **Agent architecture + game diagram** (§2)
2. **Cooperation trajectory: comm vs no-comm** (§3) — lines with seed variation
3. **Semantic window diagram** (§4) — the flagship figure
4. **Mute experiment bar chart** (§4c) — comm vs no-comm vs muted
5. **Per-sender vs aggregate effects** (§5a) — paired comparison
6. **Sender polarity heatmap** (§5b) — alignment dynamics across seeds
7. **Same-checkpoint continuation trajectories** (§4b) — 4 channel modes diverging from identical start

### Writing Process

1. **Start with §4 (flagship).** Write the semantic window section first — this is the paper's reason for existing. Get the narrative and figures right before anything else.
2. **Then §5 (aggregate metrics).** This is the methodological contribution — straightforward to write once §4 is done.
3. **Then §3 (main result).** The basic comm-helps finding. Set up the context for §4.
4. **Then §2 (setup).** Write the methods section once you know exactly what details §3–5 need.
5. **Then §1 (intro) and §6 (discussion).** Frame the story after the technical sections are stable.
6. **Appendix last.** Dump supporting tables and analyses.

---

## File Locations Reference

| File | Role |
|---|---|
| `src/experiments_pgg_v0/run_phase3_seed_expansion.py` | Training launcher (use for all new training runs) |
| `src/analysis/run_phase3_trimmed_eval.py` | Evaluation meta-runner (suite + crossplay) |
| `src/analysis/aggregate_phase3_report.py` | Aggregation into summary tables |
| `src/analysis/summarize_phase3_channel_controls.py` | Channel control comparison tables |
| `src/analysis/summarize_phase3_welfare.py` | Welfare-weighted summary |
| `src/analysis/evaluate_regime_conditional.py` | Core evaluator |
| `src/analysis/plot_phase3_fragmentation.py` | Fragmentation figures |
| `src/analysis/plot_phase3_annealed_vs_unannealed.py` | Trajectory comparison plots |
| `scripts/run_phase3_next_steps_suite.sh` | Previous orchestration script (reference for patterns) |

### Existing Data Directories

| Directory | Contents | Seeds |
|---|---|---|
| `outputs/train/phase3_annealed_trimmed/` | Learned 50k checkpoints (cond1+cond2) | 5 |
| `outputs/train/phase3_annealed_ext150k_5seeds/` | Learned 150k (cond1+cond2) | 5 |
| `outputs/eval/phase3_annealed_ext150k_5seeds/` | Eval of learned 150k | 5 |
| `outputs/train/phase3_mute_after50k_ext150k_3seeds/` | Mute-after-50k training | 3 |
| `outputs/eval/phase3_mute_after50k_ext150k_3seeds/` | Eval of mute experiment | 3 |
| `outputs/train/phase3_channel_controls_50k/` | Separate-training controls at 50k | 3 |
| `outputs/eval/phase3_channel_controls_50k/` | Eval of separate-training controls | 3 |
| `outputs/train/phase3_channel_controls_ext150k_3seeds/` | Continuation controls (from separate-training 50k) | 3 |
| `outputs/eval/phase3_channel_controls_ext150k_3seeds/` | Eval of continuation controls | 3 |
| `outputs/eval/phase3_compare/` | Trajectory comparison (annealed vs unannealed) | 2 |

### New Data Directories (to be created)

| Directory | Contents | Seeds |
|---|---|---|
| `outputs/train/phase3_sameckpt_continuation_5seeds/` | Same-checkpoint continuation training | 5 |
| `outputs/eval/phase3_sameckpt_continuation_5seeds/` | Eval of same-checkpoint continuation | 5 |
| `outputs/eval/phase3_stats/` | Statistical analysis outputs | — |

---

## Sequencing

```
Experiment 1 (same-checkpoint continuation)
├── Train: ~18h (3 modes × 5 seeds, parallelized)
├── Eval: ~4h (standard pipeline)
└── Aggregate: ~10min (channel control summary)

Experiment 2 (statistical analysis)  ← can start immediately on existing data
├── 2A: Bootstrap CIs on 5-seed main results
├── 2B: Seed-level delta tables
└── 2C: Significance conventions

Experiment 3 (interop test)  ← can start immediately
└── Sender-slot permutation on 2-3 seeds at 150k

Paper writing  ← start §4 as soon as Experiment 1 results are in
├── §4 Semantic window (flagship)
├── §5 Aggregate metrics
├── §3 Main result
├── §2 Setup
├── §1 + §6 Framing
└── Appendix
```

Experiments 2 and 3 can run in parallel with Experiment 1 training. Paper writing (§4) should begin as soon as Experiment 1 evaluation is complete, since the same-checkpoint continuation results are the paper's flagship evidence.

---

## Training Hyperparameters (for reference)

All training runs use identical hyperparameters:

| Parameter | Value |
|---|---|
| Algorithm | PPO with GAE |
| γ (discount) | 0.99 |
| λ (GAE) | 0.95 |
| clip ratio | 0.2 |
| lr (initial) | 3e-4 |
| lr (final) | 1e-5 |
| lr schedule | cosine |
| entropy_coeff (initial) | 0.01 |
| entropy_coeff (final) | 0.001 |
| entropy schedule | linear |
| msg_entropy_coeff (initial) | 0.01 |
| msg_entropy_coeff (final) | 0.0 |
| reward_scale | 20.0 |
| hidden_size | 64 |
| vocab_size | 2 |
| T (episode length) | 100 |
| F (regime levels) | {0.5, 1.5, 2.5, 3.5, 5.0} |
| ρ (regime switch prob) | 0.05 |
| σ (signal noise) | 0.5 per agent |
| ε (tremble) | 0.05 |
| checkpoint interval | 50000 episodes |
| regime log interval | 5000 episodes |

### Environment Variables (required on macOS)

```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
```
