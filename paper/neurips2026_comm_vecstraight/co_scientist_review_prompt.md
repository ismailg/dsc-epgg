# Co-Scientist Deep Review Prompt

**Role:** You are a senior reviewer and co-scientist with deep expertise in multi-agent reinforcement learning, emergent communication, game theory, and statistical methodology. Your job is to give a rigorous, honest, and constructive review of this paper before submission. You should think as hard as a Reviewer 2 at NeurIPS would — but also as a helpful collaborator who wants the paper to succeed if the science is sound.

**Target venue:** NeurIPS 2026 main conference.

---

## Paper summary

**Title:** "Useful Communication Without a Clean Code in a Hidden-Regime Social Dilemma"

**Setting:** Four agents play an Extended Public Goods Game (EPGG) over 100-round episodes. A hidden multiplier f switches stochastically among {0.5, 1.5, 2.5, 3.5, 5.0} with hazard rate 0.05. Each agent observes a noisy private signal f_hat = f + N(0, 0.25). In the communication condition, agents simultaneously broadcast a binary message m_i in {0,1} before acting (cooperate or defect). Trained with independent PPO, 15 seeds, checkpoints at 25k/50k/100k/150k episodes.

**Central claim:** Communication in this environment is useful but does not produce a clean shared symbolic code. Instead, the benefit decomposes into three layers:
1. **Channel presence** (dominant): having a varying input in the message slots provides the largest cooperation gain.
2. **Coarse shared structure** (secondary): count-of-ones or any-token summaries add smaller regime-dependent value.
3. **Rich sender-specific codes** (tertiary): within a trained seed, sender identity produces enormous behavioral effects (up to 98 pp within-seed gaps), but these codes are seed-specific private conventions that do not transfer across independently trained populations.

---

## Complete data package

### A. Base communication gap (15 seeds, greedy eval)

| Checkpoint | f=3.5 comm | f=3.5 no-comm | Gap | f=5.0 comm | f=5.0 no-comm | Gap |
|------------|-----------|--------------|-----|-----------|--------------|-----|
| 25k | 54.7% | 55.1% | -0.5 pp | 72.7% | 60.9% | +11.7 pp |
| 50k | 50.1% | 46.6% | +3.6 pp | 60.5% | 58.5% | +2.2 pp |
| 100k | 49.9% | 35.5% | +14.4 pp | 68.8% | 55.5% | +13.4 pp |
| 150k | 58.9% | 42.4% | +16.4 pp | 75.2% | 56.4% | +18.9 pp |

Note: comm and no-comm are SEPARATELY trained trajectories with independent PPO. The gap mixes message effects with broader training-path differences. Paper acknowledges this explicitly.

### B. Frozen endpoint interventions at 150k (15 seeds)

Freeze all weights, then perturb the message stream at evaluation time:

| Intervention | f=3.5 coop | vs learned | f=5.0 coop | vs learned |
|-------------|-----------|-----------|-----------|-----------|
| learned (none) | 0.587 | — | 0.751 | — |
| permute_slots | 0.548 | -3.9 pp | 0.754 | +0.2 pp |
| sender_shuffle | 0.546 | -4.1 pp | 0.748 | -0.4 pp |
| public_marginal | 0.545 | -4.2 pp | 0.740 | -1.2 pp |
| public_random | 0.541 | -4.5 pp | 0.730 | -2.1 pp |
| marginal | 0.538 | -4.8 pp | 0.733 | -1.9 pp |
| indep_random | 0.530 | -5.7 pp | 0.725 | -2.6 pp |
| fixed1 | 0.528 | -5.9 pp | **0.779** | **+2.8 pp** |
| fixed0 | 0.506 | -8.1 pp | 0.673 | -7.9 pp |
| zeros | 0.580 | -0.6 pp | 0.660 | -9.1 pp |
| no-comm baseline | 0.423 | -16.4 pp | 0.562 | -18.9 pp |

Key anomaly: fixed1 (constant token 1) outperforms learned at f=5.0 by +2.8 pp.
Note: public_marginal (token-rate matched) is only +0.3 pp above public_random at f=3.5 and +1.0 pp at f=5.0.

### C. Sender-causal probes at 150k (15 seeds)

Force one sender's token, measure receiver behavior shift:

| f | Mean |delta p(coop)| | Mean action flip rate |
|---|---------------------|----------------------|
| 0.5 | 0.002 | ~0% |
| 1.5 | 0.011 | ~1% |
| 2.5 | 0.089 | ~5% |
| 3.5 | 0.155 | ~9-10% |
| 5.0 | 0.206 | ~11-12% |

Self-effects are small and slightly negative; nonself effects are positive and large. Effects are heterogeneous across sender-receiver pairs.

### D. Low-dimensional mechanism analysis (15 seeds, 150k frozen traces)

**Count-of-ones response:**

| f | 0 ones | 1 one | 2 ones | 3 ones | any_token effect |
|---|--------|-------|--------|--------|-----------------|
| 3.5 | 0.589 | 0.584 | 0.584 | 0.646 | +0.6 pp |
| 5.0 | 0.713 | 0.775 | 0.748 | 0.832 | +4.8 pp |

Note: non-monotone at f=5.0 (count=2 < count=1).

**Within-seed sender-identity effects (holding count fixed):**

| f | count | Median within-seed pattern gap | Seeds with >10pp gap |
|---|-------|-------------------------------|---------------------|
| 3.5 | 1 | 98.4 pp | 100% |
| 5.0 | 1 | 87.0 pp | 92% |

**Cross-seed surrogate models (grouped CV by training seed):**

| f | Model | Log loss | vs count-only model |
|---|-------|---------|-------------------|
| 5.0 | history_any | 0.579 | +0.008 better |
| 5.0 | history_only | 0.581 | +0.006 better |
| 5.0 | history_count | 0.587 | (reference) |
| 5.0 | history_sender_bits | 0.607 | -0.020 worse |
| 5.0 | history_pattern | 0.695 | -0.108 worse |

Richer sender-indexed features make cross-seed prediction WORSE. Sender-specific codes are seed-specific private conventions.

### E. Observation noise sweep (evaluation-time, 15 seeds, 150k)

| Sigma | f=5.0 comm | f=5.0 no-comm | Comm gap |
|-------|-----------|--------------|---------|
| 0.0 | 0.759 | 0.586 | +17.3 pp |
| 0.25 | 0.754 | 0.580 | +17.4 pp |
| 0.5 | 0.751 | 0.562 | +18.9 pp |
| 1.0 | 0.731 | 0.527 | +20.4 pp |

Comm gap grows with noise at f=5.0, consistent with information-transfer interpretation.

### F. Message x History grid (evaluation-time, 9 msg conditions x 8 history ablations, 15 seeds, 150k)

Most striking results at f=5.0 with learned messages:

| History ablation | f=5.0 coop | vs intact (0.751) |
|-----------------|-----------|------------------|
| intact | 0.751 | — |
| zero_last_action | 0.831 | +7.9 pp |
| zero_ewma | 0.845 | +9.3 pp |
| zero_temporal | 0.838 | +8.6 pp |
| clamp_temporal_high | 0.571 | -18.0 pp |
| clamp_ewma_high | 0.626 | -12.5 pp |

Zeroing temporal history features INCREASES cooperation at f=5.0 (agents lose ability to conditionally defect). Clamping high CRUSHES cooperation (free-riding). At f=3.5, the pattern reverses — zeroing temporal features hurts.

### G. Same-checkpoint continuation controls (training-time, 15 seeds)

Branch from a checkpoint and continue training with altered message channel:

| Branch | f=3.5 at 150k | f=5.0 at 150k | vs learned (3.5) | vs learned (5.0) |
|--------|-------------|-------------|-----------------|-----------------|
| Learned 150k reference | 58.9% | 75.2% | — | — |
| sender_shuffle_50k | 73.0% | 81.0% | +14.1 pp (p=0.072) | +5.8 pp (p=0.253) |
| sender_shuffle_100k | 70.7% | 82.9% | +11.8 pp (p=0.131) | +7.7 pp (p=0.040) |
| fixed0_50k | 39.9% | 53.5% | -19.0 pp (p=0.013) | -21.7 pp (p<0.001) |
| fixed0_100k | 48.3% | 61.7% | -10.6 pp (p=0.118) | -13.5 pp (p=0.012) |
| public_random_50k | 57.1% | 60.5% | -1.8 pp (p=0.824) | -14.7 pp (p=0.049) |
| uniform_50k | 70.2% | 90.9% | +11.3 pp (p=0.271) | +15.7 pp (p=0.008) |
| No-comm 150k baseline | 42.4% | 56.4% | — | — |

P-values are sign-flip tests (exact, 15 seeds).

### H. From-scratch exogenous channel controls (training-time, 15 seeds, 150k)

Train from scratch with different channel types (not branched from learned checkpoints):

| Mode | f=3.5 (mean +/- SEM) | f=5.0 (mean +/- SEM) | vs learned (3.5/5.0) | vs no-comm (3.5/5.0) |
|------|---------------------|---------------------|---------------------|---------------------|
| learned | 58.9 +/- 6.6% | 75.2 +/- 3.5% | — | +16.4 / +18.9 pp |
| uniform | 74.8 +/- 5.9% | 92.3 +/- 2.6% | +15.9 / +17.1 pp | +32.4 / +35.9 pp |
| public_random | 64.9 +/- 6.2% | 63.9 +/- 4.7% | +6.0 / -11.3 pp | +22.5 / +7.5 pp |
| fixed0 (Hetzner) | 49.9 +/- 5.8% | 49.4 +/- 5.3% | -9.0 / -25.7 pp | +7.5 / -7.0 pp |
| fixed1 (qx6) | 44.0 +/- 5.1% | 46.9 +/- 3.4% | -14.8 / -28.3 pp | +1.6 / -9.5 pp |
| fixed0_qx6_replica | 46.3 +/- 7.0% | 54.3 +/- 5.3% | -12.6 / -20.9 pp | +3.9 / -2.1 pp |
| no_comm | 42.4 +/- 7.2% | 56.4 +/- 5.1% | — | — |

Key: uniform >> learned at both f values. Constant channels harmful, even below no-comm at f=5.0.
Note: Fixed0 Hetzner (49.4%) vs fixed0 qx6 replica (54.3%) at f=5.0 — same intervention, different hosts, 4.9 pp gap.

---

## What I need from you

Think deeply and provide:

### 1. Hard questions about methodology and identification

- The base communication gap compares SEPARATELY trained trajectories. How confident can we be this reflects communication effects rather than training-path divergence? What would strengthen identification?
- The frozen endpoint interventions are evaluation-time only. They test the policy's online dependence on its message input but cannot measure the channel's training-time value. Are the authors clear enough about this distinction?
- The sender-causal probes force one sender's token while holding everything else fixed. But in a multi-agent system, the counterfactual is not well-defined because other agents' messages are themselves endogenous. How does this affect the causal interpretation?
- The continuation controls branch from a SHARED checkpoint and continue with altered channels. But the checkpoint was TRAINED with learned messages — so continuation is not symmetric (the initial policy already "expects" learned messages). Does this bias the comparison?
- The exogenous from-scratch controls use different compute hosts (Hetzner vs qx6). The same intervention (fixed0) gives different magnitudes on different hosts. How serious is this for the claims?

### 2. Statistical adequacy and rigor

- n=15 seeds throughout. Is this sufficient for the claims being made? The SEMs are large (5-7 pp in many cases). Which specific comparisons have adequate power, and which are underpowered?
- The sign-flip p-values in the continuation table: many are >0.05 (e.g., sender_shuffle at 0.072, 0.131, 0.253). The paper seems to be making directional claims from these. Is that defensible?
- No formal multiple-comparison corrections are applied across the many intervention conditions. Should they be?
- The surrogate model comparison uses grouped cross-validation by seed. Is 15 seeds (= 15 folds) sufficient for stable cross-validated log-loss comparisons?
- The welfare column in the exogenous summary CSV shows 0.0 for learned and no_comm conditions. If this is a data pipeline gap, does it affect any claims?

### 3. The uniform puzzle — why does independent random noise beat learned communication?

This is the most surprising and important result: uniform (independent random bits per sender) reaches 92.3% cooperation at f=5.0 versus 75.2% for learned communication. Think hard about possible explanations:

- Is this a training dynamics effect? (Independent random bits might create a more exploration-friendly gradient landscape during training.)
- Is this an implicit coordination device? (Each sender gets a unique per-timestep random tag, effectively giving the group a shared random seed that enables correlated strategies without explicit semantic content.)
- Could this be an artifact of the binary vocabulary? (With only 2 tokens, learned messages may converge to a narrow behavioral repertoire, while uniform ensures maximal token entropy throughout training.)
- Does this relate to the non-stationarity of independent learning? (Learned messages create non-stationarity because the meaning of messages drifts during training; random messages are stationary by construction.)
- Is there a connection to the literature on "cheap talk" vs "costly signaling"? Random messages are the ultimate cheap talk — yet they work better than learned ones.
- What does this imply for the three-layer decomposition? If layer 1 (channel variation) alone explains the uniform result, then are layers 2 and 3 doing negative work?

### 4. Literature positioning

The paper should be situated relative to:

- **Emergent communication benchmarks:** How does this compare to the standard Lewis game / referential game literature (Lazaridou et al., 2017; Havrylov & Titov, 2017; etc.)? Those settings typically assume compositionality as the goal. This paper argues against clean codes as the headline. Is this a genuine contribution or just an artifact of the PGG setting?
- **Communication in social dilemmas:** Eccles et al. (2019) "Biases for Emergent Communication in Multi-agent Reinforcement Learning"; Jaques et al. (2019) "Social Influence as Intrinsic Motivation"; Kim et al. (2021) "Communication for Implicit Coordination". How does this paper's finding that uniform noise > learned messages relate?
- **Information aggregation in teams:** The noise sweep result (comm gap grows with noise at f=5.0) suggests information aggregation. How does this relate to the "wisdom of crowds" / information aggregation literature in mechanism design?
- **Cheap talk theory:** Crawford & Sobel (1982) showed cheap talk can be informative in equilibrium. But random noise is not equilibrium cheap talk. Does the game theory literature have anything to say about why random signals might help coordination?
- **Convention formation:** Lewis (1969) conventions, Hu et al. (2020) "Other-Play", Bullard et al. (2021) "Quasi-Equivalence". The seed-specific codes finding connects directly here.

### 5. Honest submission readiness assessment

Consider:
- Is the paper's central claim novel and interesting enough for NeurIPS main?
- Is the evidence sufficient, or are there critical gaps?
- Is the paper too long / too detailed for a conference format? (The Quarto source is ~1200 lines with extensive inline computation.)
- What are the most likely reasons a NeurIPS reviewer would reject?
- What are the 2-3 highest-impact improvements that could be made before submission?
- Would a workshop (LaReL, emergent communication workshop) be a better first venue?
- Rate submission readiness on a 1-10 scale with justification.

### 6. Specific manuscript feedback

- The abstract is very long (one continuous paragraph with many inline statistics). Is this effective?
- The paper reports many "as of March 29/30, 2026" timestamps. These are artifacts of an evolving research log. Should they be removed for submission?
- The manuscript title says "Useful Communication Without a Clean Code" but the exogenous results show uniform (pure noise) BEATS learned communication. Does the framing need to change?
- The three-layer decomposition is the paper's organizing framework. Is it the right one given that constant channels (which also have "channel presence") are HARMFUL?
- The limitations section is honest but long. For NeurIPS, should it be tightened?

---

## Important context

This paper was previously reviewed by a co-scientist in early March 2026, when it was based on the older staged-family pipeline (5 seeds). That review identified:
- Statistical methods were overconfident (bootstrap p-values unreliable at n=5)
- Per-sender effects were observational, not causal
- Recommended 15-20 seeds and exact sign-flip tests

The current version addresses those concerns (15 seeds, sign-flip tests, more cautious framing). But the scientific story has also changed fundamentally — the old staged family showed communication HURTING at f=5.0 by 150k, which is now reversed. The paper needs to own this training-path dependence honestly.

Please be thorough, honest, and specific. I would rather hear hard truths now than get desk-rejected.
