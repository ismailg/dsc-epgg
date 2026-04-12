# Phase-3 Vecstraight Didactic Overview

*Last updated: 2026-04-12*

This note explains the completed vecstraight rerun results in plain terms. It covers the
**new straight vectorized family** only (`training_family = phase3_vecstraight`, repo =
`dsc-epgg-vectorized`). If you want the old staged-family story, that lives in the original
`dsc-epgg` repo.

This file is interpretive, not the live status ledger. For current run state and ownership, use
[`PHASE3_VECSTRAIGHT_NEXT_STEPS.md`](PHASE3_VECSTRAIGHT_NEXT_STEPS.md) and
[`PHASE3_VECSTRAIGHT_PAPER_TODO.md`](PHASE3_VECSTRAIGHT_PAPER_TODO.md). For historical
implementation plans and archived task notes, use [`docs/archive/README.md`](docs/archive/README.md).

**What this covers:** Ten completed evaluation/training stages — frozen endpoints, sender-causal
probes, expanded intervention suite (with SEMs and public_marginal control), observation-noise
sweep, message-history grid, low-dimensional mechanism analysis (count-of-ones response,
sender-identity effects, surrogate model comparison), the corrected same-checkpoint continuation
ladder, the historical from-scratch exogenous-channel controls, the qx6 loss-switch repair
(quantifying the auxiliary-loss confound), the Hetzner comm x history training factorial, and the
clean direct Hetzner `msg_source_mode` family.
**What this still does not cover:** The joint evaluation grid for the comm x history factorial
cells and the observability/noise sweep remain pending on the evaluation side.

**April 12 clean training-time msg-source update:** The direct exogenous question is now answered
by a native training-time family rather than the older hybrid forced-channel path. In the completed
Hetzner `msg_source_mode` family, `uniform` finishes essentially tied with learned communication:
`75.9%` versus `77.7%` at `f=3.5` and `91.6%` versus `90.2%` at `f=5.0`. But `public_random` is
much weaker: `36.9%` and `57.6%`, close to the no-comm baseline rather than the learned channel.
The constant-channel controls are worse still: `fixed0` and `fixed1` both finish below learned at
both focal multipliers, and below no-comm at `f=5.0`.

That means the cleaner story is not "learned communication is uniquely necessary," but also not
"any random cue works." What seems to matter is rich variable slotwise input that receivers can
combine with temporal context. A single shared random bit is not enough, and a constant token is
actively harmful.

The older March 30 continuation ladder and qx6 loss-switch repair are still useful historically:
they showed that constant channels are harmful, that auxiliary-loss mismatches were real, and that
the earlier forced-channel evidence should not be over-read. But the main training-time
learned-versus-exogenous question now has a cleaner answer from the direct `msg_source_mode`
family.

Key status files:
- [`sameckpt_continuation_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_paper_pivot_20260329_status/sameckpt_continuations/sameckpt_continuation_summary.md)
- [`sameckpt_continuation_summary.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_lossswitch_controls_status_20260401/report/sameckpt_continuation_summary.csv)
- [`clean_msgsource_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_clean_msgsource_status_20260412/report/clean_msgsource_summary.md)

---

## Quick Glossary

| Term | Meaning |
| --- | --- |
| **learned** | Agents use their trained communication policy — messages flow naturally. Called `none` in the CSVs (meaning "no intervention"). Called `cond1` when distinguishing from no-comm. |
| **no-comm** (baseline) | Agents trained **without** a communication channel (no messages, otherwise identical). Called `cond2` or `baseline_none` in the CSVs. |
| **comm gap** | Learned cooperation rate minus no-comm cooperation rate. Positive = communication helped. |
| **f** | The hidden multiplier in the public goods game. Higher f makes cooperation more collectively valuable. |
| **f=3.5** | "Mixed-motive" regime — cooperation is collectively better but individual defection still tempting |
| **f=5.0** | "Cooperation-dominant" regime — cooperation clearly pays, defection costly |
| **frozen** | Freeze all network weights, then test what happens when you tamper with messages |
| **sender-causal** | Force one specific sender to send a fixed token, measure how much each receiver's behavior changes |
| **fixed0 vs zeros** | **fixed0** forces every sender to broadcast token 0 — receivers see one-hot `[1,0]` per sender (a constant *signal*). **zeros** zeroes out the entire message observation — receivers see `[0,0]` per sender (no signal at all). These are different interventions with different results. |

---

## The Headline

Agents with a communication channel cooperate substantially more than agents without one — and
this holds at **both** f=3.5 and f=5.0 by 150k episodes.

| Checkpoint | f=3.5 comm gap | f=5.0 comm gap |
| --- | ---: | ---: |
| 50k | +3.6 pp | +2.2 pp |
| 150k | **+16.4 pp** | **+18.9 pp** |

This is the single most important fact about vecstraight. In the old staged family, the late
f=5.0 gap was *negative* (communication hurt). Here it is strongly positive. **The old f=5.0
reversal does not survive the new training pipeline.**

Sources: intervention_suite_summary.csv in each frozen suite directory.

---

## The Questions

The rest of this document answers ten questions, each with its own evaluation stage:

1. **Is the channel being used at all?** → Frozen endpoint tests
2. **Does message content matter, or just the presence of a channel?** → Frozen perturbation tests
3. **Can a single sender's message causally change what others do?** → Sender-causal probes
4. **Is the comm gap statistically robust? Does it grow over training?** → Expanded intervention suite
5. **Does observation noise change the story?** → Noise sweep
6. **Which observation features drive cooperation?** → Message-history grid
7. **Is there a low-dimensional structure to how messages work?** → Count-of-ones, sender-identity, surrogate models
8. **Does token-rate matching explain the comm benefit?** → Public marginal control
9. **What do the clean direct exogenous controls show?** → Native training-time `msg_source_mode` family
10. **Does communication substitute for temporal history, or require it?** → From-scratch comm × history factorial

---

## Question 1 & 2: Frozen Endpoint Tests

### What is this test?

Take the fully trained agents. **Freeze all weights** (no more learning). Then play many
episodes under different message conditions and measure cooperation rates. If the agents
truly learned to use messages, tampering with the message stream should change behavior.

### The interventions, explained

| Intervention | What it does | What it tests |
| --- | --- | --- |
| **learned** (`none` in CSV) | Messages flow naturally from learned policy | What the agents actually do |
| **no-comm** (`baseline_none`) | Remove the channel entirely | How much does having *any* channel help? |
| **sender_shuffle** | Each sender's token is drawn from their own empirical frequency, but independently of the current state | Does content matter, or just the statistical distribution? |
| **permute_slots** | Swap which sender's message goes to which slot | Do agents care *who* said what? |
| **public_random** | All senders broadcast the same random bit | Does a shared random signal help coordinate? |
| **indep_random** | Each sender gets an independent random bit | Does any random signal help? |
| **zeros** | Zero out the message observation slots entirely — receivers see `[0,0]` per sender | What happens if the message input is completely blank? |
| **fixed0** | Force every sender to broadcast token 0 — receivers see one-hot `[1,0]` per sender | What happens with a constant "0" signal? |
| **fixed1** | Force every sender to broadcast token 1 — receivers see one-hot `[0,1]` per sender | What happens with a constant "1" signal? |
| **public_marginal** | One shared random token sampled from the learned marginal P(token) for all senders | Does matching the global token rate explain the benefit? |
| **marginal** | Each sender independently samples from the pooled marginal, breaking state-dependence | Does the per-sender frequency structure matter? |

### Results at 50k: Communication is starting to matter

| | f=3.5 | f=5.0 |
| --- | ---: | ---: |
| **learned** | 0.502 | 0.605 |
| **no-comm** (baseline) | 0.465 | 0.583 |
| **gap** | +3.6 pp | +2.2 pp |

Perturbations at 50k show small effects. The policy is beginning to rely on messages, but
the dependence is still modest. Think of 50k as an **early dependence stage** — the agents
are starting to listen, but messages don't yet dominate their decisions.

### Results at 150k: Communication now strongly matters

Raw cooperation rates:

| | f=3.5 | f=5.0 |
| --- | ---: | ---: |
| **learned** | 0.587 | 0.751 |
| **no-comm** (baseline) | 0.423 | 0.562 |
| **gap** | +16.4 pp | +18.9 pp |

What happens when you tamper with the message stream at 150k:

| Intervention | f=3.5 coop | vs. learned | f=5.0 coop | vs. learned |
| --- | ---: | ---: | ---: | ---: |
| **learned** | 0.587 | — | 0.751 | — |
| permute_slots | 0.548 | −3.9 pp | 0.754 | +0.2 pp |
| sender_shuffle | 0.546 | −4.1 pp | 0.748 | −0.4 pp |
| public_marginal | 0.545 | −4.2 pp | 0.740 | −1.2 pp |
| public_random | 0.541 | −4.5 pp | 0.730 | −2.1 pp |
| marginal | 0.538 | −4.8 pp | 0.733 | −1.9 pp |
| indep_random | 0.530 | −5.7 pp | 0.725 | −2.6 pp |
| fixed1 | 0.528 | −5.9 pp | **0.779** | **+2.8 pp** |
| fixed0 | 0.506 | −8.1 pp | 0.673 | −7.9 pp |
| zeros | 0.580 | −0.6 pp | 0.660 | −9.1 pp |
| **no-comm** (baseline) | 0.423 | −16.4 pp | 0.562 | −18.9 pp |

### How to read this table

**The channel is not irrelevant.** If agents ignored messages entirely, every row would show
the same cooperation rate. They don't — perturbations create real behavioral differences.

**But the dependence is messy, not clean.** A "clean symbolic language" would mean: learned
messages are the *only* good input, and every perturbation destroys performance equally. That
is not what we see. Instead:

- **Most disruptive:** fixed0 and zeros at f=5.0 (−8 to −9 pp). Constant silence or a
  constant "0" signal hurts. Note fixed0 ≠ zeros: fixed0 sends `[1,0]` (a definite signal),
  zeros sends `[0,0]` (blank input). Both hurt, but differently.
- **Mildly disruptive:** random, shuffled, or marginal-matched messages (−2 to −6 pp).
  Noise is worse than learned messages, but not catastrophic.
- **Barely disruptive:** sender_shuffle and permute_slots at f=5.0 (< 1 pp change).
  Swapping who-said-what barely matters here.
- **Actually better:** fixed1 at f=5.0 (+2.8 pp). Forcing "always 1" *outperforms* the
  natural learned stream. This means the learned protocol is not globally optimal — a simpler
  constant signal works better in this regime.
- **public_marginal ≈ public_random ≈ marginal:** Token-rate matching (using the learned
  global frequency of 0s vs 1s rather than a fair coin) gives at most +0.3 pp over plain
  public_random. The comm benefit is *not* explained by the aggregate token distribution.

**Bottom line:** The agents learned to depend on the channel in a real but imperfect way. The
natural message stream carries useful structure, but the protocol is not a tight, optimal code.

Key files:
- [`checkpoint_suite_main.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_frozen50k_15seeds_local_20260325/suite/checkpoint_suite_main.csv) (50k)
- [`checkpoint_suite_main.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_frozen150k_15seeds_local_20260325/suite/checkpoint_suite_main.csv) (150k)
- [`intervention_suite_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_frozen150k_15seeds_local_20260325/report/intervention_suite_summary.md)

---

## Question 3: Sender-Causal Probes

### What is this test?

The frozen tests above change *all* messages at once. The sender-causal probe is more
surgical: pick **one** sender, force their token to 0 or 1, and measure how much each
**other** agent's cooperation probability shifts. This tells you whether individual messages
have real causal power over specific receivers.

### What "causal" means here

We measure two things:

- **|delta p(cooperate)|** — How much does the receiver's probability of cooperating change
  when the sender's message is forced? Bigger = more causal influence.
- **action flip rate** — What fraction of the time does the receiver literally switch from
  cooperate to defect (or vice versa) because of the forced message? This is stricter than
  a probability shift.

### Results: Messages have real causal bite, concentrated in high-f regimes

| Regime (f) | Mean |delta p(coop)| | Interpretation |
| --- | ---: | --- |
| 0.5 | 0.002 | Essentially zero — messages don't matter here |
| 1.5 | 0.011 | Negligible |
| 2.5 | 0.089 | Starting to matter |
| **3.5** | **0.155** | **Substantial — a forced token shifts coop probability by ~15 pp** |
| **5.0** | **0.206** | **Large — ~21 pp shift** |

Action flip rates (how often does the receiver literally change their action):

| Regime | Flip rate |
| --- | ---: |
| f=3.5 | ~9–10% |
| f=5.0 | ~11–12% |

So roughly 1 in 10 decisions actually flips because of what one sender said. That is
meaningful causal influence.

### Who influences whom?

There is an important asymmetry in the effects:

- **Nonself effects** (sender → other agents): clearly positive at f=3.5 and f=5.0.
  Messages are genuinely influencing *other* agents toward cooperation.
- **Self effects** (sender → own behavior): smaller and slightly negative on average.
  The sender's own action is not simply mirroring their message.

This is consistent with the interpretation that messages are doing real communicative work
(changing others' behavior), not just reflecting the sender's own intent.

**However:** the effects are heterogeneous across sender-receiver pairs. Some pairs show
strong influence, others weak. This is not yet a picture of a uniform shared language —
it is more like an emergent, asymmetric influence network.

Key files:
- [`sender_causal_matrix.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_sender_causal_150k_15seeds_local_20260325/sender_causal_matrix.csv)
- [`sender_causal_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_sender_causal_150k_15seeds_local_20260325/report/sender_causal_summary.md)

---

## Question 4: Expanded Intervention Suite (with uncertainty)

### What is this test?

The same frozen-endpoint tests as before, but now reporting **standard errors of the mean
(SEM)** across 15 seeds and adding three new interventions: `marginal` (messages drawn from
the learned marginal distribution, destroying state-conditioned content), `zeros` (all message
observation slots set to 0.0 — *not* the same as `fixed0` which sends one-hot `[1,0]`), and
`public_marginal` (a shared random bit sampled from the learned global token rate). This
lets us say whether observed differences are real or within noise.

### What the expanded results confirm

At 150k, with SEMs in hand, the key comparisons:

| Comparison | f=3.5 | f=5.0 |
| --- | --- | --- |
| **comm gap** (learned vs baseline) | +16.4 pp (SEM overlap: none) | +18.9 pp (SEM overlap: none) |
| **fixed0 drop** (learned vs fixed0) | −8.1 pp | −7.9 pp |
| **fixed1 boost** (learned vs fixed1) | −5.9 pp | **+2.8 pp** |
| **marginal drop** (learned vs marginal) | −4.8 pp | −1.9 pp |
| **public_marginal drop** (learned vs public_marginal) | −4.2 pp | −1.2 pp |

**In plain English:**
- The comm gap is **statistically robust** — the SEMs don't overlap, so this is a real effect.
- The `marginal` intervention (which preserves the overall frequency of 0s and 1s but breaks
  the connection to the hidden state) reduces cooperation, confirming that agents use the
  *content* of messages, not just the fact that some bits are flowing.
- The `public_marginal` control (matching the learned token rate, but shared across senders)
  is only marginally better than plain `public_random` (+0.3 pp at f=3.5, +1.0 pp at f=5.0).
  **Token-rate matching does not explain the communication benefit.**
- The `fixed1` anomaly at f=5.0 persists: always sending "1" beats the learned protocol.
  This tells us the learned protocol is suboptimal — agents have not converged on the best
  possible signaling strategy.

### What 50k vs 150k tells us about development

| Intervention | f=5.0 coop at 50k | f=5.0 coop at 150k | Change |
| --- | ---: | ---: | ---: |
| learned | 0.605 | 0.751 | +14.6 pp |
| no-comm (baseline) | 0.583 | 0.562 | −2.1 pp |
| fixed0 | 0.491 | 0.673 | +18.2 pp |
| fixed1 | 0.617 | 0.779 | +16.2 pp |

The no-comm baseline actually *declined* slightly from 50k to 150k, while all comm conditions
improved. This means the communication channel is not just helpful early on — its benefit
*grows* with continued training. The agents increasingly integrate messages into their
decision-making over time.

Key files:
- [`intervention_suite_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_frozen150k_expanded_15seeds_local_20260327/report/intervention_suite_summary.md) (150k expanded)
- [`intervention_suite_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_frozen50k_expanded_15seeds_local_20260327/report/intervention_suite_summary.md) (50k expanded)

---

## Question 5: Does Observation Noise Change the Story?

### What is this test?

During training, agents see a noisy estimate of the true multiplier: f_hat = f + N(0, sigma^2),
with sigma = 0.5 by default. What if we change the noise level at **evaluation time**? If
messages carry information about the hidden state, noisier private signals should make agents
*more* reliant on communication (because their own observations become less reliable).

We tested four noise levels: sigma = 0.0 (perfect info), 0.25, 0.5 (training default), 1.0.

### Results

| Sigma | f=5.0 learned | f=5.0 no-comm | Comm gap |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.759 | 0.586 | +17.3 pp |
| 0.25 | 0.754 | 0.580 | +17.4 pp |
| 0.5 | 0.751 | 0.562 | +18.9 pp |
| 1.0 | 0.731 | 0.527 | +20.4 pp |

| Sigma | f=3.5 learned | f=3.5 no-comm | Comm gap |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.539 | 0.416 | +12.3 pp |
| 0.25 | 0.582 | 0.431 | +15.1 pp |
| 0.5 | 0.587 | 0.423 | +16.4 pp |
| 1.0 | 0.549 | 0.400 | +14.9 pp |

### In plain English

**The communication advantage grows as observations get noisier at f=5.0.** When agents
have perfect information (sigma=0), the comm gap is +17.3 pp. When their private signals are
very noisy (sigma=1.0), the gap widens to +20.4 pp. This is exactly what you would expect if
messages carry genuine information about the hidden state: the worse your private signal, the
more valuable someone else's message becomes.

At f=3.5, the pattern is less monotonic — the gap peaks around the training noise level
(sigma=0.5) and drops slightly at sigma=1.0. This likely reflects a mismatch: agents were
trained at sigma=0.5, so extreme noise at evaluation is out-of-distribution.

**The key takeaway:** The comm benefit is not fragile — it survives across a wide range of
observation quality. And the fact that it *increases* with noise at f=5.0 supports the
interpretation that messages encode regime-relevant information, not just arbitrary coordination
signals.

Key files:
- [`intervention_suite_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_noise_sweep_sigma000_150000_15seeds_local_20260327/report/intervention_suite_summary.md) (sigma=0.0)
- [`intervention_suite_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_noise_sweep_sigma025_150000_15seeds_local_20260327/report/intervention_suite_summary.md) (sigma=0.25)
- [`intervention_suite_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_noise_sweep_sigma050_150000_15seeds_local_20260327/report/intervention_suite_summary.md) (sigma=0.5)
- [`intervention_suite_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_noise_sweep_sigma100_150000_15seeds_local_20260327/report/intervention_suite_summary.md) (sigma=1.0)

---

## Question 6: Which Observation Features Drive Cooperation?

### What is this test?

The **message-history grid** is the most granular dissection of the agents' decision-making.
It crosses 9 message interventions x 8 history-feature ablations, asking: *if we zero out or
clamp specific parts of the observation, how does cooperation change?*

The history features in the observation vector are:
- **coop_fraction (temporal)**: the lagged cooperate/defect fraction (EWMA)
- **last_action**: the agent's own action last round
- **last_coop**: the group cooperation fraction last round
- **ewma**: the EWMA of cooperation

Each can be zeroed (set to 0) or clamped high (set to 1), forcing the agent to see a
specific "story" about past cooperation regardless of what actually happened.

### The most striking results (with learned messages, f=5.0)

| History ablation | f=5.0 coop | vs. learned (0.751) |
| --- | ---: | ---: |
| **learned** (intact history) | 0.751 | — |
| **zero_last_action** | 0.831 | +7.9 pp |
| **zero_ewma** | 0.845 | +9.3 pp |
| **zero_temporal** | 0.838 | +8.6 pp |
| **clamp_temporal_low** | 0.838 | +8.6 pp |
| **clamp_temporal_high** | 0.571 | −18.0 pp |
| **zero_last_coop** | 0.742 | −1.0 pp |
| **clamp_ewma_high** | 0.626 | −12.5 pp |

### In plain English

This produces a surprising and important finding: **zeroing out social history features
*increases* cooperation at f=5.0.**

When agents can see the temporal cooperation history (how cooperatively the group has
been playing), they sometimes use it to *conditionally defect* — essentially a "tit-for-tat
punishment" strategy. When we remove this information, agents cooperate more because they
can no longer condition on low past cooperation to justify defection.

Conversely, **clamping temporal high** (telling agents "everyone has been cooperating 100%")
*crushes* cooperation at f=5.0. This seems paradoxical until you realise the agents may have
learned: "if everyone is already cooperating, I can free-ride." Forced high-coop history
triggers this free-riding impulse.

At f=3.5, the pattern reverses in places — zeroing temporal features *reduces* cooperation
(to ~0.42), while it increases at f=5.0 (to ~0.84). This is the **difficulty-gated**
pattern: the role of each observation feature depends on which regime the game is in.

### How does this interact with messages?

The grid crosses history ablations with message ablations. Comparing across message
conditions for the same history ablation:

| History ablation | f=5.0 coop (learned msgs) | f=5.0 coop (zeros msgs) | Msg effect |
| --- | ---: | ---: | ---: |
| intact | 0.751 | 0.660 | −9.1 pp |
| zero_ewma | 0.845 | 0.727 | −11.7 pp |
| zero_last_action | 0.831 | 0.723 | −10.7 pp |
| clamp_temporal_low | 0.838 | 0.826 | −1.2 pp |

When social history is intact, zeroing messages costs ~9 pp. When social history is also
removed (zero_ewma), messages become *even more important* (−11.7 pp). But when the temporal
signal is clamped low (agents think cooperation has been low), messages barely matter
(−1.2 pp) — agents defect regardless of what anyone says.

**Bottom line:** Messages and social-history features are **partial substitutes**. When one
information source is missing, agents rely more heavily on the other. But when the social
signal is strongly negative (everyone defecting), messages cannot override the pessimism.

### The no-comm comparison

The `none/history_audit_summary.md` includes both cond1 (comm) and cond2 (no-comm) under
each history ablation, so we can ask: does communication's value change with different
history ablations?

| History ablation | f=5.0 learned | f=5.0 no-comm | Comm gap |
| --- | ---: | ---: | ---: |
| intact | 0.751 | 0.562 | +18.9 pp |
| zero_ewma | 0.845 | 0.757 | +8.8 pp |
| zero_last_action | 0.831 | 0.576 | +25.4 pp |
| zero_temporal | 0.838 | 0.718 | +12.0 pp |
| clamp_temporal_low | 0.838 | 0.718 | +12.0 pp |
| clamp_temporal_high | 0.571 | 0.293 | +27.8 pp |
| clamp_ewma_high | 0.626 | 0.282 | +34.4 pp |

The comm gap *explodes* when agents are given misleading social information. Under
`clamp_ewma_high`, no-comm agents collapse to 28% cooperation while comm agents hold at
63% — a 34 pp gap. Messages act as a **corrective** against misleading observations.

Key files:
- [`none/history_audit_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_message_history_grid_150000_15seeds_local_20260327/report/none/history_audit_summary.md) (comm vs no-comm)
- [`zeros/history_audit_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_message_history_grid_150000_15seeds_local_20260327/report/zeros/history_audit_summary.md)
- [`message_history_grid_meta.txt`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_message_history_grid_150000_15seeds_local_20260327/report/message_history_grid_meta.txt)

---

## Question 7: What Is the Low-Dimensional Structure of the Protocol?

### What is this test?

The frozen and causal tests tell us the channel matters, but not *how* messages work
mechanically. This analysis asks: when a receiver sees 3 binary messages from 3 other
senders, what features of that message vector drive their decision?

Three candidate structures, from simplest to richest:

1. **Any-token:** Does it matter whether *anyone* sent a 1, versus all zeros? (1 bit of info)
2. **Count-of-ones:** Does the *number* of 1s received matter? (0, 1, 2, or 3 — ordinal)
3. **Full pattern with sender identity:** Does it matter *who* sent what? (up to 8 distinct
   patterns for 3 senders × 2 tokens)

We also test these with surrogate logistic-regression models using grouped cross-validation
by training seed (so we test whether structure generalises across independently trained
populations, not just within a single trained group).

### Count-of-ones response

| f | 0 ones | 1 one | 2 ones | 3 ones | any_token effect |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3.5 | 0.589 | 0.584 | 0.584 | 0.646 | +0.6 pp |
| 5.0 | 0.713 | 0.775 | 0.748 | 0.832 | +4.8 pp |

**At f=3.5**, count barely matters. Cooperation is flat across 0/1/2 ones (~0.585) and
only rises at 3 ones (0.646). The any_token shift is just +0.6 pp. The agents are *not*
following a simple "more ones → more cooperation" rule.

**At f=5.0**, coarse message structure matters more. The any_token effect is +4.8 pp, and
3 ones gives the highest cooperation (0.832). But the response is not monotone: count=2
(0.748) is *lower* than count=1 (0.775). This non-monotonicity suggests agents aren't simply
counting ones — something else about the message pattern matters.

### Sender identity matters enormously within a trained population

Holding the count of ones constant, different *patterns* (i.e., different combinations of
which specific senders sent 1) produce very different cooperation rates:

| f | count | Median within-seed pattern gap |  Share of seeds with >10pp gap |
| ---: | ---: | ---: | ---: |
| 3.5 | 0 | 72.7 pp | 0.92 |
| 3.5 | 1 | **98.4 pp** | 1.00 |
| 3.5 | 2 | 90.7 pp | 1.00 |
| 3.5 | 3 | 69.7 pp | 1.00 |
| 5.0 | 0 | 73.3 pp | 1.00 |
| 5.0 | 1 | **87.0 pp** | 0.92 |
| 5.0 | 2 | 50.4 pp | 0.93 |
| 5.0 | 3 | 32.1 pp | 0.62 |

At f=3.5 with count=1, the median within-seed gap is **98.4 pp** — meaning within
a single trained population, receiving one "1" from sender A vs sender B can shift
cooperation from near-0% to near-100%. This is not a pooling artifact; it holds within
individual seeds.

**Example from the pooled data at f=5.0, count=1:**
- `agent_2` receiving `[agent_0:1, agent_1:0, agent_3:0]` → P(coop) = 0.926
- `agent_0` receiving `[agent_1:1, agent_2:0, agent_3:0]` → P(coop) = 0.641

Same count, completely different behaviour. The receiver has learned *which sender's*
message to trust.

### But sender-identity codes are seed-specific, not universal

The surrogate model comparison tests whether these patterns generalise across seeds
(using grouped CV where entire seeds are held out):

| f | Model | Log loss | vs count model |
| ---: | --- | ---: | ---: |
| 3.5 | history_only | 0.528 | +0.003 better |
| 3.5 | history_any | 0.528 | +0.003 better |
| 3.5 | history_count | 0.531 | (reference) |
| 3.5 | history_sender_bits | 0.539 | −0.008 worse |
| 3.5 | **history_pattern** | 0.569 | **−0.038 worse** |
| 5.0 | history_only | 0.581 | +0.006 better |
| 5.0 | history_any | 0.579 | +0.008 better |
| 5.0 | history_count | 0.587 | (reference) |
| 5.0 | history_sender_bits | 0.607 | −0.020 worse |
| 5.0 | **history_pattern** | 0.695 | **−0.108 worse** |

The richer the message representation, the **worse** the cross-seed generalisation. Adding
sender identity bits hurts. Using full sender-identity patterns hurts a lot — especially at
f=5.0 where the pattern model loses 0.108 in log loss vs the simple count model.

This means: **the rich sender-indexed codes that exist within each trained population are
idiosyncratic to that seed.** Different random seeds develop different internal conventions.
The cross-seed transferable structure is captured by the simplest features: just having any
message at all (`history_any`) or the private observation history (`history_only`).

### What this means

Messages operate on **three layers**:

1. **Channel presence** (layer 1): Just having *any* non-zero input in the message slots
   provides a coordination anchor. This is what `any_token` captures, and what `fixed1`
   exploits so effectively at f=5.0.

2. **Coarse content** (layer 2): The count or general pattern of ones carries some additional
   information at f=5.0 (+4.8 pp for any_token, non-zero count effects), but it is weak at
   f=3.5 (+0.6 pp) and not monotone.

3. **Rich sender-specific codes** (layer 3): Within a trained population, receivers have
   learned to respond very differently depending on *who* sent each token. These codes are
   real and behaviourally powerful (up to 98 pp within-seed). But they are **seed-specific**
   — they do not transfer across independently trained populations. They are private
   conventions, not a universal language.

**For the paper's claim**, the reusable cross-seed effect is dominated by layers 1–2.
Layer 3 is scientifically interesting (it shows the agents *can* develop rich protocols) but
it does not contribute to the average treatment effect reported across seeds.

Key files:
- [`lowdim_mechanism_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_frozen150k_expanded_15seeds_local_20260327/report/lowdim_mechanism/lowdim_mechanism_summary.md)
- [`pattern_seed_summary.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_frozen150k_expanded_15seeds_local_20260327/report/lowdim_mechanism/pattern_seed_summary.csv)
- [`surrogate_model_summary.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_frozen150k_expanded_15seeds_local_20260327/report/lowdim_mechanism/surrogate_model_summary.csv)

---

## Question 8: Does Token-Rate Matching Explain the Benefit?

### What is this test?

A natural worry: maybe the learned messages help just because the receiver's policy was
trained expecting a certain *distribution* of inputs (e.g., roughly 55% ones), and any
signal matching that distribution would work equally well. The `public_marginal` intervention
tests this by sampling a single shared random token from the learned global marginal P(token)
each timestep — preserving the aggregate rate but destroying content, timing, and
state-conditioned variation.

### Results

| f | learned | public_marginal | public_random | marginal | no-comm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3.5 | 0.587 | 0.545 | 0.541 | 0.538 | 0.423 |
| 5.0 | 0.751 | 0.740 | 0.730 | 0.733 | 0.562 |

`public_marginal` is only +0.4 pp above `public_random` at f=3.5 and +1.0 pp at f=5.0.
These are negligible gains from matching the token rate. The learned protocol still
outperforms by +4.2 pp (f=3.5) and +1.2 pp (f=5.0) over the best token-rate-matched
control.

### In plain English

**Token-rate matching does not explain the communication benefit.** Whether you flip a
fair coin, or a biased coin matching the learned frequency, the cooperation rates are nearly
identical. The residual benefit of learned messages (above any random-signal baseline) comes
from something other than the aggregate distribution — it comes from the state-conditioned
*content* and the receiver's ability to condition on sender-specific patterns within the
trained population.

This is consistent with the three-layer picture from Question 7: the learned protocol carries
real information (layer 3), but the bulk of the cooperation benefit comes from channel
presence (layer 1) and coarse coordination (layer 2), neither of which requires matching
the token rate.

Key files:
- [`intervention_suite_summary.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_frozen150k_expanded_15seeds_local_20260327/report/intervention_suite_summary.csv)

---

## Putting It All Together

### What we know

1. **Communication helps — a lot.** By 150k, comm agents cooperate +16–19 pp more than
   no-comm agents at both focal regimes. This is statistically robust (SEMs don't overlap).
2. **The old f=5.0 reversal is gone.** In the old staged family, communication *hurt* at
   f=5.0. In vecstraight, it helps strongly. This means the old result was specific to the
   old training pipeline, not a general property of the game.
3. **The channel is being used.** Frozen perturbations change behavior — the policy is not
   ignoring messages.
4. **Messages have real causal power.** Forcing one sender's token shifts receivers'
   cooperation by ~15–21 pp in the high-f regimes, flipping ~10% of actual decisions.
5. **But the protocol is not a clean optimal code.** Some perturbations barely matter, and
   fixed1 even beats learned messages at f=5.0. The agents are using messages, but not in a
   globally optimal way.
6. **The comm benefit grows with observation noise.** At f=5.0, the comm gap widens from
   +17 pp (perfect info) to +20 pp (sigma=1.0), consistent with messages carrying genuine
   state information.
7. **Social history features are a double-edged sword.** Temporal cooperation history enables
   conditional defection strategies. Removing it *increases* cooperation at f=5.0 — agents
   cooperate more when they cannot punish. This is the difficulty-gated pattern.
8. **Messages and history are partial substitutes.** When one information source is ablated,
   agents lean harder on the other. Messages are most valuable when private observations are
   noisy or misleading — acting as a corrective against bad social signals.
9. **Communication is most protective under adversarial information conditions.** When agents
   are given misleading cues (clamped-high EWMA), no-comm agents collapse to ~28% cooperation
   while comm agents hold at ~63%. Messages buffer against misinformation.
10. **Three forms of message dependence: slotwise variation > coarse content > sender-specific codes.**
    The clean direct msg-source family sharpens the old picture. What matters most at training time
    is not a portable shared code, but access to rich variable slotwise input. Rich
    sender-identity-indexed codes still exist within each trained population (up to 98 pp within-seed
    gaps), but they are idiosyncratic and do not generalise cleanly across independently trained seeds.
    Cross-seed surrogate models perform *worse* with richer message features.
11. **Token-rate matching does not explain the benefit.** `public_marginal` (matching the
    learned global token frequency) performs nearly identically to `public_random` (fair coin).
    The learned protocol's advantage is not about having the right *distribution* of tokens.
12. **The clean direct exogenous result is sharper than the old continuation evidence.**
    Native `uniform` messages finish essentially tied with learned communication at both focal
    multipliers, while `public_random` is much weaker and constant channels are harmful.
    So the channel's value is not explained by a single shared random cue or by constant token
    presence alone.
13. **Communication requires temporal context.** In the from-scratch comm × history factorial,
    communication provides a +15.7 pp advantage at f=3.5 under full history but only +1.9 pp
    under reduced history. At f=5.0, the advantage vanishes entirely (−0.1 pp). Messages
    don't substitute for history — they depend on it.
14. **Foreign messages transfer with a 1-bit polarity alignment.** Raw unaligned foreign
    messages perform no better than random (52.1% vs 54.1% at f=3.5). But with a global
    0↔1 flip, best-aligned foreign messages reach 60.5% — clearly above random controls
    (+6.4 pp) and comparable to same-seed natural (58.6%). Roughly half the seed pairs
    need the flip. This means the "conventions don't transfer" claim was too strong; a
    coarse polarity convention IS portable, but fine-grained sender-indexed patterns are not.

### The right mental model

Communication in this environment is useful but conditional. The agents learned to partially
depend on the message channel, but they did not converge on a tight, efficient signaling
system. The protocol is more like a "messy but helpful habit" than a "clean emergent language."

The three forms of message dependence sharpen this: **form 1** (rich variable slotwise input)
provides most of the training-time benefit. **Form 2** (coarse content) adds modest regime
information at the endpoint. **Form 3** (rich sender-specific codes) is where the
within-population "private language" lives — powerful within a seed, invisible across seeds,
and not what drives the average treatment effect.

Two critical qualifiers now apply:

**Communication requires temporal context.** The comm × history factorial shows that
communication's advantage (+15.7 pp at f=3.5) nearly vanishes under reduced history (+1.9 pp).
Messages don't substitute for history — they complement it. The channel's value is not intrinsic;
it depends on the agent having enough temporal observation scaffold to meaningfully integrate
message input.

**A single shared random cue is not enough.** The clean direct msg-source family shows a strong
split: `uniform` is near learned, but `public_random` is near no-comm, and `fixed0` / `fixed1`
are harmful. So the training-time benefit is not "portable semantics" and not "just give the
agents one public coin flip." It looks more like a history-conditioned variable-input scaffold.

---

## Question 9: What do the clean direct exogenous controls show?

*Added 2026-04-12 after the completed Hetzner clean `msg_source_mode` family.*

### What is this test?

This family asks the clean training-time question directly:

- keep the environment, receiver architecture, wrapper, optimizer, schedules, and evaluation
  contract fixed
- train from scratch
- change only the source of the message stream

The five arms are:

- `learned`: standard learned sender policy
- `uniform`: independent random bits in each sender slot
- `public_random`: one shared random bit copied into every sender slot
- `fixed0` / `fixed1`: constant tokens
- `no_comm`: the usual communication-off baseline, taken from the learned suite's `cond2`

### Results at 150k (15 seeds)

| Condition | f=3.5 coop | f=5.0 coop | How to read it |
| --- | ---: | ---: | --- |
| **learned** | 77.7% | 90.2% | native learned messages |
| **uniform** | 75.9% | 91.6% | rich slotwise random variation |
| **public_random** | 36.9% | 57.6% | one shared random bit |
| **fixed0** | 26.4% | 46.7% | constant token 0 |
| **fixed1** | 14.2% | 46.5% | constant token 1 |
| **no_comm** | 42.3% | 56.2% | no channel |

### In plain English

**Uniform random messages work almost as well as learned communication.** At `f=3.5`, `uniform`
finishes only `1.8 pp` below learned. At `f=5.0`, it finishes `1.4 pp` above learned. So the
channel's value is not coming mainly from a portable learned code.

**But not every random cue works.** `public_random` is far below learned at both focal
multipliers (`-40.8 pp` at `f=3.5`, `-32.5 pp` at `f=5.0`) and is only near the no-comm
baseline. So a single shared public bit does not reproduce the training-time benefit.

**Constant cues are actively bad.** `fixed0` and `fixed1` are below learned in all 15 seeds at
both focal multipliers, and at `f=5.0` both finish below no-comm. This means the benefit is not
just "having something in the message slots."

**The clean statement is now sharper than the old continuation story.** What seems to matter is
rich variable slotwise input that can be combined with the rest of the observation stream and with
temporal context. That is closer to a history-conditioned coordination scaffold than to a portable
shared language.

Key files:
- [`clean_msgsource_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_clean_msgsource_status_20260412/report/clean_msgsource_summary.md)
- [`channel_control_summary.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_clean_msgsource_status_20260412/report/channel_control_summary.csv)
- [`channel_control_raw.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_clean_msgsource_status_20260412/report/channel_control_raw.csv)

---

## Question 10: Does communication substitute for temporal history, or require it?

*Added 2026-04-01 after the Hetzner comm × history training factorial.*

### What is this test?

The evaluation-time history grid (Question 6) showed that messages and history features are
partial substitutes: when one is removed, agents lean harder on the other. But that was an
evaluation-time manipulation on agents **trained** with full history and full communication.
The stronger question is: what happens when agents are **trained from scratch** with or
without temporal history?

The comm × history factorial trains four cells from scratch (15 seeds each):
- `with_comm_full_history`
- `with_comm_reduced_history`
- `without_comm_full_history`
- `without_comm_reduced_history`

### Results at 150k (15 seeds per cell)

| Cell | f=3.5 coop | f=5.0 coop |
| --- | ---: | ---: |
| comm + full history | 50.4% | 65.6% |
| comm + reduced history | 23.9% | 61.7% |
| no-comm + full history | 34.7% | 51.8% |
| no-comm + reduced history | 22.0% | 61.8% |

Factorial contrasts:

| Contrast | f=3.5 | f=5.0 |
| --- | ---: | ---: |
| **Comm advantage, full history** | **+15.7 pp** | **+13.7 pp** |
| **Comm advantage, reduced history** | **+1.9 pp** | **−0.1 pp** |
| History effect with comm | +26.5 pp | +3.8 pp |
| History effect without comm | +12.7 pp | −9.9 pp |

Communication responsiveness (KL divergence):

| Cell | Responsiveness |
| --- | ---: |
| comm + full history | 0.535 ± 0.055 |
| comm + reduced history | 0.286 ± 0.027 |

### In plain English

**Communication requires temporal context to be useful.**

Under full history, communication provides a clear, large advantage: +15.7 pp at f=3.5
and +13.7 pp at f=5.0. Under reduced history, the advantage essentially vanishes:
+1.9 pp and −0.1 pp. This is a strong interaction, not a main effect.

**Messages don't substitute for history — they depend on it.** Message responsiveness
drops from 0.535 to 0.286 when history is removed. Agents trained without temporal
features don't learn to listen to messages as a replacement. Instead, they learn to
partially ignore messages because messages alone don't carry enough context to ground
meaningful conditional behavior.

**The f=5.0 reduced-history result is especially striking.** Without temporal features,
both comm (61.7%) and no-comm (61.8%) converge to the same cooperation rate — and both
*exceed* the no-comm full-history cell (51.8%). This echoes the evaluation-time finding
(Question 6) that removing temporal features at f=5.0 removes the conditional-defection
option and increases cooperation. Under reduced history, communication becomes irrelevant
because the mechanism that makes it valuable (temporal context-dependent decision-making)
has been removed.

**Note on absolute levels.** These cooperation rates are lower than the base learned
condition (71.0% at f=3.5, 79.1% at f=5.0 in the loss-switch reference). The factorial
was run from scratch with a different launch configuration. The meaningful comparisons
are the *within-factorial contrasts*, not absolute levels across experiments.

**For the paper's claim:** This result strengthens the "three forms of message dependence"
framing but adds a crucial qualifier. The first form — stable variable signal input — is
not intrinsically valuable; it requires the temporal observation scaffold to be useful.
Communication doesn't substitute for history; it *complements* it.

Key files:
- [`comm_history_factorial_summary.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_comm_history_factorial_status_20260401/report/comm_history_factorial_summary.csv)

---

---

## Question 11: Do message conventions actually transfer across seeds?

*Added 2026-04-02 after the cross-seed transfer suite with global-flip alignment.*

### What is this test?

The surrogate-model analysis (Question 7) showed that richer sender-pattern features predict
*worse* on held-out seeds, which we interpreted as evidence that conventions are seed-private.
But that was an indirect test using a simple logistic surrogate, not a direct behavioral test.

The cross-seed transfer suite directly tests portability: take a trained receiver from seed A
and feed it messages generated by seed B's trained message policy. Measure cooperation. Test
two alignment conditions:

- **identity**: use the foreign 0/1 messages as they are
- **flipall**: globally swap every 0 and 1

Then take the best of the two per receiver–donor–f pair. Compare to same-seed natural
messages, `public_random`, and `sender_shuffle`.

**Important caveat:** this is not full mixed-population cross-play. The receiver's action
policy is held fixed; only the message stream is transplanted from the donor seed. A full
cross-play test would also replace the receiver with a foreign agent, which is a stronger
test of convention portability.

### Results (15 seeds × 14 donor seeds = 210 pairs per f-value)

| Condition | f=3.5 | f=5.0 |
| --- | ---: | ---: |
| Same-seed natural | 58.6% | 75.1% |
| Foreign identity (no alignment) | 52.1% | 73.4% |
| Foreign flipall (global flip) | 52.6% | 72.2% |
| **Foreign best-aligned** (oracle max) | **60.5%** | **79.9%** |
| public_random | 54.1% | 73.0% |
| sender_shuffle | 54.6% | 74.8% |

Relative to random/shuffle controls, best-aligned foreign messages are clearly stronger:

| Contrast | f=3.5 | f=5.0 |
| --- | ---: | ---: |
| best-aligned − natural | +1.8 pp | +4.7 pp |
| best-aligned − public_random | +6.4 pp | +6.8 pp |
| best-aligned − sender_shuffle | +5.9 pp | +5.1 pp |

Best alignment usage: identity (no flip) won for 205 pairs, flipall won for 215 pairs.

### In plain English

**Foreign messages DO transfer, but often need a simple global reinterpretation of the symbols.**

Without any alignment, foreign messages (52.1% at f=3.5) perform *below* public_random
(54.1%). That is consistent with the surrogate-model finding: raw foreign conventions don't
help. But with the simplest possible alignment — a global 0↔1 flip — the best-aligned
foreign stream reaches 60.5%, clearly above both random controls and even slightly above
same-seed natural (58.6%).

**The "seed-private conventions do not transfer" story was too strong.** The corrected picture
is:

1. Raw unaligned foreign messages are no better than random noise (confirming that the
   specific token assignments are indeed seed-specific).
2. A single-bit global polarity alignment recovers substantial cross-seed transfer (6+ pp
   above random controls).
3. The alignment split is roughly 50/50 (215 flip vs 205 no-flip), consistent with the
   polarity-fragmentation finding in Appendix A.

**Important caveat about the "best-aligned" numbers.** The "best of identity vs flipall"
is an *oracle* selection — it chooses whichever alignment works better *after seeing the
outcome*. In a real zero-shot transfer scenario, you would not know which orientation to use.
Without the oracle, foreign-identity transfer (52.1%) is below public_random (54.1%) at f=3.5.
So the transfer is real, but it requires at least a one-bit coordination convention (which
polarity to use), which is itself seed-specific.

The "best-aligned slightly beats same-seed natural" result (+1.8 pp at f=3.5, +4.7 pp at
f=5.0) is likely inflated by the max-over-two-conditions selection bias and should not be
interpreted as "foreign messages are genuinely better than same-seed messages."

### What this means for the paper

The manuscript's portability claim needs to be updated. Instead of:

> "sender-specific conventions fail to transfer across seeds"

The evidence now supports:

> "raw foreign conventions perform no better than random replacement, but a single-bit
> global polarity alignment recovers substantial cross-seed transfer — suggesting the
> cross-seed portable component is a coarse polarity convention, not a fine-grained
> sender-indexed code"

This *strengthens* the paper's story rather than weakening it. It adds a positive finding
(transfer IS possible with minimal alignment) that makes the "private conventions" story
more nuanced and more interesting. The three-part decomposition becomes:

- **Portable:** low-dimensional variable input + coarse polarity convention
- **Partially portable:** global token meaning (recoverable with a 1-bit flip)
- **Private:** fine-grained sender-indexed patterns (per-sender, per-receiver pair effects)

Key files:
- [`cross_seed_transfer_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_cross_seed_transfer_flip15seeds_local_20260402/summary/cross_seed_transfer_summary.md)
- [`summary_by_f.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_cross_seed_transfer_flip15seeds_local_20260402/summary/summary_by_f.csv)
- [`best_alignment_results.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_cross_seed_transfer_flip15seeds_local_20260402/summary/best_alignment_results.csv)
- [`best_alignment_usage.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_cross_seed_transfer_flip15seeds_local_20260402/summary/best_alignment_usage.csv)

---

### What we do NOT yet know

The frozen and sender-causal tests tell us about the **endpoint** (the final trained policy).
The noise sweep, history grid, and low-dim mechanism analysis tell us *what* information
drives decisions and how. They do **not** tell us:

- **Why does uniform help specifically at f=5.0 but not f=3.5?** The loss-switch repair
  and the clean direct family together now show that `uniform` is near learned at both
  focal multipliers, but `public_random` is not. The unresolved mechanistic question is
  no longer "does exogenous input help?"; it is "why does rich slotwise variation work
  while a single shared random bit fails?"
- **When did the three layers emerge during training?** Was layer 1 (channel presence)
  established early and layers 2–3 added later? Or did they co-develop? The 50k→150k
  comparison hints at gradual development, but we need finer temporal resolution.
- **Why is the count-of-ones response non-monotone at f=5.0?** Count=2 gives lower
  cooperation than count=1. This is unexpected and may reflect interference between
  sender-specific conventions (the "wrong" pair of senders cancelling each other out
  when pooled).

The remaining open questions are:

1. **Joint evaluation grid for the comm × history factorial** (next priority): The four-cell
   training family is complete; what remains is running the frozen intervention suite and
   history audit on the factorial cells to characterise how endpoint message dependence
   differs under full vs reduced history training.
2. **Observability / noise sweep** (evaluation-time, not yet started): Crossing evaluation-time
   noise with the base and factorial families to separate information-transfer from pure
   coordination effects.
3. **Mechanism audit for the clean msg-source split** (optional): Analyse why `uniform`
   nearly matches learned communication while `public_random` and constant cues fail.
   The likely axes are slot identity, temporal conditioning, and whether receivers use
   the multi-slot pattern rather than a single shared cue.

Treat this overview as a **completed endpoint + mechanism decomposition + training-time
confound-repair + clean direct msg-source + history-interaction + cross-seed-transfer story**.
The loss-switch repair quantifies the auxiliary-loss confound; the clean direct family shows
that high-entropy slotwise exogenous input can nearly match learned communication while one
shared random bit cannot; the comm × history factorial establishes that communication requires
temporal context; and the cross-seed transfer suite shows that conventions transfer partially
with a 1-bit polarity alignment but not as raw unaligned streams. The evaluation-side audit
and observability sweep remain pending.
