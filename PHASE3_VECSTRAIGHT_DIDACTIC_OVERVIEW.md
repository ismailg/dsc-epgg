# Phase-3 Vecstraight Didactic Overview

*Last updated: 2026-04-01*

This note explains the completed vecstraight rerun results in plain terms. It covers the
**new straight vectorized family** only (`training_family = phase3_vecstraight`, repo =
`dsc-epgg-vectorized`). If you want the old staged-family story, that lives in the original
`dsc-epgg` repo.

This file is interpretive, not the live status ledger. For current run state and ownership, use
[`PHASE3_VECSTRAIGHT_NEXT_STEPS.md`](PHASE3_VECSTRAIGHT_NEXT_STEPS.md) and
[`PHASE3_VECSTRAIGHT_PAPER_TODO.md`](PHASE3_VECSTRAIGHT_PAPER_TODO.md).

**What this covers:** Ten completed evaluation/training stages — frozen endpoints, sender-causal
probes, expanded intervention suite (with SEMs and public_marginal control), observation-noise
sweep, message-history grid, low-dimensional mechanism analysis (count-of-ones response,
sender-identity effects, surrogate model comparison), the corrected same-checkpoint continuation
ladder, the from-scratch exogenous-channel controls, the qx6 loss-switch repair (quantifying the
auxiliary-loss confound), and the Hetzner comm x history training factorial.
**What this still does not cover:** The joint evaluation grid for the comm x history factorial
cells and the observability/noise sweep remain pending on the evaluation side.

**March 30 continuation and exogenous status update:** The corrected continuation story is now
complete for the intended 50k/100k vec-parity branches. In the fetched corrected qx6 100k→150k
reruns, `sender_shuffle` finishes **above** the learned 150k reference (+11.8 pp at f=3.5,
sign-flip p=0.1313; +7.7 pp at f=5.0, p=0.0395), while `fixed0` finishes **below** it
(−10.6 pp at f=3.5, p=0.1177; −13.5 pp at f=5.0, p=0.0116). The corrected qx6 50k→150k ladder
now adds three distinct outcomes: `sender_shuffle` is above learned (+14.1 pp at f=3.5,
p=0.0718; +5.8 pp at f=5.0, p=0.2531), `fixed0` is strongly below learned (−19.0 pp at f=3.5,
p=0.0132; −21.7 pp at f=5.0, p=0.0007), and `public_random` is near learned at f=3.5
(−1.8 pp, p=0.8239) but below it at f=5.0 (−14.7 pp, p=0.0492). The finished Hetzner 50k→150k
`uniform` branch is also above learned (+11.3 pp at f=3.5, p=0.2710; +15.7 pp at f=5.0,
p=0.0082). So the completed continuation ladder is still **heterogeneous**, but not in a way
that supports a simple learned-code superiority story: constant-zero channels are harmful,
shared public randomness is not uniformly sufficient, and shuffled or uniform exogenous channels
can stay above the learned reference in the mean.

The completed from-scratch exogenous controls still show that split observationally, but a same-day
trainer audit found that these forced-channel runs were not trained under exactly the same objective
as the base learned family: current `msg_training_intervention != none` runs zero `sign_lambda` and
`list_lambda`, whereas the base learned family uses `0.1` and `0.1`. So the direct training-time
`uniform > learned` contrast should now be read as **provisional** rather than final. What remains
solid already is the ranking inside the forced-channel family itself.

Observationally, the completed from-scratch exogenous controls still point the same way, but now
with a sharper split.
Canonical `public_random` training reaches 64.9% at f=3.5 and 63.9% at f=5.0, which is +6.0 pp
versus learned at f=3.5 but −11.3 pp at f=5.0. qx6 `uniform` is much stronger: 74.8% at f=3.5
and 92.3% at f=5.0, or +15.9 pp and +17.1 pp relative to learned. Canonical `fixed0` reaches only
49.9% and 49.4%, or −9.0 pp and −25.7 pp relative to learned; at f=5.0 it is below the no-comm
baseline by −6.9 pp. qx6 `fixed1` is also poor at 44.0% and 46.9%, or −14.8 pp and −28.3 pp
relative to learned; at f=5.0 it is below no-comm by −9.5 pp. The fetched qx6 `fixed0` replica
also stays below learned at both focal multipliers (−12.6 pp, −20.9 pp), so the harmful sign is
stable even though the exact `fixed0` magnitude differs by host. To repair the cross-family
learned-vs-uniform comparison, a qx6 `50k -> 150k` three-arm loss-switch continuation control is
now running: `none_base`, `none_zeroaux`, and `uniform_zeroaux`.

Key status file:
- [`sameckpt_continuation_summary.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_paper_pivot_20260329_status/sameckpt_continuations/sameckpt_continuation_summary.md)
- [`channel_control_summary.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_exogenous_channel_controls_status_20260330/report/channel_control_summary.csv)

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

The rest of this document answers eight questions, each with its own evaluation stage:

1. **Is the channel being used at all?** → Frozen endpoint tests
2. **Does message content matter, or just the presence of a channel?** → Frozen perturbation tests
3. **Can a single sender's message causally change what others do?** → Sender-causal probes
4. **Is the comm gap statistically robust? Does it grow over training?** → Expanded intervention suite
5. **Does observation noise change the story?** → Noise sweep
6. **Which observation features drive cooperation?** → Message-history grid
7. **Is there a low-dimensional structure to how messages work?** → Count-of-ones, sender-identity, surrogate models
8. **Does token-rate matching explain the comm benefit?** → Public marginal control

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
10. **Three forms of message dependence: stable variable input > coarse content > sender-specific codes.**
    The bulk of the cross-seed cooperation benefit comes from having stable variable
    input in the message slots and coarse coordination structure. Rich
    sender-identity-indexed codes exist within each trained population (up to 98 pp within-seed
    gaps) but are idiosyncratic — they do not generalise across independently trained seeds.
    Cross-seed surrogate models perform *worse* with richer message features.
11. **Token-rate matching does not explain the benefit.** `public_marginal` (matching the
    learned global token frequency) performs nearly identically to `public_random` (fair coin).
    The learned protocol's advantage is not about having the right *distribution* of tokens.
12. **The auxiliary-loss confound was real and large.** Zeroing the auxiliary communication
    losses (sign_lambda and list_lambda) while keeping everything else the same costs −16 pp
    at f=3.5. The old "uniform beats learned everywhere" does not survive; the clean statement
    is that uniform exceeds the learned reference only at f=5.0 (+7.6 pp, p=0.154) under
    matched training objectives.
13. **Communication requires temporal context.** In the from-scratch comm × history factorial,
    communication provides a +15.7 pp advantage at f=3.5 under full history but only +1.9 pp
    under reduced history. At f=5.0, the advantage vanishes entirely (−0.1 pp). Messages
    don't substitute for history — they depend on it.

### The right mental model

Communication in this environment is useful but conditional. The agents learned to partially
depend on the message channel, but they did not converge on a tight, efficient signaling
system. The protocol is more like a "messy but helpful habit" than a "clean emergent language."

The three forms of message dependence sharpen this: **form 1** (stable variable signal input)
provides a coordination anchor that transfers across seeds. **Form 2** (coarse content) adds
modest regime information at f=5.0 but is nearly inert at f=3.5. **Form 3** (rich
sender-specific codes) is where the within-population "private language" lives — powerful
within a seed, invisible across seeds, and not what drives the average treatment effect.

Two critical qualifiers now apply:

**Communication requires temporal context.** The comm × history factorial shows that
communication's advantage (+15.7 pp at f=3.5) nearly vanishes under reduced history (+1.9 pp).
Messages don't substitute for history — they complement it. The channel's value is not intrinsic;
it depends on the agent having enough temporal observation scaffold to meaningfully integrate
message input.

**Auxiliary training losses matter for training-time comparisons.** The loss-switch repair
shows that the `sign_lambda` and `list_lambda` auxiliary losses provided a substantial
training-time boost (−16 pp at f=3.5 when zeroed). Under matched training objectives,
uniform exogenous channels modestly exceed learned communication only at f=5.0 (+7.6 pp),
not at f=3.5. The regime-dependent pattern is therefore: at f=5.0 (cooperation-dominant),
exogenous variation provides a real benefit; at f=3.5 (mixed-motive), learned content
may matter more.

---

## Question 9: Did the auxiliary-loss confound drive the "uniform > learned" result?

*Added 2026-04-01 after the qx6 loss-switch repair batch.*

### What is this test?

The original from-scratch exogenous controls trained `uniform` (and other exogenous
channels) with `sign_lambda=0.0` and `list_lambda=0.0`, while the base learned condition
used `sign_lambda=0.1` and `list_lambda=0.1`. These auxiliary losses — a message-entropy
regularizer and a listening bonus — shape the training objective. The loss-switch repair
runs `none_base`, `none_zeroaux`, and `uniform_zeroaux` from the **same 50k checkpoint**
using the vectorized continuation pipeline, so the only difference between `none_base`
and `none_zeroaux` is whether the auxiliary communication losses are active.

### Results at 150k (15 seeds)

| Condition | f=3.5 coop | f=5.0 coop | Training objective |
| --- | ---: | ---: | --- |
| **none_base** (learned, full aux) | 71.0% | 79.1% | PPO + sign + listener |
| **none_zeroaux** (learned, zero aux) | 55.0% | 74.4% | PPO only |
| **uniform_zeroaux** (uniform, zero aux) | 68.7% | 86.7% | PPO only |
| **no_comm** (baseline) | 42.3% | 56.2% | PPO only (no messages) |

Paired contrasts against none_base:

| Contrast | f=3.5 delta | p | f=5.0 delta | p |
| --- | ---: | ---: | ---: | ---: |
| none_zeroaux vs none_base | −16.0 pp | 0.082 | −4.7 pp | 0.361 |
| uniform_zeroaux vs none_base | −2.4 pp | 0.801 | +7.6 pp | 0.154 |

### In plain English

**The auxiliary-loss confound was real and large.**

Removing the auxiliary communication losses while keeping everything else the same costs
**−16 pp at f=3.5** and −4.7 pp at f=5.0. This means a substantial part of the original
learned baseline's performance came from the auxiliary training signal, not just from the
learned message content.

**The old "uniform beats learned everywhere" story does not survive.** Under the matched
zero-aux objective, `uniform_zeroaux` (68.7%) does NOT beat `none_base` (71.0%) at f=3.5.
The apparent uniform dominance at f=3.5 in the original exogenous runs was substantially
inflated by the auxiliary-loss confound.

**But uniform still genuinely helps at f=5.0.** `uniform_zeroaux` (86.7%) exceeds
`none_base` (79.1%) by +7.6 pp, and it massively exceeds `none_zeroaux` (74.4%) by
+12.3 pp. So the regime-dependent story sharpens: at the cooperation-dominant regime,
exogenous high-entropy variation provides a real training-time benefit that is not explained
by the auxiliary-loss difference. At the mixed-motive regime, the evidence is much weaker.

**Caveat:** The sign-flip p-value for `uniform_zeroaux` vs `none_base` at f=5.0 is 0.154,
so the effect is suggestive but not yet statistically robust by conventional standards.
The continuation also uses a hybrid sender/delivered-message intervention path, which
introduces a secondary implementation concern.

Key files:
- [`sameckpt_continuation_summary.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_lossswitch_controls_status_20260401/report/sameckpt_continuation_summary.csv)
- [`sameckpt_continuation_paired_stats.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_lossswitch_controls_status_20260401/report/sameckpt_continuation_paired_stats.csv)

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

### What we do NOT yet know

The frozen and sender-causal tests tell us about the **endpoint** (the final trained policy).
The noise sweep, history grid, and low-dim mechanism analysis tell us *what* information
drives decisions and how. They do **not** tell us:

- **Why does uniform help specifically at f=5.0 but not f=3.5?** The loss-switch repair
  shows that under matched zero-aux training, uniform exceeds the learned reference at
  f=5.0 (+7.6 pp) but not at f=3.5 (−2.4 pp). This regime-dependent pattern needs a
  mechanistic explanation. One candidate: at f=5.0, where cooperation is individually
  rational, exogenous variation provides a low-cost coordination scaffold; at f=3.5,
  where the dilemma is sharper, state-conditioned content may matter more.
- **Is the hybrid forced-channel implementation clean enough?** The current continuation
  pipeline uses a sender/delivered-message hybrid for forced channels. A cleaner
  direct exogenous-channel implementation would strengthen the training-time claims.
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
3. **Cleaner exogenous-channel implementation** (optional): Whether to implement a direct
   exogenous-channel training path that avoids the hybrid sender/delivered-message issue
   in the current continuation pipeline.

Treat this overview as a **completed endpoint + mechanism decomposition + training-time
confound-repair + history-interaction story**. The loss-switch repair quantifies the
auxiliary-loss confound; the comm × history factorial establishes that communication
requires temporal context. The evaluation-side audit and observability sweep remain
pending.
