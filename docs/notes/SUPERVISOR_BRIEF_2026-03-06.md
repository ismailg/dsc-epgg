# Supervisor Brief — DSC-EPGG Communication Results

**Date:** 2026-03-06
**Status:** Phase 2b + Phase 3 complete. Uniform control in progress. Supervisor feedback received.

---

## Setup

- 4 PPO agents, Extended Public Goods Game, sticky regime switching (F ∈ {0.5, 1.5, 2.5, 3.5, 5.0})
- Agents observe noisy f-hat, send 1-bit messages, choose cooperate/defect
- Pilot: 2 seeds (101, 202) × 2 conditions (comm-on, no-comm), 200k episodes
- entropy_coeff = 0.01 (fixed), msg_dropout = 0.1, lr = 3e-4 (fixed)

---

## Phase 2b (200k checkpoint only)

### Comm vs no-comm

| Condition | Seed | P(C\|f=3.5) | P(C\|f=5.0) |
|---|---|---:|---:|
| Comm-on | 101 | 0.651 | 0.696 |
| No-comm | 101 | 0.414 | 0.571 |
| **Δ** | **101** | **+0.237** | **+0.125** |
| Comm-on | 202 | 0.364 | 0.858 |
| No-comm | 202 | 0.401 | 0.500 |
| **Δ** | **202** | **−0.037** | **+0.358** |

Communication helps at f=5.0 but messages themselves are not causally helping at 200k. Replacing messages with silence or noise *improves* cooperation, especially seed 202:

| Intervention | s202 P(C\|f=3.5) | Δ vs none |
|---|---:|---:|
| none (natural) | 0.364 | — |
| fixed0 | **0.862** | **+0.498** |
| flip | 0.809 | +0.445 |
| zeros | 0.671 | +0.307 |

Benefit of communication is a **training effect**, not informational.

### Mini-ablation (20k from 100k, seed 101)

| Condition | P(C\|f=5.0) | MI(m;f) |
|---|---:|---:|
| Control | 0.740 | 0.001 |
| Entropy-off | **0.781** | **0.024** |
| Dropout-off | 0.459 | 0.014 |
| ent_msg=0 | 0.517 | 0.000 |

**Data:** `outputs/eval/phase2b_suite/`, `outputs/train/phase2b/anneal_diag/`

---

## Phase 3 (50k → 100k → 150k → 200k)

### Message utility trajectory — intervention deltas

Delta = coop_none − coop_intervention. Positive = natural messages hurt.

**Seed 101:**

| Ckpt | f=3.5 zeros | f=3.5 fixed1 | f=5.0 zeros | f=5.0 fixed1 |
|---:|---:|---:|---:|---:|
| 50k | −0.067 | −0.001 | −0.021 | +0.071 |
| 100k | +0.002 | −0.006 | −0.023 | −0.008 |
| 150k | −0.068 | −0.041 | −0.004 | −0.003 |
| 200k | +0.061 | **+0.121** | **+0.054** | −0.010 |

**Seed 202:**

| Ckpt | f=3.5 fixed0 | f=3.5 fixed1 | f=5.0 fixed0 | f=5.0 fixed1 |
|---:|---:|---:|---:|---:|
| 50k | +0.212 | **−0.201** | +0.180 | **−0.270** |
| 100k | **−0.265** | +0.196 | +0.016 | −0.082 |
| 150k | −0.171 | +0.006 | −0.107 | +0.049 |
| 200k | **+0.498** | −0.273 | +0.046 | +0.046 |

### Cross-play: sender drift (delta = crossplay − matched)

**Seed 202:**

| Receiver | Sender | f=3.5 | f=5.0 |
|---|---|---:|---:|
| 200k | 50k | **+0.338** | +0.029 |
| 200k | 100k | +0.238 | +0.018 |
| 200k | 150k | +0.303 | +0.052 |

**Seed 101:**

| Receiver | Sender | f=2.5 | f=3.5 |
|---|---|---:|---:|
| 200k | 50k | +0.175 | +0.020 |
| 150k | 50k | +0.281 | +0.038 |

### Sender semantics at 200k: fragmented private codes

| Seed | Agent | Action Δ | Regime Δ |
|---|---|---:|---:|
| 101 | agent_0 | −0.303 | −0.322 |
| 101 | agent_1 | +0.125 | +0.298 |
| 101 | agent_2 | +0.290 | +0.301 |
| 101 | agent_3 | −0.102 | −0.172 |
| 202 | agent_0 | −0.260 | −0.271 |
| 202 | agent_1 | +0.289 | +0.289 |
| 202 | agent_2 | +0.101 | +0.543 |
| 202 | agent_3 | −0.437 | −0.287 |

### Receiver semantics: aggregate token effect collapses

| Seed | Ckpt | fhat 1.5–2.5 | fhat 2.5–3.5 | fhat ≥4.5 |
|---|---:|---:|---:|---:|
| 101 | 50k | **+0.452** | +0.188 | −0.022 |
| 101 | 200k | +0.004 | +0.010 | +0.003 |
| 202 | 50k | +0.010 | +0.066 | +0.199 |
| 202 | 200k | −0.017 | −0.013 | −0.013 |

**Caveat (from supervisor):** This aggregate metric may mask per-sender decoding. Per-sender receiver analysis is the top priority next step.

**Data:** `outputs/eval/phase3/report/`

---

## Bottom Line (updated with supervisor feedback)

**Paper claim:** Emergent communication in MARL can be transient: useful early, drifting later, and harmful at convergence — endpoint metrics alone are misleading.

**Strongest evidence:**
- Checkpoint-wise causal interventions (not just endpoint MI)
- Sender/receiver decomposition showing sender-side drift
- Convention polarity flips (s202, 50k→100k)
- 50pp harmful message effect at 200k (s202, f=3.5)

**Supervisor-approved next steps (priority order):**

1. **Re-analyse with sender identity preserved** — per-sender receiver response, alignment index over time (no new training, highest value per compute)
2. **Expand to 5+ seeds** on trimmed suite (50k/150k/200k; none/zeros/fixed0/fixed1; early→200k cross-play)
3. **Finish uniform control** with three-way interpretation (learned semantic vs sham zeros vs exogenous random)
4. **Entropy annealing** as primary fix (principled, no oracle checkpoint needed)
5. **Light diagnostic of 100k collapse** only if it recurs

---

## File Locations

| What | Where |
|---|---|
| Full research log (didactic) | `RESEARCH_LOG.md` |
| Phase 2b eval outputs | `outputs/eval/phase2b_suite/` |
| Phase 3 eval outputs | `outputs/eval/phase3/report/` |
| Mini-ablation outputs | `outputs/train/phase2b/anneal_diag/` |
