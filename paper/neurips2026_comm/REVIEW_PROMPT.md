# Review Prompt: NeurIPS Draft Consistency Check and Critical Review

## Your role

You are a senior ML researcher acting as an internal reviewer for a NeurIPS submission draft. You have two jobs:

1. **Fact-check every number** in the paper against the actual experimental data files.
2. **Provide a critical review** as if you were a NeurIPS Area Chair seeing this for the first time.

---

## Job 1: Consistency audit

Go through every quantitative claim in `paper/neurips2026_comm/main.tex` and verify it against the source data. The paper makes claims about cooperation rates, communication gaps, channel control results, per-sender effects, alignment patterns, and mute experiment outcomes.

### Source data files to check against

All paths relative to project root (`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/`).

**5-seed 150k main results (comm vs no-comm):**
- `outputs/eval/phase3_annealed_ext150k_5seeds/suite/checkpoint_suite_main.csv` — raw per-seed cooperation rates
- `outputs/eval/phase3_annealed_ext150k_5seeds/report/intervention_delta_table.csv` — intervention deltas
- `outputs/eval/phase3_annealed_ext150k_5seeds/report/sender_semantics_summary.csv` — sender encoding
- `outputs/eval/phase3_annealed_ext150k_5seeds/report/receiver_semantics_summary.csv` — aggregate receiver effects
- `outputs/eval/phase3_annealed_ext150k_5seeds/report/receiver_by_sender_summary.csv` — per-sender receiver effects
- `outputs/eval/phase3_annealed_ext150k_5seeds/report/sender_alignment_summary.csv` — polarity alignment across seeds/checkpoints

**Mute experiment:**
- `outputs/eval/phase3_mute_after50k_ext150k_3seeds/suite/checkpoint_suite_main.csv`
- `outputs/eval/phase3_mute_after50k_ext150k_3seeds/report/intervention_delta_table.csv`

**Channel controls — separate training at 50k:**
- `outputs/eval/phase3_channel_controls_50k/` (look for suite CSVs under each mode subdirectory)
- Or the summary: `outputs/eval/phase3_channel_controls_50k/report/` if it exists

**Channel controls — continuation to 150k (from separately-trained 50k checkpoints):**
- `outputs/eval/phase3_channel_controls_ext150k_3seeds/report/channel_control_summary.csv`

**Same-checkpoint continuation controls (if available — may still be running):**
- `outputs/eval/phase3_sameckpt_continuation_5seeds/` — check if this directory exists and has results

**Trajectory comparison:**
- `outputs/eval/phase3_compare/annealed_vs_unannealed_trajectory_mean.csv`
- `outputs/eval/phase3_compare/annealed_vs_unannealed_trajectory_raw.csv`

**50k annealed results (original Phase 4):**
- `outputs/eval/phase3_annealed_trimmed_all/suite/checkpoint_suite_main.csv`

### What to check

For each table in the paper:

- **Table 1 (comm vs no-comm gap):** Verify all P(C) values for comm_symm and no_comm_symm at 50k, 100k, 150k for f=3.5 and f=5.0. Check means and SEMs. The 50k values should come from the 5-seed eval (phase3_annealed_ext150k_5seeds, milestone 50000). Check that the comm gap arithmetic is correct.

- **Table 2 (separate controls at 50k):** Verify learned=0.809, indep_random=0.669, always_zero=0.641, public_random=0.620 at f=3.5. These come from the channel_controls_50k eval. Note this was 3 seeds only.

- **Table 3 (same-checkpoint continuation):** This may use placeholder data from the 3-seed continuation experiment rather than the 5-seed same-checkpoint experiment. If the same-checkpoint data exists, verify against it. If not, flag that the table is using the wrong experiment's data and note what needs updating.

- **Table 4 (mute experiment):** Verify comm=0.619, no_comm=0.427, mute=0.367 at f=3.5. Check welfare values. Note this is 3 seeds for the mute condition.

- **Table 5 (aggregate vs per-sender):** Verify aggregate token effect and per-sender token effect at 50k, 100k, 150k. These come from receiver_semantics_summary.csv and receiver_by_sender_summary.csv.

- **Table 6 (alignment):** Verify polarity patterns at 50k and 150k for all 5 seeds against sender_alignment_summary.csv.

### Additional claims to verify

- Abstract: "+30 percentage point cooperation benefit" — is 0.304 correctly rounded to 30?
- Abstract: "per-sender causal effects ~9× larger" — is 0.192/0.023 ≈ 8.3, and 0.157/0.018 ≈ 8.7? The paper says ~9×.
- Section 3: "range: +25.1 to +36.6pp" — verify per-seed gaps at 50k.
- Section 3: "4/5 seeds positive" at 150k — verify seed 101 is the only negative one.
- Section 4.1: "14–19pp advantage" over controls — verify 0.809−0.669=0.140, 0.809−0.620=0.189.
- Section 4.3: "6pp below" — verify 0.427−0.367=0.060.
- Section 5: ratio calculations.
- Appendix B: per-seed cooperation values at 50k, 100k, 150k.

### Output format for Job 1

Produce a table:

| Claim location | Paper value | Data value | Status | Notes |
|---|---|---|---|---|
| Table 1, comm f=3.5 50k | 0.809 | ? | ✅/❌/⚠️ | |

Use ✅ for correct, ❌ for wrong, ⚠️ for approximately correct but imprecise (e.g., rounding), 🔍 for unable to verify (data not found).

---

## Job 2: Critical NeurIPS review

After the consistency audit, write a review as if you were an expert NeurIPS reviewer. Use this structure:

### Summary (2-3 sentences)
What the paper does and its main claim.

### Strengths
List 3-5 strengths, each as a short bullet with 1-2 sentence elaboration.

### Weaknesses
List 3-5 weaknesses, each as a short bullet with 1-2 sentence elaboration. Be specific and constructive. Focus on:
- Missing experiments or controls
- Overclaiming vs evidence
- Presentation issues (is §4 really the flagship? does the narrative flow?)
- Statistical concerns (5 seeds, no CIs yet, mute is only 3 seeds)
- Scope/generalizability
- Missing related work
- Whether the 9-page limit is respected

### Questions for the authors
3-5 specific questions a reviewer would ask.

### Missing references
List any important related work not cited. Consider:
- Recent emergent communication papers (2023-2025)
- Convention formation in MARL
- Non-stationarity in independent learners
- Public goods game experiments (human or computational)
- Causal evaluation of communication
- Any paper that does per-sender or per-receiver disaggregation

### Minor issues
Typos, formatting, LaTeX issues, figure suggestions, etc.

### Overall recommendation
Score 1-10 and accept/weak accept/borderline/weak reject/reject, with 2-sentence justification.

---

## Important notes

- Read the full `DIDACTIC_OVERVIEW.md` for context on the experimental history and what each experiment means.
- Read the full `RESEARCH_LOG.md` for the chronological record of results and supervisor feedback.
- Read `docs/notes/NEXT_STEPS_IMPLEMENTATION_PROMPT.md` for what experiments are still pending.
- The paper is a FIRST DRAFT. Be constructive. Distinguish between "this is wrong" and "this needs work."
- Pay special attention to whether the paper correctly distinguishes between the two different channel control experiments:
  - **Separate-training controls** (each mode trained from scratch, 3 seeds)
  - **Same-checkpoint continuation controls** (all modes from identical learned 50k checkpoint, may not exist yet)
  - The paper's Table 3 claims to present same-checkpoint results but may actually be using the separate-training continuation data. This would be a significant error.
- Check whether the paper's narrative matches the supervisor's recommended structure (§1 Intro → §2 Setup → §3 Comm helps → §4 Semantic window → §5 Aggregate metrics → §6 Discussion). The supervisor specifically asked for the flagship result (semantic window) to come before the aggregate metrics section.
