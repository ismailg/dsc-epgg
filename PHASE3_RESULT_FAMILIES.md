# Phase-3 Result Families And Migration Plan

This note exists to prevent a very specific confusion:

- the **original manuscript-facing phase-3 results** come from one training design
- the **new straight vectorized 15-seed recheck** comes from a different training design

Fifteen seeds stabilize estimates **within** a design. They do **not** make two different designs interchangeable.

Use this file as the first stop whenever you need to answer:

- which phase-3 result family a file belongs to
- whether an old intervention/mechanism analysis can be reused for a new training run
- how new outputs should be named so old and new analyses do not get mixed

---

## Executive Rule

Do **not** completely replace the old staged/warm-start analyses yet.

Do **not** ignore them either.

Use this rule instead:

1. Treat the old staged/warm-start phase-3 family as the **current paper family**.
2. Treat the new straight vectorized phase-3 family as the **candidate successor family**.
3. Do not port intervention/mechanism conclusions from one family to the other unless that analysis has been rerun on the target family.

The reason is simple:

- old staged/warm-start family: late `f=5.0` communication gap is negative
- new straight vectorized family: late `f=5.0` communication gap is positive

So the sign of the headline result changed.

---

## Family Table

| Family token | Human label | Repo | Training design | Current status |
|---|---|---|---|---|
| `phase3_staged` | Phase-3 staged/warm-start | `dsc-epgg` | annealed `1k->50k` phase, then continued `50k->150k`; staged schedule and warm-start structure | current manuscript evidence base |
| `phase3_vecstraight` | Phase-3 straight vectorized | `dsc-epgg-vectorized` | uninterrupted `0->150k`; vectorized `num_envs=8`; subprocess rollout backend | candidate successor evidence base |

Interpretation:

- `phase3_staged` is the family on which the paper's same-checkpoint controls, frozen interventions, sender-specific probe, and history-feedback audit were run.
- `phase3_vecstraight` is the family that produced the newer cross-design recheck showing that the old late `f=5.0` reversal is not design-invariant.

---

## Canonical Source Roots

### `phase3_staged` (current paper family)

Training:
- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase3_annealed_trimmed_15seeds`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase3_annealed_trimmed_15seeds)
- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase3_annealed_ext150k_15seeds`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase3_annealed_ext150k_15seeds)

Main eval/report roots:
- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/phase3_annealed_ext150k_15seeds`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/phase3_annealed_ext150k_15seeds)
- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/paper_strengthen/iter8_base_checkpoint_25k_15seeds`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/paper_strengthen/iter8_base_checkpoint_25k_15seeds)
- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/phase3_sameckpt_continuation_15seeds_local_20260319`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/phase3_sameckpt_continuation_15seeds_local_20260319)
- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/phase3_frozen150k_15seeds_local_20260319`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/phase3_frozen150k_15seeds_local_20260319)
- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/paper_strengthen`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/paper_strengthen)

### `phase3_vecstraight` (candidate successor family)

Training roots:
- `cond1` fetched checkpoints:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323)
- `cond2` train root currently lives on IWR scratch:
  `/export/scratch/iguennou/runs/dsc-epgg-vectorized/phase3-150k-cond2-15seed-trainonly-20260324`

Main eval/report roots:
- `cond1` suite:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324)
- `cond2` suite:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325)
- cross-design communication-gap recheck:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_comm_gap_15seeds_local_20260325`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_comm_gap_15seeds_local_20260325)

---

## What Changed Scientifically

The decisive comparison is the greedy checkpoint-suite communication gap (`cond1 - cond2`) at exact manuscript multipliers:

| Family | `f=3.5, 150k` | `f=5.0, 150k` |
|---|---:|---:|
| `phase3_staged` | `+8.6 pp` | `-24.0 pp` |
| `phase3_vecstraight` | `+16.4 pp` | `+18.8 pp` |

So:

- the late positive `f=3.5` gap survives across both families
- the late negative `f=5.0` gap does **not**

This means:

- the old late `f=5.0` reversal is **trajectory-dependent**
- the old history-feedback audit explains what happened in `phase3_staged`
- it does **not** automatically explain `phase3_vecstraight`

---

## What To Keep, What To Replace

### Keep as current paper evidence

Keep these as the load-bearing evidence **for the current paper draft**:

- old staged/warm-start base comparison
- same-checkpoint continuation controls
- matched-stat sender-shuffle continuation
- frozen 150k intervention suite
- sender-specific causal probe
- history-feedback / EWMA audit

Why:

- these analyses were all run on the same family
- they are internally coherent
- the paper text is currently written around them

### Do not treat as portable across families

Do **not** assume the following can be carried from `phase3_staged` to `phase3_vecstraight` without rerunning:

- the late negative `f=5.0` communication gap
- the “state-contingent messages actively hurt at `f=5.0`” conclusion
- the EWMA-conditioned free-riding mechanism as the explanation for the new family
- any frozen intervention ranking at `150k`
- any sender-shuffle continuation effect size at `f=5.0`

### Candidate replacement path

If we want `phase3_vecstraight` to replace `phase3_staged` as the canonical family, rerun this stack on `phase3_vecstraight` first:

1. same-checkpoint continuation controls
2. matched-stat sender-shuffle continuation
3. frozen `150k` intervention suite
4. sender-specific causal probe
5. history-feature / EWMA audit

Until that happens, the correct position is:

- `phase3_staged` = canonical paper family
- `phase3_vecstraight` = serious cross-design challenge and candidate successor family

---

## Naming Convention Going Forward

The current file tree mixes old names like `phase3_annealed_*` with new names like `phase3_vectorized_*`. That is survivable now, but it will become confusing once we start rerunning interventions on the new family.

Use these family tokens in all **new** output roots:

- `phase3_staged_*` for any newly generated analysis derived from the old staged/warm-start family
- `phase3_vecstraight_*` for any newly generated analysis derived from the new straight vectorized family
- `phase3_compare_staged_vs_vecstraight_*` for direct cross-family comparison outputs

Examples:

- `outputs/eval/phase3_staged_frozen150k_15seeds_rerun_YYYYMMDD`
- `outputs/eval/phase3_vecstraight_frozen150k_15seeds_YYYYMMDD`
- `outputs/eval/phase3_vecstraight_sameckpt_continuation_15seeds_YYYYMMDD`
- `outputs/eval/phase3_compare_staged_vs_vecstraight_basegap_YYYYMMDD`

Do **not** create new ambiguous roots like:

- `phase3_frozen150k_*`
- `phase3_sameckpt_*`
- `phase3_sender_causal_*`

without a family token. Those names are now underspecified.

---

## Condition Naming Inside Families

Within either family:

- keep legacy checkpoint names `cond1` / `cond2` for backward compatibility
- use human-facing aliases `comm_symm` / `no_comm_symm` in summaries and prose when possible

But note:

- `cond1` / `cond2` tell you the **condition**
- they do **not** tell you the **training family**

You now need both.

---

## Minimum Metadata For New Reports

Every new report or summary produced from phase-3 data should state these fields explicitly near the top:

- `training_family`: `phase3_staged` or `phase3_vecstraight`
- `source_repo`: `dsc-epgg` or `dsc-epgg-vectorized`
- `source_train_root`
- `source_eval_root`
- `condition set`: `cond1/cond2` or human aliases
- `checkpoint episodes`
- `seed count`

If a report does not say which family it belongs to, treat it as incomplete.

---

## Recommended Migration Plan

### Phase A: stabilize documentation now

Do this immediately:

1. keep the current manuscript tied explicitly to `phase3_staged`
2. add family labels to all new summaries and notes
3. name all future outputs with explicit family tokens

### Phase B: choose whether to promote the new family

Only promote `phase3_vecstraight` after rerunning the core ablation/intervention stack on it.

### Phase C: if promotion happens

If `phase3_vecstraight` becomes canonical:

1. move `phase3_staged` to “legacy paper family / historical mechanism evidence”
2. replace manuscript base-comparison tables with `phase3_vecstraight`
3. replace manuscript intervention claims only after the corresponding analyses exist in `phase3_vecstraight`

---

## Bottom Line

The correct organizational stance is:

- keep the old family
- keep the new family
- never mix them implicitly
- make the family label part of the name

That is the only safe way to avoid collaborator confusion while the evidence base is in transition.
