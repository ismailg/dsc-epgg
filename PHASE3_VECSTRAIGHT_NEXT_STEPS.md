# Phase-3 Vecstraight Next Steps

This file exists so a new Codex chat or collaborator can resume the new phase-3 work without
mixing it up with the older staged/warm-start paper family.

Read this together with:

- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md)
- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/DATA_MAP.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/DATA_MAP.md)

## Scope

This file is about the **new straight vectorized family** only:

- `training_family = phase3_vecstraight`
- repo = `dsc-epgg-vectorized`

Do not use it to justify claims about the old `phase3_staged` manuscript family unless the
analysis is explicitly cross-family.

## Current Scientific Status

The new straight vectorized 15-seed recheck produced:

- `f=3.5, 150k`: `cond1-cond2 = +16.4 pp`
- `f=5.0, 150k`: `cond1-cond2 = +18.8 pp`

Relative to the old `phase3_staged` family:

- the late positive `f=3.5` communication gap survives
- the late negative `f=5.0` communication gap does **not**

Therefore:

- the old late `f=5.0` reversal is **not design-invariant**
- old intervention/mechanism results must **not** be carried over automatically

## Local Data Status

### Already local

- `cond1` straight 15-seed training tree:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323)
- `cond1` greedy checkpoint suite:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324)
- fetched `cond2` eval suite:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325)
- local `cond2` aggregated suite/report root:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325)
- local straight-run comm-gap comparison:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_comm_gap_15seeds_local_20260325`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_comm_gap_15seeds_local_20260325)

### Not fully local yet

- the **full `cond2` training checkpoint tree** is still remote on IWR scratch:
  `/export/scratch/iguennou/runs/dsc-epgg-vectorized/phase3-150k-cond2-15seed-trainonly-20260324`

This matters because same-checkpoint continuation and other new-family manipulations need the
actual `cond2` or `cond1` checkpoint trees, not just aggregated eval CSVs.

## Preflight Before New Experimental Work

Do these before starting the new intervention stack:

1. Fetch the full straight-run `cond2` training tree locally.
2. Commit or otherwise cleanly snapshot the current state of this repo before starting a new
   implementation pass.
3. Keep all new outputs family-labeled with `phase3_vecstraight_*`.
4. Start a fresh Codex session rooted in this repo for the implementation work.

## Why A Fresh Codex Session In This Repo Is Preferred

The current top-level chat has been spanning:

- old manuscript work in `dsc-epgg`
- new implementation and rechecks in `dsc-epgg-vectorized`
- cluster launch/fetch bookkeeping

For the next stage, the work is implementation-heavy and belongs in this repo. A fresh Codex
session rooted at:

- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized)

will reduce confusion and make file-local reasoning cleaner.

## Required Execution Order

Follow this order. Do not skip ahead to expensive reruns.

### Phase 1: parity infrastructure

1. Port the richer evaluator surface from the old repo into this repo.
2. Add the missing phase-3 orchestration runners in this repo.
3. Add or port tests for the new evaluator/runners.

This is necessary because the old repo currently has the full phase-3 manipulation stack, while the
vectorized repo currently has only the base checkpoint suite runner.

### Phase 2: frozen and causal analyses first

Once the evaluation layer exists, run these first on existing checkpoints:

1. `phase3_vecstraight_frozen50k_15seeds_*`
2. `phase3_vecstraight_frozen150k_15seeds_*`
3. `phase3_vecstraight_sender_causal_150k_15seeds_*`

Reason:

- these reuse existing checkpoints
- they are much cheaper than retraining
- they immediately test whether the old endpoint story survives in the new family

### Phase 3: same-checkpoint continuations

After the evaluation layer is stable, add the continuation-training layer and run:

1. branch from `50k`:
   - `fixed0`
   - `uniform`
   - `public_random`
   - `sender_shuffle`
2. branch from `100k`:
   - at minimum `sender_shuffle`
   - plus at least one basic control

Reason:

- the new family is small-gap at `50k` and large-gap at `100k/150k`
- a `50k`-only same-checkpoint suite would miss the late timing story

### Phase 4: history-feature audit

Only after the above:

1. run the history/EWMA audit on `150k`
2. optionally run it at `100k` if the timing still needs disentangling

Do not start with this. It is mechanistic and should come after the new-family endpoint and
continuation facts are established.

## Known Implementation Gap

The vectorized trainer currently supports training-time message interventions:

- `none`
- `uniform`
- `public_random`
- `fixed0`
- `fixed1`

but does **not** currently expose `sender_shuffle` in the training path. This must be added before
the full same-checkpoint continuation stack can be reproduced here.

## Output Naming Rules

Use these exact family labels in new output roots:

- `phase3_vecstraight_frozen50k_...`
- `phase3_vecstraight_frozen150k_...`
- `phase3_vecstraight_sender_causal_...`
- `phase3_vecstraight_sameckpt_continuation_50k_...`
- `phase3_vecstraight_sameckpt_continuation_100k_...`
- `phase3_compare_staged_vs_vecstraight_...`

Do not create new roots like:

- `phase3_sameckpt_*`
- `phase3_frozen150k_*`
- `phase3_sender_causal_*`

without the family token.

## Minimal Handoff Prompt For A New Chat

If starting a new Codex session in this repo, begin with something like:

> Read `PHASE3_VECSTRAIGHT_NEXT_STEPS.md`, `DATA_MAP.md`, and
> `/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md`.
> Work only in `dsc-epgg-vectorized`. First fetch the full remote `cond2` training tree if still
> missing locally. Then implement the phase-3 parity layer in this repo: evaluator interventions,
> suite runners, and `sender_shuffle` training support. Only after that run new-family frozen and
> sender-causal analyses, then same-checkpoint continuations.
