# Data Map

This is the current human-readable map for the vectorized DSC-EPGG artifacts.

This repo now contains the newer straight vectorized phase-3 family. It is **not** interchangeable
with the older staged/warm-start phase-3 family in
[`/Users/mbp17/POSTDOC/NPS26/dsc-epgg`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg).

Read this first:

- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md)

## Current Phase-3 Family In This Repo

The canonical phase-3 family here is:

- `phase3_vecstraight`:
  uninterrupted `0->150k` vectorized training with `num_envs=8`, `count_env_episodes`, and the
  subprocess rollout backend

This family is currently a **candidate successor** to the older `phase3_staged` manuscript family.
Do not reuse old staged-family intervention claims here unless that analysis has been rerun on this
family.

## Canonical Source Roots

### Training roots

- `cond1` fetched IWR checkpoints:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323)
- standalone `cond1` seed-101 run:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-straight-c1-s101-subproc-20260323`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-straight-c1-s101-subproc-20260323)
- `cond2` train root currently remains on IWR scratch:
  `/export/scratch/iguennou/runs/dsc-epgg-vectorized/phase3-150k-cond2-15seed-trainonly-20260324`

### Main eval/report roots

- `cond1` greedy checkpoint suite:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324)
- `cond2` greedy checkpoint suite:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325)
- fetched remote copy of the `cond2` eval root:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325)
- exact `f=3.5` / `f=5.0` new comm-gap recheck:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_comm_gap_15seeds_local_20260325`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_comm_gap_15seeds_local_20260325)

## Scientific Contrast That Matters

The main cross-family comparison is:

- old `phase3_staged` `cond1-cond2` gap at `150k`:
  `f=3.5 +8.6 pp`, `f=5.0 -24.0 pp`
- new `phase3_vecstraight` `cond1-cond2` gap at `150k`:
  `f=3.5 +16.4 pp`, `f=5.0 +18.8 pp`

So the late positive `f=3.5` communication gap survives, but the late negative `f=5.0` gap does
not.

## Naming Rules That Matter

Use these family tokens in all new outputs:

- `phase3_vecstraight_*`:
  analyses derived from this straight vectorized family
- `phase3_staged_*`:
  analyses derived from the older staged/warm-start family
- `phase3_compare_staged_vs_vecstraight_*`:
  direct cross-family comparisons

Do **not** create new ambiguous roots like:

- `phase3_sameckpt_*`
- `phase3_frozen150k_*`
- `phase3_sender_causal_*`

without a family token.

## Common Confusions

- `cond1` / `cond2` tell you the condition only. They do **not** identify the training family.
- This repo's current phase-3 results are not yet a drop-in replacement for the manuscript's old
  staged family. The intervention stack has not yet been fully rerun here.
- Reports should say:
  `training_family`, `source_repo`, `source_train_root`, `source_eval_root`, `checkpoint episodes`,
  and `seed count`.

