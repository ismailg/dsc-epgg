# Data Map

This is the current human-readable map for finding the main DSC-EPGG artifacts.

Use [`outputs/data_catalog/DATA_CATALOG.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/data_catalog/DATA_CATALOG.md) and [`outputs/data_catalog/data_catalog.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/data_catalog/data_catalog.csv) for the auto-generated index.

## Phase-3 Training Family Split

There are now two non-equivalent phase-3 result families:

- `phase3_staged`:
  the original staged/warm-start family in this repo; this is still the current manuscript-facing
  evidence base
- `phase3_vecstraight`:
  the newer straight `0->150k` vectorized family in
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized)

Read this first before reusing any phase-3 result:

- [`PHASE3_RESULT_FAMILIES.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md)

Use this rule:

1. Keep `phase3_staged` as the current paper family.
2. Treat `phase3_vecstraight` as a candidate successor family.
3. Do not port intervention or mechanism conclusions across those families unless the target
   analysis was rerun there.

For the vectorized family's own path guide, see:

- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/DATA_MAP.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/DATA_MAP.md)

## Current Paper (`phase3_staged`)

- Main 15-seed `50k/100k/150k` checkpoint eval:
  [`outputs/eval/phase3_annealed_ext150k_15seeds/suite/checkpoint_suite_main.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/phase3_annealed_ext150k_15seeds/suite/checkpoint_suite_main.csv)
- Main 15-seed `25k` checkpoint eval:
  [`outputs/eval/paper_strengthen/iter8_base_checkpoint_25k_15seeds/checkpoint_suite_main.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/paper_strengthen/iter8_base_checkpoint_25k_15seeds/checkpoint_suite_main.csv)
- Main 15-seed `150k` training checkpoints:
  [`outputs/train/phase3_annealed_ext150k_15seeds`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase3_annealed_ext150k_15seeds)
- Main 15-seed `1k–50k` online metrics:
  [`outputs/train/phase3_annealed_trimmed_15seeds/metrics`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase3_annealed_trimmed_15seeds/metrics)
- Main 15-seed `55k–150k` online metrics for seeds `404+`:
  [`outputs/train/phase3_annealed_ext150k_15seeds/metrics`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase3_annealed_ext150k_15seeds/metrics)
- Late-metrics supplement for seeds `101/202/303`:
  [`outputs/train/phase3_annealed_ext150k_3seeds/metrics`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase3_annealed_ext150k_3seeds/metrics)
- Main 15-seed `50k` cross-play:
  [`outputs/eval/phase3_base50k_eval_crossplay_15seeds`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/phase3_base50k_eval_crossplay_15seeds)
- Frozen `50k` intervention suite:
  [`outputs/eval/phase3_frozen50k_15seeds`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/phase3_frozen50k_15seeds)
- Frozen `150k` intervention suite used in the revised paper:
  [`outputs/eval/phase3_frozen150k_15seeds_local_20260319`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/phase3_frozen150k_15seeds_local_20260319)
- Same-checkpoint continuation eval used in the revised paper:
  [`outputs/eval/phase3_sameckpt_continuation_15seeds_local_20260319`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/phase3_sameckpt_continuation_15seeds_local_20260319)
- Ongoing/iterative paper-strengthening analyses:
  [`outputs/eval/paper_strengthen`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/paper_strengthen)
  [`outputs/train/paper_strengthen`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/paper_strengthen)
- Current dense online-vs-checkpoint figure:
  [`outputs/eval/paper_strengthen/iter9_online_vs_checkpoint_15seeds`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/paper_strengthen/iter9_online_vs_checkpoint_15seeds)

## Naming Rules That Matter

- `phase3_annealed_trimmed_*`:
  the `1k–50k` annealed training phase
- `phase3_annealed_ext150k_*`:
  the continuation from `50k` out to `150k`
- `phase3_frozen50k_*`:
  frozen-policy interventions evaluated at `50k`
- `phase3_frozen150k_*`:
  frozen-policy interventions evaluated at `150k`
- `phase3_sameckpt_continuation_*`:
  same-checkpoint continuation controls from common `50k` initializations
- `phase3_staged_*`:
  any newly generated result derived from this old staged/warm-start family
- `phase3_vecstraight_*`:
  any newly generated result derived from the new straight vectorized family
- `phase3_compare_staged_vs_vecstraight_*`:
  direct cross-family comparisons; use this instead of ambiguous names
- `paper_strengthen/iter*`:
  one-off analyses and local reruns created during the current paper revision cycle

## Common Confusions

- This repo's phase-3 results are not the only phase-3 family anymore. The new straight vectorized
  family lives in the sibling repo
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized)
  and is documented in
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/DATA_MAP.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/DATA_MAP.md).
- `cond1` / `cond2` are condition labels, not family labels. You now need both the condition and the
  training family to identify a result unambiguously.
- [`outputs/train/phase3_annealed_seed101`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase3_annealed_seed101) is stale and should not be used as a canonical root.
- [`outputs/eval/phase3_frozen_ckpt_15seeds`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/phase3_frozen_ckpt_15seeds) is a legacy path name. For the paper, prefer the explicit `phase3_frozen50k_15seeds` and `phase3_frozen150k_15seeds_local_20260319` roots.
- [`outputs/train/phase3_annealed_ext150k_15seeds/phase3_annealed_ext150k_15seeds`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase3_annealed_ext150k_15seeds/phase3_annealed_ext150k_15seeds) is a nested duplicate tree. Treat the outer root as canonical unless a specific missing file only exists in the nested copy.

## How To Find Data Quickly

- If you know the run family but not the exact path, open:
  [`outputs/data_catalog/DATA_CATALOG.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/data_catalog/DATA_CATALOG.md)
- If you want to filter programmatically by conditions, seeds, or checkpoint episodes, use:
  [`outputs/data_catalog/data_catalog.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/data_catalog/data_catalog.csv)
- If you are looking for paper plots or temporary analysis products from the current revision cycle, start in:
  [`outputs/eval/paper_strengthen`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/eval/paper_strengthen)
