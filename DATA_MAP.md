# Data Map

This is the current human-readable map for finding the main DSC-EPGG artifacts.

Use [`outputs/data_catalog/DATA_CATALOG.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/data_catalog/DATA_CATALOG.md) and [`outputs/data_catalog/data_catalog.csv`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/data_catalog/data_catalog.csv) for the auto-generated index.

## Current Paper

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
- `paper_strengthen/iter*`:
  one-off analyses and local reruns created during the current paper revision cycle

## Common Confusions

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
