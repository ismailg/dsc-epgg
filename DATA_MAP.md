# Data Map

This file is the canonical path registry for the vectorized DSC-EPGG work in this repo.

It should answer only two questions:

1. Which result family is in scope here?
2. Where do the authoritative local and remote artifacts live?

For current status, use [`PHASE3_VECSTRAIGHT_NEXT_STEPS.md`](PHASE3_VECSTRAIGHT_NEXT_STEPS.md).  
For the active manuscript checklist, use [`PHASE3_VECSTRAIGHT_PAPER_TODO.md`](PHASE3_VECSTRAIGHT_PAPER_TODO.md).  
For dated run chronology, use [`docs/archive/PHASE3_RUN_HISTORY.md`](docs/archive/PHASE3_RUN_HISTORY.md).

## Family Scope

This repo's active phase-3 family is:

- `phase3_vecstraight`: uninterrupted `0 -> 150k` vectorized training with `num_envs=8`,
  `count_env_episodes`, and the subprocess rollout backend

Do not treat it as interchangeable with the older staged/warm-start family in the sibling repo:

- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md)

## Canonical Local Roots

### Training trees

- `cond1` 15-seed train tree:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323)
- `cond2` 15-seed train tree:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond2-15seed-trainonly-20260324`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond2-15seed-trainonly-20260324)
- standalone `cond1` seed-101 train tree:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-straight-c1-s101-subproc-20260323`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-straight-c1-s101-subproc-20260323)

### Main local eval/report roots

- `cond1` greedy checkpoint suite:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324)
- `cond2` greedy checkpoint suite:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325)
- late comm-gap recheck:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_comm_gap_15seeds_local_20260325`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vectorized_comm_gap_15seeds_local_20260325)
- corrected continuation summary:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_paper_pivot_20260329_status/sameckpt_continuations`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_paper_pivot_20260329_status/sameckpt_continuations)
- historical exogenous channel summary:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_exogenous_channel_controls_status_20260330/report`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_exogenous_channel_controls_status_20260330/report)
- qx6 loss-switch repair summary:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_lossswitch_controls_status_20260401/report`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_lossswitch_controls_status_20260401/report)
- Hetzner comm × history training summary:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_comm_history_factorial_status_20260401/report`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_comm_history_factorial_status_20260401/report)
- Hetzner clean direct msg-source summary:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_clean_msgsource_status_20260412/report`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_clean_msgsource_status_20260412/report)

### Local fetched mirrors by family

Use these patterns when looking for fetched manipulation families:

- qx6 continuation mirrors:
  - `iwr-results/phase3_vecstraight_sameckpt_continuation_50000_{public_random,sender_shuffle,fixed0}_15seeds_iwr_20260327`
  - `iwr-results/phase3_vecstraight_sameckpt_continuation_100000_{sender_shuffle,fixed0}_15seeds_iwr_20260327`
- Hetzner continuation mirror:
  - `hetzner-results/phase3_vecstraight_sameckpt_continuation_50000_uniform_15seeds_hetzner_20260327`
- qx6 exogenous mirrors:
  - `iwr-results/phase3_vecstraight_exogenous_{uniform,fixed0,fixed1}_15seeds_iwr_20260328`
- Hetzner exogenous mirrors:
  - `hetzner-results/phase3_vecstraight_exogenous_{public_random,fixed0}_15seeds_hetzner_20260328`
- qx6 loss-switch mirrors:
  - `iwr-results/phase3_vecstraight_sameckpt_continuation_50000_{none_base,none_zeroaux,uniform_zeroaux}_15seeds_iwr_20260330`
- Hetzner comm × history mirrors:
  - `hetzner-results/phase3_vecstraight_comm_history_factorial_{with_comm_full_history,with_comm_reduced_history,without_comm_full_history,without_comm_reduced_history}_15seeds_hetzner_20260330par24`
- Hetzner clean direct msg-source mirrors:
  - `hetzner-results/phase3_vecstraight_clean_msgsource_{learned,public_random,uniform,fixed0,fixed1}_15seeds_hetzner_20260410`

## Canonical Remote Roots

### Hetzner

- project dir:
  `/root/compute-work/projects/dsc-epgg-vectorized`
- run-dir base:
  `/root/compute-work/runs/dsc-epgg-vectorized`

### IWR / qx6

- qx6 staging base for vecstraight reruns:
  `/export/scratch/iguennou/staging/dsc-epgg-vectorized`
- qx6 run-dir base:
  `/export/scratch/iguennou/runs/dsc-epgg-vectorized`

For dated completed-run notes, use [`docs/archive/PHASE3_RUN_HISTORY.md`](docs/archive/PHASE3_RUN_HISTORY.md).

## Fetch Conventions

- Fetch Hetzner eval or train roots from
  `/root/compute-work/projects/dsc-epgg-vectorized/outputs/{eval,train}/<basename>` into
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/hetzner-results/<basename>`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/hetzner-results)
- Fetch IWR/qx6 eval roots from the relevant staging `outputs/eval/<basename>` into
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/<basename>`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results)
- After fetching a manuscript-facing root, record it here and keep the basename unchanged.

## Naming Rules

Use explicit family tokens in new output roots:

- `phase3_vecstraight_*`: analyses derived from this straight vectorized family
- `phase3_staged_*`: analyses derived from the older staged/warm-start family
- `phase3_compare_staged_vs_vecstraight_*`: direct cross-family comparisons

Avoid ambiguous names such as:

- `phase3_sameckpt_*`
- `phase3_frozen150k_*`
- `phase3_sender_causal_*`

without a family token.

## Common Confusions

- `cond1` and `cond2` identify the condition only, not the training family.
- Use this file for path ownership, not for scientific interpretation.
- Use [`PHASE3_VECSTRAIGHT_NEXT_STEPS.md`](PHASE3_VECSTRAIGHT_NEXT_STEPS.md) for live status.
- Use [`PHASE3_VECSTRAIGHT_PAPER_TODO.md`](PHASE3_VECSTRAIGHT_PAPER_TODO.md) for the active remaining-work checklist.

