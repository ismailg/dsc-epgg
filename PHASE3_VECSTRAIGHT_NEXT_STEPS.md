# Phase-3 Vecstraight Next Steps

This file is the short handoff note for the active vecstraight phase-3 work.

It should answer three questions only:

1. What is the current scientific status?
2. What is the current gating decision?
3. Which file should I open next?

For artifact locations, use [`DATA_MAP.md`](DATA_MAP.md).  
For the active manuscript checklist, use [`PHASE3_VECSTRAIGHT_PAPER_TODO.md`](PHASE3_VECSTRAIGHT_PAPER_TODO.md).  
For the narrative interpretation, use [`PHASE3_VECSTRAIGHT_DIDACTIC_OVERVIEW.md`](PHASE3_VECSTRAIGHT_DIDACTIC_OVERVIEW.md).

## Scope

This file is about the newer straight vectorized family only:

- `training_family = phase3_vecstraight`
- repo = `dsc-epgg-vectorized`

Do not use it to justify claims about the older staged/warm-start manuscript family unless the
comparison is explicitly cross-family. For that split, read:

- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md)

## Current Scientific Status

The main vecstraight fact that changes the older manuscript story is still:

- `f=3.5, 150k`: `cond1 - cond2 = +16.4 pp`
- `f=5.0, 150k`: `cond1 - cond2 = +18.8 pp`

So, relative to the older `phase3_staged` family:

- the late positive `f=3.5` communication gap survives
- the late negative `f=5.0` reversal does not

That means older staged-family intervention and mechanism claims are not portable by default.

## Current Gating Decision

There is no active live training gate right now. The two March 30 repair families are complete and
fetched locally:

- qx6 loss-switch repair summary:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_lossswitch_controls_status_20260401/report`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_lossswitch_controls_status_20260401/report)
- Hetzner comm × history training summary:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_comm_history_factorial_status_20260401/report`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_comm_history_factorial_status_20260401/report)

Current takeaways:

- the loss-switch repair shows a real objective-switch effect:
  `none_zeroaux` falls below `none_base` at both focal multipliers, especially at `f=3.5`
- under matched zero-aux training, `uniform_zeroaux` recovers that loss at `f=3.5` and exceeds
  `none_base` at `f=5.0`, but the direct training-time learned-vs-uniform claim is still not fully
  clean because the forced-channel implementation remains a sender/delivered-message hybrid
- the comm × history factorial shows a strong history dependence:
  communication has a clear endpoint advantage under full history, but that advantage nearly
  vanishes under reduced history, and message responsiveness is much lower in the reduced-history
  branch

So the current gate is a decision, not a run:

- if the next question is about exogenous training-time controls, the remaining issue is the hybrid
  forced-message path rather than the auxiliary-loss mismatch alone
- if the next question is about mechanism, the next high-value steps are the observability sweep and
  the joint evaluation grid for the completed comm × history training family

## Current Operational Cautions

- Do not use `quadopt4` or `quadopt5` for vecstraight reruns. They failed immediately on March 27,
  2026 with host-runtime incompatibilities.
- Same-checkpoint continuation results should not be trusted until at least one saved `.run.json`
  is checked against the parity contract in [`AGENTS.md`](AGENTS.md).
- Use [`DATA_MAP.md`](DATA_MAP.md) as the path registry. This file deliberately avoids repeating
  full artifact inventories.

## Current Local Data Status

The important train trees are already local:

- `cond1` mirror:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond1-15seed-trainonly-20260323)
- `cond2` mirror:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond2-15seed-trainonly-20260324`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/iwr-results/phase3-150k-cond2-15seed-trainonly-20260324)

That means new same-checkpoint manipulations do not need to treat the full `cond2` tree as
remote-only.

## File Ownership

Use the docs this way:

- [`README.md`](README.md): repo entry point and active code map
- [`DATA_MAP.md`](DATA_MAP.md): canonical local/remote artifact locations
- [`PHASE3_VECSTRAIGHT_PAPER_TODO.md`](PHASE3_VECSTRAIGHT_PAPER_TODO.md): active paper-facing checklist
- [`PHASE3_VECSTRAIGHT_DIDACTIC_OVERVIEW.md`](PHASE3_VECSTRAIGHT_DIDACTIC_OVERVIEW.md): longer interpretation and result narrative
- [`AGENTS.md`](AGENTS.md): stable operating rules, parity contract, and cluster policy

## Minimal Handoff Prompt

If starting a new Codex session in this repo, begin with:

> Read `README.md`, `PHASE3_VECSTRAIGHT_NEXT_STEPS.md`, `DATA_MAP.md`, and
> `PHASE3_VECSTRAIGHT_PAPER_TODO.md`. Work only in `dsc-epgg-vectorized`.
> Treat `DATA_MAP.md` as the path registry, `PHASE3_VECSTRAIGHT_PAPER_TODO.md` as the active
> checklist, and this file as the short status/handoff note.
