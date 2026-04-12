# Phase-3 Vecstraight Next Steps

This file is the short status and handoff note for the active `phase3_vecstraight` work.

It should answer only three questions:

1. What is the current scientific status?
2. What is the current immediate gate?
3. Which file should I open next?

For artifact locations, use [`DATA_MAP.md`](DATA_MAP.md).  
For the active manuscript checklist, use [`PHASE3_VECSTRAIGHT_PAPER_TODO.md`](PHASE3_VECSTRAIGHT_PAPER_TODO.md).  
For the longer interpretive narrative, use [`PHASE3_VECSTRAIGHT_DIDACTIC_OVERVIEW.md`](PHASE3_VECSTRAIGHT_DIDACTIC_OVERVIEW.md).  
For historical implementation/planning material, use [`docs/archive/README.md`](docs/archive/README.md).

## Scope

This file is about the newer straight vectorized family only:

- `training_family = phase3_vecstraight`
- repo = `dsc-epgg-vectorized`

Do not use it to justify claims about the older staged/warm-start family unless the comparison is explicitly cross-family:

- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md)

## Current Scientific Status

The key vecstraight facts are:

- the late communication gap is positive at both focal multipliers by 150k:
  - `f=3.5`: `+16.4 pp`
  - `f=5.0`: `+18.8 pp`
- the clean direct message-source family shows:
  - `uniform ≈ learned`
  - `public_random ≈ no-comm`
  - `fixed0` / `fixed1` are harmful
- the comm × history factorial shows:
  - communication has a clear endpoint advantage under full history
  - that advantage nearly vanishes under reduced history
  - message responsiveness falls sharply in the reduced-history branch

So the current training-time interpretation is:

- learned sender content is not uniquely necessary
- a single shared random bit is not enough
- temporal context matters for making any message channel useful

## Current Immediate Gate

There is no active live training gate right now. The main scientific rerun packages that matter for the current paper path are complete and local:

- qx6 loss-switch repair summary:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_lossswitch_controls_status_20260401/report`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_lossswitch_controls_status_20260401/report)
- Hetzner comm × history training summary:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_comm_history_factorial_status_20260401/report`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_comm_history_factorial_status_20260401/report)
- Hetzner clean direct msg-source summary:
  [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_clean_msgsource_status_20260412/report`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/outputs/eval/phase3_vecstraight_clean_msgsource_status_20260412/report)

The current gate is therefore a paper and interpretation gate, not an implementation gate.

## Likely Next Actions

If the goal is the current submission:

1. finish the remaining manuscript polish and checklist cleanup
2. decide how much compute/license disclosure to add in Appendix G
3. rerender and verify the final PDF/checklist bundle

If the goal is follow-up science after submission:

1. run the observability/noise sweep for the clean message-source comparison
2. run the joint evaluation grid for the completed comm × history training family

## Operational Cautions

- Do not use `quadopt4` or `quadopt5` for vecstraight reruns.
- Same-checkpoint continuation results still require parity checks against [`AGENTS.md`](AGENTS.md) before reuse.
- Use [`DATA_MAP.md`](DATA_MAP.md) as the path registry; do not copy dated run inventories back into this file.

## File Ownership

Use the active docs this way:

- [`README.md`](README.md): repo entry point
- [`DATA_MAP.md`](DATA_MAP.md): canonical local/remote artifact locations
- [`PHASE3_VECSTRAIGHT_PAPER_TODO.md`](PHASE3_VECSTRAIGHT_PAPER_TODO.md): active paper-facing checklist
- [`PHASE3_VECSTRAIGHT_DIDACTIC_OVERVIEW.md`](PHASE3_VECSTRAIGHT_DIDACTIC_OVERVIEW.md): interpretation
- [`AGENTS.md`](AGENTS.md): stable operating rules and parity contract

