# MARL-EmeCom: Vectorized DSC-EPGG

**Paper:** *Learning in Public Goods Games: The Effects of Uncertainty and Communication on Cooperation* (Orzan et al. 2025)  
[Read on SpringerLink](https://link.springer.com/article/10.1007/s00521-024-10530-6)

This repository is the active home of the newer vectorized DSC-EPGG pipeline and the
`phase3_vecstraight` result family.

Do not mix it with the older staged/warm-start manuscript family in the sibling repo:

- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md)

## Read First

For current work, open these files in order:

1. [`AGENTS.md`](AGENTS.md): stable operating rules, parity contract, and cluster policy
2. [`DATA_MAP.md`](DATA_MAP.md): canonical local/remote artifact paths
3. [`PHASE3_VECSTRAIGHT_NEXT_STEPS.md`](PHASE3_VECSTRAIGHT_NEXT_STEPS.md): current state and next action
4. [`PHASE3_VECSTRAIGHT_PAPER_TODO.md`](PHASE3_VECSTRAIGHT_PAPER_TODO.md): active paper-facing checklist
5. [`PHASE3_VECSTRAIGHT_DIDACTIC_OVERVIEW.md`](PHASE3_VECSTRAIGHT_DIDACTIC_OVERVIEW.md): longer result narrative

## Active Ownership Model

Treat the root docs this way:

- [`README.md`](README.md): repo entry point only
- [`DATA_MAP.md`](DATA_MAP.md): paths only
- [`PHASE3_VECSTRAIGHT_NEXT_STEPS.md`](PHASE3_VECSTRAIGHT_NEXT_STEPS.md): current state and next step only
- [`PHASE3_VECSTRAIGHT_PAPER_TODO.md`](PHASE3_VECSTRAIGHT_PAPER_TODO.md): active manuscript checklist only
- [`PHASE3_VECSTRAIGHT_DIDACTIC_OVERVIEW.md`](PHASE3_VECSTRAIGHT_DIDACTIC_OVERVIEW.md): interpretation only

Historical implementation plans, stage notes, and older task prompts now live under:

- [`docs/archive/README.md`](docs/archive/README.md)

## Current Code Map

The active implementation lives under `src`:

- `src/environments/pgg/`: vectorized PGG environments
- `src/wrappers/`: trainer-side observation wrapper and message/history features
- `src/algos/`: PPO, GAE buffer, policy/value code
- `src/experiments_pgg_v0/`: training launchers and seed-expansion runners
- `src/analysis/`: checkpoint suites, summaries, validation, and reporting helpers

Supporting trees:

- `outputs/eval/`: local analysis outputs and summaries
- `iwr-results/`: fetched IWR mirrors
- `hetzner-results/`: fetched Hetzner mirrors
- `paper/neurips2026_comm_vecstraight/`: active manuscript workspace

## Current Family Split

Use these names consistently:

- `phase3_vecstraight` = uninterrupted `0 -> 150k` straight vectorized family in this repo
- `phase3_staged` = older staged/warm-start manuscript family in `dsc-epgg`

Do not port intervention or mechanism claims across those families unless the analysis was rerun.

