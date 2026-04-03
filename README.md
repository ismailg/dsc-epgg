# MARL-EmeCom: Multi-Agent RL with Emergent Communication in Mixed-Motive Settings

**Paper:** *Learning in Public Goods Games: The Effects of Uncertainty and Communication on Cooperation* (Orzan et al. 2025)  
[Read on SpringerLink](https://link.springer.com/article/10.1007/s00521-024-10530-6)

## Current Repo Orientation

This repository started from the broader `marl-emecom` codebase, but the active implementation and
result family here is the newer vectorized DSC-EPGG pipeline.

Use this repo for:

- the active `src/...` environment, PPO, checkpoint-suite, and analysis code
- the newer straight `0 -> 150k` vectorized phase-3 family: `phase3_vecstraight`
- the current paper-facing vecstraight follow-ups and result audits

Do not mix this with the older staged/warm-start phase-3 manuscript family in the sibling repo:

- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md)

## Read First

For current work, start with:

- [`DATA_MAP.md`](DATA_MAP.md): canonical artifact locations and fetch conventions
- [`PHASE3_VECSTRAIGHT_NEXT_STEPS.md`](PHASE3_VECSTRAIGHT_NEXT_STEPS.md): short handoff note and current gating status
- [`PHASE3_VECSTRAIGHT_PAPER_TODO.md`](PHASE3_VECSTRAIGHT_PAPER_TODO.md): active manuscript-facing checklist
- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md): cross-repo family split

Use the following rule everywhere:

- `phase3_vecstraight` = the newer straight vectorized family in this repo
- `phase3_staged` = the older staged/warm-start manuscript family in `dsc-epgg`
- do not port intervention or mechanism claims across those families unless the analysis was rerun

## Active Code Map

The current implementation lives under `src`, not the older upstream top-level layout.

- `src/environments/pgg/`: vectorized PGG environments
- `src/wrappers/`: trainer-side observation wrapper and message/history features
- `src/algos/`: PPO, GAE buffer, policy/value code
- `src/experiments_pgg_v0/`: training launchers and seed-expansion runners
- `src/analysis/`: checkpoint suites, summaries, validation, and reporting helpers
- `outputs/eval/`: local analysis outputs and summaries
- `iwr-results/` and `hetzner-results/`: fetched remote mirrors

## Historical Notes

- [`README_IMPLEMENTATION.md`](README_IMPLEMENTATION.md) is a historical Week 1-2 implementation plan. It is useful for the original staged implementation scope, but it is not the main entry point for current vecstraight phase-3 work.
- Some older upstream terminology and directory references still appear in parts of the repo. When they disagree with the current phase-3 docs above, prefer the current `src/...` layout and the vecstraight-specific Markdown files.
