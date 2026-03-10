# Condition Naming Plan

## Current semantics

These names are what the active pipeline is using today:

- `cond1`
  - communication enabled
  - symmetric uncertainty (`sigmas = 0.5, 0.5, 0.5, 0.5`)
  - tremble enabled (`epsilon_tremble = 0.05`)
- `cond2`
  - communication disabled
  - symmetric uncertainty (`sigmas = 0.5, 0.5, 0.5, 0.5`)
  - tremble enabled (`epsilon_tremble = 0.05`)

So `cond1` / `cond2` are **not** generic “comm” / “no_comm” conditions.
They are specifically the symmetric-uncertainty pair used for the main
communication contrast.

## Recommended replacement

For future code, files, and figures:

- `cond1` -> `comm_symm`
- `cond2` -> `no_comm_symm`

For human-facing plot labels:

- `Comm-On (Symmetric Uncertainty)`
- `No-Comm (Symmetric Uncertainty)`

## Why not plain `comm` / `no_comm`

Plain `comm` / `no_comm` becomes ambiguous once the repo includes:

- no-uncertainty baselines
- asymmetric-uncertainty variants
- sham/random/public-random channel controls

`comm_symm` / `no_comm_symm` keeps the main comparison readable without
colliding with those other conditions.

## What is safe while runs are active

Safe now:

- documentation updates
- post-hoc alias tables in reports
- a future compatibility layer that maps legacy names to clearer labels

Not safe until the active pipeline finishes:

- renaming checkpoint files
- changing `condition_name` values in training commands
- changing output directory names
- changing parsers/runners that the current pipeline will still execute

## Post-run migration plan

1. Add a central condition-label helper used by analysis/report scripts.
2. Keep backward compatibility for legacy checkpoints (`cond1`, `cond2`).
3. Emit clearer labels in reports immediately.
4. Switch future training runs to `comm_symm` / `no_comm_symm`.
5. Only rename on-disk files if we also keep a legacy-resolution fallback.
