# Implementation Notes (Stage 3: Logging + Regime Audit)

## Session logging

File: `src/logging/session_logger.py`

Implemented:
- Per-session `.npz` writer:
  - `true_f`, `f_hats`, `intended_actions`, `executed_actions`, `flips`,
    `rewards`, `messages`, `cooperation_count`, `welfare`.
- Consolidation:
  - Stacks per-session files into one consolidated file per `(condition, seed)`.
  - Optional deletion of shard files after consolidation.

## Regime identifiability audit

File: `src/analysis/regime_audit.py`

Implemented:
- Exact HMM forward filter over discrete `F` regimes.
- Gaussian observation likelihood matching env (`no clamping` assumption).
- Switch-triggered convergence timing:
  - At each true regime switch, count rounds until `max(pi_t) > threshold` (default `0.9`).
- Report fields:
  - `mean`, `median`, `p90`, `n_switches`, `recommendation`.
- Recommendation rule:
  - If `median <= 2`: `increase_sigma_or_reduce_F`, else `ok`.

## Validation

Added tests in `tests/test_logging_audit.py`:
- Session logger + consolidation shape checks.
- Regime audit execution and output-schema checks.

