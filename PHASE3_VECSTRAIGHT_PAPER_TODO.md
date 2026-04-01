# Phase-3 Vecstraight Paper TODO

This checklist is the current paper-facing TODO for the `phase3_vecstraight` family.

Assumption:

- the corrected vec-parity same-checkpoint continuation reruns from `2026-03-27` finish successfully and replace the earlier invalid de-vectorized continuation lineage;
- frozen endpoint suites, sender-causal probes, and base training-drift summaries already remain valid and do not need to be rerun by default.

Do not treat the older serial/single-env continuation outputs as canonical for manuscript claims.

## Status And Active TODO

### Closed Packages

These items are no longer active TODOs.

### Corrected same-checkpoint continuation package

- the corrected March 29 vec-parity continuation ladder is complete and should be treated as canonical
- do not relaunch the old invalid de-vectorized continuation lineage

### From-scratch exogenous-channel package

- Hetzner `public_random` and canonical `fixed0` are complete and fetched locally
- qx6 `uniform` and `fixed1` are complete and fetched locally
- qx6 `fixed0` is retained as a same-mode replica, not the canonical `fixed0` source
- the one-source-per-mode exogenous summary is rebuilt at
  `outputs/eval/phase3_vecstraight_exogenous_channel_controls_status_20260330/report`
- however, the direct training-time learned-vs-exogenous comparison is currently **provisional**
  rather than final, because intervention runs used a different auxiliary-loss regime

### qx6 loss-switch repair package

- the March 30 qx6 `50k -> 150k` repair batch is complete and summarized at
  `outputs/eval/phase3_vecstraight_lossswitch_controls_status_20260401/report`
- `none_zeroaux` falls below `none_base` at both focal multipliers, especially at `f=3.5`, so the
  auxiliary-loss switch was a real confound
- `uniform_zeroaux` remains close to `none_base` at `f=3.5` and above it at `f=5.0`
- however, the training-time learned-vs-uniform comparison is still not fully clean because the
  forced-channel implementation remains a sender/delivered-message hybrid

### Hetzner comm × history training package

- the four-cell training factorial is complete and summarized at
  `outputs/eval/phase3_vecstraight_comm_history_factorial_status_20260401/report`
- full-history communication clearly outperforms the no-comm branch at both focal multipliers
- under reduced history, the communication advantage nearly disappears and message responsiveness
  drops sharply
- the remaining missing piece is the joint evaluation grid and final interpretation, not the
  training run itself

## Active Main-Paper TODO

### 1. Decide whether training-time matched-random controls or a cleaner forced-channel path are still needed

Important note:

- current `indep_random` / `uniform` and `public_random` controls are plain random-bit controls, not token-rate-matched controls.
- current `sender_shuffle` already serves as the within-sender time-shuffle control; it should not be counted as a separate missing item.
- the endpoint matched-random controls are already done
- after the loss-switch repair, the remaining ambiguity is no longer the auxiliary-loss mismatch; it
  is whether the hybrid sender/delivered-message intervention path is acceptable or whether a
  cleaner direct exogenous-channel implementation is needed

- [ ] Decide whether to stop at the current loss-switch evidence or implement a cleaner direct exogenous-channel training path.
- [ ] If a new training-time exogenous family is still needed, decide whether token-rate-matched random controls are part of it.

### 2. Run the observability / noise sweep

- [ ] Run a small evaluation sweep over observation noise (`eval_sigmas`) for the main comparison families.
- [ ] Add at least one fully observable / public-multiplier condition.
- [ ] Compare learned communication, public random, and no-comm across this sweep.
- [ ] Write the resulting interpretation explicitly as information-transfer versus coordination-signal evidence.

Why this is high priority:

- this is the cleanest experiment for separating information aggregation from pure coordination/correlation effects.

### 3. Finish the communication × temporal-history interaction study

- [ ] Use the fetched four-cell training factorial as the fixed training family for this package.
- [ ] Run a joint evaluation grid that crosses message interventions with history interventions.
- [ ] Decide whether the current endpoint pattern supports substitution, complementarity, or regime-dependent interaction.
- [ ] Fold the final interpretation into the paper-facing result narrative.

Important note:

- the training-time four-cell factorial is now done; what remains is the evaluation-side audit and
  the final scientific interpretation.

### 4. Do the low-dimensional mechanism analyses from existing frozen outputs

- [ ] Compute response curves by received count-of-ones.
- [ ] Compute response curves by coarse signals such as `any_token`.
- [ ] Compare coarse summaries against exact sender-indexed receive patterns.
- [ ] Quantify sender-identity effects separately from coarse message-count effects.
- [ ] Fit a simple surrogate model predicting action from `f_hat`, message summary, sender identity, and history features.
- [ ] Report whether `f=5.0` collapses to a coarse monotone rule while `f=3.5` retains richer dependence.

Expected cost:

- this should mostly be analysis-only, because the frozen suites already contain traces and sender/receiver semantics tables.

### 5. Write the final decomposition result explicitly

- [ ] Make that table/figure separate channel-present, shared-random coordination signal, and learned sender/state-dependent code.
- [ ] Update the manuscript framing around "useful communication without a clean code."
- [ ] Make sure the final claim is tied to corrected vecstraight results only.

## Nice To Have, But Not Required For The Main Paper Path

### Protocol-stability / methodology matrix

- [ ] Compare warm-start versus from-scratch.
- [ ] Compare keep-versus-reset optimizer state.
- [ ] Compare continue-versus-reset entropy schedule.
- [ ] Compare branch points at `25k`, `50k`, and `100k`.
- [ ] Report which qualitative communication claims are design-invariant.

Use this only if the paper pivots toward protocol sensitivity / reproducibility.

## Minimal "Read This First" Set

Before doing new paper-facing phase-3 work, read:

- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/PHASE3_VECSTRAIGHT_PAPER_TODO.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/PHASE3_VECSTRAIGHT_PAPER_TODO.md)
- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/PHASE3_VECSTRAIGHT_NEXT_STEPS.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/PHASE3_VECSTRAIGHT_NEXT_STEPS.md)
- [`/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE3_RESULT_FAMILIES.md)
