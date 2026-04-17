# Zero-Aux Old-Repo Handoff

This worktree exists to carry out the compute-side portion of the zero-aux paper transition.

## Branch / worktree

- branch: `codex/zero-aux-paper-transition`
- worktree: `/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized-zeroaux-plan`

## Read first

1. `ZERO_AUX_PAPER_TRANSITION_PLAN.md`
2. `README.md`
3. `DATA_MAP.md`
4. `PHASE3_VECSTRAIGHT_NEXT_STEPS.md`
5. `PHASE3_VECSTRAIGHT_PAPER_TODO.md`

## What has already been decided

- use zero-aux `clean_msgsource_learned` as the canonical `with_comm_full` communication cell
- accept the no-comm factorial cells as functionally zero-aux controls
- drop the loss-switch / continuation-control section from the zero-aux paper path
- keep the noise-sweep appendix in scope
- keep the message-history-grid appendix off the blocking path
- target artifact-profile name in `dsc-epgg-tuned`: `zero_aux_current`

## Section 4.1 source of truth

Future agents should not re-decide the base communication-gap mapping.

- Section 4.1 / base communication gap uses zero-aux `clean_msgsource_learned` as the canonical full-history `comm` arm.
- The comparison baseline for that section is the canonical no-comm full-history factorial cell: `comm_history_factorial_without_comm_full_history`.
- Do not use the old March base-gap family (`phase3_vectorized_comm_gap_15seeds_local_20260325`) for the zero-aux paper path.
- Do not substitute the matched no-comm row bundled inside the clean-msgsource family when rebuilding the paper's base gap; that row is for the clean message-source family, not the canonical Section 4.1 baseline.
- In this worktree, the intended paper-facing zero-aux CSV for Section 4.1 is:
  `outputs/eval/phase3_vecstraight_zeroaux_base_gap_local_20260415/report/exact_f_gap_table.csv`
- The manifest-backed checkpoint sources for that CSV are:
  - `cond1`: `phase3_vecstraight_clean_msgsource_learned_15seeds_hetzner_20260410`
  - `cond2`: `phase3_vecstraight_comm_history_factorial_without_comm_full_history_15seeds_hetzner_20260330par24`
- If `paper/neurips2026_comm_vecstraight/main.qmd` points at the old March base-gap CSV, that is a zero-aux transition bug and should be corrected, not rationalized.

## What this worktree should do next

1. Add a one-cell zero-aux training path for `with_comm_reduced_history`.
   Keep the existing factorial contract fixed except set:
   - `sign_lambda=0.0`
   - `list_lambda=0.0`
   - `history_mode=reduced`
   - `comm_enabled=true`
   - `disable_comm_fallback=true`

2. Launch that one missing 15-seed training cell.

3. Build clean manifest files for the zero-aux paper path covering:
   - zero-aux `clean_msgsource_learned`
   - canonical no-comm full-history baseline
   - canonical no-comm reduced-history baseline
   - new zero-aux `with_comm_reduced_history`

4. Rebuild compute-side evaluation outputs needed for the paper:
   - comm-gap checkpoint comparison
   - clean msg-source summary
   - comm × history factorial summary
   - frozen expanded suite
   - sender-causal suite
   - cross-seed transfer suite
   - noise sweep

5. Do not block on message-history-grid unless scope is explicitly reopened.

## Guardrails

- Do not reuse the dirty main old-repo worktree for this effort.
- Do not touch the current `dsc-epgg-tuned` artifact registry from here.
- Do not reintroduce loss-switch outputs into the zero-aux paper path.
- Do not silently change any training contract beyond the one missing zero-aux cell.

## Expected deliverables back to the paper repo

- compact CSV/JSON outputs only
- enough provenance to map every zero-aux paper section to a compute-side root
- a short note stating which outputs should be vendored into `artifacts/zero_aux_current/`
