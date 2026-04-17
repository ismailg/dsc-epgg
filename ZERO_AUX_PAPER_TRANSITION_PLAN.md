# Zero-Aux Paper Transition Plan

This file is the authoritative cross-repo plan for replacing the current paper's
message-specific auxiliary-loss story with a pure-PPO story.

It is meant to be mirrored into both repos:

- paper repo: `dsc-epgg-tuned`
- compute repo: `dsc-epgg-vectorized`

## Goal

Produce a new final paper state in which the main scientific claims are based on
training runs that do not optimize `sign_lambda` or `list_lambda`, while keeping
all other core training/evaluation settings fixed unless explicitly noted.

## Scope

In scope:

- replace the paper's main communication result families with zero-aux or
  functionally zero-aux counterparts
- rerun the necessary evaluation families from those checkpoints
- create a new artifact profile in `dsc-epgg-tuned`
- compare the zero-aux manuscript state against the current final paper state

Out of scope for this pass:

- changing `msg_dropout`
- changing PPO entropy schedules
- fixing training-time intervention credit assignment
- keeping the loss-switch / continuation-control story in the main paper

## Hard Rules

- Do not modify the current baseline artifact profile in `dsc-epgg-tuned`.
- Do not overwrite or clean the existing dirty worktree in `dsc-epgg-vectorized`.
- Use a dedicated branch/worktree for zero-aux work in the old repo.
- Keep the seed set fixed at:
  `101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616`
- Keep the main milestones fixed at:
  `25000 50000 100000 150000`
- Keep the evaluation episode count fixed at `300` unless a specific family
  requires otherwise.
- Treat the current paper as the baseline comparison target, not as something to
  edit in place.

## Canonical Choices

- Canonical full-history communication baseline:
  zero-aux `clean_msgsource_learned`
- Canonical no-communication baseline:
  `comm_history_factorial_without_comm_full_history`
- Canonical reduced-history no-communication baseline:
  `comm_history_factorial_without_comm_reduced_history`
- One genuinely missing training cell:
  `with_comm_reduced_history` with `sign_lambda=0.0` and `list_lambda=0.0`
- Paper profile name in `dsc-epgg-tuned`:
  `zero_aux_current`
- Old-repo branch/worktree name:
  `codex/zero-aux-paper-transition`

## Current Status

This plan is in review mode.

Approved so far:

- write the cross-repo plan in `dsc-epgg-tuned` first
- do not create the old-repo branch/worktree yet
- validate the scientific substitutions before launching any compute
- accept `clean_msgsource_learned` as the canonical zero-aux `with_comm_full`
  cell
- accept the no-communication arms as functionally zero-aux controls because
  message-specific auxiliary terms are inactive when no sender pathway exists
- drop the loss-switch / continuation-control section from the zero-aux paper
  path
- use `zero_aux_current` as the target artifact-profile name
- keep the noise-sweep appendix in scope for the zero-aux paper path
- keep the message-history-grid appendix off the blocking path

Not yet approved:

- creating the old-repo branch/worktree
- mirroring this file into the old repo
- launching the one new zero-aux training cell
- regenerating any paper-facing artifact profile

## Validation Questions Before Execution

The main scientific substitutions have now been approved.

Execution is still paused pending operational approval for:

1. creating the old-repo branch/worktree
2. mirroring this file into the old repo
3. launching the one new zero-aux training cell
4. regenerating the zero-aux artifact profile

## Current Paper Vs Zero-Aux Target

| Family / paper use | Current-paper source | Zero-aux target | Action | New training? |
|---|---|---|---|---|
| Comm gap / Figure 1 A-B | base `cond1` vs `cond2` families | zero-aux `clean_learned` vs canonical no-comm baseline | re-evaluate and rebuild checkpoint summaries at `25k/50k/100k/150k` | no |
| Clean msg-source controls | already zero-aux clean family | same clean family plus matched no-comm baseline | rerun summary and, if desired, extend checkpoint coverage beyond `50k/150k` | no |
| Comm × history factorial | four-cell factorial with active aux config in comm cells | `with_comm_full = clean_learned`, `without_comm_* = existing factorial no-comm cells`, `with_comm_reduced = new zero-aux cell` | rebuild factorial summary from mixed existing/new raw roots | yes, one cell |
| Frozen interventions | base communication family | zero-aux `clean_learned` at `50k` and `150k` | rerun frozen expanded suite | no |
| Sender causal | base communication family | zero-aux `clean_learned` at `150k` | rerun sender-causal suite | no |
| Natural semantics / lowdim appendix | base communication family | zero-aux `clean_learned` frozen traces | regenerate from zero-aux frozen outputs | no |
| Cross-seed transfer | base communication family | zero-aux `clean_learned` at `150k` | rerun cross-seed transfer suite and summary | no |
| Loss-switch / continuation controls | currently in artifact bundle | drop from zero-aux paper path | remove from target paper scope | no |
| Noise sweep appendix | base communication family | zero-aux `clean_learned` | rerun and keep in zero-aux paper path | no |
| Message-history grid appendix | base communication family | defer from blocking zero-aux paper path | rerun later only if explicitly restored | no |

## Repo Split

### `dsc-epgg-vectorized`

Owns:

- all raw training reruns
- all raw evaluation reruns
- helper scripts or manifest generation needed to launch/validate those reruns
- provenance notes about where the zero-aux roots live

### `dsc-epgg-tuned`

Owns:

- the manuscript-facing plan
- the final compact artifact profile
- artifact registry switching
- final manuscript rewrite and render
- baseline-vs-zero-aux comparison bookkeeping

## Execution Steps

### 0. Create a safe old-repo worktree

Reason:
the current `dsc-epgg-vectorized` worktree is dirty and should not be repurposed.

Recommended commands:

```bash
cd /Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized
git worktree add -b codex/zero-aux-paper-transition ../dsc-epgg-vectorized-zeroaux-plan HEAD
cd ../dsc-epgg-vectorized-zeroaux-plan
```

After the worktree exists, mirror this file into the old repo root.

### 1. Freeze the source-root mapping

Record and keep fixed:

- zero-aux clean communication roots:
  `phase3_vecstraight_clean_msgsource_{learned,uniform,fixed0,fixed1,public_random}_15seeds_hetzner_20260410`
- canonical no-comm roots:
  `phase3_vecstraight_comm_history_factorial_without_comm_{full,reduced}_15seeds_hetzner_20260330par24`
- new training root to create:
  `phase3_vecstraight_comm_history_factorial_with_comm_reduced_history_zeroaux_15seeds_<host>_<date>`

### 2. Add one dedicated zero-aux factorial runner in the old repo

Do this on the zero-aux branch/worktree before launching compute.

Required outcome:

- a runner or command template that launches only
  `with_comm_reduced_history`
- identical to the existing factorial contract except:
  - `sign_lambda=0.0`
  - `list_lambda=0.0`

Do not change:

- `num_envs`
- `count_env_episodes`
- `env_backend`
- `env_start_method`
- entropy schedules
- learning-rate schedule
- seed set
- milestone spacing

### 3. Train the one missing cell

Launch:

- `with_comm_reduced_history`
- `comm_enabled=true`
- `history_mode=reduced`
- `sign_lambda=0.0`
- `list_lambda=0.0`
- `disable_comm_fallback=true`
- `checkpoint_interval=25000`

Expected raw outputs:

- final checkpoints at `150k`
- intermediate checkpoints at `25k/50k/75k/100k/125k`
- `.run.json` files for every checkpoint

### 4. Build clean manifests for the zero-aux paper path

Create manifest files in the old repo worktree for:

- zero-aux `cond1` full-history communication checkpoints
- zero-aux `cond2` full-history no-comm checkpoints
- zero-aux `cond1` reduced-history communication checkpoints
- zero-aux `cond2` reduced-history no-comm checkpoints

Keep these manifests separate from the current-paper baseline manifests.

### 5. Rebuild training-time result families in the old repo

Run:

- clean msg-source summary from the five zero-aux clean families plus the chosen
  no-comm baseline
- comm-gap checkpoint suite for `25k/50k/100k/150k`
- zero-aux comm × history factorial summary using:
  - `with_comm_full = clean_learned`
  - `with_comm_reduced = new zero-aux cell`
  - `without_comm_full = existing factorial no-comm full`
  - `without_comm_reduced = existing factorial no-comm reduced`

Validation rule:

- every summary must be reproducible from explicit manifests or suite outputs,
  not from hand-picked CSV edits

### 6. Rebuild endpoint families in the old repo

Run from zero-aux `clean_learned`:

- frozen expanded suite at `50k` and `150k`
- sender-causal suite at `150k`
- cross-seed transfer suite at `150k`

Then regenerate downstream summaries:

- intervention summaries / paired stats
- sender-causal report tables
- lowdim mechanism summaries
- pattern-vs-any summaries
- cross-seed transfer summaries

### 7. Run the retained noise sweep and defer message-history-grid

Rerun:

- noise sweep

Do not block the zero-aux paper path on the message-history grid / history
audit. Only rerun it later if it is explicitly restored into scope.

### 8. Vendor a new artifact profile into `dsc-epgg-tuned`

Create:

```text
artifacts/zero_aux_pure_ppo_final/
  eval/
  derived/
  manifest.json
```

Copy only compact manuscript-used outputs.

Do not copy:

- raw checkpoints
- logs
- giant traces
- full host mirrors

### 9. Compare the new paper profile against the current final paper

For each manuscript section or figure/table input, record:

- current source artifact
- new zero-aux source artifact
- whether the claim is unchanged, weakened, strengthened, or dropped

Minimum required comparison rows:

- comm gap
- clean msg-source controls
- comm × history factorial
- frozen interventions
- sender-causal endpoint results
- lowdim / pattern-vs-any appendix
- cross-seed transfer
- dropped loss-switch section

### 10. Switch the active profile only after full render validation

In `dsc-epgg-tuned`:

- add `zero_aux_pure_ppo_final` to `artifacts/registry.json`
- switch `active_profile`
- render the manuscript
- verify all figure/table chunks read only from the new profile
- verify no stale baseline paths remain in prose or captions

## To-Do Checklist

- [ ] Create the old-repo zero-aux worktree on `codex/zero-aux-paper-transition`
- [ ] Mirror this plan into the old repo root
- [ ] Add the one-cell zero-aux factorial runner in the old repo
- [ ] Train `with_comm_reduced_history_zeroaux`
- [ ] Build dedicated zero-aux manifests
- [ ] Rebuild clean msg-source summary
- [ ] Rebuild zero-aux comm-gap checkpoint comparison
- [ ] Rebuild zero-aux comm × history factorial summary
- [ ] Rebuild zero-aux frozen expanded suite
- [ ] Rebuild zero-aux sender-causal suite
- [ ] Rebuild zero-aux cross-seed transfer suite
- [ ] Rebuild zero-aux noise sweep
- [ ] Vendor `zero_aux_current` into `dsc-epgg-tuned`
- [ ] Write a section-by-section current-paper vs zero-aux comparison note
- [ ] Switch the active artifact profile
- [ ] Render and verify the zero-aux manuscript

## Validation Gates

- The new factorial summary must use only one newly trained cell.
- The no-comm baseline choice must be used consistently across all zero-aux
  training-time figures.
- Frozen, sender-causal, lowdim, and transfer analyses must all come from the
  same zero-aux `clean_learned` communication family.
- The zero-aux paper path must not depend on the loss-switch family.
- The new paper profile must render without reading from the old repo.

## Risks To Watch

- accidental mixing of current-paper baseline outputs with zero-aux outputs
- reusing the dirty old-repo worktree instead of a dedicated worktree
- changing more than the aux-loss contract in the one new training cell
- forgetting to remove loss-switch references from the zero-aux manuscript path
- silently using wrapper scripts that still assume current-paper manifests

## Success Condition

We are done when:

- the zero-aux manuscript renders from a new artifact profile in
  `dsc-epgg-tuned`
- every retained scientific claim can be traced to zero-aux or functionally
  zero-aux source runs
- the plan, raw reruns, compact artifacts, and manuscript diff are all
  documented without relying on memory
