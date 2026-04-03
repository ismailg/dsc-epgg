# Ownership Model

Use this model to keep repository docs from drifting into each other.

## Singleton Roles

Only one live file should own each of these.

### Entry Point

Question answered:
- "Where should I start in this repo?"

Good examples:
- root `README.md`

Should contain:
- current repo orientation
- active code map
- a small "read first" list

Should not contain:
- full artifact registry
- live run ledger
- long historical planning notes

### Artifact Registry

Question answered:
- "Where do the authoritative local and remote artifacts live?"

Good examples:
- `DATA_MAP.md`

Should contain:
- canonical local roots
- canonical remote roots
- fetch conventions
- naming rules

Should not contain:
- long scientific interpretation
- active TODO list
- repeated live-status narration

### Handoff / Status Note

Question answered:
- "What is the current gating run, scientific state, and next thing to look at?"

Good examples:
- `NEXT_STEPS.md`
- `STATUS.md`

Should contain:
- short current state
- active gating run or decision
- minimal handoff prompt

Should not contain:
- full artifact inventory
- broad historical archive

### Active Checklist

Question answered:
- "What remains to be done?"

Good examples:
- `PAPER_TODO.md`

Should contain:
- open tasks
- closed packages if helpful
- priority ordering

Should not contain:
- full path registry
- verbose didactic narrative

## Non-Singleton Roles

These can exist in multiple files.

### Interpretation

Use for:
- didactic overviews
- mechanism notes
- result summaries

### Historical

Use for:
- old trackers
- superseded plans
- archived execution notes

Historical docs should say so explicitly near the top.

## Common Failure Modes

- one file tries to be both artifact registry and live status note
- multiple files claim to be the active checklist
- trackers remain visible after they stop being maintained
- historical docs lack a banner and get mistaken for current guidance
- README still describes an old code layout after the code moved

## Default Fix Order

1. decide the singleton owner for each role
2. correct stale factual claims
3. mark historical docs
4. replace repeated blocks with links
5. shorten the active "read first" chain
