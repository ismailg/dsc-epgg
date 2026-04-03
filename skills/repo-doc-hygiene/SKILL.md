---
name: repo-doc-hygiene
description: Audit and streamline Markdown documentation in a code or research repository when the user wants docs reviewed for accuracy, staleness, redundancy, ownership confusion, outdated paths, broken links, or unclear source-of-truth boundaries.
---

# Repo Doc Hygiene

Use this skill when a repository has multiple Markdown files that may have drifted, duplicated each
other, or stopped reflecting the current code, outputs, or workflow.

## What This Skill Is For

- auditing Markdown files for redundancy, stale status, and ownership confusion
- identifying which file should own which kind of information
- checking local Markdown links and obvious path drift
- streamlining docs without flattening important scientific or operational nuance
- marking historical docs so they stop competing with current docs

## What This Skill Is Not For

- rewriting manuscript interpretation casually
- changing quantitative claims without checking the generating artifacts
- deleting historical docs by default
- treating heuristic audit output as proof

## Default Workflow

1. Read the repo rules first.
   Start with `AGENTS.md` or equivalent. Extract scientific, manuscript, and operational
   constraints before editing docs.
2. Inventory the Markdown surface.
   Run `scripts/md_audit.py` for a quick map of files, likely roles, broken links, and ownership
   conflicts.
3. Classify each important doc by role.
   Use the ownership model in `references/ownership-model.md`.
4. Check high-risk claims before editing.
   If a doc touches manuscript numbers, live run status, or scientific interpretation, read
   `references/high-risk-edits.md` and verify against artifacts.
5. Edit conservatively.
   Prefer:
   - clarifying a file's role
   - moving repeated path inventories to one registry file
   - adding "historical" banners
   - linking to the current source of truth
   over wholesale rewrites.
6. Summarize what changed.
   Report:
   - which files now own which information
   - which stale/conflicting claims were corrected
   - what still needs manual judgment

## Role Rules

Only one live file should own each of these roles:

- entry point
- artifact registry
- active handoff/status note
- active checklist

It is fine to have many files for:

- interpretation
- historical notes
- local run reports

If a file tries to do more than one singleton role, streamline it or split the responsibilities.

## High-Risk Edit Rules

- Do not change scientific interpretation unless the repo artifacts support the change.
- Do not change manuscript-facing numbers unless you traced them to the authoritative output.
- Do not label a file stale just because it is old; check whether it is historical by design.
- Do not treat live remote-run claims as stable unless you have current evidence.

## Bundled References

- `references/ownership-model.md`
  Read when deciding which file should own what.
- `references/high-risk-edits.md`
  Read before editing docs that mention results, live runs, manuscript framing, or remote state.

## Bundled Script

- `scripts/md_audit.py`
  Quick audit for:
  - Markdown inventory
  - broken local links
  - likely file roles
  - role-conflict warnings
  - duplicate "source of truth" or "active checklist" claims

Example:

```bash
python3 skills/repo-doc-hygiene/scripts/md_audit.py .
python3 skills/repo-doc-hygiene/scripts/md_audit.py . --include-outputs
python3 skills/repo-doc-hygiene/scripts/md_audit.py . --json
```

Treat the script as triage, not as the final judgment.
