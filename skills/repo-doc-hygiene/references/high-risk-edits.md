# High-Risk Edits

These edits need extra care.

## Manuscript-Facing Numbers

- Trace the number to the authoritative artifact before editing prose.
- Prefer generated tables, CSVs, or report outputs over hand-maintained text.
- If the repo has explicit manuscript rules, follow them first.

## Scientific Interpretation

- Separate confirmed facts from interpretation.
- Do not casually collapse distinctions between result families, training designs, or host-specific replicas.
- If a new result changes an older claim, update the ownership docs first so the current canonical claim is clear.

## Live Run Status

- Treat live remote-run information as volatile.
- Prefer a short status note over spreading the same status block across several docs.
- If a file includes live run status, make sure it is clearly the owner of that role.

## Path Corrections

- A "remote-only" claim is high-risk if a local mirror may now exist.
- Verify the local path directly before correcting the statement.
- Prefer one registry file for paths so later fetches only require one update.

## Historical Documents

- Do not delete by default.
- Add a short banner:
  - historical
  - not the current source of truth
  - which file replaced it

## Generated Or Derived Outputs

- Avoid editing generated report files just to improve wording unless that is explicitly the task.
- If a generated report is stale, fix the source workflow or point readers to the newer report root.
