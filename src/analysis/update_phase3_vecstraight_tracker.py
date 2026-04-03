from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_tracker_path(run_date: str) -> Path:
    return (
        REPO_ROOT
        / f"outputs/eval/phase3_vecstraight_next_steps_{run_date}/status/TRACKER.md"
    )


def _parse_fields(items: Sequence[str]) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--field expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if key == "":
            raise ValueError(f"empty field key in {item!r}")
        fields.append((key, value))
    return fields


def update_tracker(tracker_path: Path, stage: str, fields: Sequence[tuple[str, str]]) -> None:
    text = tracker_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    heading = f"## {stage}"

    start = None
    end = None
    for idx, line in enumerate(lines):
        if line.strip() == heading:
            start = idx
            continue
        if start is not None and line.startswith("## "):
            end = idx
            break
    if start is None:
        raise ValueError(f"missing stage heading: {heading}")
    if end is None:
        end = len(lines)

    section_lines = lines[start + 1 : end]
    existing_order: list[str] = []
    existing_values: dict[str, str] = {}
    for line in section_lines:
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        body = stripped[2:]
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        key = key.strip()
        value = value.strip()
        existing_order.append(key)
        existing_values[key] = value

    for key, value in fields:
        existing_values[key] = value
        if key not in existing_order:
            existing_order.append(key)

    new_section = [heading]
    for key in existing_order:
        new_section.append(f"- {key}: {existing_values[key]}")

    rebuilt = lines[:start] + new_section + [""] + lines[end:]
    tracker_path.write_text("\n".join(rebuilt).rstrip() + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_date",
        type=str,
        default=dt.date.today().strftime("%Y%m%d"),
    )
    parser.add_argument("--tracker_path", type=str, default="")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--field", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    tracker_path = (
        Path(args.tracker_path).resolve()
        if str(args.tracker_path).strip() != ""
        else _default_tracker_path(str(args.run_date))
    )
    fields = _parse_fields(list(args.field))
    update_tracker(tracker_path, str(args.stage), fields)
    print(f"[tracker] updated stage={args.stage} path={tracker_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
