#!/usr/bin/env python3
"""Monitor progress for a detached job and emit milestone events."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import time
from typing import Any, Dict, Optional, Tuple


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _load_json_or_jsonl(path: pathlib.Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("empty file")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("no json lines")
    obj = json.loads(lines[-1])
    if not isinstance(obj, dict):
        raise ValueError("last json line is not an object")
    return obj


def _extract_progress_from_obj(
    obj: Dict[str, Any], current_key: str, total_key: str, fixed_total: Optional[float]
) -> Tuple[float, float]:
    if current_key not in obj:
        raise KeyError(f"missing current key: {current_key}")
    current = float(obj[current_key])
    if fixed_total is not None:
        total = float(fixed_total)
    else:
        if total_key not in obj:
            raise KeyError(f"missing total key: {total_key}")
        total = float(obj[total_key])
    return current, total


def _progress_from_json_file(
    source: pathlib.Path, current_key: str, total_key: str, fixed_total: Optional[float]
) -> Tuple[float, float]:
    obj = _load_json_or_jsonl(source)
    return _extract_progress_from_obj(obj, current_key=current_key, total_key=total_key, fixed_total=fixed_total)


def _progress_from_regex_file(
    source: pathlib.Path, pattern: str, fixed_total: Optional[float]
) -> Tuple[float, float]:
    text = source.read_text(encoding="utf-8", errors="ignore")
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        raise ValueError("no regex matches")
    m = matches[-1]
    if "current" not in m.groupdict():
        raise ValueError("regex must define named group 'current'")
    current = float(m.group("current"))
    if fixed_total is not None:
        total = float(fixed_total)
    else:
        if "total" not in m.groupdict():
            raise ValueError("regex must define named group 'total' unless --fixed-total is set")
        total = float(m.group("total"))
    return current, total


def _progress_from_command(
    probe_cmd: str,
    shell: str,
    current_key: str,
    total_key: str,
    fixed_total: Optional[float],
) -> Tuple[float, float]:
    proc = subprocess.run(
        [shell, "-lc", probe_cmd],
        check=True,
        capture_output=True,
        text=True,
    )
    out = proc.stdout.strip()
    if not out:
        raise ValueError("probe command produced no stdout")
    try:
        obj = json.loads(out)
        if isinstance(obj, dict):
            return _extract_progress_from_obj(
                obj, current_key=current_key, total_key=total_key, fixed_total=fixed_total
            )
    except Exception:
        pass
    parts = out.split()
    if len(parts) >= 2:
        current = float(parts[0])
        total = float(parts[1]) if fixed_total is None else float(fixed_total)
        return current, total
    raise ValueError("probe command output must be JSON or 'current total'")


def _read_state(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(path: pathlib.Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_event(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--pidfile", type=str, default=None)
    parser.add_argument("--mode", choices=["json-file", "regex-file", "command"], required=True)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--probe-cmd", type=str, default=None)
    parser.add_argument("--shell", type=str, default="/bin/zsh")
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--current-key", type=str, default="current")
    parser.add_argument("--total-key", type=str, default="total")
    parser.add_argument("--fixed-total", type=float, default=None)
    parser.add_argument("--state-json", type=str, required=True)
    parser.add_argument("--events-jsonl", type=str, required=True)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--milestones", type=str, default="25,50,75,100")
    args = parser.parse_args()

    pid: Optional[int] = args.pid
    if pid is None and args.pidfile:
        pid_text = pathlib.Path(args.pidfile).read_text(encoding="utf-8").strip()
        pid = int(pid_text)
    if pid is None:
        parser.error("one of --pid or --pidfile is required")

    milestones = sorted({int(float(x)) for x in str(args.milestones).split(",") if str(x).strip()})
    state_path = pathlib.Path(args.state_json)
    events_path = pathlib.Path(args.events_jsonl)
    state = _read_state(state_path)
    reached = {int(x) for x in state.get("reached_milestones", [])}

    while True:
        alive = _pid_alive(int(pid))
        current = None
        total = None
        pct = None
        error = None

        try:
            if args.mode == "json-file":
                if not args.source:
                    raise ValueError("--source is required for mode=json-file")
                current, total = _progress_from_json_file(
                    pathlib.Path(args.source),
                    current_key=args.current_key,
                    total_key=args.total_key,
                    fixed_total=args.fixed_total,
                )
            elif args.mode == "regex-file":
                if not args.source or not args.pattern:
                    raise ValueError("--source and --pattern are required for mode=regex-file")
                current, total = _progress_from_regex_file(
                    pathlib.Path(args.source),
                    pattern=args.pattern,
                    fixed_total=args.fixed_total,
                )
            else:
                if not args.probe_cmd:
                    raise ValueError("--probe-cmd is required for mode=command")
                current, total = _progress_from_command(
                    probe_cmd=args.probe_cmd,
                    shell=args.shell,
                    current_key=args.current_key,
                    total_key=args.total_key,
                    fixed_total=args.fixed_total,
                )
            if total is not None and total > 0:
                pct = max(0.0, min(100.0, 100.0 * float(current) / float(total)))
        except Exception as exc:
            error = str(exc)

        state = {
            "label": args.label,
            "pid": int(pid),
            "alive": bool(alive),
            "current": current,
            "total": total,
            "pct": pct,
            "error": error,
            "reached_milestones": sorted(reached),
            "updated_at_unix": float(time.time()),
        }

        if pct is not None:
            for milestone in milestones:
                if milestone not in reached and float(pct) >= float(milestone):
                    reached.add(int(milestone))
                    _append_event(
                        events_path,
                        {
                            "event": "milestone",
                            "label": args.label,
                            "pid": int(pid),
                            "milestone": int(milestone),
                            "pct": float(pct),
                            "current": current,
                            "total": total,
                            "ts_unix": float(time.time()),
                        },
                    )
            state["reached_milestones"] = sorted(reached)

        _write_state(state_path, state)

        if not alive:
            _append_event(
                events_path,
                {
                    "event": "exit",
                    "label": args.label,
                    "pid": int(pid),
                    "pct": pct,
                    "current": current,
                    "total": total,
                    "ts_unix": float(time.time()),
                },
            )
            break

        time.sleep(max(1.0, float(args.poll_seconds)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
