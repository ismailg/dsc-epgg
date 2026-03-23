#!/usr/bin/env python3
"""Launch a local process detached from the current Codex exec session.

This avoids shell-style backgrounding (`nohup ... &`) which can still be cleaned
up when the parent exec call returns. The child is launched in a new session
with stdin closed and stdout/stderr redirected to files.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Dict, List


def _parse_env(overrides: List[str]) -> Dict[str, str]:
    env = os.environ.copy()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--env key must be non-empty, got: {item!r}")
        env[key] = value
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", type=str, default=None)
    parser.add_argument("--stdout", type=str, required=True)
    parser.add_argument("--stderr", type=str, required=True)
    parser.add_argument("--pidfile", type=str, default=None)
    parser.add_argument("--metadata-json", type=str, default=None)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("expected a command after '--'")

    cwd = os.path.abspath(args.cwd) if args.cwd else os.getcwd()
    env = _parse_env(list(args.env))

    stdout_path = pathlib.Path(args.stdout)
    stderr_path = pathlib.Path(args.stderr)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("ab", buffering=0) as out_f, stderr_path.open(
        "ab", buffering=0
    ) as err_f:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out_f,
            stderr=err_f,
            close_fds=True,
            start_new_session=True,
        )

    if args.pidfile:
        pidfile = pathlib.Path(args.pidfile)
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text(f"{proc.pid}\n", encoding="utf-8")

    if args.metadata_json:
        meta_path = pathlib.Path(args.metadata_json)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "pid": int(proc.pid),
            "cwd": cwd,
            "cmd": cmd,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "env_overrides": list(args.env),
            "launched_at_unix": float(time.time()),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(proc.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
