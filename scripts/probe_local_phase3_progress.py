#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EPISODE_RE = re.compile(r"\[episode\s+([0-9]+)\]")
CHECKPOINT_RE = re.compile(r"_ep([0-9]+)\.pt$")

SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1111, 1212, 1313, 1414, 1515, 1616]


def _read_manifest(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _last_logged_episode(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    last = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = EPISODE_RE.search(line)
            if m:
                last = max(last, int(m.group(1)))
    return last


def _last_checkpoint_episode(train_root: Path, condition: str, seed: int) -> int:
    final_ckpt = train_root / f"{condition}_seed{seed}.pt"
    if final_ckpt.exists():
        return 150000
    last = 0
    for path in train_root.glob(f"{condition}_seed{seed}_ep*.pt"):
        m = CHECKPOINT_RE.search(path.name)
        if m:
            last = max(last, int(m.group(1)))
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--condition", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manifest = _read_manifest(run_dir / "outputs" / f"phase3_{args.condition or 'cond2'}_15seeds_train_only" / "status" / "manifest.txt")
    if not manifest:
        status_dirs = sorted(run_dir.glob("outputs/*/status"))
        if not status_dirs:
            raise SystemExit("no status directory found")
        manifest = _read_manifest(status_dirs[0] / "manifest.txt")
    condition = str(args.condition or manifest.get("condition", "cond2"))
    out_root = Path(manifest.get("train_root", run_dir / "outputs" / f"phase3_{condition}_15seeds_train_only" / "train")).parent
    train_root = out_root / "train"
    logs_root = out_root / "logs"

    rows = []
    current = 0
    total = len(SEEDS) * 150000
    finished = 0
    for seed in SEEDS:
        logged = _last_logged_episode(logs_root / f"{condition}_seed{seed}.log")
        ckpt = _last_checkpoint_episode(train_root, condition, seed)
        ep = max(logged, ckpt)
        ep = min(ep, 150000)
        if ep >= 150000:
            finished += 1
        current += ep
        rows.append({"seed": seed, "episode": ep})

    print(
        json.dumps(
            {
                "current": current,
                "total": total,
                "finished": finished,
                "n_seeds": len(SEEDS),
                "condition": condition,
                "seeds": rows,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
