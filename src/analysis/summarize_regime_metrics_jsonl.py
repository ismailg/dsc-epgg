from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def _read_rows(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("scope") != "regime":
                continue
            if row.get("window") != "cumulative":
                continue
            out.append(row)
    return out


def _collect(paths: List[str]) -> Dict[Tuple[str, int], Dict[str, Dict]]:
    """
    Returns:
      keyed[(run_name, episode)][regime] = row
    """
    keyed: Dict[Tuple[str, int], Dict[str, Dict]] = defaultdict(dict)
    for path in paths:
        run_name = os.path.splitext(os.path.basename(path))[0]
        for row in _read_rows(path):
            ep = int(row.get("episode", -1))
            regime = str(row.get("key", ""))
            if ep <= 0 or regime == "":
                continue
            keyed[(run_name, ep)][regime] = row
    return keyed


def _flatten(keyed: Dict[Tuple[str, int], Dict[str, Dict]]) -> List[Dict]:
    rows = []
    for (run_name, ep), regimes in sorted(keyed.items(), key=lambda x: (x[0][0], x[0][1])):
        out = {"run": run_name, "episode": ep}
        for regime in ("competitive", "mixed", "cooperative"):
            r = regimes.get(regime, {})
            out[f"{regime}_coop"] = float(r.get("coop_rate", 0.0))
            out[f"{regime}_reward"] = float(r.get("avg_reward", 0.0))
            out[f"{regime}_rounds"] = int(r.get("n_rounds", 0))
        rows.append(out)
    return rows


def _write_csv(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fields = [
        "run",
        "episode",
        "competitive_coop",
        "mixed_coop",
        "cooperative_coop",
        "competitive_reward",
        "mixed_reward",
        "cooperative_reward",
        "competitive_rounds",
        "mixed_rounds",
        "cooperative_rounds",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _latest_by_run(rows: List[Dict]) -> List[Dict]:
    latest = {}
    for row in rows:
        run = row["run"]
        ep = int(row["episode"])
        if run not in latest or ep > int(latest[run]["episode"]):
            latest[run] = row
    return [latest[k] for k in sorted(latest.keys())]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metrics_glob",
        type=str,
        default="outputs/train/**/metrics/*.jsonl",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        default="outputs/train/analysis/regime_metrics_summary.csv",
    )
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.metrics_glob, recursive=True))
    if len(paths) == 0:
        raise FileNotFoundError(f"no files matched: {args.metrics_glob}")

    keyed = _collect(paths)
    rows = _flatten(keyed)
    _write_csv(args.out_csv, rows)

    print(f"[summary] files={len(paths)} rows={len(rows)} out_csv={args.out_csv}")
    for row in _latest_by_run(rows):
        print(
            "[latest]",
            row["run"],
            f"ep={row['episode']}",
            f"comp={row['competitive_coop']:.3f}",
            f"mixed={row['mixed_coop']:.3f}",
            f"coop={row['cooperative_coop']:.3f}",
        )


if __name__ == "__main__":
    main()

