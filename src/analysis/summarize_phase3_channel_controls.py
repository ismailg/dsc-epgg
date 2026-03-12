from __future__ import annotations

import argparse
import csv
import math
import os
import re
from statistics import stdev
from typing import Dict, List

from src.analysis.condition_labels import condition_alias, condition_display


def _read_csv_rows(path: str) -> List[Dict]:
    if path == "" or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_float(row: Dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float(default)
    return float(value)


def _as_int(row: Dict, key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return int(default)
    return int(float(value))


def _infer_condition_seed(checkpoint: str) -> tuple[str, int]:
    name = os.path.basename(str(checkpoint or ""))
    m = re.search(r"(cond[0-9]+)_seed([0-9]+)", name)
    if not m:
        return "unknown", -1
    return m.group(1), int(m.group(2))


def _collect_mode_rows(mode: str, suite_csv: str) -> List[Dict]:
    out = []
    for row in _read_csv_rows(suite_csv):
        cond = row.get("condition", "")
        train_seed = _as_int(row, "train_seed", -1)
        if cond in ("", "unknown") or train_seed < 0:
            cond, inferred_seed = _infer_condition_seed(row.get("checkpoint", ""))
            if train_seed < 0:
                train_seed = int(inferred_seed)
        if cond != "cond1":
            continue
        if row.get("scope") != "f_value":
            continue
        if row.get("eval_policy", "greedy") != "greedy":
            continue
        if row.get("ablation", "none") != "none":
            continue
        if row.get("cross_play", "none") != "none":
            continue
        if row.get("key") not in ("3.500", "5.000"):
            continue
        out.append(
            {
                "mode": mode,
                "condition": cond,
                "condition_alias": condition_alias(cond),
                "condition_display": condition_display(cond),
                "train_seed": int(train_seed),
                "checkpoint_episode": _as_int(row, "checkpoint_episode", 0),
                "f_value": row.get("key"),
                "coop_rate": _as_float(row, "coop_rate"),
                "avg_welfare": _as_float(row, "avg_welfare"),
            }
        )
    return out


def _mean(values: List[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(1, len(vals)))


def _std(values: List[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) <= 1:
        return 0.0
    return float(stdev(vals))


def _sem(values: List[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) <= 1:
        return 0.0
    return float(_std(vals) / math.sqrt(len(vals)))


def _summarize(rows: List[Dict]) -> List[Dict]:
    grouped = {}
    for row in rows:
        key = (
            row["mode"],
            row.get("condition", ""),
            row.get("condition_alias", ""),
            row.get("condition_display", ""),
            row["checkpoint_episode"],
            row["f_value"],
        )
        grouped.setdefault(key, []).append(row)

    out = []
    for (mode, condition, condition_alias_value, condition_display_value, episode, f_value), cur in sorted(grouped.items()):
        out.append(
            {
                "mode": mode,
                "condition": condition,
                "condition_alias": condition_alias_value,
                "condition_display": condition_display_value,
                "checkpoint_episode": int(episode),
                "f_value": f_value,
                "n_seeds": int(len(cur)),
                "mean_coop_rate": _mean([row["coop_rate"] for row in cur]),
                "std_coop_rate": _std([row["coop_rate"] for row in cur]),
                "sem_coop_rate": _sem([row["coop_rate"] for row in cur]),
                "mean_avg_welfare": _mean([row["avg_welfare"] for row in cur]),
                "std_avg_welfare": _std([row["avg_welfare"] for row in cur]),
                "sem_avg_welfare": _sem([row["avg_welfare"] for row in cur]),
            }
        )
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode_suite",
        nargs=2,
        action="append",
        metavar=("MODE", "CSV"),
        default=[],
        help="Arbitrary mode label and checkpoint_suite_main.csv path. If provided, these override the legacy mode-specific args.",
    )
    p.add_argument(
        "--learned_suite_csv",
        type=str,
        default="outputs/eval/phase3_annealed_trimmed_all/suite/checkpoint_suite_main.csv",
    )
    p.add_argument(
        "--zero_suite_csv",
        type=str,
        default="outputs/eval/phase3_channel_controls_50k/fixed0/suite/checkpoint_suite_main.csv",
    )
    p.add_argument(
        "--uniform_suite_csv",
        type=str,
        default="outputs/eval/phase3_channel_controls_50k/uniform/suite/checkpoint_suite_main.csv",
    )
    p.add_argument(
        "--public_suite_csv",
        type=str,
        default="outputs/eval/phase3_channel_controls_50k/public_random/suite/checkpoint_suite_main.csv",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/eval/phase3_channel_controls_50k/report",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    mode_suites = list(args.mode_suite or [])
    if len(mode_suites) == 0:
        mode_suites = [
            ("learned", args.learned_suite_csv),
            ("always_zero", args.zero_suite_csv),
            ("indep_random", args.uniform_suite_csv),
            ("public_random", args.public_suite_csv),
        ]

    rows = []
    for mode, suite_csv in mode_suites:
        rows.extend(_collect_mode_rows(str(mode), str(suite_csv)))

    summary_rows = _summarize(rows)
    _write_csv(os.path.join(out_dir, "channel_control_raw.csv"), rows)
    _write_csv(os.path.join(out_dir, "channel_control_summary.csv"), summary_rows)
    print(f"[channel-controls] out_dir={out_dir}")


if __name__ == "__main__":
    main()
