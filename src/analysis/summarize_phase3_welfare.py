from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
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


def _as_int(row: Dict, key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return int(default)
    return int(float(value))


def _as_float(row: Dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float(default)
    return float(value)


def _mean(values: List[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(1, len(vals)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite_main_csv", type=str, required=True)
    p.add_argument("--bundle_label", type=str, default="")
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--out_mean_csv", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    bundle_label = str(args.bundle_label or "")
    grouped = defaultdict(lambda: {"rounds": 0, "coop_sum": 0.0, "welfare_sum": 0.0})

    for row in _read_csv_rows(args.suite_main_csv):
        if row.get("scope") != "f_value":
            continue
        if row.get("eval_policy", "greedy") != "greedy":
            continue
        if row.get("ablation", "none") != "none":
            continue
        if row.get("cross_play", "none") != "none":
            continue
        n_rounds = _as_int(row, "n_rounds", 0)
        if n_rounds <= 0:
            continue
        cond = str(row.get("condition", ""))
        key = (
            bundle_label,
            cond,
            condition_alias(cond),
            condition_display(cond),
            _as_int(row, "checkpoint_episode", 0),
            _as_int(row, "train_seed", -1),
        )
        grouped[key]["rounds"] += n_rounds
        grouped[key]["coop_sum"] += n_rounds * _as_float(row, "coop_rate")
        grouped[key]["welfare_sum"] += n_rounds * _as_float(row, "avg_welfare")

    raw_rows = []
    for key, acc in sorted(grouped.items()):
        label, cond, cond_alias, cond_display, episode, seed = key
        total_rounds = int(acc["rounds"])
        raw_rows.append(
            {
                "bundle_label": label,
                "condition": cond,
                "condition_alias": cond_alias,
                "condition_display": cond_display,
                "checkpoint_episode": int(episode),
                "train_seed": int(seed),
                "total_rounds": total_rounds,
                "weighted_coop_rate": float(acc["coop_sum"] / max(1, total_rounds)),
                "weighted_avg_welfare": float(acc["welfare_sum"] / max(1, total_rounds)),
            }
        )

    mean_grouped = defaultdict(list)
    for row in raw_rows:
        mean_grouped[
            (
                row["bundle_label"],
                row["condition"],
                row["condition_alias"],
                row["condition_display"],
                row["checkpoint_episode"],
            )
        ].append(row)

    mean_rows = []
    for key, rows in sorted(mean_grouped.items()):
        label, cond, cond_alias, cond_display, episode = key
        mean_rows.append(
            {
                "bundle_label": label,
                "condition": cond,
                "condition_alias": cond_alias,
                "condition_display": cond_display,
                "checkpoint_episode": int(episode),
                "n_seeds": int(len(rows)),
                "mean_weighted_coop_rate": _mean([row["weighted_coop_rate"] for row in rows]),
                "mean_weighted_avg_welfare": _mean([row["weighted_avg_welfare"] for row in rows]),
            }
        )

    _write_csv(args.out_csv, raw_rows)
    _write_csv(args.out_mean_csv, mean_rows)
    print(
        f"[phase3-welfare] raw_rows={len(raw_rows)} mean_rows={len(mean_rows)} "
        f"out={args.out_mean_csv}"
    )


if __name__ == "__main__":
    main()
