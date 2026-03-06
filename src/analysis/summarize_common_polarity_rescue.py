from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple


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


def _lookup(rows: List[Dict]) -> Dict[Tuple, Dict]:
    out = {}
    for row in rows:
        if row.get("scope") != "f_value":
            continue
        out[
            (
                row.get("condition"),
                _as_int(row, "train_seed", -1),
                _as_int(row, "checkpoint_episode", 0),
                row.get("key"),
            )
        ] = row
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_main_csv", type=str, default="outputs/eval/phase3/checkpoint_suite/checkpoint_suite_main.csv")
    p.add_argument("--rescue_main_csv", type=str, default="outputs/eval/phase3/common_polarity_rescue/common_polarity_rescue_main.csv")
    p.add_argument("--out_csv", type=str, default="outputs/eval/phase3/common_polarity_rescue/common_polarity_rescue_summary.csv")
    return p.parse_args()


def main():
    args = parse_args()
    base_rows = [
        row for row in _read_csv_rows(args.base_main_csv)
        if row.get("condition") == "cond1"
        and row.get("ablation", "none") == "none"
        and row.get("eval_policy", "greedy") == "greedy"
        and row.get("cross_play", "none") == "none"
    ]
    rescue_rows = [
        row for row in _read_csv_rows(args.rescue_main_csv)
        if row.get("condition") == "cond1"
        and row.get("ablation", "none") == "none"
        and row.get("eval_policy", "greedy") == "greedy"
        and row.get("cross_play", "none") == "none"
    ]
    base_lookup = _lookup(base_rows)
    rescue_lookup = _lookup(rescue_rows)

    out = []
    keys = sorted(set(base_lookup.keys()) & set(rescue_lookup.keys()))
    for key in keys:
        condition, seed, episode, f_key = key
        if f_key not in ("3.500", "5.000"):
            continue
        base = base_lookup[key]
        rescue = rescue_lookup[key]
        out.append(
            {
                "condition": condition,
                "train_seed": int(seed),
                "checkpoint_episode": int(episode),
                "f_value": f_key,
                "base_coop": _as_float(base, "coop_rate"),
                "rescue_coop": _as_float(rescue, "coop_rate"),
                "delta_rescue_minus_base": _as_float(rescue, "coop_rate") - _as_float(base, "coop_rate"),
                "base_welfare": _as_float(base, "avg_welfare"),
                "rescue_welfare": _as_float(rescue, "avg_welfare"),
                "delta_rescue_welfare_minus_base": _as_float(rescue, "avg_welfare") - _as_float(base, "avg_welfare"),
                "sender_remap": rescue.get("sender_remap", ""),
                "n_flipped_senders": _as_int(rescue, "n_flipped_senders", 0),
            }
        )
    _write_csv(args.out_csv, out)
    print(f"[common-polarity-summary] rows={len(out)} out={args.out_csv}")


if __name__ == "__main__":
    main()
