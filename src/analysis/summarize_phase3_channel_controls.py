from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List


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


def _collect_mode_rows(mode: str, suite_csv: str) -> List[Dict]:
    out = []
    for row in _read_csv_rows(suite_csv):
        if row.get("condition") != "cond1":
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
                "train_seed": _as_int(row, "train_seed", -1),
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


def _summarize(rows: List[Dict]) -> List[Dict]:
    grouped = {}
    for row in rows:
        key = (row["mode"], row["checkpoint_episode"], row["f_value"])
        grouped.setdefault(key, []).append(row)

    out = []
    for (mode, episode, f_value), cur in sorted(grouped.items()):
        out.append(
            {
                "mode": mode,
                "checkpoint_episode": int(episode),
                "f_value": f_value,
                "n_seeds": int(len(cur)),
                "mean_coop_rate": _mean([row["coop_rate"] for row in cur]),
                "mean_avg_welfare": _mean([row["avg_welfare"] for row in cur]),
            }
        )
    return out


def parse_args():
    p = argparse.ArgumentParser()
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

    rows = []
    rows.extend(_collect_mode_rows("learned", args.learned_suite_csv))
    rows.extend(_collect_mode_rows("always_zero", args.zero_suite_csv))
    rows.extend(_collect_mode_rows("indep_random", args.uniform_suite_csv))
    rows.extend(_collect_mode_rows("public_random", args.public_suite_csv))

    summary_rows = _summarize(rows)
    _write_csv(os.path.join(out_dir, "channel_control_raw.csv"), rows)
    _write_csv(os.path.join(out_dir, "channel_control_summary.csv"), summary_rows)
    print(f"[channel-controls] out_dir={out_dir}")


if __name__ == "__main__":
    main()
