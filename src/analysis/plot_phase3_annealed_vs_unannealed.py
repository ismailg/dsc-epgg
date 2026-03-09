from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


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


def _suite_comm_gap_rows(rows: List[Dict], seeds: Iterable[int]) -> List[Dict]:
    keep_seeds = {int(v) for v in seeds}
    lookup = {}
    for row in rows:
        if row.get("scope") != "f_value":
            continue
        if row.get("key") != "3.500":
            continue
        if row.get("eval_policy", "greedy") != "greedy":
            continue
        if row.get("ablation", "none") != "none":
            continue
        if row.get("cross_play", "none") != "none":
            continue
        seed = _as_int(row, "train_seed", -1)
        if seed not in keep_seeds:
            continue
        episode = _as_int(row, "checkpoint_episode", 0)
        lookup[(row.get("condition"), seed, episode)] = _as_float(row, "coop_rate")

    out = []
    episodes = sorted({episode for (_cond, _seed, episode) in lookup.keys()})
    for seed in sorted(keep_seeds):
        for episode in episodes:
            cond1 = lookup.get(("cond1", seed, episode))
            cond2 = lookup.get(("cond2", seed, episode))
            if cond1 is None or cond2 is None:
                continue
            out.append(
                {
                    "train_seed": int(seed),
                    "checkpoint_episode": int(episode),
                    "metric": "comm_gap_f3p5",
                    "value": float(cond1 - cond2),
                }
            )
    return out


def _fragmentation_metric_rows(rows: List[Dict], seeds: Iterable[int]) -> List[Dict]:
    keep_seeds = {int(v) for v in seeds}
    metrics = {
        "aggregate_token_effect_abs_mean": "aggregate_token_effect_abs",
        "sender_specific_effect_abs_mean": "sender_specific_effect_abs",
        "alignment_regime_sign": "alignment_regime_sign",
    }
    out = []
    for row in rows:
        if row.get("condition") != "cond1":
            continue
        seed = _as_int(row, "train_seed", -1)
        if seed not in keep_seeds:
            continue
        episode = _as_int(row, "checkpoint_episode", 0)
        for key, metric_name in metrics.items():
            out.append(
                {
                    "train_seed": int(seed),
                    "checkpoint_episode": int(episode),
                    "metric": metric_name,
                    "value": _as_float(row, key),
                }
            )
    return out


def _rescue_metric_rows(rows: List[Dict], seeds: Iterable[int]) -> List[Dict]:
    keep_seeds = {int(v) for v in seeds}
    out = []
    for row in rows:
        if row.get("condition") != "cond1":
            continue
        if row.get("f_value") != "3.500":
            continue
        seed = _as_int(row, "train_seed", -1)
        if seed not in keep_seeds:
            continue
        out.append(
            {
                "train_seed": int(seed),
                "checkpoint_episode": _as_int(row, "checkpoint_episode", 0),
                "metric": "rescue_delta_f3p5",
                "value": _as_float(row, "delta_rescue_minus_base"),
            }
        )
    return out


def _collect_regime_rows(
    label: str,
    suite_paths: List[str],
    fragment_paths: List[str],
    rescue_paths: List[str],
    seeds: Iterable[int],
) -> List[Dict]:
    out: List[Dict] = []
    for path in suite_paths:
        for row in _suite_comm_gap_rows(_read_csv_rows(path), seeds=seeds):
            out.append({"regime": label, **row})
    for path in fragment_paths:
        for row in _fragmentation_metric_rows(_read_csv_rows(path), seeds=seeds):
            out.append({"regime": label, **row})
    for path in rescue_paths:
        for row in _rescue_metric_rows(_read_csv_rows(path), seeds=seeds):
            out.append({"regime": label, **row})
    return out


def _mean(values: List[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(1, len(vals)))


def _summarize(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["regime"], row["metric"], int(row["checkpoint_episode"]))].append(
            float(row["value"])
        )
    out = []
    for (regime, metric, episode), values in sorted(grouped.items()):
        out.append(
            {
                "regime": regime,
                "metric": metric,
                "checkpoint_episode": int(episode),
                "n_seeds": int(len(values)),
                "mean_value": float(_mean(values)),
            }
        )
    return out


def _maybe_plot(mean_rows: List[Dict], out_dir: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[traj-compare] matplotlib unavailable; skipped PNG plot")
        return

    metric_order = [
        "comm_gap_f3p5",
        "aggregate_token_effect_abs",
        "sender_specific_effect_abs",
        "alignment_regime_sign",
        "rescue_delta_f3p5",
    ]
    metric_titles = {
        "comm_gap_f3p5": "Comm gap at f=3.5",
        "aggregate_token_effect_abs": "Aggregate |token effect|",
        "sender_specific_effect_abs": "Sender-specific |effect|",
        "alignment_regime_sign": "Alignment index",
        "rescue_delta_f3p5": "Common-polarity rescue delta at f=3.5",
    }
    regimes = sorted({row["regime"] for row in mean_rows})
    if len(regimes) == 0:
        return

    fig, axes = plt.subplots(len(metric_order), 1, figsize=(9, 14), squeeze=False)
    for ax, metric in zip(axes[:, 0], metric_order):
        for regime in regimes:
            cur = sorted(
                [
                    row
                    for row in mean_rows
                    if row["regime"] == regime and row["metric"] == metric
                ],
                key=lambda r: int(r["checkpoint_episode"]),
            )
            if len(cur) == 0:
                continue
            xs = [int(row["checkpoint_episode"]) for row in cur]
            ys = [float(row["mean_value"]) for row in cur]
            ax.plot(xs, ys, marker="o", label=regime)
        ax.set_title(metric_titles.get(metric, metric))
        ax.set_xlabel("Checkpoint episode")
        ax.set_ylabel("Value")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "annealed_vs_unannealed_trajectory.png"), dpi=160)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--unannealed_suite_csv",
        type=str,
        default="outputs/eval/phase3/checkpoint_suite/checkpoint_suite_main.csv",
    )
    p.add_argument(
        "--unannealed_fragment_csv",
        type=str,
        default="outputs/eval/phase3/fragmentation_figures/fragmentation_over_time.csv",
    )
    p.add_argument(
        "--unannealed_rescue_csv",
        type=str,
        default="outputs/eval/phase3/common_polarity_rescue/common_polarity_rescue_summary.csv",
    )
    p.add_argument(
        "--annealed_suite_csvs",
        nargs="*",
        type=str,
        default=["outputs/eval/phase3_annealed_trimmed_all/suite/checkpoint_suite_main.csv"],
    )
    p.add_argument(
        "--annealed_fragment_csvs",
        nargs="*",
        type=str,
        default=["outputs/eval/phase3_annealed_trimmed_all/fragmentation_figures/fragmentation_over_time.csv"],
    )
    p.add_argument(
        "--annealed_rescue_csvs",
        nargs="*",
        type=str,
        default=["outputs/eval/phase3_annealed_trimmed_all/common_polarity_rescue/common_polarity_rescue_summary.csv"],
    )
    p.add_argument("--seeds", nargs="*", type=int, default=[101, 202])
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/eval/phase3_compare",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    raw_rows = []
    raw_rows.extend(
        _collect_regime_rows(
            label="unannealed",
            suite_paths=[str(args.unannealed_suite_csv)],
            fragment_paths=[str(args.unannealed_fragment_csv)],
            rescue_paths=[str(args.unannealed_rescue_csv)],
            seeds=args.seeds,
        )
    )
    raw_rows.extend(
        _collect_regime_rows(
            label="annealed",
            suite_paths=[str(v) for v in args.annealed_suite_csvs],
            fragment_paths=[str(v) for v in args.annealed_fragment_csvs],
            rescue_paths=[str(v) for v in args.annealed_rescue_csvs],
            seeds=args.seeds,
        )
    )
    mean_rows = _summarize(raw_rows)
    _write_csv(os.path.join(out_dir, "annealed_vs_unannealed_trajectory_raw.csv"), raw_rows)
    _write_csv(os.path.join(out_dir, "annealed_vs_unannealed_trajectory_mean.csv"), mean_rows)
    _maybe_plot(mean_rows, out_dir)
    print(f"[traj-compare] out_dir={out_dir}")


if __name__ == "__main__":
    main()
