from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MetricKey = Tuple[str, int, float]
SeedMetricKey = Tuple[str, int, int, float]


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_float(text: object, default: float = float("nan")) -> float:
    try:
        return float(text)
    except Exception:
        return default


def _as_int(text: object, default: int = 0) -> int:
    try:
        return int(float(text))
    except Exception:
        return default


def _mean_sem(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _filter_base_rows(
    rows: Iterable[Dict[str, str]],
    *,
    eval_policy: str = "greedy",
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        if str(row.get("ablation", "")) != "none":
            continue
        if str(row.get("scope", "")) != "f_value":
            continue
        if str(row.get("cross_play", "")) != "none":
            continue
        if str(row.get("sender_remap", "")) != "none":
            continue
        if str(row.get("eval_policy", "")) != str(eval_policy):
            continue
        out.append(dict(row))
    return out


def _aggregate_per_seed(rows: Iterable[Dict[str, str]]) -> List[Dict[str, float]]:
    grouped: Dict[SeedMetricKey, Dict[str, float]] = defaultdict(
        lambda: {
            "weight": 0.0,
            "coop_rate_sum": 0.0,
            "avg_reward_sum": 0.0,
            "avg_welfare_sum": 0.0,
        }
    )
    for row in rows:
        condition = str(row["condition"])
        checkpoint_episode = _as_int(row["checkpoint_episode"])
        train_seed = _as_int(row["train_seed"])
        f_value = _as_float(row["key"])
        weight = max(0.0, _as_float(row.get("n_rounds", 0.0), 0.0))
        if not math.isfinite(f_value) or weight <= 0.0:
            continue
        key = (condition, checkpoint_episode, train_seed, f_value)
        grouped[key]["weight"] += weight
        grouped[key]["coop_rate_sum"] += weight * _as_float(row["coop_rate"])
        grouped[key]["avg_reward_sum"] += weight * _as_float(row["avg_reward"])
        grouped[key]["avg_welfare_sum"] += weight * _as_float(row["avg_welfare"])

    out: List[Dict[str, float]] = []
    for (condition, checkpoint_episode, train_seed, f_value), acc in sorted(grouped.items()):
        weight = acc["weight"]
        out.append(
            {
                "condition": condition,
                "checkpoint_episode": checkpoint_episode,
                "train_seed": train_seed,
                "f_value": f_value,
                "coop_rate": acc["coop_rate_sum"] / weight,
                "avg_reward": acc["avg_reward_sum"] / weight,
                "avg_welfare": acc["avg_welfare_sum"] / weight,
                "n_rounds": weight,
            }
        )
    return out


def _summarize_seed_rows(seed_rows: Iterable[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[MetricKey, List[Dict[str, float]]] = defaultdict(list)
    for row in seed_rows:
        key = (
            str(row["condition"]),
            int(row["checkpoint_episode"]),
            float(row["f_value"]),
        )
        grouped[key].append(dict(row))

    out: List[Dict[str, float]] = []
    for (condition, checkpoint_episode, f_value), group in sorted(grouped.items()):
        coop_vals = [float(row["coop_rate"]) for row in group]
        rew_vals = [float(row["avg_reward"]) for row in group]
        welfare_vals = [float(row["avg_welfare"]) for row in group]
        mean_coop, sem_coop = _mean_sem(coop_vals)
        mean_reward, sem_reward = _mean_sem(rew_vals)
        mean_welfare, sem_welfare = _mean_sem(welfare_vals)
        out.append(
            {
                "condition": condition,
                "checkpoint_episode": checkpoint_episode,
                "f_value": f_value,
                "n_train_seeds": len(group),
                "mean_coop_rate": mean_coop,
                "sem_coop_rate": sem_coop,
                "mean_avg_reward": mean_reward,
                "sem_avg_reward": sem_reward,
                "mean_avg_welfare": mean_welfare,
                "sem_avg_welfare": sem_welfare,
            }
        )
    return out


def _paired_delta_rows(seed_rows: Iterable[Dict[str, float]]) -> List[Dict[str, float]]:
    by_seed: Dict[Tuple[str, int, float], Dict[int, Dict[str, float]]] = defaultdict(dict)
    for row in seed_rows:
        key = (
            str(row["condition"]),
            int(row["train_seed"]),
            float(row["f_value"]),
        )
        by_seed[key][int(row["checkpoint_episode"])] = dict(row)

    out: List[Dict[str, float]] = []
    for (condition, train_seed, f_value), per_episode in sorted(by_seed.items()):
        if 50000 not in per_episode or 150000 not in per_episode:
            continue
        start = per_episode[50000]
        end = per_episode[150000]
        out.append(
            {
                "condition": condition,
                "train_seed": train_seed,
                "f_value": f_value,
                "delta_coop_150k_minus_50k": float(end["coop_rate"]) - float(start["coop_rate"]),
                "delta_avg_reward_150k_minus_50k": float(end["avg_reward"]) - float(start["avg_reward"]),
                "delta_avg_welfare_150k_minus_50k": float(end["avg_welfare"]) - float(start["avg_welfare"]),
            }
        )
    return out


def _summarize_paired_deltas(delta_rows: Iterable[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, float]]] = defaultdict(list)
    for row in delta_rows:
        grouped[(str(row["condition"]), float(row["f_value"]))].append(dict(row))

    out: List[Dict[str, float]] = []
    for (condition, f_value), group in sorted(grouped.items()):
        coop_vals = [float(row["delta_coop_150k_minus_50k"]) for row in group]
        rew_vals = [float(row["delta_avg_reward_150k_minus_50k"]) for row in group]
        welfare_vals = [float(row["delta_avg_welfare_150k_minus_50k"]) for row in group]
        mean_coop, sem_coop = _mean_sem(coop_vals)
        mean_reward, sem_reward = _mean_sem(rew_vals)
        mean_welfare, sem_welfare = _mean_sem(welfare_vals)
        out.append(
            {
                "condition": condition,
                "f_value": f_value,
                "n_train_seeds": len(group),
                "mean_delta_coop_150k_minus_50k": mean_coop,
                "sem_delta_coop_150k_minus_50k": sem_coop,
                "mean_delta_avg_reward_150k_minus_50k": mean_reward,
                "sem_delta_avg_reward_150k_minus_50k": sem_reward,
                "mean_delta_avg_welfare_150k_minus_50k": mean_welfare,
                "sem_delta_avg_welfare_150k_minus_50k": sem_welfare,
                "n_negative_coop_deltas": int(sum(val < 0.0 for val in coop_vals)),
                "n_negative_reward_deltas": int(sum(val < 0.0 for val in rew_vals)),
                "n_negative_welfare_deltas": int(sum(val < 0.0 for val in welfare_vals)),
            }
        )
    return out


def _write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_summary(
    summary_rows: Sequence[Dict[str, float]],
    *,
    out_path: str,
    f_values: Sequence[float],
) -> None:
    metrics = [
        ("mean_coop_rate", "sem_coop_rate", "Cooperation"),
        ("mean_avg_reward", "sem_avg_reward", "Avg Reward"),
        ("mean_avg_welfare", "sem_avg_welfare", "Avg Welfare"),
    ]
    colors = {"cond1": "#1f77b4", "cond2": "#d62728"}
    labels = {"cond1": "Comm", "cond2": "No Comm"}
    checkpoints = sorted(
        {
            int(row["checkpoint_episode"])
            for row in summary_rows
            if float(row["f_value"]) in {float(v) for v in f_values}
        }
    )
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(f_values),
        figsize=(4.6 * len(f_values), 3.2 * len(metrics)),
        sharex=True,
    )
    if len(metrics) == 1 and len(f_values) == 1:
        axes = np.asarray([[axes]])
    elif len(metrics) == 1:
        axes = np.asarray([axes])
    elif len(f_values) == 1:
        axes = np.asarray([[ax] for ax in axes])

    lookup: Dict[Tuple[str, int, float], Dict[str, float]] = {}
    for row in summary_rows:
        lookup[(str(row["condition"]), int(row["checkpoint_episode"]), float(row["f_value"]))] = row

    for col_idx, f_value in enumerate(f_values):
        for row_idx, (mean_key, sem_key, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for condition in ("cond1", "cond2"):
                xs: List[int] = []
                ys: List[float] = []
                errs: List[float] = []
                for checkpoint in checkpoints:
                    row = lookup.get((condition, checkpoint, float(f_value)))
                    if row is None:
                        continue
                    xs.append(checkpoint)
                    ys.append(float(row[mean_key]))
                    errs.append(float(row[sem_key]))
                if not xs:
                    continue
                x_arr = np.asarray(xs, dtype=np.int64)
                y_arr = np.asarray(ys, dtype=np.float64)
                e_arr = np.asarray(errs, dtype=np.float64)
                ax.plot(
                    x_arr,
                    y_arr,
                    color=colors[condition],
                    marker="o",
                    linewidth=2.0,
                    label=labels[condition],
                )
                ax.fill_between(
                    x_arr,
                    y_arr - e_arr,
                    y_arr + e_arr,
                    color=colors[condition],
                    alpha=0.18,
                    linewidth=0.0,
                )
            if row_idx == 0:
                ax.set_title(f"f = {f_value:.1f}")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.set_xticks(checkpoints)
            ax.set_xticklabels([f"{int(x/1000)}k" for x in checkpoints])
    for ax in axes[-1, :]:
        ax.set_xlabel("Checkpoint")
    handles, labels_list = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Base Checkpoint Trends (mean ± SEM across train seeds)", y=1.05, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    path: str,
    *,
    delta_summary_rows: Sequence[Dict[str, float]],
    f_values: Sequence[float],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# Phase 3 Base Checkpoint Trends",
        "",
        "This diagnostic summarizes the greedy checkpoint-evaluation trend for the base `comm` and `no-comm` branches.",
        "Rows are aggregated within each train seed across eval seeds using `n_rounds` weights, then summarized across train seeds.",
        "",
    ]
    for f_value in f_values:
        lines.append(f"## f = {f_value:.1f}")
        lines.append("")
        lines.append("| Condition | Mean delta coop (150k-50k) | Mean delta reward | Mean delta welfare | Negative coop deltas |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        cur = [row for row in delta_summary_rows if float(row["f_value"]) == float(f_value)]
        for row in sorted(cur, key=lambda item: str(item["condition"])):
            label = "Comm" if str(row["condition"]) == "cond1" else "No Comm"
            lines.append(
                f"| {label} | "
                f"{100.0 * float(row['mean_delta_coop_150k_minus_50k']):+.1f} pp | "
                f"{float(row['mean_delta_avg_reward_150k_minus_50k']):+.2f} | "
                f"{float(row['mean_delta_avg_welfare_150k_minus_50k']):+.2f} | "
                f"{int(row['n_negative_coop_deltas'])}/{int(row['n_train_seeds'])} |"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--suite_csv",
        type=str,
        default="outputs/eval/phase3_annealed_ext150k_15seeds/suite/checkpoint_suite_main.csv",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/eval/paper_strengthen/iter5_base_checkpoint_trends_15seeds",
    )
    p.add_argument(
        "--f_values",
        type=float,
        nargs="*",
        default=[3.5, 5.0],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.suite_csv)
    base_rows = _filter_base_rows(rows, eval_policy="greedy")
    seed_rows = _aggregate_per_seed(base_rows)
    summary_rows = _summarize_seed_rows(seed_rows)
    delta_rows = _paired_delta_rows(seed_rows)
    delta_summary_rows = _summarize_paired_deltas(delta_rows)

    os.makedirs(args.out_dir, exist_ok=True)
    _write_csv(os.path.join(args.out_dir, "seed_level_metrics.csv"), seed_rows)
    _write_csv(os.path.join(args.out_dir, "checkpoint_trend_summary.csv"), summary_rows)
    _write_csv(os.path.join(args.out_dir, "checkpoint_trend_deltas.csv"), delta_rows)
    _write_csv(os.path.join(args.out_dir, "checkpoint_trend_delta_summary.csv"), delta_summary_rows)
    _plot_summary(
        summary_rows,
        out_path=os.path.join(args.out_dir, "base_checkpoint_trends.png"),
        f_values=args.f_values,
    )
    _write_report(
        os.path.join(args.out_dir, "BASE_CHECKPOINT_TRENDS.md"),
        delta_summary_rows=delta_summary_rows,
        f_values=args.f_values,
    )


if __name__ == "__main__":
    main()
