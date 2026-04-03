#!/usr/bin/env python3
"""Plot dense sampled training curves with sparse greedy checkpoint-eval dots.

This is the vecstraight analogue of the manuscript-style online-vs-checkpoint
figure:
- solid curves: sampled/on-policy training-window behavior from JSONL metrics
- dots: greedy checkpoint evaluation at 25k/50k/100k/150k
- separate panels for f=3.5 and f=5.0
- comm vs no-comm on the same panel
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMM_COLOR = "#1f77b4"
NOCOMM_COLOR = "#d62728"
COND_LABEL = {"cond1": "Comm", "cond2": "No-Comm"}
F_LABEL = {"3.500": "f = 3.5", "5.000": "f = 5.0"}


def _read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows for {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_sem(values: Sequence[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(len(arr)))


def _collect_online_rows(metrics_dirs: Sequence[str], f_keys: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    wanted = set(str(v) for v in f_keys)
    for metrics_dir in metrics_dirs:
        if not os.path.isdir(metrics_dir):
            raise FileNotFoundError(f"metrics directory not found: {metrics_dir}")
        for fname in sorted(os.listdir(metrics_dir)):
            if not fname.endswith(".jsonl"):
                continue
            stem = fname[:-6]
            parts = stem.split("_")
            if len(parts) != 2:
                continue
            cond = parts[0]
            seed = int(parts[1].replace("seed", ""))
            for row in _read_jsonl(os.path.join(metrics_dir, fname)):
                if row.get("scope") != "f_value":
                    continue
                if row.get("window") != "window":
                    continue
                if str(row.get("key")) not in wanted:
                    continue
                rows.append(
                    {
                        "condition": cond,
                        "seed": seed,
                        "episode": int(row["episode"]),
                        "f_key": str(row["key"]),
                        "coop_rate": float(row["coop_rate"]),
                        "avg_reward": float(row["avg_reward"]),
                        "avg_welfare": float(row["avg_reward"]) * 4.0,
                        "source_metrics_dir": os.path.abspath(metrics_dir),
                    }
                )
    rows.sort(key=lambda r: (str(r["condition"]), str(r["f_key"]), int(r["seed"]), int(r["episode"])))
    return rows


def _summarize_online(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["f_key"]), int(row["episode"]))].append(dict(row))
    out: List[Dict[str, object]] = []
    for (condition, f_key, episode), group in sorted(grouped.items()):
        coop_vals = [float(r["coop_rate"]) for r in group]
        reward_vals = [float(r["avg_reward"]) for r in group]
        welfare_vals = [float(r["avg_welfare"]) for r in group]
        mean_coop, sem_coop = _mean_sem(coop_vals)
        mean_reward, sem_reward = _mean_sem(reward_vals)
        mean_welfare, sem_welfare = _mean_sem(welfare_vals)
        out.append(
            {
                "condition": condition,
                "f_key": f_key,
                "episode": episode,
                "n_seeds": len(group),
                "mean_coop_rate": mean_coop,
                "sem_coop_rate": sem_coop,
                "mean_avg_reward": mean_reward,
                "sem_avg_reward": sem_reward,
                "mean_avg_welfare": mean_welfare,
                "sem_avg_welfare": sem_welfare,
                "seed_list": ",".join(str(int(r["seed"])) for r in sorted(group, key=lambda x: int(x["seed"]))),
            }
        )
    return out


def _collect_checkpoint_rows(
    suite_csvs: Sequence[str], f_keys: Sequence[str], episodes: Sequence[int]
) -> List[Dict[str, object]]:
    wanted_f = set(str(v) for v in f_keys)
    wanted_ep = set(int(v) for v in episodes)
    grouped: Dict[Tuple[str, str, int], List[Dict[str, object]]] = defaultdict(list)

    for suite_csv in suite_csvs:
        rows = _read_csv_rows(suite_csv)
        for row in rows:
            if str(row.get("scope")) != "f_value":
                continue
            if str(row.get("key")) not in wanted_f:
                continue
            if str(row.get("eval_policy")) != "greedy":
                continue
            if str(row.get("ablation", "none")) != "none":
                continue
            ep = int(float(row["checkpoint_episode"]))
            if ep not in wanted_ep:
                continue
            grouped[(str(row["condition"]), str(row["key"]), ep)].append(
                {
                    "condition": str(row["condition"]),
                    "seed": int(float(row["train_seed"])),
                    "f_key": str(row["key"]),
                    "episode": ep,
                    "coop_rate": float(row["coop_rate"]),
                    "avg_reward": float(row["avg_reward"]),
                    "avg_welfare": float(row.get("avg_welfare", float(row["avg_reward"]) * 4.0)),
                    "source_suite_csv": os.path.abspath(suite_csv),
                }
            )

    out: List[Dict[str, object]] = []
    for (condition, f_key, episode), group in sorted(grouped.items()):
        coop_vals = [float(r["coop_rate"]) for r in group]
        reward_vals = [float(r["avg_reward"]) for r in group]
        welfare_vals = [float(r["avg_welfare"]) for r in group]
        mean_coop, sem_coop = _mean_sem(coop_vals)
        mean_reward, sem_reward = _mean_sem(reward_vals)
        mean_welfare, sem_welfare = _mean_sem(welfare_vals)
        out.append(
            {
                "condition": condition,
                "f_key": f_key,
                "episode": episode,
                "n_seeds": len(group),
                "mean_coop_rate": mean_coop,
                "sem_coop_rate": sem_coop,
                "mean_avg_reward": mean_reward,
                "sem_avg_reward": sem_reward,
                "mean_avg_welfare": mean_welfare,
                "sem_avg_welfare": sem_welfare,
                "seed_list": ",".join(str(int(r["seed"])) for r in sorted(group, key=lambda x: int(x["seed"]))),
            }
        )
    return out


def _plot(
    online_summary: Sequence[Dict[str, object]],
    checkpoint_summary: Sequence[Dict[str, object]],
    f_keys: Sequence[str],
    out_png: str,
) -> None:
    fig, axes = plt.subplots(1, len(f_keys), figsize=(15, 5.5), sharey=True)
    if len(f_keys) == 1:
        axes = [axes]
    colors = {"cond1": COMM_COLOR, "cond2": NOCOMM_COLOR}

    for ax, f_key in zip(axes, f_keys):
        for condition in ["cond1", "cond2"]:
            line_rows = sorted(
                [r for r in online_summary if r["condition"] == condition and r["f_key"] == f_key],
                key=lambda r: int(r["episode"]),
            )
            if line_rows:
                xs = np.asarray([int(r["episode"]) / 1000.0 for r in line_rows], dtype=float)
                ys = np.asarray([float(r["mean_coop_rate"]) for r in line_rows], dtype=float)
                sems = np.asarray([float(r["sem_coop_rate"]) for r in line_rows], dtype=float)
                ax.plot(xs, ys, color=colors[condition], linewidth=2.0, alpha=0.7)
                ax.fill_between(xs, ys - sems, ys + sems, color=colors[condition], alpha=0.16)

            dot_rows = sorted(
                [r for r in checkpoint_summary if r["condition"] == condition and r["f_key"] == f_key],
                key=lambda r: int(r["episode"]),
            )
            if dot_rows:
                xs = np.asarray([int(r["episode"]) / 1000.0 for r in dot_rows], dtype=float)
                ys = np.asarray([float(r["mean_coop_rate"]) for r in dot_rows], dtype=float)
                ax.plot(
                    xs,
                    ys,
                    linestyle="--",
                    linewidth=1.4,
                    color=colors[condition],
                    alpha=0.95,
                    marker="o",
                    markersize=7,
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    label=COND_LABEL[condition] if f_key == f_keys[0] else None,
                )

        ax.axvline(50, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.axvline(100, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_xlim(0, 155)
        ax.set_ylim(0.2, 0.8)
        ax.set_title(F_LABEL.get(f_key, f_key), fontsize=13, fontweight="bold")
        ax.set_xlabel("Episode (x1k)")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Cooperation rate")
    handles = [
        plt.Line2D([0], [0], color=COMM_COLOR, linewidth=2.0, alpha=0.7, label="Comm sampled online"),
        plt.Line2D([0], [0], color=NOCOMM_COLOR, linewidth=2.0, alpha=0.7, label="No-Comm sampled online"),
        plt.Line2D([0], [0], color="black", linestyle="--", marker="o", markersize=7, linewidth=1.4, label="Greedy checkpoint eval"),
    ]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99), frameon=False)
    fig.suptitle(
        "Vecstraight online sampled curves with greedy checkpoint anchors",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        0.01,
        (
            "Solid curves show sampled training-window behavior; dots show greedy checkpoint evaluation "
            "at 25k/50k/100k/150k. Vertical lines mark 50k and 100k."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.93))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(
    checkpoint_summary: Sequence[Dict[str, object]],
    out_md: str,
    f_keys: Sequence[str],
    checkpoint_episodes: Sequence[int],
) -> None:
    lines: List[str] = []
    lines.append("# Vecstraight Online Vs Checkpoint Summary")
    lines.append("")
    lines.append("Checkpoint dots below are greedy `ablation=none` evaluation means across seeds.")
    lines.append("")
    for f_key in f_keys:
        lines.append(f"## {F_LABEL.get(f_key, f_key)}")
        lines.append("")
        episode_headers = " | ".join(f"{int(ep/1000)}k" for ep in checkpoint_episodes)
        lines.append(f"| Condition | {episode_headers} |")
        lines.append("|---|" + "|".join("---:" for _ in checkpoint_episodes) + "|")
        for condition in ["cond1", "cond2"]:
            rows = {
                int(r["episode"]): r
                for r in checkpoint_summary
                if r["condition"] == condition and r["f_key"] == f_key
            }
            vals = []
            for ep in checkpoint_episodes:
                if ep in rows:
                    vals.append(f"{100.0 * float(rows[ep]['mean_coop_rate']):.1f} pp")
                else:
                    vals.append("NA")
            lines.append(f"| {COND_LABEL[condition]} | " + " | ".join(vals) + " |")
        lines.append("")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metrics_dir",
        nargs="*",
        type=str,
        default=[
            "iwr-results/phase3-150k-cond1-15seed-trainonly-20260323/outputs/phase3_cond1_15seeds_train_only/metrics",
            "iwr-results/phase3-150k-straight-c1-s101-subproc-20260323/outputs/phase3_straight_c1_s101_subproc/metrics",
            "iwr-results/phase3-150k-cond2-15seed-trainonly-20260324/code/outputs/train/phase3-150k-cond2-15seed-trainonly-20260324/outputs/phase3_cond2_15seeds_train_only/metrics",
        ],
    )
    p.add_argument(
        "--suite_csv",
        nargs="*",
        type=str,
        default=[
            "outputs/eval/phase3_vectorized_ext150k_15seeds_local_20260324/suite/checkpoint_suite_main.csv",
            "outputs/eval/phase3_vectorized_ext150k_cond2_15seeds_iwr_20260325/suite/checkpoint_suite_main.csv",
        ],
    )
    p.add_argument("--f_keys", nargs="*", type=str, default=["3.500", "5.000"])
    p.add_argument(
        "--checkpoint_episodes",
        nargs="*",
        type=int,
        default=[25000, 50000, 75000, 100000, 125000, 150000],
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/eval/phase3_vecstraight_online_vs_checkpoint_15seeds_local_20260325",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    f_keys = [str(v) for v in args.f_keys]
    online_rows = _collect_online_rows(metrics_dirs=[str(v) for v in args.metrics_dir], f_keys=f_keys)
    online_summary = _summarize_online(online_rows)
    checkpoint_summary = _collect_checkpoint_rows(
        suite_csvs=[str(v) for v in args.suite_csv],
        f_keys=f_keys,
        episodes=[int(v) for v in args.checkpoint_episodes],
    )
    os.makedirs(args.out_dir, exist_ok=True)
    _write_csv(os.path.join(args.out_dir, "online_fvalue_rows.csv"), online_rows)
    _write_csv(os.path.join(args.out_dir, "online_fvalue_summary.csv"), online_summary)
    _write_csv(os.path.join(args.out_dir, "checkpoint_fvalue_summary.csv"), checkpoint_summary)
    _plot(
        online_summary=online_summary,
        checkpoint_summary=checkpoint_summary,
        f_keys=f_keys,
        out_png=os.path.join(args.out_dir, "online_vs_checkpoint_f35_f50.png"),
    )
    _write_markdown(
        checkpoint_summary=checkpoint_summary,
        out_md=os.path.join(args.out_dir, "ONLINE_VS_CHECKPOINT.md"),
        f_keys=f_keys,
        checkpoint_episodes=[int(v) for v in args.checkpoint_episodes],
    )
    print(f"[ok] wrote {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
