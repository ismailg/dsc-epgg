from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.plot_phase3_checkpoint_trends import (
    _aggregate_per_seed,
    _filter_base_rows,
    _mean_sem,
    _read_rows,
)


def _summarize_policy_rows(
    seed_rows: Sequence[Dict[str, float]],
    *,
    policy_label: str,
) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, int, float], List[Dict[str, float]]] = {}
    for row in seed_rows:
        key = (
            str(row["condition"]),
            int(row["checkpoint_episode"]),
            float(row["f_value"]),
        )
        grouped.setdefault(key, []).append(dict(row))

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
                "policy_label": policy_label,
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


def _paired_policy_delta_rows(
    greedy_seed_rows: Sequence[Dict[str, float]],
    sample_seed_rows: Sequence[Dict[str, float]],
) -> List[Dict[str, float]]:
    greedy_map = {
        (
            str(row["condition"]),
            int(row["checkpoint_episode"]),
            int(row["train_seed"]),
            float(row["f_value"]),
        ): dict(row)
        for row in greedy_seed_rows
    }
    sample_map = {
        (
            str(row["condition"]),
            int(row["checkpoint_episode"]),
            int(row["train_seed"]),
            float(row["f_value"]),
        ): dict(row)
        for row in sample_seed_rows
    }
    keys = sorted(set(greedy_map.keys()) & set(sample_map.keys()))
    out: List[Dict[str, float]] = []
    for key in keys:
        g = greedy_map[key]
        s = sample_map[key]
        out.append(
            {
                "condition": key[0],
                "checkpoint_episode": key[1],
                "train_seed": key[2],
                "f_value": key[3],
                "delta_coop_greedy_minus_sample": float(g["coop_rate"]) - float(s["coop_rate"]),
                "delta_avg_reward_greedy_minus_sample": float(g["avg_reward"]) - float(s["avg_reward"]),
                "delta_avg_welfare_greedy_minus_sample": float(g["avg_welfare"]) - float(s["avg_welfare"]),
            }
        )
    return out


def _summarize_policy_deltas(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, int, float], List[Dict[str, float]]] = {}
    for row in rows:
        key = (
            str(row["condition"]),
            int(row["checkpoint_episode"]),
            float(row["f_value"]),
        )
        grouped.setdefault(key, []).append(dict(row))

    out: List[Dict[str, float]] = []
    for (condition, checkpoint_episode, f_value), group in sorted(grouped.items()):
        coop_vals = [float(row["delta_coop_greedy_minus_sample"]) for row in group]
        rew_vals = [float(row["delta_avg_reward_greedy_minus_sample"]) for row in group]
        welfare_vals = [float(row["delta_avg_welfare_greedy_minus_sample"]) for row in group]
        mean_coop, sem_coop = _mean_sem(coop_vals)
        mean_reward, sem_reward = _mean_sem(rew_vals)
        mean_welfare, sem_welfare = _mean_sem(welfare_vals)
        out.append(
            {
                "condition": condition,
                "checkpoint_episode": checkpoint_episode,
                "f_value": f_value,
                "n_train_seeds": len(group),
                "mean_delta_coop_greedy_minus_sample": mean_coop,
                "sem_delta_coop_greedy_minus_sample": sem_coop,
                "mean_delta_avg_reward_greedy_minus_sample": mean_reward,
                "sem_delta_avg_reward_greedy_minus_sample": sem_reward,
                "mean_delta_avg_welfare_greedy_minus_sample": mean_welfare,
                "sem_delta_avg_welfare_greedy_minus_sample": sem_welfare,
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


def _plot(
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
    linestyles = {"greedy": "-", "sample": "--"}
    markers = {"greedy": "o", "sample": "s"}
    checkpoints = sorted({int(row["checkpoint_episode"]) for row in summary_rows})
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(f_values),
        figsize=(4.8 * len(f_values), 3.2 * len(metrics)),
        sharex=True,
    )
    if len(metrics) == 1 and len(f_values) == 1:
        axes = np.asarray([[axes]])
    elif len(metrics) == 1:
        axes = np.asarray([axes])
    elif len(f_values) == 1:
        axes = np.asarray([[ax] for ax in axes])

    lookup = {
        (
            str(row["policy_label"]),
            str(row["condition"]),
            int(row["checkpoint_episode"]),
            float(row["f_value"]),
        ): row
        for row in summary_rows
    }

    for col_idx, f_value in enumerate(f_values):
        for row_idx, (mean_key, sem_key, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for condition in ("cond1", "cond2"):
                for policy in ("greedy", "sample"):
                    xs: List[int] = []
                    ys: List[float] = []
                    errs: List[float] = []
                    for checkpoint in checkpoints:
                        row = lookup.get((policy, condition, checkpoint, float(f_value)))
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
                        linestyle=linestyles[policy],
                        marker=markers[policy],
                        linewidth=2.0,
                        label=f"{labels[condition]} {policy}",
                    )
                    ax.fill_between(
                        x_arr,
                        y_arr - e_arr,
                        y_arr + e_arr,
                        color=colors[condition],
                        alpha=0.10 if policy == "greedy" else 0.06,
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
    fig.legend(handles, labels_list, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Greedy vs Sample Checkpoint Evaluation", y=1.06, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_report(path: str, delta_summary_rows: Sequence[Dict[str, float]], f_values: Sequence[float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# Greedy vs Sample Checkpoint Evaluation",
        "",
        "Rows report `greedy - sample` deltas after aggregating within train seed across eval seeds.",
        "",
    ]
    for f_value in f_values:
        lines.append(f"## f = {f_value:.1f}")
        lines.append("")
        lines.append("| Condition | Checkpoint | Δ Coop | Δ Reward | Δ Welfare |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        cur = [row for row in delta_summary_rows if float(row["f_value"]) == float(f_value)]
        for row in sorted(cur, key=lambda item: (str(item["condition"]), int(item["checkpoint_episode"]))):
            label = "Comm" if str(row["condition"]) == "cond1" else "No Comm"
            lines.append(
                f"| {label} | {int(row['checkpoint_episode']) // 1000}k | "
                f"{100.0 * float(row['mean_delta_coop_greedy_minus_sample']):+.1f} pp | "
                f"{float(row['mean_delta_avg_reward_greedy_minus_sample']):+.2f} | "
                f"{float(row['mean_delta_avg_welfare_greedy_minus_sample']):+.2f} |"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--greedy_suite_csv",
        type=str,
        default="outputs/eval/phase3_annealed_ext150k_15seeds/suite/checkpoint_suite_main.csv",
    )
    p.add_argument(
        "--sample_suite_csv",
        type=str,
        default="outputs/eval/paper_strengthen/iter6_base_checkpoint_sample_15seeds/checkpoint_suite_main.csv",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/eval/paper_strengthen/iter6_greedy_vs_sample_compare_15seeds",
    )
    p.add_argument("--f_values", type=float, nargs="*", default=[3.5, 5.0])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    greedy_rows = _filter_base_rows(_read_rows(args.greedy_suite_csv), eval_policy="greedy")
    sample_rows = _filter_base_rows(_read_rows(args.sample_suite_csv), eval_policy="sample")
    greedy_seed_rows = _aggregate_per_seed(greedy_rows)
    sample_seed_rows = _aggregate_per_seed(sample_rows)

    combined_summary = _summarize_policy_rows(greedy_seed_rows, policy_label="greedy")
    combined_summary.extend(_summarize_policy_rows(sample_seed_rows, policy_label="sample"))
    delta_rows = _paired_policy_delta_rows(greedy_seed_rows, sample_seed_rows)
    delta_summary_rows = _summarize_policy_deltas(delta_rows)

    os.makedirs(args.out_dir, exist_ok=True)
    _write_csv(os.path.join(args.out_dir, "greedy_vs_sample_summary.csv"), combined_summary)
    _write_csv(os.path.join(args.out_dir, "greedy_vs_sample_deltas.csv"), delta_rows)
    _write_csv(os.path.join(args.out_dir, "greedy_vs_sample_delta_summary.csv"), delta_summary_rows)
    _plot(
        combined_summary,
        out_path=os.path.join(args.out_dir, "greedy_vs_sample_checkpoint_trends.png"),
        f_values=args.f_values,
    )
    _write_report(
        os.path.join(args.out_dir, "GREEDY_VS_SAMPLE_CHECKPOINTS.md"),
        delta_summary_rows=delta_summary_rows,
        f_values=args.f_values,
    )


if __name__ == "__main__":
    main()
