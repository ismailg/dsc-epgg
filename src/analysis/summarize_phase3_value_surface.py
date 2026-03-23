from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_float_key(text: str) -> bool:
    try:
        float(text)
        return True
    except Exception:
        return False


def _filter_metric_map(
    rows: Sequence[Dict[str, str]],
    *,
    condition: str,
    ablation: str,
    checkpoint_episode: int,
    suite_kind: Optional[str],
    metric: str,
) -> Dict[Tuple[int, float], float]:
    out: Dict[Tuple[int, float], float] = {}
    for row in rows:
        if str(row.get("condition", "")) != str(condition):
            continue
        if str(row.get("ablation", "")) != str(ablation):
            continue
        if int(float(row.get("checkpoint_episode", "0") or 0)) != int(checkpoint_episode):
            continue
        if suite_kind is not None and str(row.get("suite_kind", "")) != str(suite_kind):
            continue
        key = str(row.get("key", ""))
        if not _is_float_key(key):
            continue
        train_seed = int(float(row["train_seed"]))
        f_value = float(key)
        out[(train_seed, f_value)] = float(row[metric])
    return out


def _mean_sem(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _bootstrap_ci_mean(
    values: Sequence[float],
    *,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    means = np.zeros((int(n_boot),), dtype=np.float64)
    for i in range(int(n_boot)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(np.mean(sample))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _sign_flip_pvalue(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return float("nan")
    observed = abs(float(np.mean(arr)))
    if n > 20:
        rng = np.random.default_rng(1729 + n)
        reps = 200000
        null_means = np.zeros((reps,), dtype=np.float64)
        for i in range(reps):
            signs = rng.choice(np.array([-1.0, 1.0]), size=n, replace=True)
            null_means[i] = abs(float(np.mean(arr * signs)))
        return float((1.0 + np.sum(null_means >= observed)) / float(reps + 1))
    exceed = 0
    total = 1 << n
    for mask in range(total):
        signs = np.ones((n,), dtype=np.float64)
        for bit in range(n):
            if (mask >> bit) & 1:
                signs[bit] = -1.0
        if abs(float(np.mean(arr * signs))) >= observed - 1e-12:
            exceed += 1
    return float(exceed / total)


def _write_rows(path: str, rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(rows) == 0:
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


def _build_paired_rows(
    *,
    family: str,
    contrast: str,
    learned_map: Dict[Tuple[int, float], float],
    control_map: Dict[Tuple[int, float], float],
) -> List[Dict]:
    rows: List[Dict] = []
    keys = sorted(set(learned_map.keys()) & set(control_map.keys()), key=lambda item: (item[1], item[0]))
    for train_seed, f_value in keys:
        learned = float(learned_map[(train_seed, f_value)])
        control = float(control_map[(train_seed, f_value)])
        rows.append(
            {
                "family": family,
                "contrast": contrast,
                "train_seed": int(train_seed),
                "f_value": float(f_value),
                "learned_value": learned,
                "control_value": control,
                "delta_learned_minus_control": learned - control,
            }
        )
    return rows


def _summarize_paired_rows(
    rows: Sequence[Dict],
    *,
    metric: str,
    n_boot: int,
    alpha: float,
    rng_seed: int,
) -> List[Dict]:
    grouped: Dict[Tuple[str, str, float], List[Dict]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["contrast"]), float(row["f_value"]))
        grouped.setdefault(key, []).append(dict(row))

    out: List[Dict] = []
    for (family, contrast, f_value), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        learned_vals = [float(row["learned_value"]) for row in group_rows]
        control_vals = [float(row["control_value"]) for row in group_rows]
        deltas = [float(row["delta_learned_minus_control"]) for row in group_rows]
        learned_mean, learned_sem = _mean_sem(learned_vals)
        control_mean, control_sem = _mean_sem(control_vals)
        delta_mean, delta_sem = _mean_sem(deltas)
        ci_low, ci_high = _bootstrap_ci_mean(
            deltas,
            n_boot=int(n_boot),
            alpha=float(alpha),
            rng=np.random.default_rng(int(rng_seed) + int(round(100 * f_value)) + len(out)),
        )
        out.append(
            {
                "family": family,
                "contrast": contrast,
                "metric": metric,
                "f_value": float(f_value),
                "n_seeds": len(group_rows),
                "learned_mean": learned_mean,
                "learned_sem": learned_sem,
                "control_mean": control_mean,
                "control_sem": control_sem,
                "delta_mean": delta_mean,
                "delta_sem": delta_sem,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "sign_flip_p": _sign_flip_pvalue(deltas),
            }
        )
    return out


def _write_markdown_report(path: str, summary_rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    by_family: Dict[str, List[Dict]] = {}
    for row in summary_rows:
        by_family.setdefault(str(row["family"]), []).append(dict(row))

    lines = ["# Phase 3 Value Surface Summary", ""]
    for family in sorted(by_family.keys()):
        lines.append(f"## {family}")
        lines.append("")
        lines.append("| Contrast | f | Delta | 95% CI | p |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in sorted(by_family[family], key=lambda item: (str(item["contrast"]), float(item["f_value"]))):
            lines.append(
                "| "
                f"{row['contrast']} | {row['f_value']:.1f} | {row['delta_mean']:.4f} | "
                f"[{row['delta_ci_low']:.4f}, {row['delta_ci_high']:.4f}] | {row['sign_flip_p']:.5f} |"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--main_suite_csv",
        type=str,
        default="outputs/eval/phase3_annealed_ext150k_15seeds/suite/checkpoint_suite_main.csv",
    )
    p.add_argument(
        "--sameckpt_fixed0_csv",
        type=str,
        default="outputs/eval/phase3_sameckpt_continuation_15seeds_local_20260319/fixed0/suite/checkpoint_suite_main.csv",
    )
    p.add_argument(
        "--sameckpt_uniform_csv",
        type=str,
        default="outputs/eval/phase3_sameckpt_continuation_15seeds_local_20260319/uniform/suite/checkpoint_suite_main.csv",
    )
    p.add_argument(
        "--sameckpt_public_random_csv",
        type=str,
        default="outputs/eval/phase3_sameckpt_continuation_15seeds_local_20260319/public_random/suite/checkpoint_suite_main.csv",
    )
    p.add_argument(
        "--frozen_suite_csv",
        type=str,
        default="outputs/eval/phase3_frozen150k_15seeds_local_20260319/suite/checkpoint_suite_main.csv",
    )
    p.add_argument(
        "--frozen_contrasts",
        nargs="*",
        type=str,
        default=["fixed0", "indep_random", "public_random", "sender_shuffle", "permute_slots"],
    )
    p.add_argument("--metric", type=str, default="coop_rate")
    p.add_argument("--checkpoint_episode", type=int, default=150000)
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/eval/paper_strengthen/iter0_value_surface",
    )
    p.add_argument("--bootstrap_reps", type=int, default=20000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--rng_seed", type=int, default=20260319)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    main_rows = _read_rows(os.path.abspath(args.main_suite_csv))
    sameckpt_fixed0_rows = _read_rows(os.path.abspath(args.sameckpt_fixed0_csv))
    sameckpt_uniform_rows = _read_rows(os.path.abspath(args.sameckpt_uniform_csv))
    sameckpt_public_rows = _read_rows(os.path.abspath(args.sameckpt_public_random_csv))
    frozen_rows = _read_rows(os.path.abspath(args.frozen_suite_csv))

    learned_main = _filter_metric_map(
        main_rows,
        condition="cond1",
        ablation="none",
        checkpoint_episode=int(args.checkpoint_episode),
        suite_kind="comm",
        metric=str(args.metric),
    )
    baseline_main = _filter_metric_map(
        main_rows,
        condition="cond2",
        ablation="none",
        checkpoint_episode=int(args.checkpoint_episode),
        suite_kind="baseline",
        metric=str(args.metric),
    )

    paired_rows: List[Dict] = []
    paired_rows.extend(
        _build_paired_rows(
            family="main_gap",
            contrast="learned_vs_baseline",
            learned_map=learned_main,
            control_map=baseline_main,
        )
    )

    sameckpt_sources = {
        "fixed0": sameckpt_fixed0_rows,
        "uniform": sameckpt_uniform_rows,
        "public_random": sameckpt_public_rows,
    }
    for contrast, rows in sameckpt_sources.items():
        control_map = _filter_metric_map(
            rows,
            condition="cond1",
            ablation=str(contrast),
            checkpoint_episode=int(args.checkpoint_episode),
            suite_kind="comm",
            metric=str(args.metric),
        )
        paired_rows.extend(
            _build_paired_rows(
                family="sameckpt_gap",
                contrast=f"learned_vs_{contrast}",
                learned_map=learned_main,
                control_map=control_map,
            )
        )

    frozen_learned = _filter_metric_map(
        frozen_rows,
        condition="cond1",
        ablation="none",
        checkpoint_episode=int(args.checkpoint_episode),
        suite_kind="comm",
        metric=str(args.metric),
    )
    for contrast in args.frozen_contrasts:
        control_map = _filter_metric_map(
            frozen_rows,
            condition="cond1",
            ablation=str(contrast),
            checkpoint_episode=int(args.checkpoint_episode),
            suite_kind="comm",
            metric=str(args.metric),
        )
        paired_rows.extend(
            _build_paired_rows(
                family="frozen_gap",
                contrast=f"learned_vs_{contrast}",
                learned_map=frozen_learned,
                control_map=control_map,
            )
        )

    summary_rows = _summarize_paired_rows(
        paired_rows,
        metric=str(args.metric),
        n_boot=int(args.bootstrap_reps),
        alpha=float(args.alpha),
        rng_seed=int(args.rng_seed),
    )

    paired_path = os.path.join(out_dir, "value_surface_paired.csv")
    summary_path = os.path.join(out_dir, "value_surface_summary.csv")
    report_path = os.path.join(out_dir, "value_surface_summary.md")
    _write_rows(paired_path, paired_rows)
    _write_rows(summary_path, summary_rows)
    _write_markdown_report(report_path, summary_rows)
    print(f"[value-surface] paired={paired_path}")
    print(f"[value-surface] summary={summary_path}")
    print(f"[value-surface] report={report_path}")


if __name__ == "__main__":
    main()
