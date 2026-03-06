from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
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


def _mean(values: List[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / max(1, len(vals))


def _fragmentation_over_time(
    sender_alignment_rows: List[Dict],
    receiver_summary_rows: List[Dict],
    receiver_by_sender_rows: List[Dict],
) -> List[Dict]:
    align_lookup = {}
    for row in sender_alignment_rows:
        if row.get("eval_policy", "greedy") != "greedy":
            continue
        if row.get("ablation", "none") != "none":
            continue
        if row.get("cross_play", "none") != "none":
            continue
        align_lookup[
            (
                row.get("condition"),
                _as_int(row, "train_seed", -1),
                _as_int(row, "checkpoint_episode", 0),
            )
        ] = row

    agg_token = defaultdict(list)
    for row in receiver_summary_rows:
        if row.get("eval_policy", "greedy") != "greedy":
            continue
        if row.get("ablation", "none") != "none":
            continue
        if row.get("cross_play", "none") != "none":
            continue
        agg_token[
            (
                row.get("condition"),
                _as_int(row, "train_seed", -1),
                _as_int(row, "checkpoint_episode", 0),
            )
        ].append(_as_float(row, "delta_m1_minus_m0"))

    sender_token = defaultdict(list)
    for row in receiver_by_sender_rows:
        if row.get("eval_policy", "greedy") != "greedy":
            continue
        if row.get("ablation", "none") != "none":
            continue
        if row.get("cross_play", "none") != "none":
            continue
        if row.get("receiver_id") != "all_agents":
            continue
        sender_token[
            (
                row.get("condition"),
                _as_int(row, "train_seed", -1),
                _as_int(row, "checkpoint_episode", 0),
            )
        ].append(_as_float(row, "delta_m1_minus_m0"))

    out = []
    keys = set(align_lookup.keys()) | set(agg_token.keys()) | set(sender_token.keys())
    for key in sorted(keys):
        condition, seed, episode = key
        align = align_lookup.get(key, {})
        agg_vals = agg_token.get(key, [])
        sender_vals = sender_token.get(key, [])
        out.append(
            {
                "condition": condition,
                "train_seed": int(seed),
                "checkpoint_episode": int(episode),
                "aggregate_token_effect_mean": float(_mean(agg_vals)) if agg_vals else 0.0,
                "aggregate_token_effect_abs_mean": float(_mean([abs(v) for v in agg_vals])) if agg_vals else 0.0,
                "sender_specific_effect_mean": float(_mean(sender_vals)) if sender_vals else 0.0,
                "sender_specific_effect_abs_mean": float(_mean([abs(v) for v in sender_vals])) if sender_vals else 0.0,
                "alignment_action_sign": _as_float(align, "alignment_action_sign"),
                "alignment_regime_sign": _as_float(align, "alignment_regime_sign"),
                "mean_abs_action_delta": _as_float(align, "mean_abs_action_delta"),
                "mean_abs_regime_delta": _as_float(align, "mean_abs_regime_delta"),
            }
        )
    return out


def _sender_polarity_over_time(sender_rows: List[Dict]) -> List[Dict]:
    out = []
    for row in sender_rows:
        if row.get("eval_policy", "greedy") != "greedy":
            continue
        if row.get("ablation", "none") != "none":
            continue
        if row.get("cross_play", "none") != "none":
            continue
        out.append(
            {
                "condition": row.get("condition"),
                "train_seed": _as_int(row, "train_seed", -1),
                "checkpoint_episode": _as_int(row, "checkpoint_episode", 0),
                "sender_id": row.get("sender_id"),
                "delta_action1_minus_action0": _as_float(row, "delta_action1_minus_action0"),
                "delta_high_minus_low_fhat": _as_float(row, "delta_high_minus_low_fhat"),
            }
        )
    return sorted(out, key=lambda r: (r["condition"], r["train_seed"], r["sender_id"], r["checkpoint_episode"]))


def _maybe_plot(fragment_rows: List[Dict], sender_rows: List[Dict], out_dir: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[fragmentation] matplotlib unavailable; skipped PNG plots")
        return

    os.makedirs(out_dir, exist_ok=True)
    seeds = sorted({int(row["train_seed"]) for row in fragment_rows})
    if len(seeds) == 0:
        return

    fig, axes = plt.subplots(len(seeds), 1, figsize=(8, 3 * len(seeds)), squeeze=False)
    for ax, seed in zip(axes[:, 0], seeds):
        rows = [row for row in fragment_rows if int(row["train_seed"]) == int(seed)]
        xs = [int(row["checkpoint_episode"]) for row in rows]
        ax.plot(xs, [float(row["aggregate_token_effect_abs_mean"]) for row in rows], marker="o", label="aggregate |token effect|")
        ax.plot(xs, [float(row["sender_specific_effect_abs_mean"]) for row in rows], marker="s", label="sender-specific |token effect|")
        ax.plot(xs, [float(row["alignment_regime_sign"]) for row in rows], marker="^", label="regime alignment")
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("Checkpoint episode")
        ax.set_ylabel("Effect / alignment")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fragmentation_over_time.png"), dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(seeds), 2, figsize=(12, 3 * len(seeds)), squeeze=False)
    for row_idx, seed in enumerate(seeds):
        seed_rows = [row for row in sender_rows if int(row["train_seed"]) == int(seed)]
        sender_ids = sorted({row["sender_id"] for row in seed_rows})
        ax_action = axes[row_idx, 0]
        ax_regime = axes[row_idx, 1]
        for sender_id in sender_ids:
            cur = sorted(
                [row for row in seed_rows if row["sender_id"] == sender_id],
                key=lambda r: int(r["checkpoint_episode"]),
            )
            xs = [int(row["checkpoint_episode"]) for row in cur]
            ax_action.plot(xs, [float(row["delta_action1_minus_action0"]) for row in cur], marker="o", label=sender_id)
            ax_regime.plot(xs, [float(row["delta_high_minus_low_fhat"]) for row in cur], marker="o", label=sender_id)
        ax_action.set_title(f"Seed {seed} action polarity")
        ax_regime.set_title(f"Seed {seed} regime polarity")
        ax_action.set_xlabel("Checkpoint episode")
        ax_regime.set_xlabel("Checkpoint episode")
        ax_action.set_ylabel("Delta")
        ax_regime.set_ylabel("Delta")
        ax_action.axhline(0.0, color="black", linewidth=0.8)
        ax_regime.axhline(0.0, color="black", linewidth=0.8)
        ax_action.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sender_polarity_over_time.png"), dpi=160)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sender_alignment_csv", type=str, default="outputs/eval/phase3/report/sender_alignment_summary.csv")
    p.add_argument("--receiver_summary_csv", type=str, default="outputs/eval/phase3/report/receiver_semantics_summary.csv")
    p.add_argument("--receiver_by_sender_csv", type=str, default="outputs/eval/phase3/report/receiver_by_sender_summary.csv")
    p.add_argument("--sender_summary_csv", type=str, default="outputs/eval/phase3/report/sender_semantics_summary.csv")
    p.add_argument("--out_dir", type=str, default="outputs/eval/phase3/fragmentation_figures")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    sender_alignment_rows = _read_csv_rows(args.sender_alignment_csv)
    receiver_summary_rows = _read_csv_rows(args.receiver_summary_csv)
    receiver_by_sender_rows = _read_csv_rows(args.receiver_by_sender_csv)
    sender_summary_rows = _read_csv_rows(args.sender_summary_csv)

    fragmentation_rows = _fragmentation_over_time(
        sender_alignment_rows=sender_alignment_rows,
        receiver_summary_rows=receiver_summary_rows,
        receiver_by_sender_rows=receiver_by_sender_rows,
    )
    sender_polarity_rows = _sender_polarity_over_time(sender_summary_rows)

    _write_csv(os.path.join(out_dir, "fragmentation_over_time.csv"), fragmentation_rows)
    _write_csv(os.path.join(out_dir, "sender_polarity_over_time.csv"), sender_polarity_rows)
    _maybe_plot(fragmentation_rows, sender_polarity_rows, out_dir)
    print(f"[fragmentation] out_dir={out_dir}")


if __name__ == "__main__":
    main()
