from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _row_matches(
    row: Dict[str, str],
    *,
    condition: str,
    ablation: str,
    checkpoint_episode: int,
    cross_play: str,
    sender_remap: str,
) -> bool:
    if str(row.get("condition", "")) != str(condition):
        return False
    if str(row.get("ablation", "")) != str(ablation):
        return False
    if int(float(row.get("checkpoint_episode", "0") or 0)) != int(checkpoint_episode):
        return False
    if str(row.get("cross_play", "")) != str(cross_play):
        return False
    if str(row.get("sender_remap", "")) != str(sender_remap):
        return False
    return True


def _summarize_overall(rows: Sequence[Dict[str, str]], *, scope: str) -> List[Dict]:
    grouped: Dict[float, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        receiver_is_sender = int(row["receiver_is_sender"])
        if scope == "nonself" and receiver_is_sender == 1:
            continue
        if scope == "self" and receiver_is_sender == 0:
            continue
        grouped[float(row["true_f"])].append(row)

    out: List[Dict] = []
    for f_value, group in sorted(grouped.items()):
        deltas = [float(row["delta_p_cooperate_1_minus_0"]) for row in group]
        flips0 = [float(row["action_flip_rate_force0_vs_natural"]) for row in group]
        flips1 = [float(row["action_flip_rate_force1_vs_natural"]) for row in group]
        out.append(
            {
                "summary": f"overall_{scope}",
                "true_f": float(f_value),
                "n_rows": len(group),
                "mean_delta_p_cooperate": _mean(deltas),
                "mean_abs_delta_p_cooperate": _mean([abs(v) for v in deltas]),
                "mean_flip_rate_force0": _mean(flips0),
                "mean_flip_rate_force1": _mean(flips1),
            }
        )
    return out


def _summarize_pairs(rows: Sequence[Dict[str, str]], *, top_k: int) -> List[Dict]:
    grouped: Dict[Tuple[float, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (float(row["true_f"]), str(row["sender_id"]), str(row["receiver_id"]))
        grouped[key].append(row)

    out: List[Dict] = []
    for f_value in sorted({key[0] for key in grouped.keys()}):
        pair_rows = []
        for (f_key, sender_id, receiver_id), group in grouped.items():
            if float(f_key) != float(f_value):
                continue
            deltas = [float(row["delta_p_cooperate_1_minus_0"]) for row in group]
            pair_rows.append(
                {
                    "summary": "pair",
                    "true_f": float(f_value),
                    "sender_id": sender_id,
                    "receiver_id": receiver_id,
                    "n_rows": len(group),
                    "mean_delta_p_cooperate": _mean(deltas),
                    "mean_abs_delta_p_cooperate": abs(_mean(deltas)),
                }
            )
        pair_rows = sorted(
            pair_rows,
            key=lambda row: (float(row["mean_abs_delta_p_cooperate"]), row["sender_id"], row["receiver_id"]),
            reverse=True,
        )
        out.extend(pair_rows[: int(top_k)])
    return out


def _write_markdown(path: str, overall_rows: Sequence[Dict], pair_rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["# Sender Causal Summary", ""]
    lines.append("## Overall")
    lines.append("")
    lines.append("| Scope | f | Mean Delta | Mean |delta| | Flip0 | Flip1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(overall_rows, key=lambda item: (str(item["summary"]), float(item["true_f"]))):
        lines.append(
            f"| {row['summary']} | {float(row['true_f']):.1f} | "
            f"{float(row['mean_delta_p_cooperate']):.4f} | "
            f"{float(row['mean_abs_delta_p_cooperate']):.4f} | "
            f"{float(row['mean_flip_rate_force0']):.4f} | "
            f"{float(row['mean_flip_rate_force1']):.4f} |"
        )
    lines.append("")
    lines.append("## Top Pairs")
    lines.append("")
    lines.append("| f | Sender | Receiver | Mean Delta | |delta| |")
    lines.append("| ---: | --- | --- | ---: | ---: |")
    for row in sorted(pair_rows, key=lambda item: (float(item["true_f"]), -float(item["mean_abs_delta_p_cooperate"]))):
        lines.append(
            f"| {float(row['true_f']):.1f} | {row['sender_id']} | {row['receiver_id']} | "
            f"{float(row['mean_delta_p_cooperate']):.4f} | "
            f"{float(row['mean_abs_delta_p_cooperate']):.4f} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sender_causal_csv",
        type=str,
        required=True,
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
    )
    p.add_argument("--top_k_pairs", type=int, default=8)
    p.add_argument("--condition", type=str, default="cond1")
    p.add_argument("--ablation", type=str, default="none")
    p.add_argument("--checkpoint_episode", type=int, default=150000)
    p.add_argument("--cross_play", type=str, default="none")
    p.add_argument("--sender_remap", type=str, default="none")
    return p.parse_args()


def main():
    args = parse_args()
    raw_rows = _read_rows(os.path.abspath(args.sender_causal_csv))
    rows = [
        row
        for row in raw_rows
        if _row_matches(
            row,
            condition=str(args.condition),
            ablation=str(args.ablation),
            checkpoint_episode=int(args.checkpoint_episode),
            cross_play=str(args.cross_play),
            sender_remap=str(args.sender_remap),
        )
    ]
    overall_rows = _summarize_overall(rows, scope="all")
    overall_rows.extend(_summarize_overall(rows, scope="self"))
    overall_rows.extend(_summarize_overall(rows, scope="nonself"))
    pair_rows = _summarize_pairs(rows, top_k=int(args.top_k_pairs))

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    _write_rows(os.path.join(out_dir, "sender_causal_overall_summary.csv"), overall_rows)
    _write_rows(os.path.join(out_dir, "sender_causal_top_pairs.csv"), pair_rows)
    _write_markdown(os.path.join(out_dir, "sender_causal_summary.md"), overall_rows, pair_rows)
    print(f"[sender-causal-summary] out_dir={out_dir}")


if __name__ == "__main__":
    main()
