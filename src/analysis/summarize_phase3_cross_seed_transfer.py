from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


FOCAL_F_VALUES = ["3.500", "5.000"]


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _filter_transfer_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        if row.get("scope") != "f_value":
            continue
        if row.get("key") not in FOCAL_F_VALUES:
            continue
        if row.get("condition") != "cond1":
            continue
        if row.get("eval_policy", "greedy") != "greedy":
            continue
        if row.get("ablation", "none") != "none":
            continue
        if row.get("history_intervention", "none") != "none":
            continue
        out.append(row)
    return out


def _filter_reference_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        if row.get("scope") != "f_value":
            continue
        if row.get("key") not in FOCAL_F_VALUES:
            continue
        if row.get("condition") != "cond1":
            continue
        if row.get("eval_policy", "greedy") != "greedy":
            continue
        if row.get("history_intervention", "none") != "none":
            continue
        if row.get("cross_play", "none") != "none":
            continue
        if row.get("sender_remap", "none") != "none":
            continue
        out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--transfer_main_csv", type=str, required=True)
    p.add_argument("--reference_main_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    transfer_rows = _filter_transfer_rows(_read_rows(args.transfer_main_csv))
    reference_rows = _filter_reference_rows(_read_rows(args.reference_main_csv))

    transfer_pair_rows: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []
    receiver_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    reference_map: Dict[Tuple[int, str, str], float] = {}
    for row in reference_rows:
        receiver_seed = int(row["train_seed"])
        key = str(row["key"])
        ablation = str(row["ablation"])
        reference_map[(receiver_seed, key, ablation)] = float(row["coop_rate"])

    by_pair_f: Dict[Tuple[int, int, str], List[Dict[str, str]]] = defaultdict(list)
    identity_rows: Dict[Tuple[int, int, str], Dict[str, str]] = {}
    flip_rows: Dict[Tuple[int, int, str], Dict[str, str]] = {}
    for row in transfer_rows:
        receiver_seed = int(row["receiver_seed"])
        donor_seed = int(row["donor_seed"])
        f_value = str(row["key"])
        key = (receiver_seed, donor_seed, f_value)
        by_pair_f[key].append(row)
        if row.get("alignment_label") == "identity__noflip":
            identity_rows[key] = row
        if row.get("alignment_label") == "identity__flipall":
            flip_rows[key] = row

    best_alignment_counter: Counter[str] = Counter()
    for pair_key, rows in sorted(by_pair_f.items()):
        receiver_seed, donor_seed, f_value = pair_key
        natural = reference_map.get((receiver_seed, f_value, "none"))
        sender_shuffle = reference_map.get((receiver_seed, f_value, "sender_shuffle"))
        public_random = reference_map.get((receiver_seed, f_value, "public_random"))
        indep_random = reference_map.get((receiver_seed, f_value, "indep_random"))
        for row in rows:
            coop_rate = float(row["coop_rate"])
            transfer_pair_rows.append(
                {
                    "receiver_seed": receiver_seed,
                    "donor_seed": donor_seed,
                    "f_value": f_value,
                    "alignment_label": str(row["alignment_label"]),
                    "sender_remap": str(row["sender_remap"]),
                    "cross_play": str(row["cross_play"]),
                    "coop_rate": coop_rate,
                    "delta_vs_natural": (
                        "" if natural is None else float(coop_rate - float(natural))
                    ),
                    "delta_vs_sender_shuffle": (
                        "" if sender_shuffle is None else float(coop_rate - float(sender_shuffle))
                    ),
                    "delta_vs_public_random": (
                        "" if public_random is None else float(coop_rate - float(public_random))
                    ),
                    "delta_vs_indep_random": (
                        "" if indep_random is None else float(coop_rate - float(indep_random))
                    ),
                }
            )

        best = max(rows, key=lambda row: float(row["coop_rate"]))
        best_alignment_counter[str(best["alignment_label"])] += 1
        best_rate = float(best["coop_rate"])
        identity_rate = (
            float(identity_rows[pair_key]["coop_rate"])
            if pair_key in identity_rows
            else float("nan")
        )
        flip_rate = (
            float(flip_rows[pair_key]["coop_rate"])
            if pair_key in flip_rows
            else float("nan")
        )
        best_rows.append(
            {
                "receiver_seed": receiver_seed,
                "donor_seed": donor_seed,
                "f_value": f_value,
                "best_alignment_label": str(best["alignment_label"]),
                "best_coop_rate": best_rate,
                "identity_coop_rate": identity_rate,
                "flipall_coop_rate": flip_rate,
                "natural_coop_rate": natural,
                "sender_shuffle_coop_rate": sender_shuffle,
                "public_random_coop_rate": public_random,
                "indep_random_coop_rate": indep_random,
                "best_minus_natural": "" if natural is None else float(best_rate - float(natural)),
                "best_minus_public_random": (
                    "" if public_random is None else float(best_rate - float(public_random))
                ),
                "best_minus_sender_shuffle": (
                    "" if sender_shuffle is None else float(best_rate - float(sender_shuffle))
                ),
            }
        )

    best_by_f: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in best_rows:
        best_by_f[str(row["f_value"])].append(row)

    by_receiver_f: Dict[Tuple[int, str], List[Dict[str, object]]] = defaultdict(list)
    for row in best_rows:
        by_receiver_f[(int(row["receiver_seed"]), str(row["f_value"]))].append(row)

    receiver_by_f: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for (receiver_seed, f_value), rows in sorted(by_receiver_f.items()):
        natural_vals = [
            float(row["natural_coop_rate"])
            for row in rows
            if row["natural_coop_rate"] is not None
        ]
        sender_shuffle_vals = [
            float(row["sender_shuffle_coop_rate"])
            for row in rows
            if row["sender_shuffle_coop_rate"] is not None
        ]
        public_random_vals = [
            float(row["public_random_coop_rate"])
            for row in rows
            if row["public_random_coop_rate"] is not None
        ]
        indep_random_vals = [
            float(row["indep_random_coop_rate"])
            for row in rows
            if row["indep_random_coop_rate"] is not None
        ]
        receiver_row = {
            "receiver_seed": receiver_seed,
            "f_value": f_value,
            "n_donors": len(rows),
            "best_coop_rate_mean": _mean(float(row["best_coop_rate"]) for row in rows),
            "identity_coop_rate_mean": _mean(
                float(row["identity_coop_rate"])
                for row in rows
                if row["identity_coop_rate"] == row["identity_coop_rate"]
            ),
            "flipall_coop_rate_mean": _mean(
                float(row["flipall_coop_rate"])
                for row in rows
                if row["flipall_coop_rate"] == row["flipall_coop_rate"]
            ),
            "natural_coop_rate": _mean(natural_vals),
            "sender_shuffle_coop_rate": _mean(sender_shuffle_vals),
            "public_random_coop_rate": _mean(public_random_vals),
            "indep_random_coop_rate": _mean(indep_random_vals),
            "best_minus_natural_mean": _mean(
                float(row["best_minus_natural"])
                for row in rows
                if row["best_minus_natural"] != ""
            ),
            "best_minus_public_random_mean": _mean(
                float(row["best_minus_public_random"])
                for row in rows
                if row["best_minus_public_random"] != ""
            ),
            "best_minus_sender_shuffle_mean": _mean(
                float(row["best_minus_sender_shuffle"])
                for row in rows
                if row["best_minus_sender_shuffle"] != ""
            ),
        }
        receiver_rows.append(receiver_row)
        receiver_by_f[f_value].append(receiver_row)

    for f_value in FOCAL_F_VALUES:
        cur = best_by_f.get(f_value, [])
        if not cur:
            continue
        receiver_cur = receiver_by_f.get(f_value, [])
        identity_vals = [float(row["identity_coop_rate"]) for row in cur if row["identity_coop_rate"] == row["identity_coop_rate"]]
        flip_vals = [float(row["flipall_coop_rate"]) for row in cur if row["flipall_coop_rate"] == row["flipall_coop_rate"]]
        best_vals = [float(row["best_coop_rate"]) for row in cur]
        natural_vals = [float(row["natural_coop_rate"]) for row in cur if row["natural_coop_rate"] is not None]
        sender_shuffle_vals = [float(row["sender_shuffle_coop_rate"]) for row in cur if row["sender_shuffle_coop_rate"] is not None]
        public_random_vals = [float(row["public_random_coop_rate"]) for row in cur if row["public_random_coop_rate"] is not None]
        indep_random_vals = [float(row["indep_random_coop_rate"]) for row in cur if row["indep_random_coop_rate"] is not None]
        summary_rows.extend(
            [
                {"f_value": f_value, "metric": "natural_mean", "value": _mean(natural_vals), "n": len(natural_vals), "sample_unit": "ordered_pairs"},
                {"f_value": f_value, "metric": "sender_shuffle_mean", "value": _mean(sender_shuffle_vals), "n": len(sender_shuffle_vals), "sample_unit": "ordered_pairs"},
                {"f_value": f_value, "metric": "public_random_mean", "value": _mean(public_random_vals), "n": len(public_random_vals), "sample_unit": "ordered_pairs"},
                {"f_value": f_value, "metric": "indep_random_mean", "value": _mean(indep_random_vals), "n": len(indep_random_vals), "sample_unit": "ordered_pairs"},
                {"f_value": f_value, "metric": "foreign_identity_mean", "value": _mean(identity_vals), "n": len(identity_vals), "sample_unit": "ordered_pairs"},
                {"f_value": f_value, "metric": "foreign_flipall_mean", "value": _mean(flip_vals), "n": len(flip_vals), "sample_unit": "ordered_pairs"},
                {"f_value": f_value, "metric": "foreign_best_aligned_mean", "value": _mean(best_vals), "n": len(best_vals), "sample_unit": "ordered_pairs"},
                {
                    "f_value": f_value,
                    "metric": "foreign_best_minus_natural_mean",
                    "value": _mean(
                        float(row["best_minus_natural"])
                        for row in cur
                        if row["best_minus_natural"] != ""
                    ),
                    "n": sum(row["best_minus_natural"] != "" for row in cur),
                    "sample_unit": "ordered_pairs",
                },
                {
                    "f_value": f_value,
                    "metric": "foreign_best_minus_public_random_mean",
                    "value": _mean(
                        float(row["best_minus_public_random"])
                        for row in cur
                        if row["best_minus_public_random"] != ""
                    ),
                    "n": sum(row["best_minus_public_random"] != "" for row in cur),
                    "sample_unit": "ordered_pairs",
                },
                {
                    "f_value": f_value,
                    "metric": "foreign_best_minus_sender_shuffle_mean",
                    "value": _mean(
                        float(row["best_minus_sender_shuffle"])
                        for row in cur
                        if row["best_minus_sender_shuffle"] != ""
                    ),
                    "n": sum(row["best_minus_sender_shuffle"] != "" for row in cur),
                    "sample_unit": "ordered_pairs",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_natural_mean",
                    "value": _mean(float(row["natural_coop_rate"]) for row in receiver_cur),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_foreign_identity_mean",
                    "value": _mean(float(row["identity_coop_rate_mean"]) for row in receiver_cur),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_foreign_best_aligned_mean",
                    "value": _mean(float(row["best_coop_rate_mean"]) for row in receiver_cur),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_best_minus_natural_mean",
                    "value": _mean(
                        float(row["best_minus_natural_mean"]) for row in receiver_cur
                    ),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_best_minus_public_random_mean",
                    "value": _mean(
                        float(row["best_minus_public_random_mean"])
                        for row in receiver_cur
                    ),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_best_minus_sender_shuffle_mean",
                    "value": _mean(
                        float(row["best_minus_sender_shuffle_mean"])
                        for row in receiver_cur
                    ),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_positive_best_minus_natural_count",
                    "value": float(
                        sum(float(row["best_minus_natural_mean"]) > 0.0 for row in receiver_cur)
                    ),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_positive_best_minus_public_random_count",
                    "value": float(
                        sum(
                            float(row["best_minus_public_random_mean"]) > 0.0
                            for row in receiver_cur
                        )
                    ),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_positive_best_minus_sender_shuffle_count",
                    "value": float(
                        sum(
                            float(row["best_minus_sender_shuffle_mean"]) > 0.0
                            for row in receiver_cur
                        )
                    ),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
            ]
        )

    usage_rows = [
        {"alignment_label": label, "count": count}
        for label, count in sorted(best_alignment_counter.items())
    ]

    _write_rows(out_dir / "pairwise_transfer_results.csv", transfer_pair_rows)
    _write_rows(out_dir / "best_alignment_results.csv", best_rows)
    _write_rows(out_dir / "receiver_level_summary.csv", receiver_rows)
    _write_rows(out_dir / "summary_by_f.csv", summary_rows)
    _write_rows(out_dir / "best_alignment_usage.csv", usage_rows)

    metric_lookup = {(row["f_value"], row["metric"]): float(row["value"]) for row in summary_rows}
    n_lookup = {(row["f_value"], row["metric"]): int(row["n"]) for row in summary_rows}
    lines = ["# Cross-Seed Transfer Summary", ""]
    lines.append(
        "Pairwise means below average across ordered receiver-donor pairs; receiver-level robustness is summarized separately across receiver seeds."
    )
    lines.append(
        "Alignment comparisons use matched eval_seed within each receiver-donor pair."
    )
    lines.append("")
    for f_value in FOCAL_F_VALUES:
        if (f_value, "foreign_best_aligned_mean") not in metric_lookup:
            continue
        nat = metric_lookup.get((f_value, "natural_mean"), float("nan")) * 100.0
        ident = metric_lookup.get((f_value, "foreign_identity_mean"), float("nan")) * 100.0
        best = metric_lookup.get((f_value, "foreign_best_aligned_mean"), float("nan")) * 100.0
        pub = metric_lookup.get((f_value, "public_random_mean"), float("nan")) * 100.0
        shuf = metric_lookup.get((f_value, "sender_shuffle_mean"), float("nan")) * 100.0
        d_nat = metric_lookup.get((f_value, "foreign_best_minus_natural_mean"), float("nan")) * 100.0
        d_pub = metric_lookup.get((f_value, "foreign_best_minus_public_random_mean"), float("nan")) * 100.0
        d_shuf = metric_lookup.get((f_value, "foreign_best_minus_sender_shuffle_mean"), float("nan")) * 100.0
        recv_d_nat = metric_lookup.get((f_value, "receiver_best_minus_natural_mean"), float("nan")) * 100.0
        recv_pos_nat = int(
            round(metric_lookup.get((f_value, "receiver_positive_best_minus_natural_count"), float("nan")))
        )
        recv_n = int(n_lookup.get((f_value, "receiver_best_minus_natural_mean"), 0))
        lines.append(f"## f={float(f_value):.1f}")
        lines.append(
            f"- Natural same-seed mean: {nat:.1f}%"
        )
        lines.append(
            f"- Foreign identity mean: {ident:.1f}%"
        )
        lines.append(
            f"- Foreign best-aligned mean: {best:.1f}%"
        )
        lines.append(
            f"- Public random mean: {pub:.1f}%"
        )
        lines.append(
            f"- Sender shuffle mean: {shuf:.1f}%"
        )
        lines.append(
            f"- Best aligned minus natural: {d_nat:+.1f} pp"
        )
        lines.append(
            f"- Best aligned minus public random: {d_pub:+.1f} pp"
        )
        lines.append(
            f"- Best aligned minus sender shuffle: {d_shuf:+.1f} pp"
        )
        lines.append(
            f"- Receiver-level robustness: mean best aligned minus natural {recv_d_nat:+.1f} pp; positive for {recv_pos_nat}/{recv_n} receivers"
        )
        lines.append("")

    if usage_rows:
        lines.append("## Best Alignment Usage")
        for row in usage_rows:
            lines.append(f"- {row['alignment_label']}: {int(row['count'])}")
        lines.append("")

    (out_dir / "cross_seed_transfer_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[xseed-summary] out_dir={out_dir}")


if __name__ == "__main__":
    main()
