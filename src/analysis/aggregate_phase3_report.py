from __future__ import annotations

import argparse
import csv
import os
import re
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


def _receiver_sender_episode_from_row(row: Dict) -> Tuple[int, int]:
    receiver_episode = _as_int(row, "receiver_episode", 0)
    sender_episode = _as_int(row, "sender_episode", 0)
    if receiver_episode > 0 and sender_episode > 0:
        return receiver_episode, sender_episode
    checkpoint = str(row.get("checkpoint", ""))
    cross_play = str(row.get("cross_play", ""))
    m_r = re.search(r"_ep([0-9]+)\.pt$", checkpoint)
    if checkpoint.endswith(".pt") and m_r is None:
        receiver_episode = 200000
    elif m_r is not None:
        receiver_episode = int(m_r.group(1))
    m_s = re.search(r"_ep([0-9]+)\.pt$", cross_play)
    if cross_play.endswith(".pt") and m_s is None:
        sender_episode = 200000
    elif m_s is not None:
        sender_episode = int(m_s.group(1))
    return receiver_episode, sender_episode


def _build_main_lookup(rows: List[Dict]) -> Dict[Tuple, Dict]:
    lookup = {}
    for row in rows:
        lookup[
            (
                row.get("condition"),
                _as_int(row, "train_seed", -1),
                _as_int(row, "checkpoint_episode", 0),
                row.get("ablation", "none"),
                row.get("scope"),
                row.get("key"),
            )
        ] = row
    return lookup


def _intervention_delta_table(main_rows: List[Dict]) -> List[Dict]:
    lookup = _build_main_lookup(main_rows)
    out = []
    interventions = sorted({row.get("ablation", "none") for row in main_rows if row.get("condition") == "cond1"})
    interventions = [x for x in interventions if x != "none"]
    for seed in sorted({_as_int(row, "train_seed", -1) for row in main_rows if row.get("condition") == "cond1"}):
        for episode in sorted({_as_int(row, "checkpoint_episode", 0) for row in main_rows if row.get("condition") == "cond1"}):
            for f_key in ("3.500", "5.000"):
                base = lookup.get(("cond1", seed, episode, "none", "f_value", f_key))
                if base is None:
                    continue
                base_coop = _as_float(base, "coop_rate")
                base_welfare = _as_float(base, "avg_welfare")
                for intervention in interventions:
                    row = lookup.get(("cond1", seed, episode, intervention, "f_value", f_key))
                    if row is None:
                        continue
                    out.append(
                        {
                            "condition": "cond1",
                            "train_seed": int(seed),
                            "checkpoint_episode": int(episode),
                            "f_value": f_key,
                            "intervention": intervention,
                            "coop_none": base_coop,
                            "coop_intervention": _as_float(row, "coop_rate"),
                            "delta_coop_none_minus_intervention": base_coop - _as_float(row, "coop_rate"),
                            "welfare_none": base_welfare,
                            "welfare_intervention": _as_float(row, "avg_welfare"),
                            "delta_welfare_none_minus_intervention": base_welfare - _as_float(row, "avg_welfare"),
                        }
                    )
    return out


def _crossplay_delta_table(rows: List[Dict]) -> List[Dict]:
    matched = {}
    out = []
    for row in rows:
        if row.get("scope") != "f_value":
            continue
        receiver_episode, sender_episode = _receiver_sender_episode_from_row(row)
        key = (
            row.get("condition"),
            _as_int(row, "train_seed", -1),
            receiver_episode,
            row.get("key"),
        )
        if sender_episode == receiver_episode:
            matched[key] = row
    for row in rows:
        if row.get("scope") != "f_value":
            continue
        receiver_episode, sender_episode = _receiver_sender_episode_from_row(row)
        match = matched.get(
            (
                row.get("condition"),
                _as_int(row, "train_seed", -1),
                receiver_episode,
                row.get("key"),
            )
        )
        if match is None:
            continue
        out.append(
            {
                "condition": row.get("condition"),
                "train_seed": _as_int(row, "train_seed", -1),
                "receiver_episode": int(receiver_episode),
                "sender_episode": int(sender_episode),
                "f_value": row.get("key"),
                "matched_coop": _as_float(match, "coop_rate"),
                "crossplay_coop": _as_float(row, "coop_rate"),
                "delta_crossplay_minus_matched": _as_float(row, "coop_rate") - _as_float(match, "coop_rate"),
                "matched_welfare": _as_float(match, "avg_welfare"),
                "crossplay_welfare": _as_float(row, "avg_welfare"),
                "delta_crossplay_welfare_minus_matched": _as_float(row, "avg_welfare") - _as_float(match, "avg_welfare"),
            }
        )
    return out


def _sender_semantics_summary(rows: List[Dict]) -> List[Dict]:
    p_action = {}
    p_fhat = {}
    for row in rows:
        key = (
            row.get("condition"),
            _as_int(row, "train_seed", -1),
            _as_int(row, "checkpoint_episode", 0),
            row.get("sender_id"),
        )
        if row.get("summary") == "p_msg1_given_action":
            p_action[key + (str(row.get("action")),)] = _as_float(row, "p_message_1")
        elif row.get("summary") == "p_msg1_given_fhat":
            p_fhat[key + (row.get("fhat_bin"),)] = _as_float(row, "p_message_1")

    out = []
    keys = {key[:-1] for key in p_action.keys()} | {key[:-1] for key in p_fhat.keys()}
    for key in sorted(keys):
        condition, seed, episode, sender_id = key
        action_delta = p_action.get(key + ("1",), 0.0) - p_action.get(key + ("0",), 0.0)
        high_f = p_fhat.get(key + ("fhat>=4.5",), 0.0)
        low_f = p_fhat.get(key + ("fhat<1.5",), 0.0)
        out.append(
            {
                "condition": condition,
                "train_seed": int(seed),
                "checkpoint_episode": int(episode),
                "sender_id": sender_id,
                "p_msg1_given_action1": p_action.get(key + ("1",), ""),
                "p_msg1_given_action0": p_action.get(key + ("0",), ""),
                "delta_action1_minus_action0": float(action_delta),
                "p_msg1_given_high_fhat": high_f,
                "p_msg1_given_low_fhat": low_f,
                "delta_high_minus_low_fhat": float(high_f - low_f),
            }
        )
    return out


def _receiver_semantics_summary(rows: List[Dict]) -> List[Dict]:
    by_token = {}
    for row in rows:
        if row.get("summary") != "p_coop_given_any_token_fhat":
            continue
        key = (
            row.get("condition"),
            _as_int(row, "train_seed", -1),
            _as_int(row, "checkpoint_episode", 0),
            row.get("fhat_bin"),
            str(row.get("any_token")),
        )
        by_token[key] = _as_float(row, "p_cooperate")
    out = []
    keys = {key[:-1] for key in by_token.keys()}
    for key in sorted(keys):
        condition, seed, episode, fhat_bin = key
        p0 = by_token.get(key + ("0",), 0.0)
        p1 = by_token.get(key + ("1",), 0.0)
        out.append(
            {
                "condition": condition,
                "train_seed": int(seed),
                "checkpoint_episode": int(episode),
                "fhat_bin": fhat_bin,
                "p_coop_any_m0": p0,
                "p_coop_any_m1": p1,
                "delta_m1_minus_m0": float(p1 - p0),
            }
        )
    return out


def _comm_snapshot(comm_rows: List[Dict]) -> List[Dict]:
    out = []
    wanted = {
        ("mi_message_f", "all_senders"),
        ("mi_message_action", "all_senders"),
        ("responsiveness_kl", "all_agents"),
    }
    for row in comm_rows:
        key = (row.get("metric"), row.get("key"))
        if key not in wanted:
            continue
        if row.get("ablation") != "none":
            continue
        out.append(
            {
                "condition": row.get("condition"),
                "train_seed": _as_int(row, "train_seed", -1),
                "checkpoint_episode": _as_int(row, "checkpoint_episode", 0),
                "metric": row.get("metric"),
                "key": row.get("key"),
                "value": _as_float(row, "mi") if row.get("metric", "").startswith("mi_") else _as_float(row, "value"),
                "mi_significant": row.get("mi_significant", ""),
            }
        )
    return out


def _control_summary(control_main_rows: List[Dict], reference_main_rows: List[Dict], reference_episode: int) -> List[Dict]:
    if len(control_main_rows) == 0:
        return []
    ref_lookup = _build_main_lookup(reference_main_rows)
    out = []
    for row in control_main_rows:
        if row.get("scope") != "f_value" or row.get("ablation") != "none":
            continue
        f_key = row.get("key")
        if f_key not in ("3.500", "5.000"):
            continue
        seed = _as_int(row, "train_seed", -1)
        ref = ref_lookup.get(("cond1", seed, reference_episode, "none", "f_value", f_key))
        out.append(
            {
                "train_seed": int(seed),
                "f_value": f_key,
                "control_coop": _as_float(row, "coop_rate"),
                "reference_coop": _as_float(ref, "coop_rate") if ref is not None else "",
                "delta_control_minus_reference": (
                    _as_float(row, "coop_rate") - _as_float(ref, "coop_rate")
                    if ref is not None
                    else ""
                ),
                "control_welfare": _as_float(row, "avg_welfare"),
                "reference_welfare": _as_float(ref, "avg_welfare") if ref is not None else "",
            }
        )
    return out


def _mean(values: List[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / max(1, len(vals))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite_main_csv", type=str, default="outputs/eval/phase3/checkpoint_suite/checkpoint_suite_main.csv")
    p.add_argument("--suite_comm_csv", type=str, default="outputs/eval/phase3/checkpoint_suite/checkpoint_suite_comm.csv")
    p.add_argument("--suite_sender_csv", type=str, default="outputs/eval/phase3/checkpoint_suite/checkpoint_suite_sender_semantics.csv")
    p.add_argument("--suite_receiver_csv", type=str, default="outputs/eval/phase3/checkpoint_suite/checkpoint_suite_receiver_semantics.csv")
    p.add_argument("--suite_posterior_csv", type=str, default="outputs/eval/phase3/checkpoint_suite/checkpoint_suite_posterior_strat.csv")
    p.add_argument("--crossplay_main_csv", type=str, default="outputs/eval/phase3/crossplay_matrix/crossplay_matrix_main.csv")
    p.add_argument("--control_main_csv", type=str, default="")
    p.add_argument("--control_comm_csv", type=str, default="")
    p.add_argument("--control_reference_episode", type=int, default=100000)
    p.add_argument("--out_dir", type=str, default="outputs/eval/phase3/report")
    p.add_argument("--out_md", type=str, default="outputs/eval/phase3/report/PHASE3_SUMMARY.md")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    suite_main_rows = _read_csv_rows(args.suite_main_csv)
    suite_comm_rows = _read_csv_rows(args.suite_comm_csv)
    suite_sender_rows = _read_csv_rows(args.suite_sender_csv)
    suite_receiver_rows = _read_csv_rows(args.suite_receiver_csv)
    suite_posterior_rows = _read_csv_rows(args.suite_posterior_csv)
    crossplay_main_rows = _read_csv_rows(args.crossplay_main_csv)
    control_main_rows = _read_csv_rows(args.control_main_csv)
    _ = _read_csv_rows(args.control_comm_csv)  # reserved for future use

    intervention_rows = _intervention_delta_table(suite_main_rows)
    crossplay_rows = _crossplay_delta_table(crossplay_main_rows)
    sender_summary_rows = _sender_semantics_summary(suite_sender_rows)
    receiver_summary_rows = _receiver_semantics_summary(suite_receiver_rows)
    comm_snapshot_rows = _comm_snapshot(suite_comm_rows)
    control_summary_rows = _control_summary(
        control_main_rows=control_main_rows,
        reference_main_rows=suite_main_rows,
        reference_episode=int(args.control_reference_episode),
    )

    _write_csv(os.path.join(out_dir, "intervention_delta_table.csv"), intervention_rows)
    _write_csv(os.path.join(out_dir, "crossplay_delta_table.csv"), crossplay_rows)
    _write_csv(os.path.join(out_dir, "sender_semantics_summary.csv"), sender_summary_rows)
    _write_csv(os.path.join(out_dir, "receiver_semantics_summary.csv"), receiver_summary_rows)
    _write_csv(os.path.join(out_dir, "comm_snapshot.csv"), comm_snapshot_rows)
    _write_csv(os.path.join(out_dir, "control_summary.csv"), control_summary_rows)

    helpful_rows = [
        row for row in intervention_rows
        if row["f_value"] in ("3.500", "5.000")
        and float(row["delta_coop_none_minus_intervention"]) > 0.05
    ]
    harmful_late_rows = [
        row for row in intervention_rows
        if int(row["checkpoint_episode"]) == 200000
        and row["f_value"] in ("3.500", "5.000")
        and float(row["delta_coop_none_minus_intervention"]) < -0.05
    ]
    final_sender_gain = [
        float(row["delta_crossplay_minus_matched"])
        for row in crossplay_rows
        if int(row["receiver_episode"]) == 200000
        and int(row["sender_episode"]) != 200000
        and row["f_value"] in ("3.500", "5.000")
    ]
    final_receiver_gain = [
        float(row["delta_crossplay_minus_matched"])
        for row in crossplay_rows
        if int(row["sender_episode"]) == 200000
        and int(row["receiver_episode"]) != 200000
        and row["f_value"] in ("3.500", "5.000")
    ]
    drift_call = "inconclusive"
    if len(final_sender_gain) > 0 or len(final_receiver_gain) > 0:
        mean_sender = _mean(final_sender_gain) if len(final_sender_gain) > 0 else 0.0
        mean_receiver = _mean(final_receiver_gain) if len(final_receiver_gain) > 0 else 0.0
        if mean_sender > mean_receiver + 0.02:
            drift_call = "sender-side drift is more likely"
        elif mean_receiver > mean_sender + 0.02:
            drift_call = "receiver-side drift is more likely"
        else:
            drift_call = "joint or mixed drift remains more likely"

    late_sender_semantics = [
        row for row in sender_summary_rows if int(row["checkpoint_episode"]) == 200000
    ]
    late_receiver_semantics = [
        row for row in receiver_summary_rows if int(row["checkpoint_episode"]) == 200000
    ]
    action_link = _mean(
        [abs(float(row["delta_action1_minus_action0"])) for row in late_sender_semantics]
    ) if len(late_sender_semantics) > 0 else 0.0
    regime_link = _mean(
        [abs(float(row["delta_high_minus_low_fhat"])) for row in late_sender_semantics]
    ) if len(late_sender_semantics) > 0 else 0.0
    token_effect = _mean(
        [float(row["delta_m1_minus_m0"]) for row in late_receiver_semantics]
    ) if len(late_receiver_semantics) > 0 else 0.0

    lines = [
        "# Phase 3 Messaging Diagnostics Report",
        "",
        "## Overview",
        f"- checkpoint suite rows: {len(suite_main_rows)}",
        f"- cross-play rows: {len(crossplay_main_rows)}",
        f"- posterior rows: {len(suite_posterior_rows)}",
        "",
        "## Decision Readout",
        f"- messages ever causally helpful online: {'yes' if len(helpful_rows) > 0 else 'no'}",
        f"- late-stage harmful/anti-cooperative interventions detected at 200k: {'yes' if len(harmful_late_rows) > 0 else 'no'}",
        f"- cross-play localization: {drift_call}",
        (
            "- late sender semantics lean more toward action encoding than regime encoding"
            if action_link > regime_link
            else "- late sender semantics lean at least as much toward regime encoding as action encoding"
        ),
        f"- average late token effect on cooperation (m1 - m0 across fhat bins): {token_effect:.3f}",
        "",
        "## Key Files",
        f"- intervention deltas: `{os.path.join(out_dir, 'intervention_delta_table.csv')}`",
        f"- cross-play deltas: `{os.path.join(out_dir, 'crossplay_delta_table.csv')}`",
        f"- sender semantics: `{os.path.join(out_dir, 'sender_semantics_summary.csv')}`",
        f"- receiver semantics: `{os.path.join(out_dir, 'receiver_semantics_summary.csv')}`",
    ]
    if len(control_summary_rows) > 0:
        lines.extend(
            [
                "",
                "## Uniform-Message Control",
                f"- control summary: `{os.path.join(out_dir, 'control_summary.csv')}`",
                f"- reference episode used for comparison: {int(args.control_reference_episode)}",
            ]
        )

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {args.out_md}")


if __name__ == "__main__":
    main()
