from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from src.analysis.condition_labels import condition_alias, condition_display


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
                            "condition_alias": condition_alias("cond1"),
                            "condition_display": condition_display("cond1"),
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
                "condition_alias": condition_alias(row.get("condition")),
                "condition_display": condition_display(row.get("condition")),
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
            row.get("eval_policy", "greedy"),
            row.get("ablation", "none"),
            row.get("cross_play", "none"),
            row.get("sender_id"),
        )
        if row.get("summary") == "p_msg1_given_action":
            p_action[key + (str(row.get("action")),)] = _as_float(row, "p_message_1")
        elif row.get("summary") == "p_msg1_given_fhat":
            p_fhat[key + (row.get("fhat_bin"),)] = _as_float(row, "p_message_1")

    out = []
    keys = {key[:-1] for key in p_action.keys()} | {key[:-1] for key in p_fhat.keys()}
    for key in sorted(keys):
        condition, seed, episode, eval_policy, ablation, cross_play, sender_id = key
        action_delta = p_action.get(key + ("1",), 0.0) - p_action.get(key + ("0",), 0.0)
        high_f = p_fhat.get(key + ("fhat>=4.5",), 0.0)
        low_f = p_fhat.get(key + ("fhat<1.5",), 0.0)
        out.append(
            {
                "condition": condition,
                "condition_alias": condition_alias(condition),
                "condition_display": condition_display(condition),
                "train_seed": int(seed),
                "checkpoint_episode": int(episode),
                "eval_policy": eval_policy,
                "ablation": ablation,
                "cross_play": cross_play,
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
            row.get("eval_policy", "greedy"),
            row.get("ablation", "none"),
            row.get("cross_play", "none"),
            row.get("fhat_bin"),
            str(row.get("any_token")),
        )
        by_token[key] = _as_float(row, "p_cooperate")
    out = []
    keys = {key[:-1] for key in by_token.keys()}
    for key in sorted(keys):
        condition, seed, episode, eval_policy, ablation, cross_play, fhat_bin = key
        p0 = by_token.get(key + ("0",), 0.0)
        p1 = by_token.get(key + ("1",), 0.0)
        out.append(
            {
                "condition": condition,
                "condition_alias": condition_alias(condition),
                "condition_display": condition_display(condition),
                "train_seed": int(seed),
                "checkpoint_episode": int(episode),
                "eval_policy": eval_policy,
                "ablation": ablation,
                "cross_play": cross_play,
                "fhat_bin": fhat_bin,
                "p_coop_any_m0": p0,
                "p_coop_any_m1": p1,
                "delta_m1_minus_m0": float(p1 - p0),
            }
        )
    return out


def _sign_with_eps(value: float, eps: float = 1e-8) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _pairwise_alignment(signs: List[int]) -> float:
    vals = [int(v) for v in signs]
    if len(vals) < 2:
        return 0.0
    pair_scores = []
    for idx in range(len(vals)):
        for jdx in range(idx + 1, len(vals)):
            pair_scores.append(float(vals[idx] * vals[jdx]))
    return _mean(pair_scores) if len(pair_scores) > 0 else 0.0


def _sender_alignment_summary(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("condition"),
                _as_int(row, "train_seed", -1),
                _as_int(row, "checkpoint_episode", 0),
                row.get("eval_policy", "greedy"),
                row.get("ablation", "none"),
                row.get("cross_play", "none"),
            )
        ].append(row)

    out = []
    for key, grouped_rows in sorted(grouped.items()):
        condition, seed, episode, eval_policy, ablation, cross_play = key
        action_deltas = [float(row["delta_action1_minus_action0"]) for row in grouped_rows]
        regime_deltas = [float(row["delta_high_minus_low_fhat"]) for row in grouped_rows]
        action_signs = [_sign_with_eps(val) for val in action_deltas]
        regime_signs = [_sign_with_eps(val) for val in regime_deltas]
        out.append(
            {
                "condition": condition,
                "condition_alias": condition_alias(condition),
                "condition_display": condition_display(condition),
                "train_seed": int(seed),
                "checkpoint_episode": int(episode),
                "eval_policy": eval_policy,
                "ablation": ablation,
                "cross_play": cross_play,
                "n_senders": int(len(grouped_rows)),
                "alignment_action_sign": float(_pairwise_alignment(action_signs)),
                "alignment_regime_sign": float(_pairwise_alignment(regime_signs)),
                "mean_abs_action_delta": float(_mean([abs(v) for v in action_deltas])),
                "mean_abs_regime_delta": float(_mean([abs(v) for v in regime_deltas])),
                "n_action_positive": int(sum(1 for sign in action_signs if sign > 0)),
                "n_action_negative": int(sum(1 for sign in action_signs if sign < 0)),
                "n_regime_positive": int(sum(1 for sign in regime_signs if sign > 0)),
                "n_regime_negative": int(sum(1 for sign in regime_signs if sign < 0)),
            }
        )
    return out


def _receiver_by_sender_summary(trace_rows: List[Dict]) -> List[Dict]:
    if len(trace_rows) == 0:
        return []
    delivered_cols = sorted(
        {key for row in trace_rows for key in row.keys() if key.startswith("delivered_msg_")}
    )
    if len(delivered_cols) == 0:
        return []
    by_sender = defaultdict(lambda: {"n_obs": 0, "coop_sum": 0.0})
    for row in trace_rows:
        receiver_id = str(row.get("agent_id", ""))
        if receiver_id == "":
            continue
        condition = row.get("condition")
        seed = _as_int(row, "train_seed", -1)
        episode = _as_int(row, "checkpoint_episode", 0)
        eval_policy = row.get("eval_policy", "greedy")
        ablation = row.get("ablation", "none")
        cross_play = row.get("cross_play", "none")
        fhat_bin = row.get("fhat_bin")
        if not fhat_bin:
            f_hat = _as_float(row, "f_hat")
            if f_hat < 1.5:
                fhat_bin = "fhat<1.5"
            elif f_hat < 2.5:
                fhat_bin = "1.5<=fhat<2.5"
            elif f_hat < 3.5:
                fhat_bin = "2.5<=fhat<3.5"
            elif f_hat < 4.5:
                fhat_bin = "3.5<=fhat<4.5"
            else:
                fhat_bin = "fhat>=4.5"
        coop = float(_as_int(row, "action", 0))
        for delivered_col in delivered_cols:
            sender_id = str(delivered_col).replace("delivered_msg_", "", 1)
            if sender_id == receiver_id:
                continue
            raw_token = row.get(delivered_col, "")
            if raw_token in ("", None):
                continue
            token = int(float(raw_token))
            for receiver_group in (receiver_id, "all_agents"):
                by_sender[
                    (
                        condition,
                        seed,
                        episode,
                        eval_policy,
                        ablation,
                        cross_play,
                        receiver_group,
                        sender_id,
                        fhat_bin,
                        token,
                    )
                ]["n_obs"] += 1
                by_sender[
                    (
                        condition,
                        seed,
                        episode,
                        eval_policy,
                        ablation,
                        cross_play,
                        receiver_group,
                        sender_id,
                        fhat_bin,
                        token,
                    )
                ]["coop_sum"] += coop

    out = []
    grouped_keys = {key[:-1] for key in by_sender.keys()}
    for key in sorted(grouped_keys):
        (
            condition,
            seed,
            episode,
            eval_policy,
            ablation,
            cross_play,
            receiver_id,
            sender_id,
            fhat_bin,
        ) = key
        acc0 = by_sender.get(key + (0,), {"n_obs": 0, "coop_sum": 0.0})
        acc1 = by_sender.get(key + (1,), {"n_obs": 0, "coop_sum": 0.0})
        p0 = float(acc0["coop_sum"] / max(1, int(acc0["n_obs"])))
        p1 = float(acc1["coop_sum"] / max(1, int(acc1["n_obs"])))
        out.append(
            {
                "condition": condition,
                "condition_alias": condition_alias(condition),
                "condition_display": condition_display(condition),
                "train_seed": int(seed),
                "checkpoint_episode": int(episode),
                "eval_policy": eval_policy,
                "ablation": ablation,
                "cross_play": cross_play,
                "receiver_id": receiver_id,
                "sender_id": sender_id,
                "fhat_bin": fhat_bin,
                "n_obs_sender_m0": int(acc0["n_obs"]),
                "n_obs_sender_m1": int(acc1["n_obs"]),
                "p_coop_sender_m0": p0,
                "p_coop_sender_m1": p1,
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
                "condition_alias": condition_alias(row.get("condition")),
                "condition_display": condition_display(row.get("condition")),
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
                "condition": "cond1",
                "condition_alias": condition_alias("cond1"),
                "condition_display": condition_display("cond1"),
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
    p.add_argument("--suite_trace_csv", type=str, default="outputs/eval/phase3/checkpoint_suite/checkpoint_suite_trace.csv")
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
    suite_trace_rows = _read_csv_rows(args.suite_trace_csv)
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
    alignment_rows = _sender_alignment_summary(sender_summary_rows)
    receiver_by_sender_rows = _receiver_by_sender_summary(suite_trace_rows)
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
    _write_csv(os.path.join(out_dir, "sender_alignment_summary.csv"), alignment_rows)
    _write_csv(os.path.join(out_dir, "receiver_by_sender_summary.csv"), receiver_by_sender_rows)
    _write_csv(os.path.join(out_dir, "comm_snapshot.csv"), comm_snapshot_rows)
    _write_csv(os.path.join(out_dir, "control_summary.csv"), control_summary_rows)

    checkpoint_episodes = sorted(
        {
            int(row["checkpoint_episode"])
            for row in intervention_rows + sender_summary_rows + receiver_summary_rows + alignment_rows + receiver_by_sender_rows
            if str(row.get("checkpoint_episode", "")).strip() not in ("", "0")
        }
    )
    latest_episode = checkpoint_episodes[-1] if len(checkpoint_episodes) > 0 else 0

    helpful_rows = [
        row for row in intervention_rows
        if row["f_value"] in ("3.500", "5.000")
        and float(row["delta_coop_none_minus_intervention"]) > 0.05
    ]
    harmful_late_rows = [
        row for row in intervention_rows
        if latest_episode > 0
        and int(row["checkpoint_episode"]) == latest_episode
        and row["f_value"] in ("3.500", "5.000")
        and float(row["delta_coop_none_minus_intervention"]) < -0.05
    ]
    final_sender_gain = [
        float(row["delta_crossplay_minus_matched"])
        for row in crossplay_rows
        if latest_episode > 0
        and int(row["receiver_episode"]) == latest_episode
        and int(row["sender_episode"]) != latest_episode
        and row["f_value"] in ("3.500", "5.000")
    ]
    final_receiver_gain = [
        float(row["delta_crossplay_minus_matched"])
        for row in crossplay_rows
        if latest_episode > 0
        and int(row["sender_episode"]) == latest_episode
        and int(row["receiver_episode"]) != latest_episode
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
        row
        for row in sender_summary_rows
        if latest_episode > 0
        and int(row["checkpoint_episode"]) == latest_episode
        and row.get("ablation") == "none"
        and row.get("cross_play") == "none"
    ]
    late_receiver_semantics = [
        row
        for row in receiver_summary_rows
        if latest_episode > 0
        and int(row["checkpoint_episode"]) == latest_episode
        and row.get("ablation") == "none"
        and row.get("cross_play") == "none"
    ]
    late_alignment_rows = [
        row
        for row in alignment_rows
        if latest_episode > 0
        and int(row["checkpoint_episode"]) == latest_episode
        and row.get("ablation") == "none"
        and row.get("cross_play") == "none"
    ]
    late_receiver_by_sender = [
        row
        for row in receiver_by_sender_rows
        if latest_episode > 0
        and int(row["checkpoint_episode"]) == latest_episode
        and row.get("receiver_id") == "all_agents"
        and row.get("ablation") == "none"
        and row.get("cross_play") == "none"
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
    sender_specific_token_effect = _mean(
        [abs(float(row["delta_m1_minus_m0"])) for row in late_receiver_by_sender]
    ) if len(late_receiver_by_sender) > 0 else 0.0
    action_alignment = _mean(
        [float(row["alignment_action_sign"]) for row in late_alignment_rows]
    ) if len(late_alignment_rows) > 0 else 0.0
    regime_alignment = _mean(
        [float(row["alignment_regime_sign"]) for row in late_alignment_rows]
    ) if len(late_alignment_rows) > 0 else 0.0

    lines = [
        "# Phase 3 Messaging Diagnostics Report",
        "",
        "## Overview",
        f"- condition aliases: cond1 -> {condition_alias('cond1')}, cond2 -> {condition_alias('cond2')}",
        f"- latest checkpoint episode in this report: {latest_episode}",
        f"- checkpoint suite rows: {len(suite_main_rows)}",
        f"- checkpoint suite trace rows: {len(suite_trace_rows)}",
        f"- cross-play rows: {len(crossplay_main_rows)}",
        f"- posterior rows: {len(suite_posterior_rows)}",
        "",
        "## Decision Readout",
        f"- messages ever causally helpful online: {'yes' if len(helpful_rows) > 0 else 'no'}",
        f"- harmful/anti-cooperative interventions detected at the latest checkpoint ({latest_episode}): {'yes' if len(harmful_late_rows) > 0 else 'no'}",
        f"- cross-play localization: {drift_call}",
        (
            f"- sender semantics at the latest checkpoint ({latest_episode}) lean more toward action encoding than regime encoding"
            if action_link > regime_link
            else f"- sender semantics at the latest checkpoint ({latest_episode}) lean at least as much toward regime encoding as action encoding"
        ),
        f"- average token effect on cooperation at the latest checkpoint (m1 - m0 across fhat bins): {token_effect:.3f}",
        f"- average sender-specific token effect magnitude at the latest checkpoint: {sender_specific_token_effect:.3f}",
        f"- action-polarity alignment across senders at the latest checkpoint: {action_alignment:.3f}",
        f"- regime-polarity alignment across senders at the latest checkpoint: {regime_alignment:.3f}",
        "",
        "## Key Files",
        f"- intervention deltas: `{os.path.join(out_dir, 'intervention_delta_table.csv')}`",
        f"- cross-play deltas: `{os.path.join(out_dir, 'crossplay_delta_table.csv')}`",
        f"- sender semantics: `{os.path.join(out_dir, 'sender_semantics_summary.csv')}`",
        f"- receiver semantics: `{os.path.join(out_dir, 'receiver_semantics_summary.csv')}`",
        f"- sender alignment: `{os.path.join(out_dir, 'sender_alignment_summary.csv')}`",
        f"- receiver-by-sender: `{os.path.join(out_dir, 'receiver_by_sender_summary.csv')}`",
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
