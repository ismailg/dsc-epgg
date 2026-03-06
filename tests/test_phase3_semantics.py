import csv
from pathlib import Path

from src.analysis.evaluate_regime_conditional import (
    _build_received_pattern,
    _derive_receiver_semantics_rows,
    _derive_sender_semantics_rows,
    _write_trace_csv,
)


def test_build_received_pattern_excludes_self_and_is_stable():
    pattern, any_m0, any_m1 = _build_received_pattern(
        agent_id="agent_0",
        sender_ids=["agent_0", "agent_1", "agent_2"],
        delivered_messages={"agent_0": 1, "agent_1": 0, "agent_2": 1},
    )
    assert pattern == "agent_1:0|agent_2:1"
    assert any_m0 == 1
    assert any_m1 == 1


def test_trace_writer_has_stable_sender_columns(tmp_path: Path):
    path = tmp_path / "trace.csv"
    rows = [
        {
            "checkpoint": "ckpt.pt",
            "condition": "cond1",
            "train_seed": 1,
            "eval_seed": 2,
            "eval_policy": "greedy",
            "ablation": "none",
            "cross_play": "none",
            "episode": 0,
            "t": 0,
            "agent_id": "agent_0",
            "true_f": 5.0,
            "f_hat": 4.8,
            "action": 1,
            "reward": 2.0,
            "round_welfare": 8.0,
            "own_sent_msg": 1,
            "delivered_msg_agent_0": 1,
            "delivered_msg_agent_1": 0,
            "recv_any_m0": 1,
            "recv_any_m1": 0,
            "recv_pattern": "agent_1:0",
        }
    ]
    _write_trace_csv(str(path), rows)
    with open(path, "r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert "delivered_msg_agent_0" in header
    assert "delivered_msg_agent_1" in header
    assert header.index("delivered_msg_agent_0") < header.index("recv_any_m0")
    assert header.index("delivered_msg_agent_1") < header.index("recv_any_m0")


def test_sender_and_receiver_semantics_recover_known_patterns():
    trace_rows = [
        {
            "checkpoint": "cond1_seed1_ep50000.pt",
            "condition": "cond1",
            "train_seed": 1,
            "eval_seed": 10,
            "eval_policy": "greedy",
            "ablation": "none",
            "cross_play": "none",
            "agent_id": "agent_0",
            "f_hat": 4.8,
            "action": 1,
            "own_sent_msg": 1,
            "delivered_msg_agent_0": 1,
            "delivered_msg_agent_1": 1,
            "delivered_msg_agent_2": 1,
            "recv_any_m0": 0,
            "recv_any_m1": 1,
            "recv_pattern": "agent_1:1|agent_2:1",
        },
        {
            "checkpoint": "cond1_seed1_ep50000.pt",
            "condition": "cond1",
            "train_seed": 1,
            "eval_seed": 10,
            "eval_policy": "greedy",
            "ablation": "none",
            "cross_play": "none",
            "agent_id": "agent_0",
            "f_hat": 0.4,
            "action": 0,
            "own_sent_msg": 0,
            "delivered_msg_agent_0": 0,
            "delivered_msg_agent_1": 0,
            "delivered_msg_agent_2": 0,
            "recv_any_m0": 1,
            "recv_any_m1": 0,
            "recv_pattern": "agent_1:0|agent_2:0",
        },
        {
            "checkpoint": "cond1_seed1_ep50000.pt",
            "condition": "cond1",
            "train_seed": 1,
            "eval_seed": 10,
            "eval_policy": "greedy",
            "ablation": "none",
            "cross_play": "none",
            "agent_id": "agent_2",
            "f_hat": 4.9,
            "action": 1,
            "own_sent_msg": "",
            "delivered_msg_agent_0": 1,
            "delivered_msg_agent_1": 1,
            "delivered_msg_agent_2": 1,
            "recv_any_m0": 0,
            "recv_any_m1": 1,
            "recv_pattern": "agent_0:1|agent_1:1",
        },
        {
            "checkpoint": "cond1_seed1_ep50000.pt",
            "condition": "cond1",
            "train_seed": 1,
            "eval_seed": 10,
            "eval_policy": "greedy",
            "ablation": "none",
            "cross_play": "none",
            "agent_id": "agent_2",
            "f_hat": 0.8,
            "action": 0,
            "own_sent_msg": "",
            "delivered_msg_agent_0": 0,
            "delivered_msg_agent_1": 0,
            "delivered_msg_agent_2": 0,
            "recv_any_m0": 1,
            "recv_any_m1": 0,
            "recv_pattern": "agent_0:0|agent_1:0",
        },
    ]
    sender_rows = _derive_sender_semantics_rows(trace_rows)
    receiver_rows = _derive_receiver_semantics_rows(trace_rows)

    by_sender = {
        (row["summary"], row["fhat_bin"], str(row["action"])): row for row in sender_rows
    }
    assert by_sender[("p_msg1_given_fhat", "fhat>=4.5", "")]["p_message_1"] == 1.0
    assert by_sender[("p_msg1_given_fhat", "fhat<1.5", "")]["p_message_1"] == 0.0
    assert by_sender[("p_msg1_given_action", "", "1")]["p_message_1"] == 1.0
    assert by_sender[("p_msg1_given_action", "", "0")]["p_message_1"] == 0.0

    by_receiver = {
        (row["summary"], row["recv_pattern"], str(row["any_token"]), row["fhat_bin"]): row
        for row in receiver_rows
    }
    assert (
        by_receiver[("p_coop_given_recv_pattern_fhat", "agent_1:1|agent_2:1", "", "fhat>=4.5")]["p_cooperate"]
        == 1.0
    )
    assert (
        by_receiver[("p_coop_given_any_token_fhat", "", "0", "fhat<1.5")]["p_cooperate"]
        == 0.0
    )
    sender_specific = {
        (
            row["summary"],
            row["receiver_id"],
            row["sender_id"],
            str(row["sender_token"]),
            row["fhat_bin"],
        ): row
        for row in receiver_rows
        if row["summary"] == "p_coop_given_sender_token_fhat"
    }
    assert (
        sender_specific[
            ("p_coop_given_sender_token_fhat", "agent_2", "agent_0", "1", "fhat>=4.5")
        ]["p_cooperate"]
        == 1.0
    )
    assert (
        sender_specific[
            ("p_coop_given_sender_token_fhat", "agent_2", "agent_0", "0", "fhat<1.5")
        ]["p_cooperate"]
        == 0.0
    )
