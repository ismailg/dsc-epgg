from src.analysis.aggregate_phase3_report import (
    _receiver_by_sender_summary,
    _sender_alignment_summary,
)


def test_sender_alignment_summary_detects_fragmented_codes():
    sender_rows = [
        {
            "condition": "cond1",
            "train_seed": 101,
            "checkpoint_episode": 200000,
            "eval_policy": "greedy",
            "ablation": "none",
            "cross_play": "none",
            "sender_id": "agent_0",
            "delta_action1_minus_action0": 0.4,
            "delta_high_minus_low_fhat": 0.6,
        },
        {
            "condition": "cond1",
            "train_seed": 101,
            "checkpoint_episode": 200000,
            "eval_policy": "greedy",
            "ablation": "none",
            "cross_play": "none",
            "sender_id": "agent_1",
            "delta_action1_minus_action0": -0.2,
            "delta_high_minus_low_fhat": -0.5,
        },
    ]
    out = _sender_alignment_summary(sender_rows)
    assert len(out) == 1
    row = out[0]
    assert row["alignment_action_sign"] == -1.0
    assert row["alignment_regime_sign"] == -1.0
    assert row["n_action_positive"] == 1
    assert row["n_action_negative"] == 1


def test_receiver_by_sender_summary_preserves_sender_identity():
    trace_rows = [
        {
            "condition": "cond1",
            "train_seed": 101,
            "checkpoint_episode": 50000,
            "eval_policy": "greedy",
            "ablation": "none",
            "cross_play": "none",
            "agent_id": "agent_2",
            "action": 1,
            "f_hat": 4.8,
            "delivered_msg_agent_0": 1,
            "delivered_msg_agent_1": 0,
            "delivered_msg_agent_2": 1,
        },
        {
            "condition": "cond1",
            "train_seed": 101,
            "checkpoint_episode": 50000,
            "eval_policy": "greedy",
            "ablation": "none",
            "cross_play": "none",
            "agent_id": "agent_2",
            "action": 0,
            "f_hat": 0.7,
            "delivered_msg_agent_0": 0,
            "delivered_msg_agent_1": 1,
            "delivered_msg_agent_2": 0,
        },
    ]
    out = _receiver_by_sender_summary(trace_rows)
    lookup = {
        (row["receiver_id"], row["sender_id"], row["fhat_bin"]): row
        for row in out
        if row["receiver_id"] == "agent_2"
    }
    high = lookup[("agent_2", "agent_0", "fhat>=4.5")]
    low = lookup[("agent_2", "agent_0", "fhat<1.5")]
    assert high["p_coop_sender_m1"] == 1.0
    assert low["p_coop_sender_m0"] == 0.0
