from src.analysis.plot_phase3_fragmentation import _fragmentation_over_time
from src.analysis.run_phase3_common_polarity_rescue import _build_flip_maps


def test_build_flip_maps_uses_negative_polarity_as_flip():
    rows = [
        {
            "condition": "cond1",
            "train_seed": 101,
            "checkpoint_episode": 50000,
            "eval_policy": "greedy",
            "ablation": "none",
            "sender_remap": "none",
            "cross_play": "none",
            "sender_id": "agent_0",
            "delta_high_minus_low_fhat": -0.4,
            "delta_action1_minus_action0": 0.2,
        },
        {
            "condition": "cond1",
            "train_seed": 101,
            "checkpoint_episode": 50000,
            "eval_policy": "greedy",
            "ablation": "none",
            "sender_remap": "none",
            "cross_play": "none",
            "sender_id": "agent_1",
            "delta_high_minus_low_fhat": 0.3,
            "delta_action1_minus_action0": -0.1,
        },
    ]
    flip_maps = _build_flip_maps(rows, basis="regime")
    assert flip_maps[("cond1", 101, 50000)] == {"agent_0": 1, "agent_1": 0}


def test_fragmentation_over_time_compares_aggregate_and_sender_specific_effects():
    sender_alignment_rows = [
        {
            "condition": "cond1",
            "train_seed": 101,
            "checkpoint_episode": 200000,
            "eval_policy": "greedy",
            "ablation": "none",
            "cross_play": "none",
            "alignment_action_sign": -0.33,
            "alignment_regime_sign": -0.66,
            "mean_abs_action_delta": 0.2,
            "mean_abs_regime_delta": 0.4,
        }
    ]
    receiver_summary_rows = [
        {
            "condition": "cond1",
            "train_seed": 101,
            "checkpoint_episode": 200000,
            "eval_policy": "greedy",
            "ablation": "none",
            "cross_play": "none",
            "fhat_bin": "fhat<1.5",
            "delta_m1_minus_m0": 0.05,
        },
        {
            "condition": "cond1",
            "train_seed": 101,
            "checkpoint_episode": 200000,
            "eval_policy": "greedy",
            "ablation": "none",
            "cross_play": "none",
            "fhat_bin": "fhat>=4.5",
            "delta_m1_minus_m0": -0.05,
        },
    ]
    receiver_by_sender_rows = [
        {
            "condition": "cond1",
            "train_seed": 101,
            "checkpoint_episode": 200000,
            "eval_policy": "greedy",
            "ablation": "none",
            "cross_play": "none",
            "receiver_id": "all_agents",
            "sender_id": "agent_0",
            "delta_m1_minus_m0": 0.4,
        },
        {
            "condition": "cond1",
            "train_seed": 101,
            "checkpoint_episode": 200000,
            "eval_policy": "greedy",
            "ablation": "none",
            "cross_play": "none",
            "receiver_id": "all_agents",
            "sender_id": "agent_1",
            "delta_m1_minus_m0": -0.4,
        },
    ]
    out = _fragmentation_over_time(
        sender_alignment_rows=sender_alignment_rows,
        receiver_summary_rows=receiver_summary_rows,
        receiver_by_sender_rows=receiver_by_sender_rows,
    )
    assert len(out) == 1
    row = out[0]
    assert row["aggregate_token_effect_mean"] == 0.0
    assert row["aggregate_token_effect_abs_mean"] == 0.05
    assert row["sender_specific_effect_abs_mean"] == 0.4
    assert row["alignment_regime_sign"] == -0.66
