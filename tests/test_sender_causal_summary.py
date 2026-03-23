from src.analysis import summarize_phase3_sender_causal as sc


def test_summarize_overall_emits_self_and_nonself_scopes():
    rows = [
        {
            "receiver_is_sender": "1",
            "true_f": "3.5",
            "delta_p_cooperate_1_minus_0": "0.06",
            "action_flip_rate_force0_vs_natural": "0.10",
            "action_flip_rate_force1_vs_natural": "0.12",
        },
        {
            "receiver_is_sender": "0",
            "true_f": "3.5",
            "delta_p_cooperate_1_minus_0": "0.04",
            "action_flip_rate_force0_vs_natural": "0.08",
            "action_flip_rate_force1_vs_natural": "0.09",
        },
        {
            "receiver_is_sender": "0",
            "true_f": "3.5",
            "delta_p_cooperate_1_minus_0": "-0.02",
            "action_flip_rate_force0_vs_natural": "0.06",
            "action_flip_rate_force1_vs_natural": "0.07",
        },
    ]

    overall_all = sc._summarize_overall(rows, scope="all")
    overall_self = sc._summarize_overall(rows, scope="self")
    overall_nonself = sc._summarize_overall(rows, scope="nonself")

    assert overall_all[0]["summary"] == "overall_all"
    assert overall_self[0]["summary"] == "overall_self"
    assert overall_nonself[0]["summary"] == "overall_nonself"
    assert overall_self[0]["n_rows"] == 1
    assert overall_nonself[0]["n_rows"] == 2
    assert abs(overall_self[0]["mean_abs_delta_p_cooperate"] - 0.06) < 1e-12
    assert abs(overall_nonself[0]["mean_abs_delta_p_cooperate"] - 0.03) < 1e-12


def test_row_matches_filters_sender_causal_rows():
    row = {
        "condition": "cond1",
        "ablation": "none",
        "checkpoint_episode": "150000",
        "cross_play": "none",
        "sender_remap": "none",
    }

    assert sc._row_matches(
        row,
        condition="cond1",
        ablation="none",
        checkpoint_episode=150000,
        cross_play="none",
        sender_remap="none",
    )
    assert not sc._row_matches(
        row,
        condition="cond2",
        ablation="none",
        checkpoint_episode=150000,
        cross_play="none",
        sender_remap="none",
    )
