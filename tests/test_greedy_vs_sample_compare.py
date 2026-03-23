from src.analysis import plot_phase3_greedy_vs_sample as gvs


def test_paired_policy_deltas_align_by_condition_checkpoint_seed_and_f():
    greedy_seed_rows = [
        {
            "condition": "cond1",
            "checkpoint_episode": 50000,
            "train_seed": 101,
            "f_value": 3.5,
            "coop_rate": 0.8,
            "avg_reward": 10.0,
            "avg_welfare": 40.0,
        },
        {
            "condition": "cond1",
            "checkpoint_episode": 150000,
            "train_seed": 101,
            "f_value": 3.5,
            "coop_rate": 0.5,
            "avg_reward": 8.0,
            "avg_welfare": 32.0,
        },
    ]
    sample_seed_rows = [
        {
            "condition": "cond1",
            "checkpoint_episode": 50000,
            "train_seed": 101,
            "f_value": 3.5,
            "coop_rate": 0.6,
            "avg_reward": 9.0,
            "avg_welfare": 36.0,
        },
        {
            "condition": "cond1",
            "checkpoint_episode": 150000,
            "train_seed": 101,
            "f_value": 3.5,
            "coop_rate": 0.55,
            "avg_reward": 8.5,
            "avg_welfare": 34.0,
        },
    ]

    delta_rows = gvs._paired_policy_delta_rows(greedy_seed_rows, sample_seed_rows)
    assert len(delta_rows) == 2
    assert abs(delta_rows[0]["delta_coop_greedy_minus_sample"] - 0.2) < 1e-12
    assert abs(delta_rows[1]["delta_avg_welfare_greedy_minus_sample"] - (-2.0)) < 1e-12

    summary_rows = gvs._summarize_policy_deltas(delta_rows)
    assert len(summary_rows) == 2
    first = next(row for row in summary_rows if row["checkpoint_episode"] == 50000)
    assert abs(first["mean_delta_avg_reward_greedy_minus_sample"] - 1.0) < 1e-12
