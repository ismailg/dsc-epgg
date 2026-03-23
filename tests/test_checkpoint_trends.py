from src.analysis import plot_phase3_checkpoint_trends as pct


def test_aggregate_per_seed_weights_eval_rows_before_seed_summary():
    rows = [
        {
            "condition": "cond1",
            "checkpoint_episode": "50000",
            "train_seed": "101",
            "key": "3.5",
            "n_rounds": "10",
            "coop_rate": "0.8",
            "avg_reward": "11.0",
            "avg_welfare": "44.0",
        },
        {
            "condition": "cond1",
            "checkpoint_episode": "50000",
            "train_seed": "101",
            "key": "3.5",
            "n_rounds": "30",
            "coop_rate": "0.4",
            "avg_reward": "7.0",
            "avg_welfare": "28.0",
        },
        {
            "condition": "cond1",
            "checkpoint_episode": "50000",
            "train_seed": "202",
            "key": "3.5",
            "n_rounds": "20",
            "coop_rate": "0.6",
            "avg_reward": "9.0",
            "avg_welfare": "36.0",
        },
    ]

    seed_rows = pct._aggregate_per_seed(rows)
    assert len(seed_rows) == 2
    first = next(row for row in seed_rows if int(row["train_seed"]) == 101)
    assert abs(first["coop_rate"] - 0.5) < 1e-12
    assert abs(first["avg_reward"] - 8.0) < 1e-12
    assert abs(first["avg_welfare"] - 32.0) < 1e-12

    summary_rows = pct._summarize_seed_rows(seed_rows)
    assert len(summary_rows) == 1
    summary = summary_rows[0]
    assert abs(summary["mean_coop_rate"] - 0.55) < 1e-12
    assert abs(summary["mean_avg_reward"] - 8.5) < 1e-12
    assert abs(summary["mean_avg_welfare"] - 34.0) < 1e-12


def test_paired_delta_summary_counts_negative_changes():
    seed_rows = [
        {
            "condition": "cond1",
            "checkpoint_episode": 50000,
            "train_seed": 101,
            "f_value": 3.5,
            "coop_rate": 0.8,
            "avg_reward": 10.0,
            "avg_welfare": 40.0,
            "n_rounds": 100.0,
        },
        {
            "condition": "cond1",
            "checkpoint_episode": 150000,
            "train_seed": 101,
            "f_value": 3.5,
            "coop_rate": 0.5,
            "avg_reward": 8.0,
            "avg_welfare": 32.0,
            "n_rounds": 100.0,
        },
        {
            "condition": "cond1",
            "checkpoint_episode": 50000,
            "train_seed": 202,
            "f_value": 3.5,
            "coop_rate": 0.6,
            "avg_reward": 9.0,
            "avg_welfare": 36.0,
            "n_rounds": 100.0,
        },
        {
            "condition": "cond1",
            "checkpoint_episode": 150000,
            "train_seed": 202,
            "f_value": 3.5,
            "coop_rate": 0.7,
            "avg_reward": 9.5,
            "avg_welfare": 38.0,
            "n_rounds": 100.0,
        },
    ]

    delta_rows = pct._paired_delta_rows(seed_rows)
    assert len(delta_rows) == 2
    summary_rows = pct._summarize_paired_deltas(delta_rows)
    assert len(summary_rows) == 1
    summary = summary_rows[0]
    assert abs(summary["mean_delta_coop_150k_minus_50k"] - (-0.1)) < 1e-12
    assert abs(summary["mean_delta_avg_reward_150k_minus_50k"] - (-0.75)) < 1e-12
    assert abs(summary["mean_delta_avg_welfare_150k_minus_50k"] - (-3.0)) < 1e-12
    assert summary["n_negative_coop_deltas"] == 1
    assert summary["n_negative_reward_deltas"] == 1
    assert summary["n_negative_welfare_deltas"] == 1
