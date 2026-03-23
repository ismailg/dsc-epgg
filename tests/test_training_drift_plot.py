import math

from src.analysis.plot_phase3_training_drift import _summarize_rows


def test_summarize_rows_mean_and_sem():
    rows = [
        {
            "condition": "cond1",
            "seed": 101,
            "episode": 1000,
            "scope": "overall",
            "coop_rate": 0.2,
            "avg_reward": 5.0,
            "avg_welfare": 20.0,
            "n_rounds": 100,
        },
        {
            "condition": "cond1",
            "seed": 202,
            "episode": 1000,
            "scope": "overall",
            "coop_rate": 0.4,
            "avg_reward": 7.0,
            "avg_welfare": 28.0,
            "n_rounds": 120,
        },
    ]
    summary = _summarize_rows(rows)
    assert len(summary) == 1
    row = summary[0]
    assert math.isclose(float(row["mean_coop_rate"]), 0.3)
    assert math.isclose(float(row["mean_avg_reward"]), 6.0)
    assert math.isclose(float(row["mean_avg_welfare"]), 24.0)
    assert int(row["n_seeds"]) == 2
    assert float(row["sem_coop_rate"]) > 0.0
