import numpy as np

from src.experiments_pgg_v0.train_ppo import minimal_test_config, train


def test_full_loop_runs(tmp_path):
    cfg = minimal_test_config(
        n_episodes=5,
        T=8,
        save_path=str(tmp_path / "agents.pt"),
        seed=123,
    )
    metrics = train(cfg)
    assert len(metrics) == 5
    assert (tmp_path / "agents.pt").exists()


def test_cooperation_changes(tmp_path):
    cfg = minimal_test_config(
        n_episodes=30,
        T=8,
        save_path=str(tmp_path / "agents2.pt"),
        seed=123,
    )
    metrics = train(cfg)
    coop_rates = np.array([m["coop_rate"] for m in metrics], dtype=np.float32)
    assert np.std(coop_rates) > 1e-3

