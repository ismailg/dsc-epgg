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


def test_train_with_session_logging(tmp_path):
    session_dir = tmp_path / "sessions"
    cfg = minimal_test_config(
        n_episodes=3,
        T=6,
        save_path=str(tmp_path / "agents3.pt"),
        seed=123,
        log_sessions=True,
        session_log_dir=str(session_dir),
        condition_name="ci",
        consolidate_sessions=True,
    )
    _ = train(cfg)
    parts = list(session_dir.glob("data_ci_123_*.npz"))
    assert len(parts) >= 3
    consolidated = session_dir / "data_ci_123_consolidated.npz"
    assert consolidated.exists()


def test_full_loop_runs_vectorized(tmp_path):
    cfg = minimal_test_config(
        n_episodes=3,
        T=6,
        num_envs=2,
        save_path=str(tmp_path / "agents_vec.pt"),
        seed=321,
    )
    metrics = train(cfg)
    assert len(metrics) == 3
    assert (tmp_path / "agents_vec.pt").exists()
    assert all(int(row["num_envs"]) == 2 for row in metrics)
    assert all(int(row["steps"]) == 12 for row in metrics)
