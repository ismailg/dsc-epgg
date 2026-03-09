import csv
import os
import subprocess
import sys
from pathlib import Path

from src.experiments_pgg_v0.train_ppo import minimal_test_config, train


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_checkpoint(tmp_path: Path, condition: str, seed: int, comm_enabled: bool) -> Path:
    ckpt = tmp_path / f"{condition}_seed{seed}.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=3,
        T=4,
        comm_enabled=comm_enabled,
        n_senders=4 if comm_enabled else 0,
        seed=seed,
        save_path=str(ckpt),
        checkpoint_interval=1,
        regime_log_interval=1,
        condition_name=condition,
    )
    train(cfg)
    assert ckpt.exists()
    assert (tmp_path / f"{condition}_seed{seed}_ep1.pt").exists()
    assert (tmp_path / f"{condition}_seed{seed}_ep2.pt").exists()
    return ckpt


def test_checkpoint_suite_runner_aggregates_outputs(tmp_path: Path):
    _make_checkpoint(tmp_path, "cond1", 111, comm_enabled=True)
    _make_checkpoint(tmp_path, "cond2", 111, comm_enabled=False)
    out_dir = tmp_path / "suite_out"
    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.run_phase3_checkpoint_suite",
            "--checkpoint_dir",
            str(tmp_path),
            "--out_dir",
            str(out_dir),
            "--seeds",
            "111",
            "--milestones",
            "1",
            "2",
            "--interventions",
            "none",
            "fixed0",
            "--n_eval_episodes",
            "1",
            "--max_workers",
            "1",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    main_csv = out_dir / "checkpoint_suite_main.csv"
    sender_csv = out_dir / "checkpoint_suite_sender_semantics.csv"
    assert main_csv.exists()
    assert sender_csv.exists()
    with open(main_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert {"1", "2"} <= {row["checkpoint_episode"] for row in rows}
    assert {"comm", "baseline"} <= {row["suite_kind"] for row in rows}


def test_crossplay_runner_aggregates_matrix(tmp_path: Path):
    _make_checkpoint(tmp_path, "cond1", 222, comm_enabled=True)
    out_dir = tmp_path / "crossplay_out"
    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.run_phase3_crossplay_matrix",
            "--checkpoint_dir",
            str(tmp_path),
            "--out_dir",
            str(out_dir),
            "--condition",
            "cond1",
            "--seeds",
            "222",
            "--milestones",
            "1",
            "2",
            "--n_eval_episodes",
            "1",
            "--max_workers",
            "1",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    main_csv = out_dir / "crossplay_matrix_main.csv"
    assert main_csv.exists()
    with open(main_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    pairs = {(row["sender_episode"], row["receiver_episode"]) for row in rows if row["scope"] == "f_value"}
    assert ("1", "1") in pairs
    assert ("1", "2") in pairs
    assert ("2", "1") in pairs
    assert ("2", "2") in pairs


def test_seed_expansion_runner_launches_trimmed_jobs(tmp_path: Path):
    warm_ckpt = tmp_path / "fixedf_5p0_seed333.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=4,
        comm_enabled=False,
        n_senders=0,
        seed=333,
        save_path=str(warm_ckpt),
        condition_name="fixedf",
    )
    train(cfg)
    out_dir = tmp_path / "phase3_train"
    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.experiments_pgg_v0.run_phase3_seed_expansion",
            "--fixed_f_dir",
            str(tmp_path),
            "--out_dir",
            str(out_dir),
            "--conditions",
            "cond1",
            "cond2",
            "--seeds",
            "333",
            "--n_episodes",
            "2",
            "--T",
            "4",
            "--log_interval",
            "1",
            "--regime_log_interval",
            "1",
            "--checkpoint_interval",
            "1",
            "--max_workers",
            "1",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    assert (out_dir / "cond1_seed333.pt").exists()
    assert (out_dir / "cond2_seed333.pt").exists()
    assert (out_dir / "phase3_seed_expansion_manifest.json").exists()


def test_seed_expansion_runner_supports_checkpoint_continuation_and_public_random(tmp_path: Path):
    base_dir = tmp_path / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    _make_checkpoint(base_dir, "cond1", 555, comm_enabled=True)
    out_dir = tmp_path / "phase3_train_ext"
    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.experiments_pgg_v0.run_phase3_seed_expansion",
            "--out_dir",
            str(out_dir),
            "--init_checkpoint_dir",
            str(base_dir),
            "--init_episode",
            "3",
            "--episode_offset",
            "3",
            "--schedule_total_episodes",
            "5",
            "--conditions",
            "cond1",
            "--seeds",
            "555",
            "--n_episodes",
            "2",
            "--T",
            "4",
            "--msg_training_intervention",
            "public_random",
            "--log_interval",
            "1",
            "--regime_log_interval",
            "1",
            "--checkpoint_interval",
            "1",
            "--max_workers",
            "1",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    assert (out_dir / "cond1_seed555_public_random.pt").exists()
    assert (out_dir / "cond1_seed555_public_random_ep4.pt").exists()


def test_trimmed_eval_runner_orchestrates_suite_and_crossplay(tmp_path: Path):
    _make_checkpoint(tmp_path, "cond1", 444, comm_enabled=True)
    _make_checkpoint(tmp_path, "cond2", 444, comm_enabled=False)
    out_root = tmp_path / "trimmed_eval"
    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.run_phase3_trimmed_eval",
            "--checkpoint_dir",
            str(tmp_path),
            "--suite_out_dir",
            str(out_root / "suite"),
            "--crossplay_out_dir",
            str(out_root / "crossplay"),
            "--seeds",
            "444",
            "--milestones",
            "1",
            "2",
            "--interventions",
            "none",
            "fixed0",
            "--crossplay_sender_milestones",
            "1",
            "--crossplay_receiver_milestones",
            "2",
            "--n_eval_episodes",
            "1",
            "--max_workers",
            "1",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    assert (out_root / "suite" / "checkpoint_suite_main.csv").exists()
    assert (out_root / "crossplay" / "crossplay_matrix_main.csv").exists()
