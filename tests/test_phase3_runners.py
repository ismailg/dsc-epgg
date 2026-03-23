import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.analysis.checkpoint_artifacts import infer_absolute_milestones
from src.analysis.run_phase3_checkpoint_suite import _checkpoint_path as suite_checkpoint_path
from src.analysis.run_phase3_crossplay_matrix import _checkpoint_path as crossplay_checkpoint_path
from src.analysis.run_phase3_sender_causal_suite import _checkpoint_path as sender_checkpoint_path
from src.analysis.summarize_phase3_channel_controls import _collect_mode_rows
from src.analysis.validate_checkpoint_suite_outputs import validate_checkpoint_suite_outputs
from src.experiments_pgg_v0.train_ppo import minimal_test_config, train


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_fake_ckpt(path: Path, episode_offset: int, n_episodes: int):
    payload = {"config": {"episode_offset": episode_offset, "n_episodes": n_episodes}}
    torch.save(payload, path)


@pytest.mark.parametrize(
    "resolver",
    [suite_checkpoint_path, crossplay_checkpoint_path, sender_checkpoint_path],
)
def test_checkpoint_path_uses_absolute_episode_under_continuation(tmp_path: Path, resolver):
    final_ckpt = tmp_path / "cond1_seed111.pt"
    mid_ckpt = tmp_path / "cond1_seed111_ep50000.pt"
    _write_fake_ckpt(final_ckpt, episode_offset=50000, n_episodes=100000)
    _write_fake_ckpt(mid_ckpt, episode_offset=50000, n_episodes=100000)

    assert resolver(str(tmp_path), "cond1", 111, 100000) == str(mid_ckpt)
    assert resolver(str(tmp_path), "cond1", 111, 150000) == str(final_ckpt)

    with pytest.raises(FileNotFoundError):
        resolver(str(tmp_path), "cond1", 111, 50000)


def test_infer_absolute_milestones_deduplicates_final_and_intermediate(tmp_path: Path):
    _write_fake_ckpt(tmp_path / "cond1_seed111.pt", episode_offset=50000, n_episodes=100000)
    _write_fake_ckpt(tmp_path / "cond1_seed111_ep50000.pt", episode_offset=50000, n_episodes=100000)
    _write_fake_ckpt(tmp_path / "cond1_seed111_ep100000.pt", episode_offset=50000, n_episodes=100000)
    assert infer_absolute_milestones(str(tmp_path), condition="cond1", seeds=[111]) == [100000, 150000]


def test_validate_checkpoint_suite_outputs_rejects_header_only_raw_csv(tmp_path: Path):
    suite_dir = tmp_path / "suite"
    raw_dir = suite_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = suite_dir / "checkpoint_suite_manifest.json"
    payload = [
        {
            "name": "cond1_seed111_ep100000_none",
            "checkpoint": "dummy.pt",
            "episode": 100000,
            "intervention": "none",
            "out_csv": str(raw_dir / "cond1_seed111_ep100000_none.csv"),
            "out_comm_csv": str(raw_dir / "cond1_seed111_ep100000_none_comm.csv"),
            "out_condition_csv": str(raw_dir / "cond1_seed111_ep100000_none_condition.csv"),
            "out_trace_csv": "",
            "out_sender_csv": "",
            "out_receiver_csv": "",
            "out_posterior_csv": "",
        }
    ]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    for path in [
        raw_dir / "cond1_seed111_ep100000_none.csv",
        raw_dir / "cond1_seed111_ep100000_none_comm.csv",
        raw_dir / "cond1_seed111_ep100000_none_condition.csv",
    ]:
        path.write_text("a,b\n", encoding="utf-8")
    for path in [
        suite_dir / "checkpoint_suite_main.csv",
        suite_dir / "checkpoint_suite_comm.csv",
        suite_dir / "checkpoint_suite_condition.csv",
    ]:
        path.write_text("a,b\n", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_checkpoint_suite_outputs(
            manifest_path,
            suite_dir=suite_dir,
            expected_seeds=[111],
            expected_episodes=[100000],
            expected_interventions=["none"],
        )


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
            "public_random",
            "sender_shuffle",
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


def test_checkpoint_suite_runner_passes_history_intervention(tmp_path: Path):
    _make_checkpoint(tmp_path, "cond1", 112, comm_enabled=True)
    _make_checkpoint(tmp_path, "cond2", 112, comm_enabled=False)
    out_dir = tmp_path / "suite_hist_out"
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
            "112",
            "--milestones",
            "1",
            "--interventions",
            "none",
            "--history_intervention",
            "clamp_temporal_high",
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
    assert main_csv.exists()
    with open(main_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert {row["history_intervention"] for row in rows} == {"clamp_temporal_high"}
    raw_names = {path.name for path in (out_dir / "raw").glob("*.csv")}
    assert any("_hist_clamp_temporal_high.csv" in name for name in raw_names)


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
            "public_random",
            "sender_shuffle",
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


def test_channel_control_summary_recovers_condition_seed_from_checkpoint(tmp_path: Path):
    suite_csv = tmp_path / "suite.csv"
    with open(suite_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "condition",
                "train_seed",
                "scope",
                "eval_policy",
                "ablation",
                "cross_play",
                "key",
                "checkpoint_episode",
                "coop_rate",
                "avg_welfare",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "checkpoint": "outputs/train/phase3_channel_controls_50k/fixed0/cond1_seed101_fixed0_ep25000.pt",
                "condition": "unknown",
                "train_seed": "-1",
                "scope": "f_value",
                "eval_policy": "greedy",
                "ablation": "none",
                "cross_play": "none",
                "key": "3.500",
                "checkpoint_episode": "25000",
                "coop_rate": "0.5",
                "avg_welfare": "10.0",
            }
        )
    rows = _collect_mode_rows("always_zero", str(suite_csv))
    assert len(rows) == 1
    assert rows[0]["train_seed"] == 101
