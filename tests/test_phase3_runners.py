import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.algos.PPO import PPOAgentV2
from src.analysis import evaluate_regime_conditional
from src.analysis.checkpoint_artifacts import (
    infer_absolute_milestones,
    infer_manifest_absolute_milestones,
    resolve_checkpoint_path,
    resolve_manifest_checkpoint_path,
)
from src.analysis.validate_checkpoint_suite_outputs import validate_checkpoint_suite_outputs
from src.analysis.validate_sender_causal_outputs import validate_sender_causal_outputs
from src.experiments_pgg_v0 import run_phase3_seed_expansion
from src.experiments_pgg_v0.train_ppo import minimal_test_config, train
from src.wrappers.observation_wrapper import ObservationWrapper


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_fake_ckpt(path: Path, episode_offset: int, n_episodes: int):
    payload = {"config": {"episode_offset": episode_offset, "n_episodes": n_episodes}}
    torch.save(payload, path)


def _write_manifest(path: Path, paths: list[Path]) -> Path:
    path.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    return path


def _make_checkpoint(tmp_path: Path, condition: str, seed: int, comm_enabled: bool) -> Path:
    ckpt = tmp_path / f"{condition}_seed{seed}.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=4,
        comm_enabled=comm_enabled,
        n_senders=4 if comm_enabled else 0,
        seed=seed,
        save_path=str(ckpt),
        checkpoint_interval=1,
        regime_log_interval=1,
        condition_name=condition,
        sign_lambda=0.0,
        list_lambda=0.0,
    )
    train(cfg)
    assert ckpt.exists()
    assert (tmp_path / f"{condition}_seed{seed}_ep1.pt").exists()
    return ckpt


def test_manifest_checkpoint_path_uses_absolute_episode_under_continuation(tmp_path: Path):
    final_ckpt = tmp_path / "cond1_seed111.pt"
    mid_ckpt = tmp_path / "cond1_seed111_ep100000.pt"
    _write_fake_ckpt(final_ckpt, episode_offset=50000, n_episodes=100000)
    _write_fake_ckpt(mid_ckpt, episode_offset=50000, n_episodes=100000)
    manifest = _write_manifest(tmp_path / "cond1.txt", [mid_ckpt, final_ckpt])

    assert resolve_manifest_checkpoint_path(manifest, "cond1", 111, 100000) == str(mid_ckpt)
    assert resolve_manifest_checkpoint_path(manifest, "cond1", 111, 150000) == str(final_ckpt)
    assert infer_manifest_absolute_milestones(manifest, condition="cond1", seeds=[111]) == [
        100000,
        150000,
    ]
    assert resolve_checkpoint_path(str(tmp_path), "cond1", 111, 100000) == str(mid_ckpt)
    assert resolve_checkpoint_path(str(tmp_path), "cond1", 111, 150000) == str(final_ckpt)
    assert infer_absolute_milestones(str(tmp_path), condition="cond1", seeds=[111]) == [
        100000,
        150000,
    ]

    with pytest.raises(FileNotFoundError):
        resolve_manifest_checkpoint_path(manifest, "cond1", 111, 50000)


def test_manifest_checkpoint_path_prefers_run_json_sidecar(tmp_path: Path):
    final_ckpt = tmp_path / "cond1_seed111.pt"
    mid_ckpt = tmp_path / "cond1_seed111_ep100000.pt"
    final_ckpt.write_text("not-a-torch-checkpoint\n", encoding="utf-8")
    mid_ckpt.write_text("not-a-torch-checkpoint\n", encoding="utf-8")
    (tmp_path / "cond1_seed111.run.json").write_text(
        json.dumps({"config": {"episode_offset": 50000, "n_episodes": 100000}}),
        encoding="utf-8",
    )
    (tmp_path / "cond1_seed111_ep100000.run.json").write_text(
        json.dumps({"config": {"episode_offset": 50000, "n_episodes": 100000}}),
        encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path / "cond1.txt", [mid_ckpt, final_ckpt])

    assert resolve_manifest_checkpoint_path(manifest, "cond1", 111, 100000) == str(mid_ckpt)
    assert resolve_manifest_checkpoint_path(manifest, "cond1", 111, 150000) == str(final_ckpt)
    assert infer_manifest_absolute_milestones(manifest, condition="cond1", seeds=[111]) == [
        100000,
        150000,
    ]


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
            "out_sender_causal_csv": "",
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


def test_validate_checkpoint_suite_outputs_allows_baseline_without_comm_csv(tmp_path: Path):
    suite_dir = tmp_path / "suite"
    raw_dir = suite_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = suite_dir / "checkpoint_suite_manifest.json"
    payload = [
        {
            "name": "cond2_seed111_ep100000_none",
            "checkpoint": "dummy.pt",
            "episode": 100000,
            "intervention": "none",
            "suite_kind": "baseline",
            "out_csv": str(raw_dir / "cond2_seed111_ep100000_none.csv"),
            "out_comm_csv": str(raw_dir / "cond2_seed111_ep100000_none_comm.csv"),
            "out_condition_csv": str(raw_dir / "cond2_seed111_ep100000_none_condition.csv"),
            "out_trace_csv": "",
            "out_sender_csv": "",
            "out_receiver_csv": "",
            "out_posterior_csv": "",
            "out_sender_causal_csv": "",
        }
    ]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    (raw_dir / "cond2_seed111_ep100000_none.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (raw_dir / "cond2_seed111_ep100000_none_condition.csv").write_text(
        "a,b\n1,2\n", encoding="utf-8"
    )
    (suite_dir / "checkpoint_suite_main.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (suite_dir / "checkpoint_suite_comm.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (suite_dir / "checkpoint_suite_condition.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    validate_checkpoint_suite_outputs(
        manifest_path,
        suite_dir=suite_dir,
        expected_seeds=[111],
        expected_episodes=[100000],
        expected_interventions=["none"],
    )


def test_checkpoint_suite_runner_supports_manifests_and_baseline_only(tmp_path: Path):
    _make_checkpoint(tmp_path, "cond1", 111, comm_enabled=True)
    _make_checkpoint(tmp_path, "cond2", 111, comm_enabled=False)
    comm_manifest = _write_manifest(
        tmp_path / "cond1_manifest.txt",
        [tmp_path / "cond1_seed111_ep1.pt", tmp_path / "cond1_seed111.pt"],
    )
    baseline_manifest = _write_manifest(
        tmp_path / "cond2_manifest.txt",
        [tmp_path / "cond2_seed111_ep1.pt", tmp_path / "cond2_seed111.pt"],
    )
    out_dir = tmp_path / "suite_out"
    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.run_phase3_checkpoint_suite",
            "--comm_checkpoint_manifest",
            str(comm_manifest),
            "--baseline_checkpoint_manifest",
            str(baseline_manifest),
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
            "public_marginal",
            "--n_eval_episodes",
            "1",
            "--max_workers",
            "1",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    manifest_json = out_dir / "checkpoint_suite_manifest.json"
    assert manifest_json.exists()
    tasks = json.loads(manifest_json.read_text(encoding="utf-8"))
    baseline_tasks = [task for task in tasks if task["suite_kind"] == "baseline"]
    comm_tasks = [task for task in tasks if task["suite_kind"] == "comm"]
    assert {task["intervention"] for task in baseline_tasks} == {"none"}
    assert {task["intervention"] for task in comm_tasks} == {
        "none",
        "public_random",
        "public_marginal",
    }

    main_csv = out_dir / "checkpoint_suite_main.csv"
    assert main_csv.exists()
    with main_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert {"1", "2"} <= {row["checkpoint_episode"] for row in rows}
    cond2_rows = [row for row in rows if row["condition"] == "cond2"]
    assert len(cond2_rows) > 0
    assert {row["ablation"] for row in cond2_rows} == {"none"}


def test_public_marginal_eval_intervention_uses_shared_average_sender_marginal(monkeypatch):
    wrapper = ObservationWrapper(
        n_agents=4,
        comm_enabled=True,
        n_senders=4,
        sender_ids=["agent_0", "agent_1", "agent_2", "agent_3"],
        vocab_size=2,
    )
    wrapper.reset(agent_ids=["agent_0", "agent_1", "agent_2", "agent_3"])
    wrapper.msg_marginals = {
        "agent_0": np.asarray([0.9, 0.1], dtype=np.float32),
        "agent_1": np.asarray([0.8, 0.2], dtype=np.float32),
        "agent_2": np.asarray([0.7, 0.3], dtype=np.float32),
        "agent_3": np.asarray([0.2, 0.8], dtype=np.float32),
    }
    captured: dict[str, np.ndarray] = {}

    def fake_choice(n: int, p=None):
        captured["p"] = np.asarray(p, dtype=np.float64)
        return 1

    monkeypatch.setattr(evaluate_regime_conditional.np.random, "choice", fake_choice)
    out, force_zero = evaluate_regime_conditional._apply_message_intervention(
        intervention="public_marginal",
        delivered={"agent_0": 0, "agent_1": 1, "agent_2": 0, "agent_3": 1},
        wrapper=wrapper,
        sender_ids=["agent_0", "agent_1", "agent_2", "agent_3"],
        vocab_size=2,
    )

    assert not force_zero
    assert out == {"agent_0": 1, "agent_1": 1, "agent_2": 1, "agent_3": 1}
    assert np.allclose(captured["p"], np.asarray([0.65, 0.35], dtype=np.float64))


def test_zeros_eval_intervention_blanks_message_slice_not_token_zero():
    wrapper = ObservationWrapper(
        n_agents=4,
        comm_enabled=True,
        n_senders=4,
        sender_ids=["agent_0", "agent_1", "agent_2", "agent_3"],
        vocab_size=2,
    )
    agent_ids = ["agent_0", "agent_1", "agent_2", "agent_3"]
    wrapper.reset(agent_ids=agent_ids)
    raw_obs = {agent_id: np.asarray([3.5, 4.0], dtype=np.float32) for agent_id in agent_ids}
    f_hat = {agent_id: 3.5 for agent_id in agent_ids}

    fixed0_messages, fixed0_force_zero = evaluate_regime_conditional._apply_message_intervention(
        intervention="fixed0",
        delivered={"agent_0": 1, "agent_1": 1, "agent_2": 0, "agent_3": 1},
        wrapper=wrapper,
        sender_ids=agent_ids,
        vocab_size=2,
    )
    zeros_messages, zeros_force_zero = evaluate_regime_conditional._apply_message_intervention(
        intervention="zeros",
        delivered={"agent_0": 1, "agent_1": 1, "agent_2": 0, "agent_3": 1},
        wrapper=wrapper,
        sender_ids=agent_ids,
        vocab_size=2,
    )

    assert fixed0_messages == {agent_id: 0 for agent_id in agent_ids}
    assert not fixed0_force_zero
    assert zeros_messages == {"agent_0": 1, "agent_1": 1, "agent_2": 0, "agent_3": 1}
    assert zeros_force_zero

    fixed0_obs = evaluate_regime_conditional._build_aug_obs(
        wrapper=wrapper,
        raw_obs=raw_obs,
        agent_ids=agent_ids,
        current_messages=fixed0_messages,
        observed_f_hat_by_agent=f_hat,
        history_intervention="none",
    )
    zeros_obs = evaluate_regime_conditional._build_aug_obs(
        wrapper=wrapper,
        raw_obs=raw_obs,
        agent_ids=agent_ids,
        current_messages=zeros_messages,
        observed_f_hat_by_agent=f_hat,
        history_intervention="none",
    )
    msg_start = int(wrapper.message_start_idx)
    for agent_id in agent_ids:
        zeros_obs[agent_id][msg_start:] = 0.0

    assert np.allclose(fixed0_obs["agent_0"][msg_start:], np.asarray([1.0, 0.0] * 4))
    assert np.allclose(zeros_obs["agent_0"][msg_start:], np.zeros(8, dtype=np.float32))
    assert not np.allclose(fixed0_obs["agent_0"][msg_start:], zeros_obs["agent_0"][msg_start:])


def test_sender_remap_requires_full_bijection():
    sender_ids = ["agent_0", "agent_1", "agent_2", "agent_3"]

    with pytest.raises(ValueError):
        evaluate_regime_conditional._parse_sender_remap(
            json.dumps({"agent_0": "agent_1"}),
            sender_ids,
        )

    with pytest.raises(ValueError):
        evaluate_regime_conditional._parse_sender_remap(
            json.dumps(
                {
                    "agent_0": "agent_0",
                    "agent_1": "agent_0",
                    "agent_2": "agent_2",
                    "agent_3": "agent_3",
                }
            ),
            sender_ids,
        )


def test_sender_remap_relabels_delivered_slots():
    delivered = {"agent_0": 0, "agent_1": 1, "agent_2": 0, "agent_3": 1}
    remap = {
        "agent_0": "agent_1",
        "agent_1": "agent_2",
        "agent_2": "agent_3",
        "agent_3": "agent_0",
    }

    out = evaluate_regime_conditional._apply_sender_remap(delivered, remap)

    assert out == {"agent_1": 0, "agent_2": 1, "agent_3": 0, "agent_0": 1}


def test_public_exact_observability_overrides_all_agent_f_hats():
    raw_obs = {
        "agent_0": np.asarray([0.5, 4.0], dtype=np.float32),
        "agent_1": np.asarray([1.5, 4.0], dtype=np.float32),
        "agent_2": np.asarray([2.5, 4.0], dtype=np.float32),
        "agent_3": np.asarray([3.5, 4.0], dtype=np.float32),
    }

    out = evaluate_regime_conditional._observed_f_hat_by_agent(
        raw_obs=raw_obs,
        agent_ids=["agent_0", "agent_1", "agent_2", "agent_3"],
        true_f=5.0,
        observability_mode="public_exact",
        public_signal_sigma=None,
    )

    assert out == {
        "agent_0": 5.0,
        "agent_1": 5.0,
        "agent_2": 5.0,
        "agent_3": 5.0,
    }


def test_public_noisy_observability_uses_one_shared_sample(monkeypatch):
    raw_obs = {
        "agent_0": np.asarray([0.5, 4.0], dtype=np.float32),
        "agent_1": np.asarray([1.5, 4.0], dtype=np.float32),
        "agent_2": np.asarray([2.5, 4.0], dtype=np.float32),
        "agent_3": np.asarray([3.5, 4.0], dtype=np.float32),
    }

    monkeypatch.setattr(evaluate_regime_conditional.np.random, "normal", lambda loc, scale: 4.25)
    out = evaluate_regime_conditional._observed_f_hat_by_agent(
        raw_obs=raw_obs,
        agent_ids=["agent_0", "agent_1", "agent_2", "agent_3"],
        true_f=3.5,
        observability_mode="public_noisy",
        public_signal_sigma=0.25,
    )

    assert out == {
        "agent_0": 4.25,
        "agent_1": 4.25,
        "agent_2": 4.25,
        "agent_3": 4.25,
    }


def test_eval_checkpoint_uses_native_exogenous_message_source(monkeypatch, tmp_path: Path):
    ckpt = tmp_path / "cond1_public_random_seed445.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=4,
        comm_enabled=True,
        n_senders=4,
        seed=445,
        save_path=str(ckpt),
        condition_name="cond1",
        sign_lambda=0.0,
        list_lambda=0.0,
        msg_dropout=0.0,
        msg_source_mode="public_random",
    )
    train(cfg)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("sample_message should not be called for exogenous msg_source_mode")

    monkeypatch.setattr(PPOAgentV2, "sample_message", fail_if_called)

    rows, comm_rows, *_rest = evaluate_regime_conditional._eval_checkpoint(
        checkpoint_path=str(ckpt),
        n_eval_episodes=1,
        eval_seed=123,
        greedy=False,
    )

    assert len(rows) > 0
    assert len(comm_rows) > 0
    all_sender_rows = [
        row for row in comm_rows if row.get("key") == "all_senders" and "n_pairs" in row
    ]
    assert len(all_sender_rows) > 0
    assert all(int(row["n_pairs"]) > 0 for row in all_sender_rows)


def test_checkpoint_suite_runner_records_public_exact_observability(tmp_path: Path):
    _make_checkpoint(tmp_path, "cond1", 444, comm_enabled=True)
    _make_checkpoint(tmp_path, "cond2", 444, comm_enabled=False)
    comm_manifest = _write_manifest(
        tmp_path / "cond1_public_exact_manifest.txt",
        [tmp_path / "cond1_seed444_ep1.pt", tmp_path / "cond1_seed444.pt"],
    )
    baseline_manifest = _write_manifest(
        tmp_path / "cond2_public_exact_manifest.txt",
        [tmp_path / "cond2_seed444_ep1.pt", tmp_path / "cond2_seed444.pt"],
    )
    out_dir = tmp_path / "public_exact_suite_out"
    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.run_phase3_checkpoint_suite",
            "--comm_checkpoint_manifest",
            str(comm_manifest),
            "--baseline_checkpoint_manifest",
            str(baseline_manifest),
            "--out_dir",
            str(out_dir),
            "--seeds",
            "444",
            "--milestones",
            "1",
            "--interventions",
            "none",
            "--n_eval_episodes",
            "1",
            "--max_workers",
            "1",
            "--observability_mode",
            "public_exact",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )

    tasks = json.loads((out_dir / "checkpoint_suite_manifest.json").read_text(encoding="utf-8"))
    assert len(tasks) == 2
    assert {task["observability_mode"] for task in tasks} == {"public_exact"}
    assert all("_obs_public_exact" in task["name"] for task in tasks)

    with (out_dir / "checkpoint_suite_main.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert {row["observability_mode"] for row in rows} == {"public_exact"}


def test_sender_causal_runner_supports_manifest(tmp_path: Path):
    _make_checkpoint(tmp_path, "cond1", 222, comm_enabled=True)
    manifest = _write_manifest(
        tmp_path / "cond1_sender_manifest.txt",
        [tmp_path / "cond1_seed222_ep1.pt", tmp_path / "cond1_seed222.pt"],
    )
    out_dir = tmp_path / "sender_causal_out"
    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.run_phase3_sender_causal_suite",
            "--checkpoint_manifest",
            str(manifest),
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
    main_csv = out_dir / "sender_causal_matrix.csv"
    assert main_csv.exists()
    with open(main_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert {"1", "2"} <= {row["checkpoint_episode"] for row in rows}
    validate_sender_causal_outputs(
        out_dir / "sender_causal_manifest.json",
        suite_dir=out_dir,
        expected_seeds=[222],
        expected_episodes=[1, 2],
    )


def test_cross_seed_transfer_runner_supports_manifest_and_flip_alignment(tmp_path: Path):
    _make_checkpoint(tmp_path, "cond1", 555, comm_enabled=True)
    _make_checkpoint(tmp_path, "cond1", 666, comm_enabled=True)
    manifest = _write_manifest(
        tmp_path / "cond1_xseed_manifest.txt",
        [tmp_path / "cond1_seed555.pt", tmp_path / "cond1_seed666.pt"],
    )
    out_dir = tmp_path / "xseed_out"
    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.run_phase3_cross_seed_transfer_suite",
            "--checkpoint_manifest",
            str(manifest),
            "--out_dir",
            str(out_dir),
            "--condition",
            "cond1",
            "--episode",
            "2",
            "--alignment_mode",
            "flip",
            "--n_eval_episodes",
            "1",
            "--max_workers",
            "1",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )

    manifest_json = out_dir / "cross_seed_transfer_manifest.json"
    assert manifest_json.exists()
    tasks = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert len(tasks) == 4
    assert {"identity__noflip", "identity__flipall"} == {
        task["alignment_label"] for task in tasks
    }
    pair_eval_seeds = {}
    for task in tasks:
        pair_key = (task["receiver_seed"], task["donor_seed"])
        pair_eval_seeds.setdefault(pair_key, set()).add(int(task["eval_seed"]))
    assert all(len(seeds) == 1 for seeds in pair_eval_seeds.values())

    main_csv = out_dir / "cross_seed_transfer_main.csv"
    assert main_csv.exists()
    with main_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert {"555", "666"} == {row["receiver_seed"] for row in rows}
    assert {"555", "666"} == {row["donor_seed"] for row in rows}
    assert {"foreign"} == {row["transfer_kind"] for row in rows}
    assert {row["sender_remap"] for row in rows} == {"identity"}
    assert {
        "cond1_seed555.pt",
        "cond1_seed666.pt",
    } == {row["cross_play"] for row in rows}

    comm_csv = out_dir / "cross_seed_transfer_comm.csv"
    assert comm_csv.exists()
    with comm_csv.open("r", encoding="utf-8") as f:
        comm_rows = list(csv.DictReader(f))
    assert len(comm_rows) > 0
    assert {row["alignment_label"] for row in comm_rows} == {"identity__noflip"}


def test_cross_seed_transfer_summary_selects_best_alignment(tmp_path: Path):
    transfer_csv = tmp_path / "transfer_main.csv"
    reference_csv = tmp_path / "reference_main.csv"
    out_dir = tmp_path / "summary_out"

    transfer_rows = [
        {
            "condition": "cond1",
            "eval_policy": "greedy",
            "ablation": "none",
            "history_intervention": "none",
            "scope": "f_value",
            "key": "3.500",
            "coop_rate": "0.70",
            "receiver_seed": "101",
            "donor_seed": "202",
            "alignment_label": "identity__noflip",
            "sender_remap": "identity",
            "cross_play": "cond1_seed202.pt",
        },
        {
            "condition": "cond1",
            "eval_policy": "greedy",
            "ablation": "none",
            "history_intervention": "none",
            "scope": "f_value",
            "key": "3.500",
            "coop_rate": "0.74",
            "receiver_seed": "101",
            "donor_seed": "202",
            "alignment_label": "identity__flipall",
            "sender_remap": "identity",
            "cross_play": "cond1_seed202.pt",
        },
        {
            "condition": "cond1",
            "eval_policy": "greedy",
            "ablation": "none",
            "history_intervention": "none",
            "scope": "f_value",
            "key": "5.000",
            "coop_rate": "0.51",
            "receiver_seed": "101",
            "donor_seed": "202",
            "alignment_label": "identity__noflip",
            "sender_remap": "identity",
            "cross_play": "cond1_seed202.pt",
        },
        {
            "condition": "cond1",
            "eval_policy": "greedy",
            "ablation": "none",
            "history_intervention": "none",
            "scope": "f_value",
            "key": "5.000",
            "coop_rate": "0.67",
            "receiver_seed": "101",
            "donor_seed": "202",
            "alignment_label": "identity__flipall",
            "sender_remap": "identity",
            "cross_play": "cond1_seed202.pt",
        },
    ]
    reference_rows = [
        {
            "condition": "cond1",
            "train_seed": "101",
            "eval_policy": "greedy",
            "ablation": "none",
            "history_intervention": "none",
            "sender_remap": "none",
            "cross_play": "none",
            "scope": "f_value",
            "key": "3.500",
            "coop_rate": "0.80",
        },
        {
            "condition": "cond1",
            "train_seed": "101",
            "eval_policy": "greedy",
            "ablation": "public_random",
            "history_intervention": "none",
            "sender_remap": "none",
            "cross_play": "none",
            "scope": "f_value",
            "key": "3.500",
            "coop_rate": "0.73",
        },
        {
            "condition": "cond1",
            "train_seed": "101",
            "eval_policy": "greedy",
            "ablation": "sender_shuffle",
            "history_intervention": "none",
            "sender_remap": "none",
            "cross_play": "none",
            "scope": "f_value",
            "key": "3.500",
            "coop_rate": "0.72",
        },
        {
            "condition": "cond1",
            "train_seed": "101",
            "eval_policy": "greedy",
            "ablation": "indep_random",
            "history_intervention": "none",
            "sender_remap": "none",
            "cross_play": "none",
            "scope": "f_value",
            "key": "3.500",
            "coop_rate": "0.71",
        },
        {
            "condition": "cond1",
            "train_seed": "101",
            "eval_policy": "greedy",
            "ablation": "none",
            "history_intervention": "none",
            "sender_remap": "none",
            "cross_play": "none",
            "scope": "f_value",
            "key": "5.000",
            "coop_rate": "0.77",
        },
        {
            "condition": "cond1",
            "train_seed": "101",
            "eval_policy": "greedy",
            "ablation": "public_random",
            "history_intervention": "none",
            "sender_remap": "none",
            "cross_play": "none",
            "scope": "f_value",
            "key": "5.000",
            "coop_rate": "0.64",
        },
        {
            "condition": "cond1",
            "train_seed": "101",
            "eval_policy": "greedy",
            "ablation": "sender_shuffle",
            "history_intervention": "none",
            "sender_remap": "none",
            "cross_play": "none",
            "scope": "f_value",
            "key": "5.000",
            "coop_rate": "0.66",
        },
        {
            "condition": "cond1",
            "train_seed": "101",
            "eval_policy": "greedy",
            "ablation": "indep_random",
            "history_intervention": "none",
            "sender_remap": "none",
            "cross_play": "none",
            "scope": "f_value",
            "key": "5.000",
            "coop_rate": "0.65",
        },
    ]

    with transfer_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(transfer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(transfer_rows)
    with reference_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(reference_rows[0].keys()))
        writer.writeheader()
        writer.writerows(reference_rows)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.summarize_phase3_cross_seed_transfer",
            "--transfer_main_csv",
            str(transfer_csv),
            "--reference_main_csv",
            str(reference_csv),
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )

    best_csv = out_dir / "best_alignment_results.csv"
    assert best_csv.exists()
    with best_csv.open("r", encoding="utf-8") as f:
        best_rows = list(csv.DictReader(f))
    assert len(best_rows) == 2
    by_f = {row["f_value"]: row for row in best_rows}
    assert by_f["3.500"]["best_alignment_label"] == "identity__flipall"
    assert float(by_f["3.500"]["best_coop_rate"]) == pytest.approx(0.74)
    assert float(by_f["3.500"]["identity_coop_rate"]) == pytest.approx(0.70)
    assert float(by_f["3.500"]["flipall_coop_rate"]) == pytest.approx(0.74)
    assert by_f["5.000"]["best_alignment_label"] == "identity__flipall"
    assert float(by_f["5.000"]["best_coop_rate"]) == pytest.approx(0.67)
    assert float(by_f["5.000"]["identity_coop_rate"]) == pytest.approx(0.51)
    assert float(by_f["5.000"]["flipall_coop_rate"]) == pytest.approx(0.67)

    summary_csv = out_dir / "summary_by_f.csv"
    assert summary_csv.exists()
    with summary_csv.open("r", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    lookup = {(row["f_value"], row["metric"]): row for row in summary_rows}
    assert lookup[("3.500", "natural_mean")]["sample_unit"] == "ordered_pairs"
    assert lookup[("3.500", "receiver_best_minus_natural_mean")]["sample_unit"] == "receivers"
    assert float(lookup[("3.500", "receiver_best_minus_natural_mean")]["value"]) == pytest.approx(-0.06)
    assert float(lookup[("3.500", "receiver_positive_best_minus_natural_count")]["value"]) == pytest.approx(0.0)


def test_phase3_seed_expansion_supports_init_manifest(tmp_path: Path):
    _make_checkpoint(tmp_path, "cond1", 333, comm_enabled=True)
    manifest = _write_manifest(
        tmp_path / "cond1_cont_manifest.txt",
        [tmp_path / "cond1_seed333_ep1.pt", tmp_path / "cond1_seed333.pt"],
    )
    out_dir = tmp_path / "continuation_out"
    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.experiments_pgg_v0.run_phase3_seed_expansion",
            "--out_dir",
            str(out_dir),
            "--init_checkpoint_manifest",
            str(manifest),
            "--init_episode",
            "2",
            "--conditions",
            "cond1",
            "--seeds",
            "333",
            "--n_episodes",
            "1",
            "--num_envs",
            "1",
            "--env_backend",
            "serial",
            "--no_count_env_episodes",
            "--checkpoint_interval",
            "1",
            "--episode_offset",
            "2",
            "--schedule_total_episodes",
            "3",
            "--msg_training_intervention",
            "sender_shuffle",
            "--max_workers",
            "1",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    assert (out_dir / "cond1_seed333.pt").exists()
    assert (out_dir / "phase3_seed_expansion_manifest.json").exists()


def test_phase3_seed_expansion_defaults_match_vecstraight_base_contract(tmp_path: Path):
    final_ckpt = tmp_path / "cond1_seed333.pt"
    mid_ckpt = tmp_path / "cond1_seed333_ep50000.pt"
    _write_fake_ckpt(final_ckpt, episode_offset=0, n_episodes=150000)
    _write_fake_ckpt(mid_ckpt, episode_offset=0, n_episodes=150000)
    manifest = _write_manifest(tmp_path / "cond1_manifest.txt", [mid_ckpt, final_ckpt])
    defaults = run_phase3_seed_expansion.parse_args(
        [
            "--out_dir",
            str(tmp_path / "continuation_out"),
            "--init_checkpoint_manifest",
            str(manifest),
            "--init_episode",
            "50000",
            "--conditions",
            "cond1",
            "--seeds",
            "333",
            "--n_episodes",
            "100000",
            "--episode_offset",
            "50000",
            "--schedule_total_episodes",
            "150000",
        ]
    )
    job = run_phase3_seed_expansion._build_job(defaults, condition="cond1", seed=333)
    cmd = job.cmd

    assert "--num_envs" in cmd
    assert cmd[cmd.index("--num_envs") + 1] == "8"
    assert "--count_env_episodes" in cmd
    assert "--env_backend" in cmd
    assert cmd[cmd.index("--env_backend") + 1] == "subproc"
    assert "--env_start_method" in cmd
    assert cmd[cmd.index("--env_start_method") + 1] == "spawn"
    assert "--msg_entropy_coeff" in cmd
    assert cmd[cmd.index("--msg_entropy_coeff") + 1] == "0.01"
    assert "--msg_entropy_coeff_final" in cmd
    assert cmd[cmd.index("--msg_entropy_coeff_final") + 1] == "0.0"
    assert "--sign_lambda" in cmd
    assert cmd[cmd.index("--sign_lambda") + 1] == "0.1"
    assert "--list_lambda" in cmd
    assert cmd[cmd.index("--list_lambda") + 1] == "0.1"
    assert "--regime_log_interval" in cmd
    assert cmd[cmd.index("--regime_log_interval") + 1] == "400"
    assert "--checkpoint_interval" in cmd
    assert cmd[cmd.index("--checkpoint_interval") + 1] == "25000"
    assert "--disable_comm_fallback" in cmd


def test_comm_history_factorial_script_preserves_vecstraight_base_contract():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_comm_history_factorial.sh"
    text = script_path.read_text(encoding="utf-8")

    assert 'TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-auto}"' in text
    assert 'SIGN_LAMBDA="${SIGN_LAMBDA:-0.1}"' in text
    assert 'LIST_LAMBDA="${LIST_LAMBDA:-0.1}"' in text
    assert "detect_default_train_workers()" in text
    assert 'if [[ "${TRAIN_MAX_WORKERS}" == "auto" ]]; then' in text
    assert 'suggested=$(( nproc_val / 2 ))' in text
    assert 'must resolve to a positive integer' in text
    assert "--num_envs 8" in text
    assert "--count_env_episodes" in text
    assert "--env_backend subproc" in text
    assert "--env_start_method spawn" in text
    assert "--entropy_schedule linear" in text
    assert "--lr_schedule cosine" in text
    assert "--msg_entropy_coeff 0.01" in text
    assert "--msg_entropy_coeff_final 0.0" in text
    assert "--history_mode \"${history_mode}\"" in text
    assert '--sign_lambda "${SIGN_LAMBDA}"' in text
    assert '--list_lambda "${LIST_LAMBDA}"' in text
    assert 'local out_root="${OUT_ROOT:-${REPO_ROOT}/outputs/train/phase3_vecstraight_comm_history_factorial_${cell}_15seeds_${RUN_KIND}_${RUN_DATE}}"' in text
    assert "must be divisible by 8 to preserve the base vectorized count_env_episodes contract" in text
    assert 'if (( ${#active_pids[@]} == 0 )); then' in text
    assert 'local -a kept=()' in text
    assert 'for pid in "${active_pids[@]}";' in text
    assert 'if (( ${#kept[@]} == 0 )); then' in text
    assert 'active_pids=()' in text
    assert 'active_pids=("${kept[@]}")' in text


def test_comm_history_factorial_parallel_script_splits_host_budget_across_all_cells():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_comm_history_factorial_parallel.sh"
    text = script_path.read_text(encoding="utf-8")

    assert 'TOTAL_WORKERS="${TOTAL_WORKERS:-auto}"' in text
    assert "detect_default_total_workers()" in text
    assert 'suggested=$(( nproc_val / 2 ))' in text
    assert 'if (( suggested < ${#CELLS[@]} )); then' in text
    assert 'must resolve to an integer >= ${#CELLS[@]} to run all cells in parallel' in text
    assert 'base_workers=$(( TOTAL_WORKERS / ${#CELLS[@]} ))' in text
    assert 'remainder=$(( TOTAL_WORKERS % ${#CELLS[@]} ))' in text
    assert 'export TRAIN_MAX_WORKERS="${cell_workers}"' in text
    assert './scripts/run_phase3_vecstraight_comm_history_factorial.sh "${cell}" "${RUN_KIND}"' in text
    assert '[parallel factorial done] status=ok' in text


def test_zeroaux_reduced_history_wrapper_only_changes_aux_lambdas_and_output_root():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_comm_history_zeroaux_reduced.sh"
    text = script_path.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert 'export SIGN_LAMBDA="${SIGN_LAMBDA:-0.0}"' in text
    assert 'export LIST_LAMBDA="${LIST_LAMBDA:-0.0}"' in text
    assert 'if [[ -n "${IWR_RUN_DIR:-}" ]]; then' in text
    assert 'OUTPUT_BASE="${IWR_RUN_DIR}/outputs/train"' in text
    assert 'OUTPUT_BASE="${REPO_ROOT}/outputs/train"' in text
    assert 'phase3_vecstraight_comm_history_factorial_with_comm_reduced_history_zeroaux_15seeds_${RUN_KIND}_${RUN_DATE}' in text
    assert 'exec ./scripts/run_phase3_vecstraight_comm_history_factorial.sh with_comm_reduced_history "${RUN_KIND}"' in text


def test_iwr_zeroaux_reduced_history_job_uses_staged_venv_and_wrapper():
    script_path = REPO_ROOT / "scripts" / "iwr_jobs" / "phase3_vecstraight_zeroaux_comm_reduced_iwr.sh"
    text = script_path.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert 'export RUN_KIND="${RUN_KIND:-iwr}"' in text
    assert 'export TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-15}"' in text
    assert 'export PYTHON_BIN="${PYTHON_BIN:-${IWR_PROJECT_DIR:-$(pwd)}/.venv/bin/python}"' in text
    assert 'missing python interpreter: ${PYTHON_BIN}' in text
    assert './scripts/run_phase3_vecstraight_comm_history_zeroaux_reduced.sh "${RUN_KIND}"' in text


def test_phase3_seed_expansion_build_job_respects_explicit_comm_lambdas(tmp_path):
    ckpt = tmp_path / "cond1_seed333_ep50000.pt"
    _write_fake_ckpt(ckpt, episode_offset=0, n_episodes=50000)
    manifest = _write_manifest(tmp_path / "cond1_manifest.txt", [ckpt])
    defaults = run_phase3_seed_expansion.parse_args(
        [
            "--out_dir",
            str(tmp_path / "continuation_out"),
            "--init_checkpoint_manifest",
            str(manifest),
            "--init_episode",
            "50000",
            "--conditions",
            "cond1",
            "--seeds",
            "333",
            "--n_episodes",
            "100000",
            "--episode_offset",
            "50000",
            "--schedule_total_episodes",
            "150000",
            "--msg_training_intervention",
            "uniform",
            "--sign_lambda",
            "0.0",
            "--list_lambda",
            "0.0",
        ]
    )
    job = run_phase3_seed_expansion._build_job(defaults, condition="cond1", seed=333)
    cmd = job.cmd

    assert "--msg_training_intervention" in cmd
    assert cmd[cmd.index("--msg_training_intervention") + 1] == "uniform"
    assert "--sign_lambda" in cmd
    assert cmd[cmd.index("--sign_lambda") + 1] == "0.0"
    assert "--list_lambda" in cmd
    assert cmd[cmd.index("--list_lambda") + 1] == "0.0"


def test_lossswitch_control_arm_script_encodes_three_matched_arms():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_lossswitch_control_arm.sh"
    text = script_path.read_text(encoding="utf-8")

    assert 'none_base)' in text
    assert 'MODE="none"' in text
    assert 'SIGN_LAMBDA="0.1"' in text
    assert 'LIST_LAMBDA="0.1"' in text
    assert 'none_zeroaux)' in text
    assert 'uniform_zeroaux)' in text
    assert './scripts/run_phase3_vecstraight_continuation_mode.sh "${BRANCH_EP}" "${MODE}" "${RUN_KIND}"' in text


def test_qx6_lossswitch_batch_script_preserves_vecstraight_base_contract():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_qx6_lossswitch_controls_20260330.sh"
    text = script_path.read_text(encoding="utf-8")

    assert 'BRANCH_EP="${BRANCH_EP:-50000}"' in text
    assert 'ARMS=(none_base none_zeroaux uniform_zeroaux)' in text
    assert 'PARALLEL_ARMS="${PARALLEL_ARMS:-1}"' in text
    assert 'ARM_LOG_DIR="${ARM_LOG_DIR:-${STATUS_DIR}/arm_logs}"' in text
    assert 'launch_arm() {' in text
    assert 'if [[ "${PARALLEL_ARMS}" == "1" ]]; then' in text
    assert 'arm failed arm=${arm} exit=${code}' in text
    assert 'TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-15}"' in text
    assert 'EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-15}"' in text
    assert 'NUM_ENVS="${NUM_ENVS:-8}"' in text
    assert 'ENV_BACKEND="${ENV_BACKEND:-subproc}"' in text
    assert 'ENV_START_METHOD="${ENV_START_METHOD:-spawn}"' in text
    assert 'REGIME_LOG_INTERVAL="${REGIME_LOG_INTERVAL:-400}"' in text
    assert 'CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25000}"' in text
    assert 'MSG_ENTROPY_COEFF="${MSG_ENTROPY_COEFF:-0.01}"' in text
    assert 'MSG_ENTROPY_COEFF_FINAL="${MSG_ENTROPY_COEFF_FINAL:-0.0}"' in text


def test_clean_msgsource_family_script_preserves_zeroaux_vecstraight_contract():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_clean_msgsource_family.sh"
    text = script_path.read_text(encoding="utf-8")

    assert 'learned|fixed0|fixed1|public_random|uniform' in text
    assert 'OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_clean_msgsource_${MODE}_15seeds_${RUN_KIND}_${RUN_DATE}}"' in text
    assert 'TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-auto}"' in text
    assert "detect_default_train_workers()" in text
    assert 'suggested=$(( nproc_val / 2 ))' in text
    assert 'must resolve to a positive integer' in text
    assert '--num_envs "${NUM_ENVS}"' in text
    assert "--count_env_episodes" in text
    assert '--env_backend "${ENV_BACKEND}"' in text
    assert '--env_start_method "${ENV_START_METHOD}"' in text
    assert "--entropy_schedule linear" in text
    assert "--lr_schedule cosine" in text
    assert "--msg_entropy_coeff 0.01" in text
    assert "--msg_entropy_coeff_final 0.0" in text
    assert "--sign_lambda 0.0" in text
    assert "--list_lambda 0.0" in text
    assert '--msg_dropout "${MSG_DROPOUT}"' in text
    assert '--msg_source_mode "${MODE}"' in text
    assert 'msg_training_intervention=none' in text
    assert 'if (( ${#active_pids[@]} == 0 )); then' in text
    assert 'local -a kept=()' in text
    assert 'for pid in "${active_pids[@]}";' in text
    assert 'if (( ${#kept[@]} == 0 )); then' in text
    assert 'active_pids=()' in text
    assert 'active_pids=("${kept[@]}")' in text


def test_clean_msgsource_queue_script_runs_all_modes_on_hetzner():
    script_path = (
        REPO_ROOT / "scripts" / "hetzner_jobs" / "phase3_vecstraight_clean_msgsource_queue_20260410.sh"
    )
    text = script_path.read_text(encoding="utf-8")

    assert 'RUN_DATE="${RUN_DATE:-20260410}"' in text
    assert 'HETZNER_PROJECT_DIR="${HETZNER_PROJECT_DIR:-/root/compute-work/projects/dsc-epgg-vectorized}"' in text
    assert 'BASELINE_MANIFEST="${BASELINE_MANIFEST:-${NEXT_ROOT}/manifests/cond2_all_15seeds.txt}"' in text
    assert "learned" in text
    assert "public_random" in text
    assert "uniform" in text
    assert "fixed0" in text
    assert "fixed1" in text
    assert 'TRAIN_MAX_WORKERS="${TRAIN_MAX_WORKERS:-auto}"' in text
    assert './scripts/run_phase3_vecstraight_clean_msgsource_family.sh "${mode}" hetzner' in text
    assert 'RUN_DATE="${RUN_DATE}" "${HETZNER_PROJECT_DIR}/scripts/run_phase3_vecstraight_setup.sh"' in text
    assert 'log "all clean msg_source families complete"' in text


def test_vecstraight_setup_script_is_bash_portable():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_setup.sh"
    text = script_path.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert "/bin/zsh -lc" not in text
    assert '"${PYTHON_BIN}" -m src.analysis.prepare_phase3_vecstraight_manifests --run_date "${RUN_DATE}"' in text


def test_zeroaux_manifest_prepare_script_invokes_python_entrypoint():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_prepare_zeroaux_manifests.sh"
    text = script_path.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert 'RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"' in text
    assert '"${PYTHON_BIN}" -m src.analysis.prepare_phase3_zeroaux_manifests --run_date "${RUN_DATE}"' in text


def test_zeroaux_manifest_builder_tracks_sizes_and_sampled_run_json_contract():
    script_path = REPO_ROOT / "src" / "analysis" / "prepare_phase3_zeroaux_manifests.py"
    text = script_path.read_text(encoding="utf-8")

    assert "phase3_vecstraight_clean_msgsource_learned_15seeds_hetzner_20260410" in text
    assert "phase3_vecstraight_comm_history_factorial_without_comm_full_history_15seeds_hetzner_20260330par24" in text
    assert '"size_bytes": path.stat().st_size if path.exists() else None' in text
    assert '"sample_run_json": {' in text
    assert '"sign_lambda": config.get("sign_lambda")' in text
    assert '"list_lambda": config.get("list_lambda")' in text
    assert '"comm_enabled": config.get("comm_enabled")' in text
    assert "agent.can_send" in text
    assert "functionally identical to a zero-aux no-comm training path" in text


def test_frozen_suite_expanded_runner_supports_manifest_and_label_overrides():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_frozen_suite_expanded.sh"
    text = script_path.read_text(encoding="utf-8")

    assert 'COMM_MANIFEST="${COMM_MANIFEST:-${NEXT_ROOT}/manifests/cond1_all_15seeds.txt}"' in text
    assert 'BASELINE_MANIFEST="${BASELINE_MANIFEST:-${NEXT_ROOT}/manifests/cond2_all_15seeds.txt}"' in text
    assert 'OUT_LABEL_PREFIX="${OUT_LABEL_PREFIX:-phase3_vecstraight}"' in text
    assert 'RUN_KIND_LABEL="${RUN_KIND_LABEL:-${RUN_KIND}}"' in text
    assert 'OUT_ROOT="${REPO_ROOT}/outputs/eval/${OUT_LABEL_PREFIX}_${label}_15seeds_${RUN_KIND_LABEL}_${RUN_DATE}"' in text
    assert '--eval_seed "${EVAL_SEED}"' in text


def test_sender_causal_runner_supports_manifest_and_label_overrides():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_sender_causal.sh"
    text = script_path.read_text(encoding="utf-8")

    assert 'COMM_MANIFEST="${COMM_MANIFEST:-${NEXT_ROOT}/manifests/cond1_all_15seeds.txt}"' in text
    assert 'OUT_LABEL_PREFIX="${OUT_LABEL_PREFIX:-phase3_vecstraight}"' in text
    assert 'RUN_KIND_LABEL="${RUN_KIND_LABEL:-${RUN_KIND}}"' in text
    assert 'OUT_ROOT="${REPO_ROOT}/outputs/eval/${OUT_LABEL_PREFIX}_sender_causal_150k_15seeds_${RUN_KIND_LABEL}_${RUN_DATE}"' in text
    assert '--eval_seed "${EVAL_SEED}"' in text


def test_noise_sweep_runner_supports_manifest_and_label_overrides():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_noise_sweep_eval.sh"
    text = script_path.read_text(encoding="utf-8")

    assert 'COMM_MANIFEST="${COMM_MANIFEST:-${NEXT_ROOT}/manifests/cond1_all_15seeds.txt}"' in text
    assert 'BASELINE_MANIFEST="${BASELINE_MANIFEST:-${NEXT_ROOT}/manifests/cond2_all_15seeds.txt}"' in text
    assert 'OUT_LABEL_PREFIX="${OUT_LABEL_PREFIX:-phase3_vecstraight}"' in text
    assert 'RUN_KIND_LABEL="${RUN_KIND_LABEL:-${RUN_KIND}}"' in text
    assert 'out_root="${REPO_ROOT}/outputs/eval/${OUT_LABEL_PREFIX}_noise_sweep_${obs_mode}_${sigma_label}_${MILESTONE}_15seeds_${RUN_KIND_LABEL}_${RUN_DATE}"' in text
    assert '--eval_seed "${EVAL_SEED}" \\' in text


def test_zeroaux_endpoint_bundle_runs_frozen_sender_causal_and_lowdim():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_zeroaux_endpoint_bundle.sh"
    text = script_path.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert 'COMM_MANIFEST="${COMM_MANIFEST:-${MANIFEST_ROOT}/manifests/cond1_zeroaux_clean_msgsource_learned_full_history_all_15seeds_25k_50k_100k_150k.txt}"' in text
    assert 'BASELINE_MANIFEST="${BASELINE_MANIFEST:-${MANIFEST_ROOT}/manifests/cond2_zeroaux_nocomm_full_history_all_15seeds_25k_50k_100k_150k.txt}"' in text
    assert 'echo "missing zero-aux manifests" >&2' in text
    assert 'RUN_DATE=${RUN_DATE} ./scripts/run_phase3_vecstraight_prepare_zeroaux_manifests.sh' in text
    assert './scripts/run_phase3_vecstraight_frozen_suite_expanded.sh "${RUN_KIND}"' in text
    assert './scripts/run_phase3_vecstraight_sender_causal.sh' in text
    assert '"${PYTHON_BIN}" -m src.analysis.summarize_phase3_lowdim_mechanism \\' in text
    assert '--trace_csv "${TRACE_CSV}" \\' in text
    assert '--checkpoint_episode 150000 \\' in text
    assert '--suite_kind comm' in text


def test_zeroaux_cross_seed_transfer_wrapper_uses_manifest_reference_and_sender_semantics():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_zeroaux_cross_seed_transfer.sh"
    text = script_path.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert 'COMM_MANIFEST="${COMM_MANIFEST:-${MANIFEST_ROOT}/manifests/cond1_zeroaux_clean_msgsource_learned_full_history_all_15seeds_25k_50k_100k_150k.txt}"' in text
    assert 'REFERENCE_MAIN_CSV="${REFERENCE_MAIN_CSV:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_zeroaux_frozen150k_expanded_15seeds_local_${RUN_DATE}/suite/checkpoint_suite_main.csv}"' in text
    assert 'SENDER_SEMANTICS_CSV="${SENDER_SEMANTICS_CSV:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_zeroaux_frozen150k_expanded_15seeds_local_${RUN_DATE}/suite/checkpoint_suite_sender_semantics.csv}"' in text
    assert 'OUT_ROOT="${REPO_ROOT}/outputs/eval/${OUT_LABEL_PREFIX}_cross_seed_transfer_flip15seeds_matched_${RUN_KIND_LABEL}_${RUN_DATE}"' in text
    assert "from src.analysis.checkpoint_artifacts import resolve_manifest_checkpoint_path" in text
    assert 'SKIP_EXISTING="${SKIP_EXISTING:-1}"' in text
    assert 'INPUT_CHECKPOINTS="${INPUTS_OUT}/checkpoints.txt"' in text
    assert '--checkpoint_manifest "${INPUT_CHECKPOINTS}" \\' in text
    assert '--episode "${EPISODE}" \\' in text
    assert '--alignment_mode "${ALIGNMENT_MODE}" \\' in text
    assert '--eval_seed "${EVAL_SEED}" \\' in text
    assert '--reference_main_csv "${REFERENCE_MAIN_CSV}" \\' in text
    assert '--sender_semantics_csv "${SENDER_SEMANTICS_CSV}" \\' in text


def test_zeroaux_sender_encoding_wrapper_runs_natural_trace_and_summary():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_zeroaux_sender_encoding.sh"
    text = script_path.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert 'COMM_MANIFEST="${COMM_MANIFEST:-${MANIFEST_ROOT}/manifests/cond1_zeroaux_clean_msgsource_learned_full_history_all_15seeds_25k_50k_100k_150k.txt}"' in text
    assert 'OUT_ROOT="${REPO_ROOT}/outputs/eval/${OUT_LABEL_PREFIX}_frozen150k_natural_intended_15seeds_${RUN_KIND_LABEL}_${RUN_DATE}"' in text
    assert '--baseline_condition "" \\' in text
    assert '--milestones "${EPISODE}" \\' in text
    assert '--interventions none \\' in text
    assert '--eval_seed "${EVAL_SEED}" \\' in text
    assert '"${PYTHON_BIN}" -m src.analysis.summarize_phase3_sender_encoding \\' in text
    assert '--trace_csv "${SUITE_OUT}/checkpoint_suite_trace.csv" \\' in text
    assert '--checkpoint_episode "${EPISODE}" \\' in text
    assert '--eval_policy greedy \\' in text
    assert '--suite_kind comm' in text


def test_zeroaux_message_history_grid_uses_zeroaux_manifests_and_history_matrix():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_zeroaux_message_history_grid.sh"
    text = script_path.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert 'COMM_MANIFEST="${COMM_MANIFEST:-${MANIFEST_ROOT}/manifests/cond1_zeroaux_clean_msgsource_learned_full_history_all_15seeds_25k_50k_100k_150k.txt}"' in text
    assert 'BASELINE_MANIFEST="${BASELINE_MANIFEST:-${MANIFEST_ROOT}/manifests/cond2_zeroaux_nocomm_full_history_all_15seeds_25k_50k_100k_150k.txt}"' in text
    assert "phase3_vecstraight_message_history_grid_150000_15seeds_local_20260327" not in text
    assert 'OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/eval/${OUT_LABEL_PREFIX}_message_history_grid_${MILESTONE}_15seeds_${RUN_KIND_LABEL}_${RUN_DATE}}"' in text
    assert "INTERVENTIONS=(none zeros marginal fixed0 fixed1 indep_random public_random sender_shuffle permute_slots)" in text
    assert "HISTORYS=(none zero_temporal clamp_temporal_high clamp_temporal_low zero_last_coop zero_last_action zero_ewma clamp_ewma_high)" in text
    assert '--history_interventions "${HISTORYS[@]}" \\' in text
    assert '"${PYTHON_BIN}" -m src.analysis.summarize_phase3_history_audit \\' in text
    assert "message_history_grid_meta.txt" in text


def test_iwr_zeroaux_message_history_grid_job_uses_run_dir_inputs_and_project_script():
    script_path = REPO_ROOT / "scripts" / "iwr_jobs" / "phase3_vecstraight_zeroaux_message_history_grid_iwr.sh"
    text = script_path.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert ': "${IWR_RUN_DIR:?IWR_RUN_DIR must be set by the IWR launcher}"' in text
    assert ': "${IWR_PROJECT_DIR:?IWR_PROJECT_DIR must be set by the IWR launcher}"' in text
    assert 'INPUT_ROOT="${INPUT_ROOT:-${IWR_RUN_DIR}/inputs/zeroaux_message_history_grid}"' in text
    assert 'COMM_CKPT_ROOT="${COMM_CKPT_ROOT:-${INPUT_ROOT}/cond1}"' in text
    assert 'BASELINE_CKPT_ROOT="${BASELINE_CKPT_ROOT:-${INPUT_ROOT}/cond2}"' in text
    assert "export COMM_CKPT_ROOT" in text
    assert "export BASELINE_CKPT_ROOT" in text
    assert 'OUT_ROOT="${OUT_ROOT:-${IWR_RUN_DIR}/outputs/eval/phase3_vecstraight_zeroaux_message_history_grid_150000_15seeds_iwr_${RUN_DATE}}"' in text
    assert 'MAX_WORKERS="${MAX_WORKERS:-24}"' in text
    assert './scripts/run_phase3_vecstraight_zeroaux_message_history_grid.sh iwr' in text


def test_recommended_smoke_script_targets_subset_not_full_matrix():
    script_path = REPO_ROOT / "scripts" / "run_phase3_vecstraight_recommended_smoke.sh"
    text = script_path.read_text(encoding="utf-8")

    assert 'OBS_MODES_STR="private public_noisy public_exact"' in text
    assert 'SIGMA_VALUES_STR="0.5"' in text
    assert 'SEEDS_STR="101"' in text
    assert 'SEEDS_STR="101 202"' in text
    assert 'N_EPISODES="${N_EPISODES:-16}"' in text
    assert 'CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-8}"' in text
