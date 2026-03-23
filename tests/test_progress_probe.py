import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "probe_phase3_progress.py"


def _run_probe(*args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout.strip())


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_probe_seed_expansion_from_launcher_metadata(tmp_path: Path):
    out_dir = tmp_path / "train"
    _write(out_dir / "metrics" / "cond1_seed111.jsonl", '{"episode": 80}\n')
    _write(out_dir / "cond2_seed111.pt", "done")

    meta = {
        "cwd": str(REPO_ROOT),
        "cmd": [
            sys.executable,
            "-m",
            "src.experiments_pgg_v0.run_phase3_seed_expansion",
            "--out_dir",
            str(out_dir),
            "--conditions",
            "cond1",
            "cond2",
            "--seeds",
            "111",
            "--n_episodes",
            "100",
            "--episode_offset",
            "50",
        ],
    }
    meta_path = tmp_path / "seedexp_job.json"
    _write(meta_path, json.dumps(meta))

    result = _run_probe("from-launcher-metadata", "--metadata-json", str(meta_path))
    assert result["kind"] == "seed-expansion"
    assert result["units"] == "episodes"
    assert result["current"] == 130.0
    assert result["total"] == 200.0
    assert result["pct"] == 65.0
    assert result["jobs_complete"] == 1


def test_probe_checkpoint_suite_from_launcher_metadata(tmp_path: Path):
    out_dir = tmp_path / "suite"
    raw_dir = out_dir / "raw"
    base = "cond1_seed111_ep50000_none"
    for suffix in [
        ".csv",
        "_condition.csv",
        "_comm.csv",
        "_posterior_strat.csv",
        "_trace.csv",
        "_sender_semantics.csv",
        "_receiver_semantics.csv",
    ]:
        _write(raw_dir / f"{base}{suffix}", "x")

    zeros = "cond1_seed111_ep50000_zeros"
    for suffix in [".csv", "_condition.csv"]:
        _write(raw_dir / f"{zeros}{suffix}", "x")

    baseline = "cond2_seed111_ep50000_none"
    for suffix in [".csv", "_condition.csv", "_comm.csv", "_posterior_strat.csv"]:
        _write(raw_dir / f"{baseline}{suffix}", "x")

    meta = {
        "cwd": str(REPO_ROOT),
        "cmd": [
            sys.executable,
            "-m",
            "src.analysis.run_phase3_checkpoint_suite",
            "--out_dir",
            str(out_dir),
            "--comm_condition",
            "cond1",
            "--baseline_condition",
            "cond2",
            "--seeds",
            "111",
            "--milestones",
            "50000",
            "--interventions",
            "none",
            "zeros",
        ],
    }
    meta_path = tmp_path / "suite_job.json"
    _write(meta_path, json.dumps(meta))

    result = _run_probe("from-launcher-metadata", "--metadata-json", str(meta_path))
    assert result["kind"] == "checkpoint-suite"
    assert result["tasks_total"] == 3
    assert result["tasks_complete"] == 2
    assert result["tasks_started"] == 3
    assert abs(result["current"] - (2.0 + 2.0 / 3.0)) < 1e-9
    assert abs(result["pct"] - (100.0 * (2.0 + 2.0 / 3.0) / 3.0)) < 1e-9


def test_probe_trimmed_eval_from_launcher_metadata(tmp_path: Path):
    suite_out = tmp_path / "suite"
    crossplay_out = tmp_path / "crossplay"
    suite_raw = suite_out / "raw"
    cross_raw = crossplay_out / "raw"

    base = "cond1_seed111_ep50000_none"
    for suffix in [
        ".csv",
        "_condition.csv",
        "_comm.csv",
        "_posterior_strat.csv",
        "_trace.csv",
        "_sender_semantics.csv",
        "_receiver_semantics.csv",
    ]:
        _write(suite_raw / f"{base}{suffix}", "x")

    complete_cross = "cond1_seed111_sender50000_receiver150000"
    for suffix in [".csv", "_condition.csv", "_comm.csv"]:
        _write(cross_raw / f"{complete_cross}{suffix}", "x")

    partial_cross = "cond1_seed111_sender100000_receiver150000"
    _write(cross_raw / f"{partial_cross}.csv", "x")

    meta = {
        "cwd": str(REPO_ROOT),
        "cmd": [
            sys.executable,
            "-m",
            "src.analysis.run_phase3_trimmed_eval",
            "--suite_out_dir",
            str(suite_out),
            "--crossplay_out_dir",
            str(crossplay_out),
            "--comm_condition",
            "cond1",
            "--baseline_condition",
            "",
            "--seeds",
            "111",
            "--milestones",
            "50000",
            "--interventions",
            "none",
            "zeros",
            "--crossplay_sender_milestones",
            "50000",
            "100000",
            "--crossplay_receiver_milestones",
            "150000",
        ],
    }
    meta_path = tmp_path / "trimmed_job.json"
    _write(meta_path, json.dumps(meta))

    result = _run_probe("from-launcher-metadata", "--metadata-json", str(meta_path))
    assert result["kind"] == "trimmed-eval"
    assert abs(result["current"] - (1.0 + 1.0 + 1.0 / 3.0)) < 1e-9
    assert result["total"] == 4.0
    assert abs(result["pct"] - (100.0 * (2.0 + 1.0 / 3.0) / 4.0)) < 1e-9
    assert result["suite"]["tasks_total"] == 2
    assert result["crossplay"]["tasks_total"] == 2
