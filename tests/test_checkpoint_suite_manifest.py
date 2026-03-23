import json
from pathlib import Path

import pytest

from src.analysis.checkpoint_suite_manifest import (
    summarize_checkpoint_suite_manifest,
    validate_checkpoint_suite_manifest,
)


def _write_manifest(tmp_path: Path, tasks: list[dict]) -> Path:
    manifest = tmp_path / "checkpoint_suite_manifest.json"
    manifest.write_text(json.dumps(tasks), encoding="utf-8")
    return manifest


def test_summarize_checkpoint_suite_manifest_extracts_design(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "name": "cond1_seed101_ep50000_none",
                "checkpoint": "outputs/train/cond1_seed101.pt",
                "episode": 50000,
                "intervention": "none",
            },
            {
                "name": "cond1_seed202_ep150000_public_random",
                "checkpoint": "outputs/train/cond1_seed202.pt",
                "episode": 150000,
                "intervention": "public_random",
            },
        ],
    )

    summary = summarize_checkpoint_suite_manifest(manifest)

    assert summary.task_count == 2
    assert summary.seeds == (101, 202)
    assert summary.episodes == (50000, 150000)
    assert summary.interventions == ("none", "public_random")


def test_validate_checkpoint_suite_manifest_rejects_mismatched_seed_set(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "name": "cond1_seed101_ep50000_none",
                "checkpoint": "outputs/train/cond1_seed101.pt",
                "episode": 50000,
                "intervention": "none",
            }
        ],
    )

    with pytest.raises(ValueError, match="seed set mismatch"):
        validate_checkpoint_suite_manifest(
            manifest,
            expected_seeds=[101, 202],
            expected_episodes=[50000],
            expected_interventions=["none"],
        )


def test_validate_checkpoint_suite_manifest_accepts_exact_design(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "name": "cond1_seed101_ep50000_none",
                "checkpoint": "outputs/train/cond1_seed101.pt",
                "episode": 50000,
                "intervention": "none",
            },
            {
                "name": "cond1_seed202_ep150000_public_random",
                "checkpoint": "outputs/train/cond1_seed202.pt",
                "episode": 150000,
                "intervention": "public_random",
            },
        ],
    )

    summary = validate_checkpoint_suite_manifest(
        manifest,
        expected_seeds=[101, 202],
        expected_episodes=[50000, 150000],
        expected_interventions=["none", "public_random"],
    )

    assert summary.task_count == 2
