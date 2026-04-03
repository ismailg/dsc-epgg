from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Iterable

from src.analysis.checkpoint_artifacts import (
    atomic_write_json,
    atomic_write_text,
    infer_manifest_absolute_milestones,
    resolve_manifest_checkpoint_path,
    scan_checkpoint_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1111, 1212, 1313, 1414, 1515, 1616]
EPISODES = [50000, 100000, 150000]

COND1_BATCH_ROOT = (
    REPO_ROOT
    / "iwr-results/phase3-150k-cond1-15seed-trainonly-20260323/outputs/phase3_cond1_15seeds_train_only/train"
)
COND1_STANDALONE_ROOT = (
    REPO_ROOT
    / "iwr-results/phase3-150k-straight-c1-s101-subproc-20260323/outputs/phase3_straight_c1_s101_subproc/checkpoints"
)
COND2_ROOT = (
    REPO_ROOT
    / "iwr-results/phase3-150k-cond2-15seed-trainonly-20260324/code/outputs/train/"
    "phase3-150k-cond2-15seed-trainonly-20260324/outputs/phase3_cond2_15seeds_train_only/train"
)


def _run_root(run_date: str) -> Path:
    return REPO_ROOT / f"outputs/eval/phase3_vecstraight_next_steps_{run_date}"


def _checkpoint_path(root: Path, condition: str, seed: int, episode: int) -> Path:
    if int(episode) == 150000:
        return root / f"{condition}_seed{int(seed)}.pt"
    return root / f"{condition}_seed{int(seed)}_ep{int(episode)}.pt"


def _cond1_paths() -> list[str]:
    paths: list[str] = []
    for seed in SEEDS:
        root = COND1_STANDALONE_ROOT if int(seed) == 101 else COND1_BATCH_ROOT
        for episode in EPISODES:
            path = _checkpoint_path(root, "cond1", seed, episode)
            if not path.exists():
                raise FileNotFoundError(path)
            paths.append(str(path.resolve()))
    return paths


def _cond2_paths() -> list[str]:
    paths: list[str] = []
    for seed in SEEDS:
        for episode in EPISODES:
            path = _checkpoint_path(COND2_ROOT, "cond2", seed, episode)
            if not path.exists():
                raise FileNotFoundError(path)
            paths.append(str(path.resolve()))
    return paths


def _write_manifest(path: Path, checkpoint_paths: Iterable[str]) -> None:
    atomic_write_text(path, "\n".join(str(p) for p in checkpoint_paths) + "\n")


def _tracker_text(run_root: Path, cond1_manifest: Path, cond2_manifest: Path) -> str:
    return "\n".join(
        [
            "# Phase-3 Vecstraight Next Steps Tracker",
            "",
            f"- run_root: `{run_root}`",
            "- tracking_rule: update this file after each stage completes or fails",
            "",
            "## Stage 0A",
            "- status: done",
            f"- cond1_manifest: `{cond1_manifest}`",
            f"- cond2_manifest: `{cond2_manifest}`",
            "- validation: 15 seeds x 3 milestones resolved for cond1 and cond2",
            "- next_action: Stage 0B evaluation parity",
            "",
            "## Stage 0B",
            "- status: pending",
            "- inputs: manifest-driven cond1/cond2 checkpoint resolution plus evaluation parity modules",
            "- command: pending",
            "- output_root: pending",
            "- validation: targeted tests + smoke suite + smoke sender-causal",
            "- next_action: Stage 1 frozen 50k",
            "",
            "## Stage 1",
            "- status: pending",
            "- command: pending",
            "- output_root: pending",
            "- validation: suite outputs + summary note",
            "- next_action: Stage 2 frozen 150k",
            "",
            "## Stage 2",
            "- status: pending",
            "- command: pending",
            "- output_root: pending",
            "- validation: suite outputs + learned-vs-controls summary",
            "- next_action: Stage 3 sender-causal 150k",
            "",
            "## Stage 3",
            "- status: pending",
            "- command: pending",
            "- output_root: pending",
            "- validation: sender_causal_matrix.csv + summary markdown",
            "- next_action: Stage 4 trainer sender_shuffle parity",
            "",
            "## Stage 4",
            "- status: pending",
            "- command: pending",
            "- output_root: pending",
            "- validation: trainer tests + continuation smoke",
            "- next_action: Stage 5 same-checkpoint continuations from 50k",
            "",
            "## Stage 5",
            "- status: pending",
            "- command: pending",
            "- output_root: pending",
            "- validation: validated suite outputs for fixed0/uniform/public_random/sender_shuffle",
            "- next_action: Stage 6 same-checkpoint continuations from 100k",
            "",
            "## Stage 6",
            "- status: pending",
            "- command: pending",
            "- output_root: pending",
            "- validation: validated suite outputs for sender_shuffle and fixed0",
            "- next_action: Stage 7 history audit",
            "",
            "## Stage 7",
            "- status: pending",
            "- command: pending",
            "- output_root: pending",
            "- validation: history audit summary",
            "- next_action: done",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_date",
        type=str,
        default=dt.date.today().strftime("%Y%m%d"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = _run_root(str(args.run_date))
    manifests_dir = run_root / "manifests"
    status_dir = run_root / "status"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    cond1_manifest = manifests_dir / "cond1_all_15seeds.txt"
    cond2_manifest = manifests_dir / "cond2_all_15seeds.txt"
    _write_manifest(cond1_manifest, _cond1_paths())
    _write_manifest(cond2_manifest, _cond2_paths())

    cond1_scan = scan_checkpoint_manifest(cond1_manifest, condition="cond1")
    cond2_scan = scan_checkpoint_manifest(cond2_manifest, condition="cond2")
    cond1_milestones = infer_manifest_absolute_milestones(cond1_manifest, condition="cond1")
    cond2_milestones = infer_manifest_absolute_milestones(cond2_manifest, condition="cond2")

    resolve_manifest_checkpoint_path(cond1_manifest, "cond1", 101, 50000)
    resolve_manifest_checkpoint_path(cond1_manifest, "cond1", 101, 100000)
    resolve_manifest_checkpoint_path(cond1_manifest, "cond1", 101, 150000)
    for seed in [202, 303, 404, 505, 606, 707, 808, 909, 1111, 1212, 1313, 1414, 1515, 1616]:
        for episode in EPISODES:
            resolve_manifest_checkpoint_path(cond1_manifest, "cond1", seed, episode)
    for seed in SEEDS:
        for episode in EPISODES:
            resolve_manifest_checkpoint_path(cond2_manifest, "cond2", seed, episode)

    sanity_payload = {
        "run_root": str(run_root),
        "cond1_manifest": str(cond1_manifest),
        "cond2_manifest": str(cond2_manifest),
        "cond1_task_count": len(cond1_scan),
        "cond2_task_count": len(cond2_scan),
        "cond1_seeds": sorted({item.seed for item in cond1_scan}),
        "cond2_seeds": sorted({item.seed for item in cond2_scan}),
        "cond1_milestones": cond1_milestones,
        "cond2_milestones": cond2_milestones,
        "cond1_seed101_root": str(COND1_STANDALONE_ROOT),
        "cond1_batch_root": str(COND1_BATCH_ROOT),
        "cond2_root": str(COND2_ROOT),
    }
    atomic_write_json(status_dir / "manifest_sanity.json", sanity_payload)
    tracker_path = status_dir / "TRACKER.md"
    if not tracker_path.exists():
        atomic_write_text(tracker_path, _tracker_text(run_root, cond1_manifest, cond2_manifest))

    print(f"[vecstraight-manifests] run_root={run_root}")
    print(f"[vecstraight-manifests] cond1_manifest={cond1_manifest}")
    print(f"[vecstraight-manifests] cond2_manifest={cond2_manifest}")
    print(f"[vecstraight-manifests] cond1_count={len(cond1_scan)} milestones={cond1_milestones}")
    print(f"[vecstraight-manifests] cond2_count={len(cond2_scan)} milestones={cond2_milestones}")


if __name__ == "__main__":
    main()
