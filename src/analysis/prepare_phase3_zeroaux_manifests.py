from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, Iterable, List

from src.analysis.checkpoint_artifacts import (
    atomic_write_json,
    atomic_write_text,
    infer_manifest_absolute_milestones,
    resolve_manifest_checkpoint_path,
    scan_checkpoint_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1111, 1212, 1313, 1414, 1515, 1616]
EPISODES = [25000, 50000, 100000, 150000]

COND1_ROOT = (
    Path("/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/hetzner-results")
    / "phase3_vecstraight_clean_msgsource_learned_15seeds_hetzner_20260410"
    / "train"
)
COND2_ROOT = (
    Path("/Users/mbp17/POSTDOC/NPS26/dsc-epgg-vectorized/hetzner-results")
    / "phase3_vecstraight_comm_history_factorial_without_comm_full_history_15seeds_hetzner_20260330par24"
    / "train"
)


def _run_root(run_date: str) -> Path:
    return REPO_ROOT / f"outputs/eval/phase3_vecstraight_zeroaux_manifests_{run_date}"


def _checkpoint_path(root: Path, condition: str, seed: int, episode: int) -> Path:
    if int(episode) == 150000:
        return root / f"{condition}_seed{int(seed)}.pt"
    return root / f"{condition}_seed{int(seed)}_ep{int(episode)}.pt"


def _collect_paths(root: Path, condition: str) -> list[str]:
    paths: list[str] = []
    for seed in SEEDS:
        for episode in EPISODES:
            path = _checkpoint_path(root, condition, seed, episode)
            if not path.exists():
                raise FileNotFoundError(path)
            paths.append(str(path.resolve()))
    return paths


def _write_manifest(path: Path, checkpoint_paths: Iterable[str]) -> None:
    atomic_write_text(path, "\n".join(str(p) for p in checkpoint_paths) + "\n")


def _sample_run_json_payload(root: Path, condition: str, seed: int, episode: int) -> Dict:
    checkpoint = _checkpoint_path(root, condition, seed, episode)
    run_json = checkpoint.with_suffix(".run.json")
    if not run_json.exists():
        raise FileNotFoundError(run_json)
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    return {
        "checkpoint": str(checkpoint.resolve()),
        "run_json": str(run_json.resolve()),
        "seed": int(seed),
        "episode": int(episode),
        "sign_lambda": config.get("sign_lambda"),
        "list_lambda": config.get("list_lambda"),
        "comm_enabled": config.get("comm_enabled"),
        "history_mode": config.get("history_mode"),
        "disable_comm_fallback": config.get("disable_comm_fallback"),
    }


def _path_metadata(paths: Iterable[str]) -> list[Dict]:
    out: list[Dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        out.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    return out


def _provenance_text(run_root: Path, cond1_manifest: Path, cond2_manifest: Path) -> str:
    return "\n".join(
        [
            "# Zero-Aux Manifest Provenance",
            "",
            f"- run_root: `{run_root}`",
            f"- cond1_manifest: `{cond1_manifest}`",
            f"- cond2_manifest: `{cond2_manifest}`",
            f"- cond1_root: `{COND1_ROOT}`",
            f"- cond2_root: `{COND2_ROOT}`",
            "- cond1_semantics: canonical zero-aux full-history communication source for the paper path",
            "- cond2_semantics: canonical no-comm full-history baseline for the paper path",
            "",
            "## Why cond2 is a valid zero-aux substitute",
            "",
            "- `cond2` uses the factorial `without_comm_full_history` batch, whose config still contains nominal sign/list lambda values.",
            "- Its sampled `.run.json` should show `comm_enabled=false` and no sender pathway.",
            "- In `src/algos/PPO.py`, the sender-specific auxiliary path is gated by `agent.can_send` before message loss and message-sign loss are computed (see the guarded block around the `agent.can_send` check and the subsequent `sign_lambda` / `list_lambda` additions).",
            "- Therefore the no-comm factorial arm is functionally identical to a zero-aux no-comm training path even if the stored config still records nominal lambda values.",
            "- This manifest bundle records sampled `.run.json` fields so later agents can verify the no-comm contract directly before reusing the checkpoints.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_date", type=str, default=dt.date.today().strftime("%Y%m%d"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = _run_root(str(args.run_date))
    manifests_dir = run_root / "manifests"
    status_dir = run_root / "status"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    cond1_manifest = manifests_dir / "cond1_zeroaux_clean_msgsource_learned_full_history_all_15seeds_25k_50k_100k_150k.txt"
    cond2_manifest = manifests_dir / "cond2_zeroaux_nocomm_full_history_all_15seeds_25k_50k_100k_150k.txt"

    cond1_paths = _collect_paths(COND1_ROOT, "cond1")
    cond2_paths = _collect_paths(COND2_ROOT, "cond2")
    _write_manifest(cond1_manifest, cond1_paths)
    _write_manifest(cond2_manifest, cond2_paths)

    cond1_scan = scan_checkpoint_manifest(cond1_manifest, condition="cond1")
    cond2_scan = scan_checkpoint_manifest(cond2_manifest, condition="cond2")
    cond1_milestones = infer_manifest_absolute_milestones(cond1_manifest, condition="cond1")
    cond2_milestones = infer_manifest_absolute_milestones(cond2_manifest, condition="cond2")

    for seed in SEEDS:
        for episode in EPISODES:
            resolve_manifest_checkpoint_path(cond1_manifest, "cond1", seed, episode)
            resolve_manifest_checkpoint_path(cond2_manifest, "cond2", seed, episode)

    sanity_payload = {
        "run_root": str(run_root),
        "cond1_manifest": str(cond1_manifest),
        "cond2_manifest": str(cond2_manifest),
        "cond1_count": len(cond1_scan),
        "cond2_count": len(cond2_scan),
        "cond1_milestones": cond1_milestones,
        "cond2_milestones": cond2_milestones,
        "seeds": SEEDS,
        "episodes": EPISODES,
        "cond1_root": str(COND1_ROOT),
        "cond2_root": str(COND2_ROOT),
        "path_metadata": {
            "cond1": _path_metadata(cond1_paths),
            "cond2": _path_metadata(cond2_paths),
        },
        "sample_run_json": {
            "cond1": _sample_run_json_payload(COND1_ROOT, "cond1", 101, 150000),
            "cond2": _sample_run_json_payload(COND2_ROOT, "cond2", 101, 150000),
        },
    }
    atomic_write_json(status_dir / "manifest_sanity.json", sanity_payload)
    atomic_write_text(status_dir / "PROVENANCE.md", _provenance_text(run_root, cond1_manifest, cond2_manifest))

    print(f"[zeroaux-manifests] run_root={run_root}")
    print(f"[zeroaux-manifests] cond1_manifest={cond1_manifest}")
    print(f"[zeroaux-manifests] cond2_manifest={cond2_manifest}")
    print(f"[zeroaux-manifests] cond1_count={len(cond1_scan)} milestones={cond1_milestones}")
    print(f"[zeroaux-manifests] cond2_count={len(cond2_scan)} milestones={cond2_milestones}")


if __name__ == "__main__":
    main()
