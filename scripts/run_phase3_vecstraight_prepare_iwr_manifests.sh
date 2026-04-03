#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_DATE="${1:-${RUN_DATE:-$(date +%Y%m%d)}}"
NEXT_ROOT="${NEXT_ROOT:-${REPO_ROOT}/outputs/eval/phase3_vecstraight_next_steps_${RUN_DATE}}"
MANIFEST_DIR="${MANIFEST_DIR:-${NEXT_ROOT}/manifests_iwr}"
COMM_MANIFEST="${COMM_MANIFEST:-${MANIFEST_DIR}/cond1_all_15seeds_iwr.txt}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${MANIFEST_DIR}/cond2_all_15seeds_iwr.txt}"

export COMM_MANIFEST
export BASELINE_MANIFEST
export COND1_BATCH_ROOT="${COND1_BATCH_ROOT:-/export/scratch/iguennou/runs/dsc-epgg-vectorized/phase3-150k-cond1-15seed-trainonly-20260323/outputs/phase3_cond1_15seeds_train_only/train}"
export COND1_STANDALONE_ROOT="${COND1_STANDALONE_ROOT:-/export/scratch/iguennou/runs/dsc-epgg-vectorized/phase3-150k-straight-c1-s101-subproc-20260323/outputs/phase3_straight_c1_s101_subproc/checkpoints}"
export COND2_ROOT="${COND2_ROOT:-/export/scratch/iguennou/runs/dsc-epgg-vectorized/phase3-150k-cond2-15seed-trainonly-20260324/code/outputs/train/phase3-150k-cond2-15seed-trainonly-20260324/outputs/phase3_cond2_15seeds_train_only/train}"

mkdir -p "${MANIFEST_DIR}"

python3 - <<'PYEOF'
import os
from pathlib import Path

seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1111, 1212, 1313, 1414, 1515, 1616]
episodes = [50000, 100000, 150000]


def checkpoint_path(root: Path, condition: str, seed: int, episode: int) -> Path:
    if int(episode) == 150000:
        return root / f"{condition}_seed{int(seed)}.pt"
    return root / f"{condition}_seed{int(seed)}_ep{int(episode)}.pt"


def write_manifest(path_str: str, items: list[str]) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(items) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


cond1_batch_root = Path(os.environ["COND1_BATCH_ROOT"])
cond1_standalone_root = Path(os.environ["COND1_STANDALONE_ROOT"])
cond2_root = Path(os.environ["COND2_ROOT"])
comm_manifest = os.environ["COMM_MANIFEST"]
baseline_manifest = os.environ["BASELINE_MANIFEST"]

cond1_paths = []
for seed in seeds:
    root = cond1_standalone_root if int(seed) == 101 else cond1_batch_root
    for episode in episodes:
        path = checkpoint_path(root, "cond1", seed, episode)
        if not path.exists():
            raise FileNotFoundError(path)
        cond1_paths.append(str(path))

cond2_paths = []
for seed in seeds:
    for episode in episodes:
        path = checkpoint_path(cond2_root, "cond2", seed, episode)
        if not path.exists():
            raise FileNotFoundError(path)
        cond2_paths.append(str(path))

write_manifest(comm_manifest, cond1_paths)
write_manifest(baseline_manifest, cond2_paths)
print(f"[iwr-manifests] comm_manifest={comm_manifest} count={len(cond1_paths)}")
print(f"[iwr-manifests] baseline_manifest={baseline_manifest} count={len(cond2_paths)}")
PYEOF
