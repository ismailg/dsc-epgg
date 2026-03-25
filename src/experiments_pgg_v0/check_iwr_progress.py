from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from typing import Any, Dict, List


DEFAULT_RUN_DIR = (
    "/export/scratch/iguennou/runs/dsc-epgg-vectorized/"
    "phase3-150k-cond1-15seed-trainonly-20260323"
)


REMOTE_PROBE = textwrap.dedent(
    r"""
    import json
    import re
    import subprocess
    import sys
    from collections import deque
    from pathlib import Path


    def _run_text(cmd):
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return ""
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if out:
            return out
        return err


    def _tail_lines(path: Path, limit: int):
        if not path.exists():
            return []
        dq = deque(maxlen=max(1, int(limit)))
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                dq.append(line.rstrip("\n"))
        return list(dq)


    def _tail_matching_lines(path: Path, patterns, limit: int):
        if not path.exists():
            return []
        dq = deque(maxlen=max(1, int(limit)))
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if any(pat in line for pat in patterns):
                    dq.append(line.rstrip("\n"))
        return list(dq)


    def _extract_episode(line: str):
        m = re.search(r"episode\s+([0-9]+)", line or "")
        if not m:
            return None
        return int(m.group(1))


    def _parse_manifest(path: Path):
        data = {}
        if not path.exists():
            return data
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "=" not in line:
                    continue
                k, v = line.rstrip("\n").split("=", 1)
                data[str(k).strip()] = str(v).strip()
        return data


    def _choose_sample_seeds(manifest, requested):
        if requested:
            return [int(x) for x in requested]
        batch = str(manifest.get("seeds_batch", "")).split()
        seeds = sorted({int(x) for x in batch if str(x).strip()})
        if not seeds:
            return []
        if len(seeds) <= 3:
            return seeds
        return [seeds[0], seeds[len(seeds) // 2], seeds[-1]]


    def _summarize_seed(seed: int, log_path: Path, train_root: Path, tail_lines: int, condition: str):
        milestone_lines = _tail_matching_lines(
            log_path,
            patterns=("[episode ", "[checkpoint]", "[regime @ episode"),
            limit=tail_lines,
        )
        last_line = milestone_lines[-1] if milestone_lines else ""
        ckpt_glob = f"{condition}_seed{int(seed)}_ep*.pt"
        ckpt_count = len(list(train_root.glob(ckpt_glob))) if train_root.exists() else 0
        final_ckpt = train_root / f"{condition}_seed{int(seed)}.pt"
        return {
            "seed": int(seed),
            "episode": _extract_episode(last_line),
            "checkpoint_count": int(ckpt_count),
            "final_exists": bool(final_ckpt.exists()),
            "last_line": str(last_line),
            "milestones": milestone_lines,
            "log_path": str(log_path),
        }


    def _summarize_standalone_run(run_dir: Path, tail_lines: int, condition: str):
        run_log = run_dir / "run.log"
        checkpoint_dirs = sorted((run_dir / "outputs").glob("*/checkpoints"))
        ckpt_count = 0
        if checkpoint_dirs:
            ckpt_count = len(list(checkpoint_dirs[0].glob(f"{condition}_seed*_ep*.pt")))
        milestone_lines = _tail_matching_lines(
            run_log,
            patterns=("[episode ", "[checkpoint]", "[regime @ episode"),
            limit=tail_lines,
        )
        last_line = milestone_lines[-1] if milestone_lines else ""
        return {
            "run_dir": str(run_dir),
            "kind": "standalone",
            "episode": _extract_episode(last_line),
            "checkpoint_count": int(ckpt_count),
            "last_line": str(last_line),
            "milestones": milestone_lines,
        }


    def _summarize_batch_run(run_dir: Path, tail_lines: int, condition: str, requested_seeds):
        progress_logs = sorted((run_dir / "outputs").glob("*/status/progress.log"))
        batches = []
        linked_runs = []
        for progress_log in progress_logs:
            out_root = progress_log.parent.parent
            manifest = _parse_manifest(progress_log.parent / "manifest.txt")
            sample_seeds = _choose_sample_seeds(manifest, requested_seeds)
            train_root = out_root / "train"
            logs_root = out_root / "logs"
            progress_tail = _tail_lines(progress_log, max(20, tail_lines * 3))
            seed_rows = []
            for seed in sample_seeds:
                seed_rows.append(
                    _summarize_seed(
                        seed=int(seed),
                        log_path=logs_root / f"{condition}_seed{int(seed)}.log",
                        train_root=train_root,
                        tail_lines=tail_lines,
                        condition=condition,
                    )
                )
            linked = str(manifest.get("existing_seed101_run_dir", "")).strip()
            if linked:
                linked_runs.append(linked)
            batches.append(
                {
                    "run_dir": str(run_dir),
                    "kind": "batch",
                    "out_root": str(out_root),
                    "progress_log": str(progress_log),
                    "progress_tail": progress_tail,
                    "manifest": manifest,
                    "sample_seeds": seed_rows,
                }
            )
        return batches, linked_runs


    def _collect_report(cfg):
        condition = str(cfg.get("condition_name", "cond1"))
        tail_lines = int(cfg.get("tail_lines", 8))
        sample_seeds = [int(x) for x in cfg.get("sample_seeds", [])]
        run_dirs = [Path(x) for x in cfg.get("run_dirs", [])]
        seen_standalone = set()
        standalone = []
        batches = []
        pending_linked = []

        for run_dir in run_dirs:
            if not run_dir.exists():
                continue
            batch_rows, linked = _summarize_batch_run(
                run_dir=run_dir,
                tail_lines=tail_lines,
                condition=condition,
                requested_seeds=sample_seeds,
            )
            batches.extend(batch_rows)
            pending_linked.extend(linked)
            if (
                len(batch_rows) == 0
                and (run_dir / "run.log").exists()
                and str(run_dir) not in seen_standalone
            ):
                standalone.append(_summarize_standalone_run(run_dir, tail_lines, condition))
                seen_standalone.add(str(run_dir))

        for linked in pending_linked:
            if linked in seen_standalone:
                continue
            linked_dir = Path(linked)
            if linked_dir.exists() and (linked_dir / "run.log").exists():
                standalone.append(_summarize_standalone_run(linked_dir, tail_lines, condition))
                seen_standalone.add(str(linked_dir))

        host = {
            "hostname": _run_text(["hostname"]),
            "uptime": _run_text(["uptime"]),
            "mpstat": _run_text(["bash", "-lc", "mpstat 1 1 | tail -n 1"]),
            "trainer_count": _run_text(
                [
                    "bash",
                    "-lc",
                    f"pgrep -fa 'src\\.experiments_pgg_v0\\.train_ppo.*--condition_name {condition}' | wc -l",
                ]
            ),
        }
        return {"host": host, "standalone_runs": standalone, "batch_runs": batches}


    print(json.dumps(_collect_report(CFG)))
    """
).strip()


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="quadxeon8")
    p.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        default=[],
        help="Remote run directory to inspect. Can be passed multiple times.",
    )
    p.add_argument("--sample-seeds", nargs="*", type=int, default=[])
    p.add_argument("--tail-lines", type=int, default=8)
    p.add_argument("--condition-name", type=str, default="cond1")
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def _probe_remote(host: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    script = "import json\nCFG = json.loads(" + repr(json.dumps(payload)) + ")\n" + REMOTE_PROBE
    proc = subprocess.run(
        ["ssh", str(host), "python3", "-"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or proc.stdout or "")
        raise SystemExit(proc.returncode)
    return json.loads(proc.stdout)


def _render_host(report: Dict[str, Any]) -> List[str]:
    host = report.get("host", {}) or {}
    out = ["**Host**"]
    if host.get("hostname"):
        out.append(f"host={host['hostname']}")
    if host.get("uptime"):
        out.append(str(host["uptime"]))
    if host.get("mpstat"):
        out.append(str(host["mpstat"]))
    if host.get("trainer_count"):
        out.append(f"trainer_count={host['trainer_count']}")
    return out


def _render_standalone(row: Dict[str, Any]) -> List[str]:
    out = ["**Standalone Run**", f"run_dir={row.get('run_dir', '')}"]
    if row.get("episode") is not None:
        out.append(f"latest_episode={int(row['episode'])}")
    out.append(f"checkpoint_count={int(row.get('checkpoint_count', 0))}")
    last = str(row.get("last_line", "")).strip()
    if last:
        out.append(f"last={last}")
    return out


def _render_batch(row: Dict[str, Any]) -> List[str]:
    out = ["**Batch Run**", f"run_dir={row.get('run_dir', '')}"]
    out.append(f"out_root={row.get('out_root', '')}")
    manifest = row.get("manifest", {}) or {}
    if manifest:
        seeds_total = manifest.get("seeds_total_target", "")
        seeds_batch = manifest.get("seeds_batch", "")
        if seeds_total:
            out.append(f"target_seeds={seeds_total}")
        if seeds_batch:
            out.append(f"batch_seeds={seeds_batch}")
    progress_tail = row.get("progress_tail", []) or []
    if progress_tail:
        out.append("progress_tail:")
        out.extend(str(line) for line in progress_tail[-12:])
    for seed_row in row.get("sample_seeds", []) or []:
        seed = int(seed_row.get("seed", -1))
        episode = seed_row.get("episode")
        last = str(seed_row.get("last_line", "")).strip()
        ckpts = int(seed_row.get("checkpoint_count", 0))
        final_exists = bool(seed_row.get("final_exists", False))
        out.append(
            f"sample_seed={seed} episode={episode if episode is not None else 'na'} "
            f"ckpts={ckpts} final={int(final_exists)}"
        )
        if last:
            out.append(f"sample_last={last}")
    return out


def main():
    args = _parse_args()
    run_dirs = list(args.run_dirs)
    if len(run_dirs) == 0:
        run_dirs = [DEFAULT_RUN_DIR]
    payload = {
        "run_dirs": run_dirs,
        "sample_seeds": list(args.sample_seeds),
        "tail_lines": int(args.tail_lines),
        "condition_name": str(args.condition_name),
    }
    report = _probe_remote(host=str(args.host), payload=payload)
    if bool(args.json):
        print(json.dumps(report, indent=2))
        return
    chunks: List[str] = []
    chunks.extend(_render_host(report))
    for row in report.get("standalone_runs", []) or []:
        chunks.append("")
        chunks.extend(_render_standalone(row))
    for row in report.get("batch_runs", []) or []:
        chunks.append("")
        chunks.extend(_render_batch(row))
    print("\n".join(chunks))


if __name__ == "__main__":
    main()
