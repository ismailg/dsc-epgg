from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from src.analysis.checkpoint_artifacts import (
    atomic_write_json,
    atomic_write_rows,
    csv_has_data_rows,
    resolve_checkpoint_path,
)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _checkpoint_path(checkpoint_dir: str, condition: str, seed: int, episode: int) -> str:
    return resolve_checkpoint_path(checkpoint_dir, condition, seed, episode)


def _task_name(
    condition: str,
    sender_seed: int,
    receiver_seed: int,
    sender_episode: int,
    receiver_episode: int,
) -> str:
    if int(sender_seed) == int(receiver_seed):
        return (
            f"{condition}_seed{int(sender_seed)}_sender{int(sender_episode)}_receiver{int(receiver_episode)}"
        )
    return (
        f"{condition}_sseed{int(sender_seed)}_sender{int(sender_episode)}"
        f"_rseed{int(receiver_seed)}_receiver{int(receiver_episode)}"
    )


def _run_task(task: Dict, raw_dir: str, log_dir: str, skip_existing: bool):
    out_csv = os.path.join(raw_dir, f"{task['name']}.csv")
    out_condition_csv = os.path.join(raw_dir, f"{task['name']}_condition.csv")
    out_comm_csv = os.path.join(raw_dir, f"{task['name']}_comm.csv")
    log_path = os.path.join(log_dir, f"{task['name']}.log")
    expected = [out_csv, out_condition_csv, out_comm_csv]
    if skip_existing and all(csv_has_data_rows(path) for path in expected):
        return {
            **task,
            "out_csv": out_csv,
            "out_condition_csv": out_condition_csv,
            "out_comm_csv": out_comm_csv,
            "skipped": True,
        }

    cmd = [
        sys.executable,
        "-m",
        "src.analysis.evaluate_regime_conditional",
        "--checkpoints_glob",
        task["receiver_checkpoint"],
        "--n_eval_episodes",
        str(int(task["n_eval_episodes"])),
        "--eval_seed",
        str(int(task["eval_seed"])),
        "--greedy",
        "--msg_intervention",
        "none",
        "--cross_play_checkpoint",
        task["sender_checkpoint"],
        "--out_csv",
        out_csv,
        "--out_comm_csv",
        out_comm_csv,
        "--out_condition_csv",
        out_condition_csv,
    ]
    eval_sigmas = task.get("eval_sigmas")
    if eval_sigmas is not None and len(eval_sigmas) > 0:
        cmd.extend(["--eval_sigmas", *[str(float(v)) for v in eval_sigmas]])
    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    with open(log_path, "w", encoding="utf-8") as log_f:
        subprocess.run(
            cmd,
            cwd=_ROOT,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return {
        **task,
        "out_csv": out_csv,
        "out_condition_csv": out_condition_csv,
        "out_comm_csv": out_comm_csv,
        "skipped": False,
    }


def _read_csv_rows(path: str) -> List[Dict]:
    if path == "" or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    atomic_write_rows(path, rows, fieldnames)


def _aggregate_results(results: List[Dict], out_dir: str):
    main_rows: List[Dict] = []
    comm_rows: List[Dict] = []
    condition_rows: List[Dict] = []
    for result in results:
        extra = {
            "sender_seed": int(result["sender_seed"]),
            "receiver_seed": int(result["receiver_seed"]),
            "sender_episode": int(result["sender_episode"]),
            "receiver_episode": int(result["receiver_episode"]),
        }
        for row in _read_csv_rows(result["out_csv"]):
            main_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_comm_csv"]):
            comm_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_condition_csv"]):
            condition_rows.append({**row, **extra})
    _write_rows(os.path.join(out_dir, "crossplay_matrix_main.csv"), main_rows)
    _write_rows(os.path.join(out_dir, "crossplay_matrix_comm.csv"), comm_rows)
    _write_rows(os.path.join(out_dir, "crossplay_matrix_condition.csv"), condition_rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, default="outputs/train/phase2b")
    p.add_argument("--out_dir", type=str, default="outputs/eval/phase3/crossplay_matrix")
    p.add_argument("--condition", type=str, default="cond1")
    p.add_argument("--seeds", nargs="*", type=int, default=[101, 202])
    p.add_argument("--sender_seeds", nargs="*", type=int, default=None)
    p.add_argument("--receiver_seeds", nargs="*", type=int, default=None)
    p.add_argument(
        "--milestones",
        nargs="*",
        type=int,
        default=[50000, 100000, 150000, 200000],
    )
    p.add_argument("--sender_milestones", nargs="*", type=int, default=None)
    p.add_argument("--receiver_milestones", nargs="*", type=int, default=None)
    p.add_argument("--n_eval_episodes", type=int, default=300)
    p.add_argument("--eval_seed", type=int, default=9001)
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--eval_sigmas", nargs="*", type=float, default=None)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    raw_dir = os.path.join(out_dir, "raw")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    sender_milestones = (
        [int(v) for v in args.sender_milestones]
        if args.sender_milestones is not None and len(args.sender_milestones) > 0
        else [int(v) for v in args.milestones]
    )
    receiver_milestones = (
        [int(v) for v in args.receiver_milestones]
        if args.receiver_milestones is not None and len(args.receiver_milestones) > 0
        else [int(v) for v in args.milestones]
    )
    sender_seeds = (
        [int(v) for v in args.sender_seeds]
        if args.sender_seeds is not None and len(args.sender_seeds) > 0
        else [int(v) for v in args.seeds]
    )
    receiver_seeds = (
        [int(v) for v in args.receiver_seeds]
        if args.receiver_seeds is not None and len(args.receiver_seeds) > 0
        else [int(v) for v in args.seeds]
    )

    tasks = []
    for sender_seed in sender_seeds:
        for sender_episode in sender_milestones:
            sender_ckpt = _checkpoint_path(
                checkpoint_dir=args.checkpoint_dir,
                condition=args.condition,
                seed=int(sender_seed),
                episode=int(sender_episode),
            )
            for receiver_seed in receiver_seeds:
                for receiver_episode in receiver_milestones:
                    receiver_ckpt = _checkpoint_path(
                        checkpoint_dir=args.checkpoint_dir,
                        condition=args.condition,
                        seed=int(receiver_seed),
                        episode=int(receiver_episode),
                    )
                    tasks.append(
                        {
                            "name": _task_name(
                                args.condition,
                                sender_seed,
                                receiver_seed,
                                sender_episode,
                                receiver_episode,
                            ),
                            "sender_seed": int(sender_seed),
                            "receiver_seed": int(receiver_seed),
                            "sender_episode": int(sender_episode),
                            "receiver_episode": int(receiver_episode),
                            "sender_checkpoint": sender_ckpt,
                            "receiver_checkpoint": receiver_ckpt,
                            "n_eval_episodes": int(args.n_eval_episodes),
                            "eval_seed": int(args.eval_seed),
                            "eval_sigmas": (
                                [float(v) for v in args.eval_sigmas]
                                if args.eval_sigmas is not None and len(args.eval_sigmas) > 0
                                else None
                            ),
                        }
                    )

    results = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        future_to_task = {
            ex.submit(_run_task, task, raw_dir, log_dir, bool(args.skip_existing)): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results.append(result)
            print(
                "[crossplay] done "
                f"{task['name']} skipped={bool(result.get('skipped', False))}"
            )

    results = sorted(results, key=lambda item: item["name"])
    _aggregate_results(results=results, out_dir=out_dir)
    atomic_write_json(os.path.join(out_dir, "crossplay_matrix_manifest.json"), results)
    print(f"[crossplay] tasks={len(results)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
