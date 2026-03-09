from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _checkpoint_path(checkpoint_dir: str, condition: str, seed: int, episode: int) -> str:
    base = os.path.join(checkpoint_dir, f"{condition}_seed{int(seed)}.pt")
    if os.path.exists(base):
        payload = None
        try:
            import torch  # local import to keep startup cheap when not needed

            payload = torch.load(base, map_location="cpu")
        except Exception:
            payload = None
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        final_episodes = int(config.get("n_episodes", 0) or 0)
        if int(episode) == 200000 or (final_episodes > 0 and int(episode) == final_episodes):
            return base
    ep_path = os.path.join(
        checkpoint_dir, f"{condition}_seed{int(seed)}_ep{int(episode)}.pt"
    )
    if os.path.exists(ep_path):
        return ep_path
    if os.path.exists(base):
        return base
    raise FileNotFoundError(
        f"checkpoint missing for {condition} seed={seed} episode={episode}"
    )


def _task_name(condition: str, seed: int, episode: int, intervention: str) -> str:
    return f"{condition}_seed{int(seed)}_ep{int(episode)}_{intervention}"


def _run_task(task: Dict, raw_dir: str, log_dir: str, skip_existing: bool):
    out_csv = os.path.join(raw_dir, f"{task['name']}.csv")
    out_condition_csv = os.path.join(raw_dir, f"{task['name']}_condition.csv")
    out_comm_csv = os.path.join(raw_dir, f"{task['name']}_comm.csv")
    out_trace_csv = os.path.join(raw_dir, f"{task['name']}_trace.csv")
    out_sender_csv = os.path.join(raw_dir, f"{task['name']}_sender_semantics.csv")
    out_receiver_csv = os.path.join(raw_dir, f"{task['name']}_receiver_semantics.csv")
    out_posterior_csv = os.path.join(raw_dir, f"{task['name']}_posterior_strat.csv")
    log_path = os.path.join(log_dir, f"{task['name']}.log")

    expected = [out_csv, out_condition_csv, out_comm_csv]
    if bool(task.get("posterior_strat")):
        expected.append(out_posterior_csv)
    if bool(task.get("collect_semantics")):
        expected.extend([out_trace_csv, out_sender_csv, out_receiver_csv])
    if skip_existing and all(os.path.exists(path) for path in expected):
        return {
            **task,
            "out_csv": out_csv,
            "out_condition_csv": out_condition_csv,
            "out_comm_csv": out_comm_csv,
            "out_trace_csv": out_trace_csv if bool(task.get("collect_semantics")) else "",
            "out_sender_csv": out_sender_csv if bool(task.get("collect_semantics")) else "",
            "out_receiver_csv": out_receiver_csv if bool(task.get("collect_semantics")) else "",
            "out_posterior_csv": out_posterior_csv if bool(task.get("posterior_strat")) else "",
            "skipped": True,
        }

    cmd = [
        sys.executable,
        "-m",
        "src.analysis.evaluate_regime_conditional",
        "--checkpoints_glob",
        task["checkpoint"],
        "--n_eval_episodes",
        str(int(task["n_eval_episodes"])),
        "--eval_seed",
        str(int(task["eval_seed"])),
        "--greedy",
        "--msg_intervention",
        str(task["intervention"]),
        "--out_csv",
        out_csv,
        "--out_comm_csv",
        out_comm_csv,
        "--out_condition_csv",
        out_condition_csv,
    ]
    if bool(task.get("posterior_strat")):
        cmd.append("--posterior_strat")
    if bool(task.get("collect_semantics")):
        cmd.extend(
            [
                "--out_trace_csv",
                out_trace_csv,
                "--out_sender_semantics_csv",
                out_sender_csv,
                "--out_receiver_semantics_csv",
                out_receiver_csv,
            ]
        )

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
        "out_trace_csv": out_trace_csv if bool(task.get("collect_semantics")) else "",
        "out_sender_csv": out_sender_csv if bool(task.get("collect_semantics")) else "",
        "out_receiver_csv": out_receiver_csv if bool(task.get("collect_semantics")) else "",
        "out_posterior_csv": out_posterior_csv if bool(task.get("posterior_strat")) else "",
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_results(results: List[Dict], out_dir: str):
    main_rows: List[Dict] = []
    comm_rows: List[Dict] = []
    condition_rows: List[Dict] = []
    posterior_rows: List[Dict] = []
    trace_rows: List[Dict] = []
    sender_rows: List[Dict] = []
    receiver_rows: List[Dict] = []

    for result in results:
        extra = {
            "checkpoint_episode": int(result["episode"]),
            "suite_kind": str(result["suite_kind"]),
        }
        for row in _read_csv_rows(result["out_csv"]):
            main_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_comm_csv"]):
            comm_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_condition_csv"]):
            condition_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_posterior_csv"]):
            posterior_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_trace_csv"]):
            trace_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_sender_csv"]):
            sender_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_receiver_csv"]):
            receiver_rows.append({**row, **extra})

    _write_rows(os.path.join(out_dir, "checkpoint_suite_main.csv"), main_rows)
    _write_rows(os.path.join(out_dir, "checkpoint_suite_comm.csv"), comm_rows)
    _write_rows(
        os.path.join(out_dir, "checkpoint_suite_condition.csv"),
        condition_rows,
    )
    _write_rows(
        os.path.join(out_dir, "checkpoint_suite_posterior_strat.csv"),
        posterior_rows,
    )
    _write_rows(os.path.join(out_dir, "checkpoint_suite_trace.csv"), trace_rows)
    _write_rows(
        os.path.join(out_dir, "checkpoint_suite_sender_semantics.csv"),
        sender_rows,
    )
    _write_rows(
        os.path.join(out_dir, "checkpoint_suite_receiver_semantics.csv"),
        receiver_rows,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, default="outputs/train/phase2b")
    p.add_argument("--out_dir", type=str, default="outputs/eval/phase3/checkpoint_suite")
    p.add_argument("--comm_condition", type=str, default="cond1")
    p.add_argument("--baseline_condition", type=str, default="cond2")
    p.add_argument("--seeds", nargs="*", type=int, default=[101, 202])
    p.add_argument(
        "--milestones",
        nargs="*",
        type=int,
        default=[50000, 100000, 150000, 200000],
    )
    p.add_argument(
        "--interventions",
        nargs="*",
        type=str,
        default=["none", "zeros", "marginal", "fixed0", "fixed1"],
    )
    p.add_argument("--n_eval_episodes", type=int, default=300)
    p.add_argument("--eval_seed", type=int, default=9001)
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    raw_dir = os.path.join(out_dir, "raw")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    tasks = []
    for seed in args.seeds:
        for episode in args.milestones:
            comm_ckpt = _checkpoint_path(
                checkpoint_dir=args.checkpoint_dir,
                condition=args.comm_condition,
                seed=int(seed),
                episode=int(episode),
            )
            for intervention in args.interventions:
                tasks.append(
                    {
                        "name": _task_name(args.comm_condition, seed, episode, intervention),
                        "suite_kind": "comm",
                        "checkpoint": comm_ckpt,
                        "episode": int(episode),
                        "intervention": str(intervention),
                        "posterior_strat": str(intervention) == "none",
                        "collect_semantics": str(intervention) == "none",
                        "n_eval_episodes": int(args.n_eval_episodes),
                        "eval_seed": int(args.eval_seed),
                    }
                )
            if str(args.baseline_condition or "").strip() != "":
                baseline_ckpt = _checkpoint_path(
                    checkpoint_dir=args.checkpoint_dir,
                    condition=args.baseline_condition,
                    seed=int(seed),
                    episode=int(episode),
                )
                tasks.append(
                    {
                        "name": _task_name(args.baseline_condition, seed, episode, "none"),
                        "suite_kind": "baseline",
                        "checkpoint": baseline_ckpt,
                        "episode": int(episode),
                        "intervention": "none",
                        "posterior_strat": True,
                        "collect_semantics": False,
                        "n_eval_episodes": int(args.n_eval_episodes),
                        "eval_seed": int(args.eval_seed),
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
                "[suite] done "
                f"{task['name']} skipped={bool(result.get('skipped', False))}"
            )

    results = sorted(results, key=lambda item: item["name"])
    _aggregate_results(results=results, out_dir=out_dir)
    with open(os.path.join(out_dir, "checkpoint_suite_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[suite] tasks={len(results)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
