from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


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


def _as_int(row: Dict, key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return int(default)
    return int(float(value))


def _as_float(row: Dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float(default)
    return float(value)


def _checkpoint_path(checkpoint_dir: str, condition: str, seed: int, episode: int) -> str:
    base = os.path.join(checkpoint_dir, f"{condition}_seed{int(seed)}.pt")
    if int(episode) == 200000 and os.path.exists(base):
        return base
    ep_path = os.path.join(checkpoint_dir, f"{condition}_seed{int(seed)}_ep{int(episode)}.pt")
    if os.path.exists(ep_path):
        return ep_path
    if os.path.exists(base):
        return base
    raise FileNotFoundError(f"checkpoint missing for {condition} seed={seed} episode={episode}")


def _build_flip_maps(rows: List[Dict], basis: str) -> Dict[Tuple[str, int, int], Dict[str, int]]:
    key_name = (
        "delta_high_minus_low_fhat"
        if str(basis).strip().lower() == "regime"
        else "delta_action1_minus_action0"
    )
    out: Dict[Tuple[str, int, int], Dict[str, int]] = {}
    for row in rows:
        if row.get("eval_policy", "greedy") != "greedy":
            continue
        if row.get("ablation", "none") != "none":
            continue
        if row.get("cross_play", "none") != "none":
            continue
        if row.get("sender_remap", "none") != "none":
            continue
        condition = str(row.get("condition"))
        seed = _as_int(row, "train_seed", -1)
        episode = _as_int(row, "checkpoint_episode", 0)
        sender_id = str(row.get("sender_id"))
        delta = _as_float(row, key_name, 0.0)
        out.setdefault((condition, seed, episode), {})[sender_id] = int(delta < 0.0)
    return out


def _task_name(condition: str, seed: int, episode: int, basis: str) -> str:
    return f"{condition}_seed{int(seed)}_ep{int(episode)}_align_{basis}"


def _run_task(task: Dict, raw_dir: str, log_dir: str, skip_existing: bool) -> Dict:
    out_csv = os.path.join(raw_dir, f"{task['name']}.csv")
    out_condition_csv = os.path.join(raw_dir, f"{task['name']}_condition.csv")
    out_comm_csv = os.path.join(raw_dir, f"{task['name']}_comm.csv")
    log_path = os.path.join(log_dir, f"{task['name']}.log")
    expected = [out_csv, out_condition_csv, out_comm_csv]
    if skip_existing and all(os.path.exists(path) for path in expected):
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
        task["checkpoint"],
        "--n_eval_episodes",
        str(int(task["n_eval_episodes"])),
        "--eval_seed",
        str(int(task["eval_seed"])),
        "--greedy",
        "--msg_intervention",
        "none",
        "--sender_flip_map_json",
        json.dumps(task["flip_map"], separators=(",", ":")),
        "--sender_remap_label",
        str(task["sender_remap_label"]),
        "--out_csv",
        out_csv,
        "--out_comm_csv",
        out_comm_csv,
        "--out_condition_csv",
        out_condition_csv,
    ]
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


def _aggregate_results(results: List[Dict], out_dir: str):
    main_rows: List[Dict] = []
    comm_rows: List[Dict] = []
    condition_rows: List[Dict] = []
    for result in results:
        extra = {
            "checkpoint_episode": int(result["episode"]),
            "sender_remap_basis": str(result["basis"]),
            "n_flipped_senders": int(sum(int(v) for v in result["flip_map"].values())),
        }
        for row in _read_csv_rows(result["out_csv"]):
            main_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_comm_csv"]):
            comm_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_condition_csv"]):
            condition_rows.append({**row, **extra})
    _write_rows(os.path.join(out_dir, "common_polarity_rescue_main.csv"), main_rows)
    _write_rows(os.path.join(out_dir, "common_polarity_rescue_comm.csv"), comm_rows)
    _write_rows(os.path.join(out_dir, "common_polarity_rescue_condition.csv"), condition_rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, default="outputs/train/phase2b")
    p.add_argument("--sender_summary_csv", type=str, default="outputs/eval/phase3/report/sender_semantics_summary.csv")
    p.add_argument("--out_dir", type=str, default="outputs/eval/phase3/common_polarity_rescue")
    p.add_argument("--condition", type=str, default="cond1")
    p.add_argument("--basis", type=str, choices=["regime", "action"], default="regime")
    p.add_argument("--seeds", nargs="*", type=int, default=[101, 202])
    p.add_argument("--milestones", nargs="*", type=int, default=[50000, 100000, 150000, 200000])
    p.add_argument("--n_eval_episodes", type=int, default=300)
    p.add_argument("--eval_seed", type=int, default=9001)
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    sender_rows = _read_csv_rows(args.sender_summary_csv)
    flip_maps = _build_flip_maps(sender_rows, basis=str(args.basis))
    out_dir = os.path.abspath(args.out_dir)
    raw_dir = os.path.join(out_dir, "raw")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    tasks = []
    for seed in args.seeds:
        for episode in args.milestones:
            key = (str(args.condition), int(seed), int(episode))
            flip_map = flip_maps.get(key)
            if flip_map is None or len(flip_map) == 0:
                continue
            tasks.append(
                {
                    "name": _task_name(str(args.condition), int(seed), int(episode), str(args.basis)),
                    "condition": str(args.condition),
                    "seed": int(seed),
                    "episode": int(episode),
                    "basis": str(args.basis),
                    "sender_remap_label": f"align_{str(args.basis)}",
                    "flip_map": flip_map,
                    "checkpoint": _checkpoint_path(args.checkpoint_dir, str(args.condition), int(seed), int(episode)),
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
                "[common-polarity-rescue] done "
                f"{task['name']} flipped={sum(int(v) for v in task['flip_map'].values())} "
                f"skipped={bool(result.get('skipped', False))}"
            )

    results = sorted(results, key=lambda item: item["name"])
    _aggregate_results(results, out_dir)
    with open(os.path.join(out_dir, "common_polarity_rescue_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[common-polarity-rescue] tasks={len(results)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
