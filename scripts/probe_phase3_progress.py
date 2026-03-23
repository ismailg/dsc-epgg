#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_int_list(values: Iterable[Any]) -> List[int]:
    return [int(v) for v in values]


def _as_str_list(values: Iterable[Any]) -> List[str]:
    return [str(v) for v in values]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(path_str: str, cwd: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (cwd / path).resolve()


def _read_last_jsonl_row(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    last: Optional[Dict[str, Any]] = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                last = obj
    return last


def _parse_abs_episode_from_log(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"(?:\[episode\s+|\[regime @ episode\s+)(\d+)\]", text)
    if not matches:
        return None
    return max(int(v) for v in matches)


def _parse_checkpoint_episode_candidates(out_dir: Path, stem: str) -> Optional[int]:
    pattern = re.compile(re.escape(stem) + r"_ep(\d+)\.pt$")
    best: Optional[int] = None
    for path in out_dir.glob(f"{stem}_ep*.pt"):
        m = pattern.search(path.name)
        if m is None:
            continue
        ep = int(m.group(1))
        if best is None or ep > best:
            best = ep
    return best


def _parse_cli_opts(argv: Sequence[str]) -> Dict[str, Any]:
    opts: Dict[str, Any] = {}
    i = 0
    while i < len(argv):
        token = str(argv[i])
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:].replace("-", "_")
        i += 1
        values: List[str] = []
        while i < len(argv) and not str(argv[i]).startswith("--"):
            values.append(str(argv[i]))
            i += 1
        if len(values) == 0:
            opts[key] = True
        elif len(values) == 1:
            opts[key] = values[0]
        else:
            opts[key] = values
    return opts


def _module_from_cmd(cmd: Sequence[str]) -> Tuple[str, List[str]]:
    cmd_list = [str(v) for v in cmd]
    if "-m" in cmd_list:
        idx = cmd_list.index("-m")
        if idx + 1 >= len(cmd_list):
            raise ValueError("malformed command: -m without module")
        return cmd_list[idx + 1], cmd_list[idx + 2 :]
    if len(cmd_list) >= 2 and cmd_list[1].endswith(".py"):
        return Path(cmd_list[1]).name, cmd_list[2:]
    raise ValueError("unsupported command format; expected python -m module ...")


def _job_name(condition: str, seed: int, msg_training_intervention: str) -> str:
    suffix = ""
    if str(msg_training_intervention).strip().lower() != "none":
        suffix = f"_{str(msg_training_intervention).strip().lower()}"
    return f"{condition}_seed{int(seed)}{suffix}"


def _checkpoint_task_name(condition: str, seed: int, episode: int, intervention: str) -> str:
    return f"{condition}_seed{int(seed)}_ep{int(episode)}_{intervention}"


def _crossplay_task_name(condition: str, seed: int, sender_episode: int, receiver_episode: int) -> str:
    return (
        f"{condition}_seed{int(seed)}_sender{int(sender_episode)}_receiver{int(receiver_episode)}"
    )


def _expected_checkpoint_suite_files(out_dir: Path, task: Dict[str, Any]) -> List[Path]:
    raw_dir = out_dir / "raw"
    name = str(task["name"])
    files = [
        raw_dir / f"{name}.csv",
        raw_dir / f"{name}_condition.csv",
        raw_dir / f"{name}_comm.csv",
    ]
    if bool(task.get("posterior_strat")):
        files.append(raw_dir / f"{name}_posterior_strat.csv")
    if bool(task.get("collect_semantics")):
        files.extend(
            [
                raw_dir / f"{name}_trace.csv",
                raw_dir / f"{name}_sender_semantics.csv",
                raw_dir / f"{name}_receiver_semantics.csv",
            ]
        )
    return files


def _expected_crossplay_files(out_dir: Path, task_name: str) -> List[Path]:
    raw_dir = out_dir / "raw"
    return [
        raw_dir / f"{task_name}.csv",
        raw_dir / f"{task_name}_condition.csv",
        raw_dir / f"{task_name}_comm.csv",
    ]


def _task_fraction(expected: Sequence[Path]) -> float:
    if len(expected) == 0:
        return 0.0
    return float(sum(1 for path in expected if path.exists())) / float(len(expected))


def _seed_expansion_jobs_from_manifest(path: Path, cwd: Path) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    jobs: List[Dict[str, Any]] = []
    for item in payload:
        job = item["job"]
        cmd = [str(v) for v in job["cmd"]]
        opts = _parse_cli_opts(cmd)
        jobs.append(
            {
                "cwd": cwd,
                "out_dir": _resolve_path(str(Path(job["save_path"]).parent), cwd),
                "condition": str(job["condition"]),
                "seed": int(job["seed"]),
                "name": Path(job["save_path"]).stem,
                "n_episodes": int(opts.get("n_episodes", 0) or 0),
                "episode_offset": int(opts.get("episode_offset", 0) or 0),
            }
        )
    return jobs


def _seed_expansion_jobs_from_args(
    cwd: Path,
    out_dir: str,
    conditions: Sequence[str],
    seeds: Sequence[int],
    n_episodes: int,
    episode_offset: int,
    msg_training_intervention: str,
) -> List[Dict[str, Any]]:
    resolved_out = _resolve_path(out_dir, cwd)
    return [
        {
            "cwd": cwd,
            "out_dir": resolved_out,
            "condition": str(condition),
            "seed": int(seed),
            "name": _job_name(str(condition), int(seed), str(msg_training_intervention)),
            "n_episodes": int(n_episodes),
            "episode_offset": int(episode_offset),
        }
        for condition in conditions
        for seed in seeds
    ]


def _probe_seed_expansion(jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    current = 0.0
    total = 0.0
    complete = 0
    running = 0
    started = 0
    not_started = 0
    for job in jobs:
        out_dir = Path(job["out_dir"])
        name = str(job["name"])
        n_episodes = int(job["n_episodes"])
        episode_offset = int(job["episode_offset"])
        total += float(n_episodes)

        save_path = out_dir / f"{name}.pt"
        lock_path = out_dir / f"{name}.pt.lock"
        metrics_path = out_dir / "metrics" / f"{name}.jsonl"
        log_path = out_dir / "logs" / f"{name}.log"

        if save_path.exists():
            local_done = float(n_episodes)
            complete += 1
            started += 1
        else:
            abs_episode: Optional[int] = None
            last_row = _read_last_jsonl_row(metrics_path)
            if last_row is not None and "episode" in last_row:
                abs_episode = int(last_row["episode"])
            if abs_episode is None:
                abs_episode = _parse_abs_episode_from_log(log_path)
            if abs_episode is None:
                abs_episode = _parse_checkpoint_episode_candidates(out_dir, name)
            if abs_episode is None:
                local_done = 0.0
            else:
                local_done = float(max(0, min(n_episodes, int(abs_episode) - episode_offset)))

            if lock_path.exists():
                running += 1
            if local_done > 0.0 or metrics_path.exists() or log_path.exists():
                started += 1
            else:
                not_started += 1

        current += local_done

    pct = 0.0 if total <= 0 else 100.0 * current / total
    return {
        "kind": "seed-expansion",
        "units": "episodes",
        "current": float(current),
        "total": float(total),
        "pct": float(pct),
        "jobs_total": int(len(jobs)),
        "jobs_complete": int(complete),
        "jobs_running": int(running),
        "jobs_started": int(started),
        "jobs_not_started": int(not_started),
    }


def _checkpoint_suite_tasks_from_manifest(path: Path) -> List[Dict[str, Any]]:
    return list(_load_json(path))


def _checkpoint_suite_tasks_from_args(
    comm_condition: str,
    baseline_condition: str,
    seeds: Sequence[int],
    milestones: Sequence[int],
    interventions: Sequence[str],
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for seed in seeds:
        for episode in milestones:
            for intervention in interventions:
                tasks.append(
                    {
                        "name": _checkpoint_task_name(comm_condition, int(seed), int(episode), str(intervention)),
                        "suite_kind": "comm",
                        "episode": int(episode),
                        "intervention": str(intervention),
                        "posterior_strat": str(intervention) == "none",
                        "collect_semantics": str(intervention) == "none",
                    }
                )
            if str(baseline_condition).strip() != "":
                tasks.append(
                    {
                        "name": _checkpoint_task_name(str(baseline_condition), int(seed), int(episode), "none"),
                        "suite_kind": "baseline",
                        "episode": int(episode),
                        "intervention": "none",
                        "posterior_strat": True,
                        "collect_semantics": False,
                    }
                )
    return tasks


def _probe_checkpoint_suite(out_dir: Path, tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    current = 0.0
    total = float(len(tasks))
    complete = 0
    started = 0
    for task in tasks:
        frac = _task_fraction(_expected_checkpoint_suite_files(out_dir, task))
        current += frac
        if frac >= 1.0:
            complete += 1
        if frac > 0.0:
            started += 1
    pct = 0.0 if total <= 0 else 100.0 * current / total
    return {
        "kind": "checkpoint-suite",
        "units": "tasks",
        "current": float(current),
        "total": float(total),
        "pct": float(pct),
        "tasks_total": int(len(tasks)),
        "tasks_complete": int(complete),
        "tasks_started": int(started),
        "tasks_not_started": int(len(tasks) - started),
        "out_dir": str(out_dir),
    }


def _crossplay_tasks_from_manifest(path: Path) -> List[Dict[str, Any]]:
    return list(_load_json(path))


def _crossplay_tasks_from_args(
    condition: str,
    seeds: Sequence[int],
    sender_milestones: Sequence[int],
    receiver_milestones: Sequence[int],
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for seed in seeds:
        for sender_episode in sender_milestones:
            for receiver_episode in receiver_milestones:
                tasks.append(
                    {
                        "name": _crossplay_task_name(condition, int(seed), int(sender_episode), int(receiver_episode)),
                        "sender_episode": int(sender_episode),
                        "receiver_episode": int(receiver_episode),
                    }
                )
    return tasks


def _probe_crossplay(out_dir: Path, tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    current = 0.0
    total = float(len(tasks))
    complete = 0
    started = 0
    for task in tasks:
        frac = _task_fraction(_expected_crossplay_files(out_dir, str(task["name"])))
        current += frac
        if frac >= 1.0:
            complete += 1
        if frac > 0.0:
            started += 1
    pct = 0.0 if total <= 0 else 100.0 * current / total
    return {
        "kind": "crossplay",
        "units": "tasks",
        "current": float(current),
        "total": float(total),
        "pct": float(pct),
        "tasks_total": int(len(tasks)),
        "tasks_complete": int(complete),
        "tasks_started": int(started),
        "tasks_not_started": int(len(tasks) - started),
        "out_dir": str(out_dir),
    }


def _probe_trimmed_eval(
    suite_out_dir: Path,
    crossplay_out_dir: Path,
    comm_condition: str,
    baseline_condition: str,
    seeds: Sequence[int],
    milestones: Sequence[int],
    interventions: Sequence[str],
    crossplay_sender_milestones: Sequence[int],
    crossplay_receiver_milestones: Sequence[int],
) -> Dict[str, Any]:
    suite_tasks = _checkpoint_suite_tasks_from_args(
        comm_condition=comm_condition,
        baseline_condition=baseline_condition,
        seeds=seeds,
        milestones=milestones,
        interventions=interventions,
    )
    crossplay_tasks = _crossplay_tasks_from_args(
        condition=comm_condition,
        seeds=seeds,
        sender_milestones=crossplay_sender_milestones,
        receiver_milestones=crossplay_receiver_milestones,
    )
    suite = _probe_checkpoint_suite(out_dir=suite_out_dir, tasks=suite_tasks)
    crossplay = _probe_crossplay(out_dir=crossplay_out_dir, tasks=crossplay_tasks)
    current = float(suite["current"]) + float(crossplay["current"])
    total = float(suite["total"]) + float(crossplay["total"])
    pct = 0.0 if total <= 0 else 100.0 * current / total
    return {
        "kind": "trimmed-eval",
        "units": "tasks",
        "current": float(current),
        "total": float(total),
        "pct": float(pct),
        "suite": suite,
        "crossplay": crossplay,
    }


def _probe_from_launcher_metadata(path: Path) -> Dict[str, Any]:
    meta = _load_json(path)
    cwd = Path(str(meta.get("cwd", REPO_ROOT))).resolve()
    module, argv = _module_from_cmd(meta["cmd"])
    opts = _parse_cli_opts(argv)

    if module == "src.experiments_pgg_v0.run_phase3_seed_expansion" or module == "run_phase3_seed_expansion.py":
        jobs = _seed_expansion_jobs_from_args(
            cwd=cwd,
            out_dir=str(opts.get("out_dir", "outputs/train/phase3_trimmed")),
            conditions=_as_str_list(_ensure_list(opts.get("conditions", ["cond1", "cond2"]))),
            seeds=_as_int_list(_ensure_list(opts.get("seeds", [101, 202, 303, 404, 505]))),
            n_episodes=int(opts.get("n_episodes", 200000)),
            episode_offset=int(opts.get("episode_offset", 0)),
            msg_training_intervention=str(opts.get("msg_training_intervention", "none")),
        )
        return _probe_seed_expansion(jobs)

    if module == "src.analysis.run_phase3_checkpoint_suite" or module == "run_phase3_checkpoint_suite.py":
        out_dir = _resolve_path(str(opts.get("out_dir", "outputs/eval/phase3/checkpoint_suite")), cwd)
        tasks = _checkpoint_suite_tasks_from_args(
            comm_condition=str(opts.get("comm_condition", "cond1")),
            baseline_condition=str(opts.get("baseline_condition", "cond2")),
            seeds=_as_int_list(_ensure_list(opts.get("seeds", [101, 202, 303, 404, 505]))),
            milestones=_as_int_list(_ensure_list(opts.get("milestones", [50000, 150000, 200000]))),
            interventions=_as_str_list(
                _ensure_list(
                    opts.get(
                        "interventions",
                        [
                            "none",
                            "zeros",
                            "marginal",
                            "fixed0",
                            "fixed1",
                            "indep_random",
                            "public_random",
                            "sender_shuffle",
                            "permute_slots",
                        ],
                    )
                )
            ),
        )
        return _probe_checkpoint_suite(out_dir=out_dir, tasks=tasks)

    if module == "src.analysis.run_phase3_crossplay_matrix" or module == "run_phase3_crossplay_matrix.py":
        milestones = _as_int_list(_ensure_list(opts.get("milestones", [50000, 100000, 150000, 200000])))
        sender_milestones = _as_int_list(_ensure_list(opts.get("sender_milestones", milestones)))
        receiver_milestones = _as_int_list(_ensure_list(opts.get("receiver_milestones", milestones)))
        out_dir = _resolve_path(str(opts.get("out_dir", "outputs/eval/phase3/crossplay_matrix")), cwd)
        tasks = _crossplay_tasks_from_args(
            condition=str(opts.get("condition", "cond1")),
            seeds=_as_int_list(_ensure_list(opts.get("seeds", [101, 202]))),
            sender_milestones=sender_milestones,
            receiver_milestones=receiver_milestones,
        )
        return _probe_crossplay(out_dir=out_dir, tasks=tasks)

    if module == "src.analysis.run_phase3_trimmed_eval" or module == "run_phase3_trimmed_eval.py":
        milestones = _as_int_list(_ensure_list(opts.get("milestones", [50000, 150000, 200000])))
        sender_milestones = _as_int_list(_ensure_list(opts.get("crossplay_sender_milestones", milestones)))
        receiver_milestones = _as_int_list(_ensure_list(opts.get("crossplay_receiver_milestones", [200000])))
        return _probe_trimmed_eval(
            suite_out_dir=_resolve_path(str(opts.get("suite_out_dir", "outputs/eval/phase3_trimmed/checkpoint_suite")), cwd),
            crossplay_out_dir=_resolve_path(str(opts.get("crossplay_out_dir", "outputs/eval/phase3_trimmed/crossplay")), cwd),
            comm_condition=str(opts.get("comm_condition", "cond1")),
            baseline_condition=str(opts.get("baseline_condition", "cond2")),
            seeds=_as_int_list(_ensure_list(opts.get("seeds", [101, 202, 303, 404, 505]))),
            milestones=milestones,
            interventions=_as_str_list(
                _ensure_list(
                    opts.get(
                        "interventions",
                        [
                            "none",
                            "zeros",
                            "marginal",
                            "fixed0",
                            "fixed1",
                            "indep_random",
                            "public_random",
                            "sender_shuffle",
                            "permute_slots",
                        ],
                    )
                )
            ),
            crossplay_sender_milestones=sender_milestones,
            crossplay_receiver_milestones=receiver_milestones,
        )

    raise ValueError(f"unsupported launched module for progress probe: {module}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("from-launcher-metadata")
    p.add_argument("--metadata-json", type=str, required=True)

    p = sub.add_parser("seed-expansion")
    p.add_argument("--manifest-json", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--conditions", nargs="*", type=str, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--n-episodes", type=int, default=0)
    p.add_argument("--episode-offset", type=int, default=0)
    p.add_argument("--msg-training-intervention", type=str, default="none")

    p = sub.add_parser("checkpoint-suite")
    p.add_argument("--manifest-json", type=str, default="")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--comm-condition", type=str, default="cond1")
    p.add_argument("--baseline-condition", type=str, default="")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--milestones", nargs="*", type=int, default=None)
    p.add_argument("--interventions", nargs="*", type=str, default=None)
    # Accepted for compatibility with watcher commands; progress inference for
    # checkpoint suites already knows which tasks emit semantics/posterior files.
    p.add_argument("--collect-semantics", action="store_true")
    p.add_argument("--posterior-strat", action="store_true")

    p = sub.add_parser("crossplay")
    p.add_argument("--manifest-json", type=str, default="")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--condition", type=str, default="cond1")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--sender-milestones", nargs="*", type=int, default=None)
    p.add_argument("--receiver-milestones", nargs="*", type=int, default=None)
    p.add_argument("--milestones", nargs="*", type=int, default=None)

    p = sub.add_parser("trimmed-eval")
    p.add_argument("--manifest-json", type=str, default="")
    p.add_argument("--suite-out-dir", type=str, required=True)
    p.add_argument("--crossplay-out-dir", type=str, required=True)
    p.add_argument("--comm-condition", type=str, default="cond1")
    p.add_argument("--baseline-condition", type=str, default="cond2")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--milestones", nargs="*", type=int, default=None)
    p.add_argument("--interventions", nargs="*", type=str, default=None)
    p.add_argument("--crossplay-sender-milestones", nargs="*", type=int, default=None)
    p.add_argument("--crossplay-receiver-milestones", nargs="*", type=int, default=None)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "from-launcher-metadata":
        result = _probe_from_launcher_metadata(Path(args.metadata_json))
    elif args.cmd == "seed-expansion":
        manifest_json = str(args.manifest_json or "").strip()
        use_explicit = (
            args.conditions is not None and args.seeds is not None and int(args.n_episodes) > 0
        )
        if manifest_json == "" and not use_explicit and str(args.out_dir or "").strip() != "":
            candidate = Path(args.out_dir) / "phase3_seed_expansion_manifest.json"
            manifest_json = str(candidate) if candidate.exists() else ""
        if manifest_json != "":
            jobs = _seed_expansion_jobs_from_manifest(Path(manifest_json), cwd=REPO_ROOT)
        else:
            if str(args.out_dir or "").strip() == "":
                raise ValueError("seed-expansion requires --out-dir or --manifest-json")
            if args.conditions is None or args.seeds is None or int(args.n_episodes) <= 0:
                raise ValueError(
                    "seed-expansion without a manifest requires --conditions --seeds --n-episodes"
                )
            jobs = _seed_expansion_jobs_from_args(
                cwd=REPO_ROOT,
                out_dir=str(args.out_dir),
                conditions=list(args.conditions),
                seeds=list(args.seeds),
                n_episodes=int(args.n_episodes),
                episode_offset=int(args.episode_offset),
                msg_training_intervention=str(args.msg_training_intervention),
            )
        result = _probe_seed_expansion(jobs)
    elif args.cmd == "checkpoint-suite":
        manifest_json = str(args.manifest_json or "").strip()
        use_explicit = (
            args.seeds is not None and args.milestones is not None and args.interventions is not None
        )
        if manifest_json == "" and not use_explicit and (Path(args.out_dir) / "checkpoint_suite_manifest.json").exists():
            manifest_json = str(Path(args.out_dir) / "checkpoint_suite_manifest.json")
        if manifest_json != "":
            tasks = _checkpoint_suite_tasks_from_manifest(Path(manifest_json))
        else:
            tasks = _checkpoint_suite_tasks_from_args(
                comm_condition=str(args.comm_condition),
                baseline_condition=str(args.baseline_condition),
                seeds=list(args.seeds or []),
                milestones=list(args.milestones or []),
                interventions=list(args.interventions or []),
            )
        result = _probe_checkpoint_suite(out_dir=Path(args.out_dir).resolve(), tasks=tasks)
    elif args.cmd == "crossplay":
        manifest_json = str(args.manifest_json or "").strip()
        use_explicit = (
            args.seeds is not None
            and (
                args.sender_milestones is not None
                or args.receiver_milestones is not None
                or args.milestones is not None
            )
        )
        if manifest_json == "" and not use_explicit and (Path(args.out_dir) / "crossplay_matrix_manifest.json").exists():
            manifest_json = str(Path(args.out_dir) / "crossplay_matrix_manifest.json")
        if manifest_json != "":
            tasks = _crossplay_tasks_from_manifest(Path(manifest_json))
        else:
            base_milestones = list(args.milestones or [])
            sender = list(args.sender_milestones or base_milestones)
            receiver = list(args.receiver_milestones or base_milestones)
            tasks = _crossplay_tasks_from_args(
                condition=str(args.condition),
                seeds=list(args.seeds or []),
                sender_milestones=sender,
                receiver_milestones=receiver,
            )
        result = _probe_crossplay(out_dir=Path(args.out_dir).resolve(), tasks=tasks)
    else:
        manifest_json = str(args.manifest_json or "").strip()
        use_explicit = (
            args.seeds is not None
            and args.milestones is not None
            and args.interventions is not None
            and args.crossplay_sender_milestones is not None
            and args.crossplay_receiver_milestones is not None
        )
        if manifest_json != "":
            manifest = _load_json(Path(manifest_json))
            result = _probe_trimmed_eval(
                suite_out_dir=Path(manifest["suite_out_dir"]).resolve(),
                crossplay_out_dir=Path(manifest["crossplay_out_dir"]).resolve(),
                comm_condition=str(manifest["comm_condition"]),
                baseline_condition=str(manifest["baseline_condition"]),
                seeds=_as_int_list(manifest["seeds"]),
                milestones=_as_int_list(manifest["milestones"]),
                interventions=_as_str_list(manifest["interventions"]),
                crossplay_sender_milestones=_as_int_list(manifest["crossplay_sender_milestones"]),
                crossplay_receiver_milestones=_as_int_list(manifest["crossplay_receiver_milestones"]),
            )
        else:
            result = _probe_trimmed_eval(
                suite_out_dir=Path(args.suite_out_dir).resolve(),
                crossplay_out_dir=Path(args.crossplay_out_dir).resolve(),
                comm_condition=str(args.comm_condition),
                baseline_condition=str(args.baseline_condition),
                seeds=list(args.seeds or []),
                milestones=list(args.milestones or []),
                interventions=list(args.interventions or []),
                crossplay_sender_milestones=list(args.crossplay_sender_milestones or []),
                crossplay_receiver_milestones=list(args.crossplay_receiver_milestones or []),
            )

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
