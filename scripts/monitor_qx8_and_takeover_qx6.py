#!/usr/bin/env python3
"""Poll qx8 stage-6 completion, then migrate slow qx6 50k jobs onto qx8.

This is a one-shot monitor:
- every poll interval, check whether qx8 100k sender-shuffle and fixed0 are fully done
- once both are done, stop qx6 50k uniform/public_random
- immediately relaunch fresh 50k uniform/public_random runs on qx8
- record state to JSON and append a line-oriented event log
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


REMOTE_PROJECT_DIR = (
    "/export/scratch/iguennou/staging/dsc-epgg-vectorized/"
    "phase3-vecstraight-continuations-20260325"
)
REMOTE_PYTHON = (
    "/export/scratch/iguennou/runs/dsc-epgg/"
    "phase3-15seed-main-story-iwr-20260317/venv-py310/bin/python"
)

QX8_STAGE6 = {
    "sender_shuffle_100k": {
        "host": "iguennou@quadxeon8",
        "run_dir": "/export/scratch/iguennou/runs/dsc-epgg-vectorized/"
        "phase3-vecstraight-quadxeon8-stage6-sender-shuffle-20260325",
        "out_root": REMOTE_PROJECT_DIR
        + "/outputs/eval/phase3_vecstraight_sameckpt_continuation_100000_sender_shuffle_15seeds_iwr_20260325",
    },
    "fixed0_100k": {
        "host": "iguennou@quadxeon8",
        "run_dir": "/export/scratch/iguennou/runs/dsc-epgg-vectorized/"
        "phase3-vecstraight-quadxeon8-stage6-fixed0-20260325",
        "out_root": REMOTE_PROJECT_DIR
        + "/outputs/eval/phase3_vecstraight_sameckpt_continuation_100000_fixed0_15seeds_iwr_20260325",
    },
}

QX6_TO_STOP = {
    "uniform": {
        "host": "iguennou@quadxeon6",
        "run_dir": "/export/scratch/iguennou/runs/dsc-epgg-vectorized/"
        "phase3-vecstraight-quadxeon6-uniform-20260325",
        "out_tag": "phase3_vecstraight_sameckpt_continuation_50000_uniform_15seeds_iwr_20260325/train",
    },
    "public_random": {
        "host": "iguennou@quadxeon6",
        "run_dir": "/export/scratch/iguennou/runs/dsc-epgg-vectorized/"
        "phase3-vecstraight-quadxeon6-public-random-20260325",
        "out_tag": "phase3_vecstraight_sameckpt_continuation_50000_public_random_15seeds_iwr_20260325/train",
    },
}

QX8_RETAKE = {
    "uniform": {
        "host": "iguennou@quadxeon8",
        "queue_name": "qx8_retake_uniform",
        "run_dir": "/export/scratch/iguennou/runs/dsc-epgg-vectorized/"
        "phase3-vecstraight-quadxeon8-retake-uniform-20260326",
        "mode": "uniform",
        "out_root": REMOTE_PROJECT_DIR
        + "/outputs/eval/phase3_vecstraight_sameckpt_continuation_50000_uniform_15seeds_iwr_20260326",
    },
    "public_random": {
        "host": "iguennou@quadxeon8",
        "queue_name": "qx8_retake_public_random",
        "run_dir": "/export/scratch/iguennou/runs/dsc-epgg-vectorized/"
        "phase3-vecstraight-quadxeon8-retake-public-random-20260326",
        "mode": "public_random",
        "out_root": REMOTE_PROJECT_DIR
        + "/outputs/eval/phase3_vecstraight_sameckpt_continuation_50000_public_random_15seeds_iwr_20260326",
    },
}


def _run_ssh(host: str, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", host, f"bash -lc {shlex.quote(command)}"],
        text=True,
        capture_output=True,
        check=False,
    )


def _must_ssh(host: str, command: str) -> str:
    proc = _run_ssh(host, command)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ssh {host} failed ({proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    return (proc.stdout or "").strip()


def _write_state(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_event(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _worker_count(host: str, tag: str) -> int:
    cmd = (
        "ps -eo args | "
        f"grep {shlex.quote(tag)} | "
        "grep 'python -m src.experiments_pgg_v0.train_ppo' | "
        "grep -v grep | wc -l"
    )
    out = _must_ssh(host, cmd)
    return int(out or "0")


def _path_exists(host: str, path: str) -> bool:
    out = _must_ssh(host, f"test -f {shlex.quote(path)} && echo 1 || echo 0")
    return str(out).strip() == "1"


def _count_files(host: str, pattern: str) -> int:
    cmd = (
        "python3 - <<'PY'\n"
        "import glob\n"
        f"print(len(glob.glob({pattern!r})))\n"
        "PY"
    )
    out = _must_ssh(host, cmd)
    return int(out or "0")


def _latest_metric_episode(host: str, metrics_dir: str) -> int | None:
    cmd = (
        "python3 - <<'PY'\n"
        "import json\n"
        "from pathlib import Path\n"
        f"metrics_dir = Path({metrics_dir!r})\n"
        "best = None\n"
        "for path in metrics_dir.glob('cond1_seed*.jsonl'):\n"
        "    try:\n"
        "        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()\n"
        "    except Exception:\n"
        "        continue\n"
        "    for line in reversed(lines):\n"
        "        try:\n"
        "            row = json.loads(line)\n"
        "        except Exception:\n"
        "            continue\n"
        "        ep = row.get('episode')\n"
        "        if isinstance(ep, int):\n"
        "            best = ep if best is None else max(best, ep)\n"
        "            break\n"
        "print('' if best is None else best)\n"
        "PY"
    )
    out = _must_ssh(host, cmd)
    out = str(out).strip()
    if not out:
        return None
    return int(out)


def _qx8_idle_pct(host: str) -> float | None:
    cmd = "mpstat 1 1 | tail -n 1"
    out = _must_ssh(host, cmd)
    fields = str(out).split()
    if not fields:
        return None
    try:
        return float(fields[-1])
    except ValueError:
        match = re.search(r"([0-9]+(?:\\.[0-9]+)?)\\s*$", str(out))
        if not match:
            return None
        return float(match.group(1))


def _logical_cpu_count(host: str) -> int:
    out = _must_ssh(host, "nproc")
    return int(out or "0")


def _required_idle_pct(
    logical_cpus: int,
    n_modes: int,
    workers_per_mode: int,
    reserve_cpus: int,
    floor_pct: float,
) -> float:
    if logical_cpus <= 0:
        return float(floor_pct)
    required_cpus = int(n_modes) * int(workers_per_mode) + int(reserve_cpus)
    pct = 100.0 * float(required_cpus) / float(logical_cpus)
    return max(float(floor_pct), round(pct, 2))


def _wait_for_workers(
    host: str,
    tag: str,
    min_workers: int = 1,
    timeout_seconds: int = 180,
    poll_seconds: int = 5,
) -> int:
    deadline = time.time() + float(timeout_seconds)
    last_count = 0
    while time.time() < deadline:
        last_count = _worker_count(host, tag)
        if last_count >= int(min_workers):
            return last_count
        time.sleep(float(poll_seconds))
    return last_count


def _mode_done_on_qx8(spec: Dict[str, str]) -> Dict[str, object]:
    host = spec["host"]
    report_path = f"{spec['out_root']}/report/intervention_suite_summary.md"
    train_root = f"{spec['out_root']}/train"
    trainers = _worker_count(host, train_root)
    report_exists = _path_exists(host, report_path)
    final_ckpts = _count_files(host, train_root + "/cond1_seed*.pt")
    lock_files = _count_files(host, train_root + "/cond1_seed*.pt.lock")
    latest_metric_episode = _latest_metric_episode(host, train_root + "/metrics")
    train_complete = bool(final_ckpts == 15 and lock_files == 0)
    stalled_partial_training = bool(trainers == 0 and not train_complete and latest_metric_episode is not None)
    post_train_pending = bool(trainers == 0 and train_complete and not report_exists)
    return {
        "trainers": trainers,
        "report_exists": report_exists,
        "final_ckpts": final_ckpts,
        "lock_files": lock_files,
        "latest_metric_episode": latest_metric_episode,
        "train_complete": train_complete,
        "stalled_partial_training": stalled_partial_training,
        "post_train_pending": post_train_pending,
        "done": bool(trainers == 0 and train_complete and report_exists),
    }


def _collect_stop_pids(host: str, run_dir: str, out_tag: str) -> List[int]:
    cmd = (
        "ps -eo pid=,args= | "
        f"grep -E {shlex.quote(run_dir + '|' + out_tag)} | "
        "grep -v grep | awk '{print $1}'"
    )
    out = _must_ssh(host, cmd)
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return sorted(set(pids))


def _stop_qx6_mode(name: str, spec: Dict[str, str]) -> List[int]:
    pids = _collect_stop_pids(spec["host"], spec["run_dir"], spec["out_tag"])
    if pids:
        _must_ssh(spec["host"], "kill -TERM " + " ".join(str(pid) for pid in pids))
    return pids


def _launch_qx8_mode(spec: Dict[str, str]) -> str:
    host = spec["host"]
    run_dir = spec["run_dir"]
    queue_name = spec["queue_name"]
    mode = spec["mode"]
    job_script = f"{run_dir}/job.sh"
    run_log = f"{run_dir}/run.log"
    pidfile = f"{run_dir}/pid"
    remote = f"""
set -euo pipefail
mkdir -p {shlex.quote(run_dir)} {shlex.quote(run_dir + "/status")}
cat > {shlex.quote(job_script)} <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(REMOTE_PROJECT_DIR)}
export RUN_DATE=20260326
export RUN_KIND=iwr
export NEXT_ROOT={shlex.quote(REMOTE_PROJECT_DIR + "/outputs/eval/phase3_vecstraight_next_steps_20260325")}
export TRAIN_MAX_WORKERS=15
export EVAL_MAX_WORKERS=15
export PYTHON_BIN={shlex.quote(REMOTE_PYTHON)}
./scripts/run_phase3_vecstraight_continuation_queue_iwr.sh {shlex.quote(queue_name)} 50000:{shlex.quote(mode)}
EOS
chmod +x {shlex.quote(job_script)}
setsid bash {shlex.quote(job_script)} > {shlex.quote(run_log)} 2>&1 < /dev/null &
echo $! > {shlex.quote(pidfile)}
cat {shlex.quote(pidfile)}
"""
    return _must_ssh(host, remote)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--min-qx8-idle-pct", type=float, default=20.0)
    parser.add_argument("--required-ready-polls", type=int, default=2)
    parser.add_argument("--workers-per-mode", type=int, default=15)
    parser.add_argument("--capacity-reserve-cpus", type=int, default=12)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--events-jsonl", type=Path, required=True)
    args = parser.parse_args()

    qx8_host = QX8_STAGE6["sender_shuffle_100k"]["host"]
    logical_cpus = _logical_cpu_count(qx8_host)
    pending_modes: List[str] = list(QX8_RETAKE.keys())
    stopped: Dict[str, List[int]] = {}
    launched: Dict[str, Dict[str, object]] = {}
    ready_streak_one_mode = 0
    ready_streak_two_modes = 0

    state: Dict[str, object] = {
        "phase": "waiting_for_qx8",
        "capacity_reserve_cpus": int(args.capacity_reserve_cpus),
        "launched_qx8": launched,
        "logical_cpus": int(logical_cpus),
        "min_qx8_idle_pct": float(args.min_qx8_idle_pct),
        "pending_modes": pending_modes,
        "poll_seconds": int(args.poll_seconds),
        "required_ready_polls": int(args.required_ready_polls),
        "stopped_qx6": stopped,
        "workers_per_mode": int(args.workers_per_mode),
        "updated_at_unix": float(time.time()),
    }
    _write_state(args.state_json, state)

    while True:
        sender_status = _mode_done_on_qx8(QX8_STAGE6["sender_shuffle_100k"])
        fixed_status = _mode_done_on_qx8(QX8_STAGE6["fixed0_100k"])
        qx8_idle_pct = _qx8_idle_pct(qx8_host)
        both_done = bool(sender_status["done"]) and bool(fixed_status["done"])
        idle_threshold_one_mode = _required_idle_pct(
            logical_cpus=logical_cpus,
            n_modes=1,
            workers_per_mode=int(args.workers_per_mode),
            reserve_cpus=int(args.capacity_reserve_cpus),
            floor_pct=float(args.min_qx8_idle_pct),
        )
        idle_threshold_two_modes = _required_idle_pct(
            logical_cpus=logical_cpus,
            n_modes=2,
            workers_per_mode=int(args.workers_per_mode),
            reserve_cpus=int(args.capacity_reserve_cpus),
            floor_pct=float(args.min_qx8_idle_pct),
        )
        idle_ready_one_mode = bool(
            qx8_idle_pct is not None and qx8_idle_pct >= idle_threshold_one_mode
        )
        idle_ready_two_modes = bool(
            qx8_idle_pct is not None and qx8_idle_pct >= idle_threshold_two_modes
        )
        if both_done:
            ready_streak_one_mode = (
                ready_streak_one_mode + 1 if idle_ready_one_mode else 0
            )
            ready_streak_two_modes = (
                ready_streak_two_modes + 1 if idle_ready_two_modes else 0
            )
        else:
            ready_streak_one_mode = 0
            ready_streak_two_modes = 0

        if bool(sender_status["stalled_partial_training"]) or bool(
            fixed_status["stalled_partial_training"]
        ):
            phase = "blocked_qx8_partial_training"
        elif bool(sender_status["post_train_pending"]) or bool(
            fixed_status["post_train_pending"]
        ):
            phase = "blocked_qx8_missing_report"
        elif both_done and pending_modes:
            if len(pending_modes) >= 2 and ready_streak_two_modes >= int(
                args.required_ready_polls
            ):
                phase = "ready_for_two_mode_retake"
            elif ready_streak_one_mode >= int(args.required_ready_polls):
                phase = "ready_for_one_mode_retake"
            else:
                phase = "waiting_for_qx8_capacity"
        elif not pending_modes:
            phase = "launched_retake"
        else:
            phase = "waiting_for_qx8"

        state = {
            "phase": phase,
            "capacity_reserve_cpus": int(args.capacity_reserve_cpus),
            "idle_ready_one_mode": idle_ready_one_mode,
            "idle_ready_two_modes": idle_ready_two_modes,
            "idle_threshold_one_mode": idle_threshold_one_mode,
            "idle_threshold_two_modes": idle_threshold_two_modes,
            "launched_qx8": launched,
            "logical_cpus": int(logical_cpus),
            "min_qx8_idle_pct": float(args.min_qx8_idle_pct),
            "pending_modes": pending_modes,
            "sender_shuffle_100k": sender_status,
            "fixed0_100k": fixed_status,
            "qx8_idle_pct": qx8_idle_pct,
            "ready_streak_one_mode": int(ready_streak_one_mode),
            "ready_streak_two_modes": int(ready_streak_two_modes),
            "required_ready_polls": int(args.required_ready_polls),
            "stopped_qx6": stopped,
            "workers_per_mode": int(args.workers_per_mode),
            "updated_at_unix": float(time.time()),
        }
        _write_state(args.state_json, state)
        _append_event(
            args.events_jsonl,
            {
                "event": "poll",
                "sender_shuffle_100k": sender_status,
                "fixed0_100k": fixed_status,
                "idle_ready_one_mode": idle_ready_one_mode,
                "idle_ready_two_modes": idle_ready_two_modes,
                "idle_threshold_one_mode": idle_threshold_one_mode,
                "idle_threshold_two_modes": idle_threshold_two_modes,
                "pending_modes": pending_modes,
                "qx8_idle_pct": qx8_idle_pct,
                "ready_streak_one_mode": int(ready_streak_one_mode),
                "ready_streak_two_modes": int(ready_streak_two_modes),
                "ts_unix": float(time.time()),
            },
        )

        launch_budget = 0
        if both_done and pending_modes:
            if len(pending_modes) >= 2 and ready_streak_two_modes >= int(
                args.required_ready_polls
            ):
                launch_budget = 2
            elif ready_streak_one_mode >= int(args.required_ready_polls):
                launch_budget = 1

        if launch_budget <= 0:
            if not pending_modes:
                return 0
            time.sleep(float(args.poll_seconds))
            continue

        while launch_budget > 0 and pending_modes:
            name = pending_modes[0]
            stop_spec = QX6_TO_STOP[name]
            launch_spec = QX8_RETAKE[name]

            stopped[name] = _stop_qx6_mode(name, stop_spec)
            _append_event(
                args.events_jsonl,
                {
                    "event": "stopped_qx6_mode",
                    "mode": name,
                    "pids": stopped[name],
                    "ts_unix": float(time.time()),
                },
            )

            time.sleep(5.0)

            launch_pid = _launch_qx8_mode(launch_spec)
            verified_workers = _wait_for_workers(
                host=launch_spec["host"],
                tag=launch_spec["out_root"] + "/train",
                min_workers=1,
                timeout_seconds=180,
                poll_seconds=5,
            )
            launched[name] = {
                "launcher_pid": launch_pid,
                "verified_trainers": int(verified_workers),
                "launched_at_unix": float(time.time()),
            }
            _append_event(
                args.events_jsonl,
                {
                    "event": "launched_qx8_mode",
                    "mode": name,
                    "launcher_pid": launch_pid,
                    "verified_trainers": int(verified_workers),
                    "ts_unix": float(time.time()),
                },
            )
            pending_modes.pop(0)
            ready_streak_one_mode = 0
            ready_streak_two_modes = 0
            launch_budget -= 1

            state = {
                "phase": "launched_partial_retake"
                if pending_modes
                else "launched_retake",
                "capacity_reserve_cpus": int(args.capacity_reserve_cpus),
                "launched_qx8": launched,
                "logical_cpus": int(logical_cpus),
                "min_qx8_idle_pct": float(args.min_qx8_idle_pct),
                "pending_modes": pending_modes,
                "poll_seconds": int(args.poll_seconds),
                "required_ready_polls": int(args.required_ready_polls),
                "stopped_qx6": stopped,
                "workers_per_mode": int(args.workers_per_mode),
                "updated_at_unix": float(time.time()),
            }
            _write_state(args.state_json, state)

            if launch_budget <= 0 or not pending_modes:
                break

            qx8_idle_pct = _qx8_idle_pct(qx8_host)
            idle_ready_one_mode = bool(
                qx8_idle_pct is not None and qx8_idle_pct >= idle_threshold_one_mode
            )
            _append_event(
                args.events_jsonl,
                {
                    "event": "post_launch_capacity_check",
                    "pending_modes": pending_modes,
                    "qx8_idle_pct": qx8_idle_pct,
                    "idle_ready_one_mode": idle_ready_one_mode,
                    "idle_threshold_one_mode": idle_threshold_one_mode,
                    "ts_unix": float(time.time()),
                },
            )
            if not idle_ready_one_mode:
                break

        if not pending_modes:
            return 0

        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    sys.exit(main())
