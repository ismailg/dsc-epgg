from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _run(cmd):
    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run(cmd, cwd=_ROOT, env=env, check=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, default="outputs/train/phase3_trimmed")
    p.add_argument("--suite_out_dir", type=str, default="outputs/eval/phase3_trimmed/checkpoint_suite")
    p.add_argument("--crossplay_out_dir", type=str, default="outputs/eval/phase3_trimmed/crossplay")
    p.add_argument("--comm_condition", type=str, default="cond1")
    p.add_argument("--baseline_condition", type=str, default="cond2")
    p.add_argument("--seeds", nargs="*", type=int, default=[101, 202, 303, 404, 505])
    p.add_argument("--milestones", nargs="*", type=int, default=[50000, 150000, 200000])
    p.add_argument("--interventions", nargs="*", type=str, default=["none", "zeros", "fixed0", "fixed1"])
    p.add_argument("--crossplay_sender_milestones", nargs="*", type=int, default=[50000, 150000, 200000])
    p.add_argument("--crossplay_receiver_milestones", nargs="*", type=int, default=[200000])
    p.add_argument("--n_eval_episodes", type=int, default=300)
    p.add_argument("--eval_seed", type=int, default=9001)
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.abspath(args.suite_out_dir), exist_ok=True)
    os.makedirs(os.path.abspath(args.crossplay_out_dir), exist_ok=True)

    suite_cmd = [
        sys.executable,
        "-m",
        "src.analysis.run_phase3_checkpoint_suite",
        "--checkpoint_dir",
        str(args.checkpoint_dir),
        "--out_dir",
        str(args.suite_out_dir),
        "--comm_condition",
        str(args.comm_condition),
        "--baseline_condition",
        str(args.baseline_condition),
        "--seeds",
        *[str(int(v)) for v in args.seeds],
        "--milestones",
        *[str(int(v)) for v in args.milestones],
        "--interventions",
        *[str(v) for v in args.interventions],
        "--n_eval_episodes",
        str(int(args.n_eval_episodes)),
        "--eval_seed",
        str(int(args.eval_seed)),
        "--max_workers",
        str(int(args.max_workers)),
    ]
    if bool(args.skip_existing):
        suite_cmd.append("--skip_existing")
    _run(suite_cmd)

    crossplay_cmd = [
        sys.executable,
        "-m",
        "src.analysis.run_phase3_crossplay_matrix",
        "--checkpoint_dir",
        str(args.checkpoint_dir),
        "--out_dir",
        str(args.crossplay_out_dir),
        "--condition",
        str(args.comm_condition),
        "--seeds",
        *[str(int(v)) for v in args.seeds],
        "--sender_milestones",
        *[str(int(v)) for v in args.crossplay_sender_milestones],
        "--receiver_milestones",
        *[str(int(v)) for v in args.crossplay_receiver_milestones],
        "--n_eval_episodes",
        str(int(args.n_eval_episodes)),
        "--eval_seed",
        str(int(args.eval_seed)),
        "--max_workers",
        str(int(args.max_workers)),
    ]
    if bool(args.skip_existing):
        crossplay_cmd.append("--skip_existing")
    _run(crossplay_cmd)

    manifest = {
        "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
        "suite_out_dir": os.path.abspath(args.suite_out_dir),
        "crossplay_out_dir": os.path.abspath(args.crossplay_out_dir),
        "comm_condition": str(args.comm_condition),
        "baseline_condition": str(args.baseline_condition),
        "seeds": [int(v) for v in args.seeds],
        "milestones": [int(v) for v in args.milestones],
        "interventions": [str(v) for v in args.interventions],
        "crossplay_sender_milestones": [int(v) for v in args.crossplay_sender_milestones],
        "crossplay_receiver_milestones": [int(v) for v in args.crossplay_receiver_milestones],
    }
    manifest_path = os.path.join(os.path.abspath(os.path.dirname(args.suite_out_dir)), "phase3_trimmed_eval_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[phase3-trimmed-eval] manifest={manifest_path}")


if __name__ == "__main__":
    main()
