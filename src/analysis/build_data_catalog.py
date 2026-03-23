#!/usr/bin/env python3
"""Build a lightweight catalog of train/eval artifacts under outputs/.

The goal is practical discoverability, not a perfect ontology. We summarize:
- train roots that contain checkpoints / metrics
- eval roots that contain suite CSVs / crossplay matrices / reports

This writes:
- CSV catalog for filtering/querying
- Markdown catalog for quick human browsing
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CHECKPOINT_RE = re.compile(r"^(cond[0-9]+)_seed([0-9]+)(?:_ep([0-9]+))?\.pt$")
METRICS_RE = re.compile(r"^(cond[0-9]+)_seed([0-9]+)\.jsonl$")


def _write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows for {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _rel(path: str, repo_root: str) -> str:
    return os.path.relpath(path, repo_root)


def _collect_train_roots(outputs_train_root: str) -> List[str]:
    roots: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(outputs_train_root):
        if os.path.abspath(dirpath) == os.path.abspath(outputs_train_root):
            continue
        if any(name.endswith(".pt") for name in filenames):
            roots.add(dirpath)
    return sorted(roots)


def _collect_eval_roots(outputs_eval_root: str) -> List[str]:
    roots: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(outputs_eval_root):
        fnames = set(filenames)
        if "checkpoint_suite_main.csv" in fnames:
            if os.path.basename(dirpath) == "suite":
                roots.add(os.path.dirname(dirpath))
            else:
                roots.add(dirpath)
        if "crossplay_matrix_main.csv" in fnames:
            if os.path.basename(dirpath) == "crossplay":
                roots.add(os.path.dirname(dirpath))
            else:
                roots.add(dirpath)
    return sorted(roots)


def _summarize_train_root(root: str, repo_root: str) -> Dict[str, object]:
    conditions: set[str] = set()
    seeds: set[int] = set()
    final_seeds: Dict[str, set[int]] = defaultdict(set)
    ckpt_episodes: set[int] = set()
    metrics_conditions: set[str] = set()
    metrics_seeds: set[int] = set()
    logs = 0
    run_manifests = 0
    checkpoints = 0

    for dirpath, _dirnames, filenames in os.walk(root):
        rel_dir = _rel(dirpath, repo_root)
        for fname in filenames:
            m = CHECKPOINT_RE.match(fname)
            if m:
                checkpoints += 1
                cond = m.group(1)
                seed = int(m.group(2))
                ep = m.group(3)
                conditions.add(cond)
                seeds.add(seed)
                if ep is None:
                    final_seeds[cond].add(seed)
                else:
                    ckpt_episodes.add(int(ep))
                continue
            m2 = METRICS_RE.match(fname)
            if m2 and os.path.basename(dirpath) == "metrics":
                metrics_conditions.add(m2.group(1))
                metrics_seeds.add(int(m2.group(2)))
                continue
            if fname.endswith(".run.json"):
                run_manifests += 1
            if fname.endswith(".log"):
                logs += 1

    return {
        "kind": "train",
        "root": _rel(root, repo_root),
        "conditions": ",".join(sorted(conditions)),
        "n_conditions": len(conditions),
        "n_unique_seeds": len(seeds),
        "seed_list": ",".join(str(s) for s in sorted(seeds)),
        "final_checkpoint_seed_count": sum(len(v) for v in final_seeds.values()),
        "checkpoint_episodes": ",".join(str(ep) for ep in sorted(ckpt_episodes)),
        "n_checkpoint_files": checkpoints,
        "metrics_seed_count": len(metrics_seeds),
        "metrics_seed_list": ",".join(str(s) for s in sorted(metrics_seeds)),
        "n_run_manifests": run_manifests,
        "n_log_files": logs,
    }


def _suite_main_path(root: str) -> Optional[str]:
    direct = os.path.join(root, "checkpoint_suite_main.csv")
    if os.path.exists(direct):
        return direct
    suite = os.path.join(root, "suite", "checkpoint_suite_main.csv")
    if os.path.exists(suite):
        return suite
    return None


def _crossplay_main_path(root: str) -> Optional[str]:
    direct = os.path.join(root, "crossplay_matrix_main.csv")
    if os.path.exists(direct):
        return direct
    nested = os.path.join(root, "crossplay", "crossplay_matrix_main.csv")
    if os.path.exists(nested):
        return nested
    return None


def _summarize_eval_root(root: str, repo_root: str) -> Dict[str, object]:
    suite_path = _suite_main_path(root)
    crossplay_path = _crossplay_main_path(root)
    report_dir = os.path.join(root, "report")
    raw_dir = os.path.join(root, "raw")

    conditions: set[str] = set()
    seeds: set[int] = set()
    checkpoint_episodes: set[int] = set()
    ablations: set[str] = set()
    eval_policies: set[str] = set()
    suite_rows = 0
    crossplay_rows = 0
    has_sender_summary = False
    has_receiver_summary = False

    if suite_path is not None:
        rows = _read_csv_rows(suite_path)
        suite_rows = len(rows)
        for row in rows:
            if "condition" in row:
                conditions.add(str(row["condition"]))
            if "train_seed" in row and str(row["train_seed"]).strip() != "":
                seeds.add(int(row["train_seed"]))
            if "checkpoint_episode" in row and str(row["checkpoint_episode"]).strip() != "":
                checkpoint_episodes.add(int(row["checkpoint_episode"]))
            if "ablation" in row and str(row["ablation"]).strip() != "":
                ablations.add(str(row["ablation"]))
            if "eval_policy" in row and str(row["eval_policy"]).strip() != "":
                eval_policies.add(str(row["eval_policy"]))
    if crossplay_path is not None:
        crossplay_rows = len(_read_csv_rows(crossplay_path))
    if os.path.exists(os.path.join(report_dir, "sender_semantics_summary.csv")):
        has_sender_summary = True
    if os.path.exists(os.path.join(report_dir, "receiver_semantics_summary.csv")):
        has_receiver_summary = True

    return {
        "kind": "eval",
        "root": _rel(root, repo_root),
        "conditions": ",".join(sorted(conditions)),
        "n_conditions": len(conditions),
        "n_unique_seeds": len(seeds),
        "seed_list": ",".join(str(s) for s in sorted(seeds)),
        "checkpoint_episodes": ",".join(str(ep) for ep in sorted(checkpoint_episodes)),
        "ablations": ",".join(sorted(ablations)),
        "eval_policies": ",".join(sorted(eval_policies)),
        "suite_main_csv": _rel(suite_path, repo_root) if suite_path else "",
        "suite_rows": suite_rows,
        "crossplay_main_csv": _rel(crossplay_path, repo_root) if crossplay_path else "",
        "crossplay_rows": crossplay_rows,
        "has_report_dir": os.path.isdir(report_dir),
        "has_raw_dir": os.path.isdir(raw_dir),
        "has_sender_summary": has_sender_summary,
        "has_receiver_summary": has_receiver_summary,
    }


def build_catalog(repo_root: str) -> List[Dict[str, object]]:
    outputs_train_root = os.path.join(repo_root, "outputs", "train")
    outputs_eval_root = os.path.join(repo_root, "outputs", "eval")

    rows: List[Dict[str, object]] = []
    for root in _collect_train_roots(outputs_train_root):
        rows.append(_summarize_train_root(root, repo_root))
    for root in _collect_eval_roots(outputs_eval_root):
        rows.append(_summarize_eval_root(root, repo_root))
    rows.sort(key=lambda r: (str(r["kind"]), str(r["root"])))
    return rows


def _write_markdown(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines: List[str] = []
    lines.append("# Data Catalog")
    lines.append("")
    lines.append("Auto-generated summary of train/eval artifact roots under `outputs/`.")
    lines.append("")
    lines.append("## Train Roots")
    lines.append("")
    lines.append("| Root | Conditions | Seeds | Checkpoint episodes | Metrics seeds | Notes |")
    lines.append("|---|---:|---:|---|---:|---|")
    for row in rows:
        if row["kind"] != "train":
            continue
        notes = []
        if int(row["n_run_manifests"]) > 0:
            notes.append(f"{row['n_run_manifests']} run manifests")
        if int(row["n_log_files"]) > 0:
            notes.append(f"{row['n_log_files']} logs")
        lines.append(
            f"| `{row['root']}` | `{row['conditions']}` | {row['n_unique_seeds']} | "
            f"`{row['checkpoint_episodes']}` | {row['metrics_seed_count']} | "
            f"{'; '.join(notes) if notes else ''} |"
        )
    lines.append("")
    lines.append("## Eval Roots")
    lines.append("")
    lines.append("| Root | Conditions | Seeds | Checkpoint episodes | Policies | Suite rows | Crossplay |")
    lines.append("|---|---:|---:|---|---|---:|---:|")
    for row in rows:
        if row["kind"] != "eval":
            continue
        lines.append(
            f"| `{row['root']}` | `{row['conditions']}` | {row['n_unique_seeds']} | "
            f"`{row['checkpoint_episodes']}` | `{row['eval_policies']}` | "
            f"{row['suite_rows']} | {row['crossplay_rows']} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", type=str, default=".")
    p.add_argument("--out_dir", type=str, default="outputs/data_catalog")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = os.path.abspath(args.repo_root)
    out_dir = os.path.abspath(os.path.join(repo_root, args.out_dir))
    rows = build_catalog(repo_root=repo_root)
    _write_csv(os.path.join(out_dir, "data_catalog.csv"), rows)
    _write_markdown(os.path.join(out_dir, "DATA_CATALOG.md"), rows)
    print(f"[ok] wrote {out_dir}")


if __name__ == "__main__":
    main()
