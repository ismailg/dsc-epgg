from __future__ import annotations

import argparse
import csv
import math
import os
import random
from collections import defaultdict
from statistics import stdev
from typing import Dict, Iterable, List, Tuple

from src.analysis.condition_labels import condition_alias, condition_display


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(sorted_vals: List[float], q: float) -> float:
    if len(sorted_vals) == 0:
        return float('nan')
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _bootstrap_mean(values: List[float], n_boot: int, seed: int) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        draw = rng.choices(values, k=len(values))
        samples.append(sum(draw) / len(draw))
    samples.sort()
    return (
        float(sum(values) / len(values)),
        _percentile(samples, 0.025),
        _percentile(samples, 0.975),
    )


def _bootstrap_paired_delta(values: List[float], n_boot: int, seed: int) -> Tuple[float, float, float, float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        draw = rng.choices(values, k=len(values))
        samples.append(sum(draw) / len(draw))
    samples.sort()
    p_neg = sum(1 for x in samples if x <= 0.0) / len(samples)
    p_pos = sum(1 for x in samples if x >= 0.0) / len(samples)
    p_value = min(1.0, 2.0 * min(p_neg, p_pos))
    return (
        float(sum(values) / len(values)),
        _percentile(samples, 0.025),
        _percentile(samples, 0.975),
        float(p_value),
    )


def _sem(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(stdev(values) / math.sqrt(len(values)))


def _stars(p_value: float | None) -> str:
    if p_value is None or p_value != p_value:
        return ''
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return ''


def _filtered_rows(path: str) -> List[Dict[str, str]]:
    out = []
    for row in _read_rows(path):
        if row.get('scope') != 'f_value':
            continue
        if row.get('eval_policy', 'greedy') != 'greedy':
            continue
        if row.get('ablation', 'none') != 'none':
            continue
        if row.get('cross_play', 'none') != 'none':
            continue
        if row.get('sender_remap', 'none') != 'none':
            continue
        out.append(row)
    return out


def _main_rows(path: str) -> List[Dict]:
    out = []
    for row in _filtered_rows(path):
        out.append(
            {
                'group': 'main',
                'label': condition_alias(row['condition']),
                'display': condition_display(row['condition']),
                'condition': row['condition'],
                'train_seed': int(float(row['train_seed'])),
                'checkpoint_episode': int(float(row['checkpoint_episode'])),
                'f_value': row['key'],
                'coop_rate': float(row['coop_rate']),
                'avg_welfare': float(row['avg_welfare']),
            }
        )
    return out


def _sameckpt_rows(mode: str, path: str) -> List[Dict]:
    label_map = {
        'learned': 'learned',
        'always_zero': 'always_zero',
        'fixed0': 'always_zero',
        'indep_random': 'indep_random',
        'uniform': 'indep_random',
        'public_random': 'public_random',
    }
    label = label_map.get(mode, mode)
    out = []
    for row in _filtered_rows(path):
        out.append(
            {
                'group': 'sameckpt',
                'label': label,
                'display': label,
                'condition': row['condition'],
                'train_seed': int(float(row['train_seed'])),
                'checkpoint_episode': int(float(row['checkpoint_episode'])),
                'f_value': row['key'],
                'coop_rate': float(row['coop_rate']),
                'avg_welfare': float(row['avg_welfare']),
            }
        )
    return out


def _sameckpt_learned_reference(path: str) -> List[Dict]:
    out = []
    for row in _filtered_rows(path):
        if row['condition'] != 'cond1':
            continue
        out.append(
            {
                'group': 'sameckpt',
                'label': 'learned',
                'display': 'learned',
                'condition': row['condition'],
                'train_seed': int(float(row['train_seed'])),
                'checkpoint_episode': int(float(row['checkpoint_episode'])),
                'f_value': row['key'],
                'coop_rate': float(row['coop_rate']),
                'avg_welfare': float(row['avg_welfare']),
            }
        )
    return out


def _summaries(rows: List[Dict], n_boot: int, seed: int) -> List[Dict]:
    grouped: Dict[Tuple[str, str, int, str], List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[(row['group'], row['label'], row['checkpoint_episode'], row['f_value'])].append(row)

    out = []
    for (group, label, episode, f_value), cur in sorted(grouped.items()):
        for metric in ('coop_rate', 'avg_welfare'):
            vals = [float(r[metric]) for r in cur]
            mean, lo, hi = _bootstrap_mean(vals, n_boot=n_boot, seed=seed + int(episode))
            out.append(
                {
                    'group': group,
                    'label': label,
                    'checkpoint_episode': int(episode),
                    'f_value': f_value,
                    'metric': metric,
                    'mean': mean,
                    'ci_lower': lo,
                    'ci_upper': hi,
                    'sem': _sem(vals),
                    'n_seeds': len(vals),
                }
            )
    return out


def _paired_main(rows: List[Dict], comm_label: str, baseline_label: str, n_boot: int, seed: int) -> List[Dict]:
    by_key: Dict[Tuple[int, int, str, str], Dict[str, Dict]] = defaultdict(dict)
    for row in rows:
        if row['group'] != 'main':
            continue
        key = (row['train_seed'], row['checkpoint_episode'], row['f_value'], 'main_comm_vs_nocomm')
        by_key[key][row['label']] = row

    out = []
    metric_map = {'coop_rate': 'P(C)', 'avg_welfare': 'welfare'}
    for episode in sorted({k[1] for k in by_key.keys()}):
        for f_value in sorted({k[2] for k in by_key.keys()}, key=float):
            paired = [v for k, v in by_key.items() if k[1] == episode and k[2] == f_value and comm_label in v and baseline_label in v]
            if len(paired) == 0:
                continue
            for metric in ('coop_rate', 'avg_welfare'):
                deltas = [float(v[comm_label][metric]) - float(v[baseline_label][metric]) for v in paired]
                mean, lo, hi, p_value = _bootstrap_paired_delta(deltas, n_boot=n_boot, seed=seed + int(episode) + (1 if metric == 'avg_welfare' else 0))
                out.append(
                    {
                        'comparison_group': 'main_comm_vs_nocomm',
                        'lhs_label': comm_label,
                        'rhs_label': baseline_label,
                        'checkpoint_episode': int(episode),
                        'f_value': f_value,
                        'metric': metric_map[metric],
                        'delta_mean': mean,
                        'delta_ci_lower': lo,
                        'delta_ci_upper': hi,
                        'p_value': p_value,
                        'n_seed_pairs': len(deltas),
                    }
                )
    return out


def _paired_sameckpt(rows: List[Dict], n_boot: int, seed: int) -> List[Dict]:
    by_key: Dict[Tuple[int, int, str], Dict[str, Dict]] = defaultdict(dict)
    for row in rows:
        if row['group'] != 'sameckpt':
            continue
        key = (row['train_seed'], row['checkpoint_episode'], row['f_value'])
        by_key[key][row['label']] = row

    out = []
    metric_map = {'coop_rate': 'P(C)', 'avg_welfare': 'welfare'}
    modes = ['always_zero', 'indep_random', 'public_random']
    mode_seed_offsets = {'always_zero': 11, 'indep_random': 17, 'public_random': 23}
    for episode in sorted({k[1] for k in by_key.keys()}):
        for f_value in sorted({k[2] for k in by_key.keys()}, key=float):
            for mode in modes:
                paired = [v for k, v in by_key.items() if k[1] == episode and k[2] == f_value and 'learned' in v and mode in v]
                if len(paired) == 0:
                    continue
                for metric in ('coop_rate', 'avg_welfare'):
                    deltas = [float(v[mode][metric]) - float(v['learned'][metric]) for v in paired]
                    mean, lo, hi, p_value = _bootstrap_paired_delta(
                        deltas,
                        n_boot=n_boot,
                        seed=seed + int(episode) + mode_seed_offsets[mode] + (101 if metric == 'avg_welfare' else 0),
                    )
                    out.append(
                        {
                            'comparison_group': 'sameckpt_vs_learned',
                            'lhs_label': mode,
                            'rhs_label': 'learned',
                            'checkpoint_episode': int(episode),
                            'f_value': f_value,
                            'metric': metric_map[metric],
                            'delta_mean': mean,
                            'delta_ci_lower': lo,
                            'delta_ci_upper': hi,
                            'p_value': p_value,
                            'n_seed_pairs': len(deltas),
                        }
                    )
    return out


def _paper_rows(summary_rows: List[Dict], paired_rows: List[Dict]) -> List[Dict]:
    paired_lookup = {}
    for row in paired_rows:
        paired_lookup[(row['comparison_group'], row['lhs_label'], row['checkpoint_episode'], row['f_value'], row['metric'])] = row

    groups: Dict[Tuple[str, int, str, str], List[Dict]] = defaultdict(list)
    for row in summary_rows:
        metric_name = 'P(C)' if row['metric'] == 'coop_rate' else 'welfare'
        if row['group'] == 'main' and row['label'] in ('comm_symm', 'no_comm_symm'):
            group_name = 'main_comm_vs_nocomm'
        elif row['group'] == 'sameckpt' and row['label'] in ('learned', 'always_zero', 'indep_random', 'public_random'):
            group_name = 'sameckpt_vs_learned'
        else:
            continue
        groups[(group_name, row['checkpoint_episode'], row['f_value'], metric_name)].append(row)

    out = []
    for key, cur in sorted(groups.items()):
        group_name, episode, f_value, metric_name = key
        best = max(float(row['mean']) for row in cur)
        for row in cur:
            pair = paired_lookup.get((group_name, row['label'], row['checkpoint_episode'], row['f_value'], metric_name))
            out.append(
                {
                    'comparison_group': group_name,
                    'label': row['label'],
                    'checkpoint_episode': int(row['checkpoint_episode']),
                    'f_value': row['f_value'],
                    'metric': metric_name,
                    'mean': row['mean'],
                    'sem': row['sem'],
                    'ci_lower': row['ci_lower'],
                    'ci_upper': row['ci_upper'],
                    'p_value': '' if pair is None else pair['p_value'],
                    'stars': '' if pair is None else _stars(float(pair['p_value'])),
                    'is_best_in_group': int(abs(float(row['mean']) - best) < 1e-12),
                }
            )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--main_suite_csv', type=str, default='outputs/eval/phase3_annealed_ext150k_5seeds/suite/checkpoint_suite_main.csv')
    p.add_argument('--comm_label', type=str, default='comm_symm')
    p.add_argument('--baseline_label', type=str, default='no_comm_symm')
    p.add_argument('--sameckpt_suite', nargs=2, action='append', metavar=('MODE', 'CSV'), default=[])
    p.add_argument('--n_boot', type=int, default=10000)
    p.add_argument('--seed', type=int, default=9001)
    p.add_argument('--out_dir', type=str, default='outputs/eval/phase3_stats')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = _main_rows(args.main_suite_csv)
    for mode, path in args.sameckpt_suite:
        rows.extend(_sameckpt_rows(mode, path))
    if len(args.sameckpt_suite) > 0:
        rows.extend(_sameckpt_learned_reference(args.main_suite_csv))

    summary_rows = _summaries(rows, n_boot=int(args.n_boot), seed=int(args.seed))
    paired_rows = []
    paired_rows.extend(_paired_main(rows, comm_label=str(args.comm_label), baseline_label=str(args.baseline_label), n_boot=int(args.n_boot), seed=int(args.seed) + 100))
    if len(args.sameckpt_suite) > 0:
        paired_rows.extend(_paired_sameckpt(rows, n_boot=int(args.n_boot), seed=int(args.seed) + 200))
    paper_rows = _paper_rows(summary_rows, paired_rows)

    _write_csv(os.path.join(out_dir, 'bootstrap_ci_table.csv'), summary_rows)
    _write_csv(os.path.join(out_dir, 'paired_delta_table.csv'), paired_rows)
    _write_csv(os.path.join(out_dir, 'paper_stats_table.csv'), paper_rows)
    print(f'[bootstrap-cis] out_dir={out_dir}')


if __name__ == '__main__':
    main()
