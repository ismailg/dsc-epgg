from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple


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


def _mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _filtered(path: str) -> List[Dict]:
    out = []
    for row in _read_rows(path):
        if row.get('scope') != 'f_value':
            continue
        if row.get('eval_policy', 'greedy') != 'greedy':
            continue
        if row.get('cross_play', 'none') != 'none':
            continue
        if row.get('sender_remap', 'none') != 'none':
            continue
        if row.get('key') not in ('3.500', '5.000'):
            continue
        out.append(row)
    return out


def _collect_bundle(bundle: str, suite_csv: str) -> List[Dict]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Dict]] = defaultdict(dict)
    for row in _filtered(suite_csv):
        key = (
            str(row['train_seed']),
            str(row['checkpoint_episode']),
            str(row['key']),
            str(row.get('condition_alias') or row.get('condition')),
        )
        grouped[key][row['ablation']] = row

    per_seed = []
    for (seed, episode, f_value, condition), ablations in sorted(grouped.items()):
        if 'none' not in ablations or 'permute_slots' not in ablations:
            continue
        base = ablations['none']
        perm = ablations['permute_slots']
        per_seed.append(
            {
                'bundle': bundle,
                'condition': condition,
                'train_seed': int(seed),
                'checkpoint_episode': int(float(episode)),
                'f_value': f_value,
                'coop_none': float(base['coop_rate']),
                'coop_permute_slots': float(perm['coop_rate']),
                'delta_coop_none_minus_permute': float(base['coop_rate']) - float(perm['coop_rate']),
                'welfare_none': float(base['avg_welfare']),
                'welfare_permute_slots': float(perm['avg_welfare']),
                'delta_welfare_none_minus_permute': float(base['avg_welfare']) - float(perm['avg_welfare']),
            }
        )
    return per_seed


def _summarize(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, int, str], List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[(row['bundle'], row['condition'], row['checkpoint_episode'], row['f_value'])].append(row)

    out = []
    for (bundle, condition, episode, f_value), cur in sorted(grouped.items()):
        out.append(
            {
                'bundle': bundle,
                'condition': condition,
                'checkpoint_episode': int(episode),
                'f_value': f_value,
                'n_seeds': len(cur),
                'mean_delta_coop_none_minus_permute': _mean([float(r['delta_coop_none_minus_permute']) for r in cur]),
                'mean_delta_welfare_none_minus_permute': _mean([float(r['delta_welfare_none_minus_permute']) for r in cur]),
            }
        )
    return out


def _write_md(path: str, summary_rows: List[Dict]):
    lines = ['# Sender-Slot Permutation Summary', '']
    if len(summary_rows) == 0:
        lines.append('No rows with both `none` and `permute_slots` were found.')
    else:
        lines.append('Positive deltas mean permutation hurts performance, which supports sender-conditioned decoding.')
        lines.append('')
        for row in summary_rows:
            if int(row['checkpoint_episode']) != 150000:
                continue
            lines.append(
                f"- {row['bundle']} / {row['condition']} / f={row['f_value']}: "
                f"delta P(C)={float(row['mean_delta_coop_none_minus_permute']):+.3f}, "
                f"delta welfare={float(row['mean_delta_welfare_none_minus_permute']):+.3f}, n={int(row['n_seeds'])}"
            )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', nargs=2, action='append', metavar=('LABEL', 'CSV'), default=[])
    p.add_argument('--out_dir', type=str, default='outputs/eval/phase3_interoperability')
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.bundle) == 0:
        raise ValueError('at least one --bundle LABEL CSV is required')

    rows = []
    for label, csv_path in args.bundle:
        rows.extend(_collect_bundle(str(label), str(csv_path)))
    summary_rows = _summarize(rows)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, 'permute_slots_raw.csv'), rows)
    _write_csv(os.path.join(out_dir, 'permute_slots_summary.csv'), summary_rows)
    _write_md(os.path.join(out_dir, 'permute_slots_summary.md'), summary_rows)
    print(f'[permute-slots] out_dir={out_dir}')


if __name__ == '__main__':
    main()
