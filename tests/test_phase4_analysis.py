import csv
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
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


BASE_FIELDS = {
    'scope': 'f_value',
    'eval_policy': 'greedy',
    'ablation': 'none',
    'cross_play': 'none',
    'sender_remap': 'none',
}


def test_channel_control_summary_supports_generic_mode_suite(tmp_path: Path):
    learned_csv = tmp_path / 'learned.csv'
    zero_csv = tmp_path / 'zero.csv'
    rows = [
        {
            **BASE_FIELDS,
            'condition': 'cond1',
            'train_seed': '101',
            'checkpoint_episode': '50000',
            'key': '3.500',
            'coop_rate': '0.5',
            'avg_welfare': '10.0',
        },
        {
            **BASE_FIELDS,
            'condition': 'cond1',
            'train_seed': '202',
            'checkpoint_episode': '50000',
            'key': '3.500',
            'coop_rate': '0.7',
            'avg_welfare': '14.0',
        },
    ]
    _write_csv(learned_csv, rows)
    _write_csv(zero_csv, rows)
    out_dir = tmp_path / 'report'
    subprocess.run(
        [
            sys.executable,
            '-m',
            'src.analysis.summarize_phase3_channel_controls',
            '--mode_suite',
            'learned',
            str(learned_csv),
            '--mode_suite',
            'always_zero',
            str(zero_csv),
            '--out_dir',
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    summary = list(csv.DictReader(open(out_dir / 'channel_control_summary.csv', 'r', encoding='utf-8')))
    assert len(summary) == 2
    learned_row = next(row for row in summary if row['mode'] == 'learned')
    assert float(learned_row['mean_coop_rate']) == 0.6
    assert float(learned_row['std_coop_rate']) > 0.0
    assert float(learned_row['sem_coop_rate']) > 0.0


def test_compute_bootstrap_cis_outputs_main_and_sameckpt_tables(tmp_path: Path):
    main_csv = tmp_path / 'main.csv'
    sameckpt_csv = tmp_path / 'sameckpt.csv'
    main_rows = []
    for seed, comm, base in [(101, (0.6, 20.0), (0.4, 16.0)), (202, (0.7, 22.0), (0.5, 18.0))]:
        for condition, vals in [('cond1', comm), ('cond2', base)]:
            main_rows.append(
                {
                    **BASE_FIELDS,
                    'condition': condition,
                    'train_seed': str(seed),
                    'checkpoint_episode': '150000',
                    'key': '3.500',
                    'coop_rate': str(vals[0]),
                    'avg_welfare': str(vals[1]),
                }
            )
    _write_csv(main_csv, main_rows)
    sameckpt_rows = []
    for seed, val in [(101, 0.55), (202, 0.65)]:
        sameckpt_rows.append(
            {
                **BASE_FIELDS,
                'condition': 'cond1',
                'train_seed': str(seed),
                'checkpoint_episode': '150000',
                'key': '3.500',
                'coop_rate': str(val),
                'avg_welfare': str(19.0 + val),
            }
        )
    _write_csv(sameckpt_csv, sameckpt_rows)
    out_dir = tmp_path / 'stats'
    subprocess.run(
        [
            sys.executable,
            '-m',
            'src.analysis.compute_bootstrap_cis',
            '--main_suite_csv',
            str(main_csv),
            '--sameckpt_suite',
            'always_zero',
            str(sameckpt_csv),
            '--n_boot',
            '200',
            '--seed',
            '7',
            '--out_dir',
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    paired = list(csv.DictReader(open(out_dir / 'paired_delta_table.csv', 'r', encoding='utf-8')))
    groups = {row['comparison_group'] for row in paired}
    assert 'main_comm_vs_nocomm' in groups
    assert 'sameckpt_vs_learned' in groups
    paper = list(csv.DictReader(open(out_dir / 'paper_stats_table.csv', 'r', encoding='utf-8')))
    assert any(row['label'] == 'comm_symm' for row in paper)
    assert any(row['label'] == 'always_zero' for row in paper)


def test_summarize_sender_slot_permutation_compares_none_vs_permute(tmp_path: Path):
    suite_csv = tmp_path / 'suite.csv'
    rows = []
    for seed, none_val, perm_val in [(101, 0.7, 0.6), (202, 0.5, 0.45)]:
        for ablation, coop, welfare in [('none', none_val, 40.0), ('permute_slots', perm_val, 36.0)]:
            rows.append(
                {
                    **BASE_FIELDS,
                    'condition': 'cond1',
                    'condition_alias': 'comm_symm',
                    'train_seed': str(seed),
                    'checkpoint_episode': '150000',
                    'key': '3.500',
                    'ablation': ablation,
                    'coop_rate': str(coop),
                    'avg_welfare': str(welfare),
                }
            )
    _write_csv(suite_csv, rows)
    out_dir = tmp_path / 'interop'
    subprocess.run(
        [
            sys.executable,
            '-m',
            'src.analysis.summarize_sender_slot_permutation',
            '--bundle',
            'learned',
            str(suite_csv),
            '--out_dir',
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    summary = list(csv.DictReader(open(out_dir / 'permute_slots_summary.csv', 'r', encoding='utf-8')))
    assert len(summary) == 1
    row = summary[0]
    assert row['bundle'] == 'learned'
    assert abs(float(row['mean_delta_coop_none_minus_permute']) - 0.075) < 1e-9
    assert (out_dir / 'permute_slots_summary.md').exists()
