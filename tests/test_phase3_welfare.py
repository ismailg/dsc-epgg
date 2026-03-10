from __future__ import annotations

import csv
from pathlib import Path

from src.analysis.summarize_phase3_welfare import main


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_weighted_welfare_summary(tmp_path, monkeypatch):
    suite_csv = tmp_path / "suite" / "checkpoint_suite_main.csv"
    _write_csv(
        suite_csv,
        [
            {
                "condition": "cond1",
                "scope": "f_value",
                "key": "3.500",
                "n_rounds": "30",
                "coop_rate": "0.5",
                "avg_welfare": "40",
                "eval_policy": "greedy",
                "ablation": "none",
                "cross_play": "none",
                "checkpoint_episode": "50000",
                "train_seed": "101",
            },
            {
                "condition": "cond1",
                "scope": "f_value",
                "key": "5.000",
                "n_rounds": "10",
                "coop_rate": "0.9",
                "avg_welfare": "80",
                "eval_policy": "greedy",
                "ablation": "none",
                "cross_play": "none",
                "checkpoint_episode": "50000",
                "train_seed": "101",
            },
            {
                "condition": "cond2",
                "scope": "f_value",
                "key": "3.500",
                "n_rounds": "20",
                "coop_rate": "0.2",
                "avg_welfare": "30",
                "eval_policy": "greedy",
                "ablation": "none",
                "cross_play": "none",
                "checkpoint_episode": "50000",
                "train_seed": "101",
            },
        ],
    )

    out_csv = tmp_path / "report" / "welfare_raw.csv"
    out_mean_csv = tmp_path / "report" / "welfare_mean.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_phase3_welfare.py",
            "--suite_main_csv",
            str(suite_csv),
            "--bundle_label",
            "demo",
            "--out_csv",
            str(out_csv),
            "--out_mean_csv",
            str(out_mean_csv),
        ],
    )
    main()

    raw_rows = list(csv.DictReader(out_csv.open()))
    assert len(raw_rows) == 2
    cond1 = next(row for row in raw_rows if row["condition"] == "cond1")
    assert cond1["condition_alias"] == "comm_symm"
    assert abs(float(cond1["weighted_coop_rate"]) - 0.6) < 1e-9
    assert abs(float(cond1["weighted_avg_welfare"]) - 50.0) < 1e-9

    mean_rows = list(csv.DictReader(out_mean_csv.open()))
    assert len(mean_rows) == 2
    mean_cond1 = next(row for row in mean_rows if row["condition"] == "cond1")
    assert mean_cond1["bundle_label"] == "demo"
    assert abs(float(mean_cond1["mean_weighted_avg_welfare"]) - 50.0) < 1e-9
