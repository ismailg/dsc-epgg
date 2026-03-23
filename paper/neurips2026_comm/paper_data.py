"""Single source of truth for every number in the paper.

Every quantitative claim in ``main.qmd`` traces back to a function or
attribute in this module. This file also carries the explicit manuscript
source map: which experiment output folder feeds each paper component, and
which components are stitched from multiple sources.

Run ``python paper_data.py`` to execute self-checks.
Run ``python paper_data.py --gen-source-map`` to write a manuscript-facing
source-map markdown file next to the paper.
"""
from __future__ import annotations

import csv
import itertools
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import stdev
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional acceleration only
    np = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
DATA_ROOT = _HERE / ".." / ".." / "outputs" / "eval"
SOURCE_MAP_MD = _HERE / "paper_source_map.md"

MAIN_CSV = DATA_ROOT / "phase3_annealed_ext150k_15seeds" / "suite" / "checkpoint_suite_main.csv"
INTERVENTION_CSV = DATA_ROOT / "phase3_annealed_ext150k_15seeds" / "report" / "intervention_delta_table.csv"
RECV_SEMANTICS_CSV = DATA_ROOT / "phase3_annealed_ext150k_15seeds" / "report" / "receiver_semantics_summary.csv"
RECV_BY_SENDER_CSV = DATA_ROOT / "phase3_annealed_ext150k_15seeds" / "report" / "receiver_by_sender_summary.csv"
ALIGNMENT_CSV = DATA_ROOT / "phase3_annealed_ext150k_15seeds" / "report" / "sender_alignment_summary.csv"

# Frozen-checkpoint interventions: 50k and 150k from separate suites
FROZEN_50K_CSV = DATA_ROOT / "phase3_frozen50k_15seeds" / "report" / "intervention_delta_table.csv"
FROZEN_150K_CSV = DATA_ROOT / "phase3_frozen150k_15seeds_local_20260319" / "report" / "intervention_delta_table.csv"

CTRL_50K_CSV = DATA_ROOT / "phase3_channel_controls_50k" / "report" / "channel_control_summary.csv"
CTRL_CONT_CSV = DATA_ROOT / "phase3_channel_controls_ext150k_3seeds" / "report" / "channel_control_summary.csv"

MUTE_INTERVENTION_CSV = DATA_ROOT / "phase3_mute_after50k_ext150k_3seeds" / "report" / "intervention_delta_table.csv"

# Same-checkpoint continuation (15-seed)
SAMECKPT_DIR = DATA_ROOT / "phase3_sameckpt_continuation_15seeds_local_20260319"
SAMECKPT_SENDER_SHUFFLE_50K_CSV = (
    DATA_ROOT
    / "paper_strengthen"
    / "iter4_sameckpt_sender_shuffle_50k_fixed"
    / "sender_shuffle"
    / "suite"
    / "checkpoint_suite_main.csv"
)
VALUE_SURFACE_CSV = DATA_ROOT / "paper_strengthen" / "iter0_value_surface" / "value_surface_summary.csv"
SENDER_CAUSAL_MATRIX_CSV = (
    DATA_ROOT / "paper_strengthen" / "iter2_sender_causal_15seeds" / "sender_causal_matrix.csv"
)


# ---------------------------------------------------------------------------
# Explicit paper component source map
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceMapEntry:
    """One manuscript component and the concrete output folders it uses."""

    component: str
    old_source: str
    new_source: str
    paths: Tuple[Path, ...]
    note: str = ""


def _path_label(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_ROOT))
    except ValueError:
        return str(path)


def build_source_map() -> List[SourceMapEntry]:
    """Return the explicit paper-component to data-source mapping.

    The goal is not only to expose the active path constants, but also to make
    clear which manuscript components are intentionally unchanged and which are
    stitched from multiple experiment folders.
    """

    return [
        SourceMapEntry(
            component="Table 1 (comm vs no-comm)",
            old_source="phase3_annealed_ext150k_5seeds",
            new_source="phase3_annealed_ext150k_15seeds",
            paths=(MAIN_CSV,),
            note="Uses the 15-seed base checkpoint-suite main CSV.",
        ),
        SourceMapEntry(
            component="Table 2 (separate controls 50k)",
            old_source="phase3_channel_controls_50k",
            new_source="unchanged (no new data)",
            paths=(CTRL_50K_CSV,),
            note="Still the older 50k separate-training control experiment; not rerun at 15 seeds.",
        ),
        SourceMapEntry(
            component="Table 3 (same-checkpoint continuation)",
            old_source="phase3_sameckpt_continuation_5seeds",
            new_source="phase3_sameckpt_continuation_15seeds_local_20260319",
            paths=(SAMECKPT_DIR / "report" / "channel_control_summary.csv",),
            note="15-seed summary regenerated locally from the completed same-checkpoint continuations.",
        ),
        SourceMapEntry(
            component="Table 4 (frozen-checkpoint interventions)",
            old_source="phase3_frozen_ckpt_5seeds",
            new_source=(
                "merge: 50k from phase3_frozen50k_15seeds; "
                "150k from phase3_frozen150k_15seeds_local_20260319; "
                "fixed0/fixed1 from phase3_annealed_ext150k_15seeds"
            ),
            paths=(FROZEN_50K_CSV, FROZEN_150K_CSV, INTERVENTION_CSV),
            note=(
                "This table is intentionally stitched. The reduced frozen suites "
                "supply zeros/public/indep/shuffle/permute; fixed0/fixed1 remain in "
                "the base 15-seed eval report."
            ),
        ),
        SourceMapEntry(
            component="Value-surface figure",
            old_source="not previously shown",
            new_source="paper_strengthen/iter0_value_surface",
            paths=(VALUE_SURFACE_CSV,),
            note="Full multiplier-surface summary used for the new three-panel synthesis figure.",
        ),
        SourceMapEntry(
            component="Endpoint sender-causal table",
            old_source="not previously shown",
            new_source="paper_strengthen/iter2_sender_causal_15seeds",
            paths=(SENDER_CAUSAL_MATRIX_CSV,),
            note=(
                "Seed-aggregated frozen-checkpoint sender->receiver intervention summary "
                "computed from the full 15-seed causal matrix."
            ),
        ),
        SourceMapEntry(
            component="Table 5 (mute)",
            old_source="phase3_mute_after50k_ext150k_3seeds",
            new_source="unchanged (no new data)",
            paths=(MUTE_INTERVENTION_CSV,),
            note="Still exploratory and still 3 seeds.",
        ),
        SourceMapEntry(
            component="Table 6 (per-sender disaggregation)",
            old_source="phase3_annealed_ext150k_5seeds",
            new_source="phase3_annealed_ext150k_15seeds",
            paths=(RECV_SEMANTICS_CSV, RECV_BY_SENDER_CSV),
            note="Aggregate and per-sender receiver-response summaries now come from the 15-seed base report.",
        ),
        SourceMapEntry(
            component="Alignment appendix",
            old_source="phase3_annealed_ext150k_5seeds",
            new_source="phase3_annealed_ext150k_15seeds",
            paths=(ALIGNMENT_CSV,),
            note="Uses the 15-seed sender alignment summary from the base report.",
        ),
    ]


def generate_source_map_markdown(entries: Optional[Sequence[SourceMapEntry]] = None) -> str:
    """Render the explicit manuscript source map as markdown."""

    rows = list(entries) if entries is not None else build_source_map()
    lines = [
        "# Paper Source Map",
        "",
        "This file is auto-generated by `paper_data.py --gen-source-map`.",
        "It records which experiment outputs feed each paper component.",
        "",
        "| Paper component | Old source | New source | Concrete path(s) | Note |",
        "|---|---|---|---|---|",
    ]
    for entry in rows:
        paths = "<br>".join(f"`{_path_label(path)}`" for path in entry.paths)
        note = entry.note.replace("\n", " ").strip()
        lines.append(
            f"| {entry.component} | `{entry.old_source}` | {entry.new_source} | {paths} | {note} |"
        )
    lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: Dict, key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", None) else default


def _int(row: Dict, key: str, default: int = 0) -> int:
    v = row.get(key, "")
    return int(float(v)) if v not in ("", None) else default

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _sem(vals: List[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    return stdev(vals) / math.sqrt(len(vals))


def _bootstrap_mean(
    values: List[float], n_boot: int = 10_000, seed: int = 9001
) -> Tuple[float, float, float]:
    """Return (mean, ci_lower_2.5%, ci_upper_97.5%)."""
    if np is not None:
        arr = np.asarray(values, dtype=float)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
        samples = np.sort(arr[idx].mean(axis=1))
        lo = float(samples[int(0.025 * len(samples))])
        hi = float(samples[int(0.975 * len(samples))])
        return float(arr.mean()), lo, hi
    rng = random.Random(seed)
    samples = sorted(
        _mean(rng.choices(values, k=len(values))) for _ in range(n_boot)
    )
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples))]
    return _mean(values), lo, hi


def _bootstrap_paired_delta(
    values: List[float], n_boot: int = 10_000, seed: int = 9001
) -> Tuple[float, float, float, float]:
    """Return (delta_mean, ci_lower, ci_upper, bootstrap p_value)."""
    if np is not None:
        arr = np.asarray(values, dtype=float)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
        samples = np.sort(arr[idx].mean(axis=1))
        lo = float(samples[int(0.025 * len(samples))])
        hi = float(samples[int(0.975 * len(samples))])
        p_neg = float(np.mean(samples <= 0.0))
        p_pos = float(np.mean(samples >= 0.0))
        p_val = min(1.0, 2.0 * min(p_neg, p_pos))
        return float(arr.mean()), lo, hi, p_val
    rng = random.Random(seed)
    samples = sorted(
        _mean(rng.choices(values, k=len(values))) for _ in range(n_boot)
    )
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples))]
    p_neg = sum(1 for x in samples if x <= 0) / len(samples)
    p_pos = sum(1 for x in samples if x >= 0) / len(samples)
    p_val = min(1.0, 2.0 * min(p_neg, p_pos))
    return _mean(values), lo, hi, p_val


def _exact_sign_flip_p_value(values: List[float]) -> float:
    """Exact two-sided paired sign-flip p-value for the mean delta."""
    vals = [float(v) for v in values]
    if len(vals) == 0:
        return 1.0
    if np is not None:
        arr = np.asarray(vals, dtype=float)
        n = arr.size
        if n > 20:
            observed = abs(float(arr.mean()))
            rng = np.random.default_rng(9001)
            signs = rng.choice((-1.0, 1.0), size=(200_000, n))
            flipped = np.sum(signs * arr[None, :], axis=1) / float(n)
            exceed = np.count_nonzero(np.abs(flipped) + 1e-12 >= observed)
            return float(exceed) / float(flipped.size)
        sign_bits = (
            (np.arange(1 << n, dtype=np.uint32)[:, None] >> np.arange(n, dtype=np.uint32)) & 1
        )
        signs = sign_bits.astype(np.float64) * 2.0 - 1.0
        flipped = np.sum(signs * arr[None, :], axis=1) / float(n)
        observed = abs(float(arr.mean()))
        exceed = np.count_nonzero(np.abs(flipped) + 1e-12 >= observed)
        return float(exceed) / float(flipped.size)
    observed = abs(_mean(vals))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(vals)):
        total += 1
        flipped = sum(sign * val for sign, val in zip(signs, vals)) / float(len(vals))
        if abs(flipped) + 1e-12 >= observed:
            exceed += 1
    return float(exceed) / float(total)


def _paired_delta_summary(
    values: List[float], n_boot: int = 10_000, seed: int = 9001
) -> Tuple[float, float, float, float]:
    """Return bootstrap CI for magnitude and exact sign-flip p-value."""
    delta_mean, lo, hi, _bootstrap_p = _bootstrap_paired_delta(
        values, n_boot=n_boot, seed=seed
    )
    return delta_mean, lo, hi, _exact_sign_flip_p_value(values)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(val: float, d: int = 3) -> str:
    """Format a float to *d* decimal places."""
    return f"{val:.{d}f}"


def fmt_pm(mean: float, sem: float, d: int = 3) -> str:
    return f"{mean:.{d}f} \\pm {sem:.{d}f}"


def fmt_ci(mean: float, lo: float, hi: float, d: int = 3) -> str:
    return f"{mean:.{d}f}\\;[{lo:.{d}f},\\,{hi:.{d}f}]"


def fmt_delta(val: float, d: int = 3) -> str:
    return f"{val:+.{d}f}"


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

# ---------------------------------------------------------------------------
# Table 1:  comm vs no-comm gap
# ---------------------------------------------------------------------------

@dataclass
class CoopCell:
    """One cell of Table 1: per-seed values + summary stats."""
    seeds: List[int]
    values: List[float]
    mean: float = 0.0
    sem: float = 0.0
    ci_lo: float = 0.0
    ci_hi: float = 0.0

    def __post_init__(self):
        self.mean = _mean(self.values)
        self.sem = _sem(self.values)
        self.mean, self.ci_lo, self.ci_hi = _bootstrap_mean(self.values)


@dataclass
class GapCell:
    """Comm gap = comm - no_comm, paired by seed."""
    deltas: List[float]
    mean: float = 0.0
    ci_lo: float = 0.0
    ci_hi: float = 0.0
    p_value: float = 1.0

    def __post_init__(self):
        self.mean, self.ci_lo, self.ci_hi, self.p_value = (
            _paired_delta_summary(self.deltas)
        )


@dataclass
class Table1:
    """All data for Table 1 and Appendix B."""
    # keys: (f_value_str, checkpoint_episode)
    comm: Dict[Tuple[str, int], CoopCell] = field(default_factory=dict)
    no_comm: Dict[Tuple[str, int], CoopCell] = field(default_factory=dict)
    gap: Dict[Tuple[str, int], GapCell] = field(default_factory=dict)
    # Per-seed gap for Appendix B: (seed, f_value_str, episode) -> delta
    per_seed_gap: Dict[Tuple[int, str, int], float] = field(default_factory=dict)


def load_table1() -> Table1:
    rows = _read_csv(MAIN_CSV)
    # Group coop_rate by (condition, f_value, episode, seed)
    vals: Dict[Tuple[str, str, int, int], float] = {}
    for r in rows:
        if r.get("ablation") != "none":
            continue
        if r.get("scope") != "f_value":
            continue
        fv = r.get("key", "")
        if fv not in ("3.500", "5.000"):
            continue
        cond = r.get("condition", "")
        seed = _int(r, "train_seed")
        ep = _int(r, "checkpoint_episode")
        vals[(cond, fv, ep, seed)] = _float(r, "coop_rate")

    t = Table1()
    episodes = [50_000, 100_000, 150_000]
    f_values = ["3.500", "5.000"]
    seeds = sorted({k[3] for k in vals})

    for fv in f_values:
        for ep in episodes:
            comm_vals = [(s, vals.get(("cond1", fv, ep, s), float("nan"))) for s in seeds]
            nc_vals = [(s, vals.get(("cond2", fv, ep, s), float("nan"))) for s in seeds]
            comm_vals = [(s, v) for s, v in comm_vals if not math.isnan(v)]
            nc_vals = [(s, v) for s, v in nc_vals if not math.isnan(v)]

            if comm_vals:
                t.comm[(fv, ep)] = CoopCell(
                    seeds=[s for s, _ in comm_vals],
                    values=[v for _, v in comm_vals],
                )
            if nc_vals:
                t.no_comm[(fv, ep)] = CoopCell(
                    seeds=[s for s, _ in nc_vals],
                    values=[v for _, v in nc_vals],
                )
            # Paired gap (only for seeds present in both)
            comm_dict = dict(comm_vals)
            nc_dict = dict(nc_vals)
            paired_seeds = sorted(set(comm_dict) & set(nc_dict))
            if paired_seeds:
                deltas = [comm_dict[s] - nc_dict[s] for s in paired_seeds]
                t.gap[(fv, ep)] = GapCell(deltas=deltas)
                for s in paired_seeds:
                    t.per_seed_gap[(s, fv, ep)] = comm_dict[s] - nc_dict[s]

    return t

# ---------------------------------------------------------------------------
# Table 2:  separate-training channel controls at 50k
# ---------------------------------------------------------------------------

@dataclass
class ControlRow:
    mode: str
    mean_coop: float
    n_seeds: int
    ci_lo: float = float("nan")
    ci_hi: float = float("nan")


def load_table2() -> List[ControlRow]:
    rows = _read_csv(CTRL_50K_CSV)
    out = []
    for r in rows:
        if _int(r, "checkpoint_episode") != 50_000:
            continue
        if r.get("f_value") != "3.500":
            continue
        out.append(ControlRow(
            mode=r["mode"],
            mean_coop=_float(r, "mean_coop_rate"),
            n_seeds=_int(r, "n_seeds"),
        ))
    out.sort(key=lambda x: -x.mean_coop)
    return out

# ---------------------------------------------------------------------------
# Table 3:  same-checkpoint continuation controls
# ---------------------------------------------------------------------------

@dataclass
class Table3Row:
    mode: str
    ep50k: float
    ep100k: Optional[float]
    ep150k: float
    n_seeds: int
    # SEM for each checkpoint
    ep50k_sem: float = float("nan")
    ep100k_sem: float = float("nan")
    ep150k_sem: float = float("nan")
    # f=5.0 values for difficulty-gating comparison
    ep50k_f50: float = float("nan")
    ep150k_f50: float = float("nan")
    ep50k_f50_sem: float = float("nan")
    ep150k_f50_sem: float = float("nan")
    # Paired gap vs learned at 150k f=3.5 (for bootstrap CI)
    gap_vs_learned_150k: float = float("nan")
    gap_vs_learned_150k_ci_lo: float = float("nan")
    gap_vs_learned_150k_ci_hi: float = float("nan")
    gap_vs_learned_150k_p: float = float("nan")


def load_table3() -> List[Table3Row]:
    """Load continuation controls.

    Preferred source is the regenerated 15-seed same-checkpoint summary in
    ``phase3_sameckpt_continuation_15seeds_local_20260319``. If that summary is
    absent, fall back to the legacy continuation-control CSV.
    """
    # Check if same-checkpoint data exists
    sameckpt_csv = SAMECKPT_DIR / "report" / "channel_control_summary.csv"
    csv_path = sameckpt_csv if sameckpt_csv.exists() else CTRL_CONT_CSV

    rows = _read_csv(csv_path)
    by_mode_35: Dict[str, Dict[int, float]] = defaultdict(dict)
    by_mode_50: Dict[str, Dict[int, float]] = defaultdict(dict)
    sem_35: Dict[str, Dict[int, float]] = defaultdict(dict)
    sem_50: Dict[str, Dict[int, float]] = defaultdict(dict)
    n_seeds_map: Dict[str, int] = {}
    for r in rows:
        mode = r["mode"]
        ep = _int(r, "checkpoint_episode")
        fv = r.get("f_value", "")
        if fv == "3.500":
            by_mode_35[mode][ep] = _float(r, "mean_coop_rate")
            sem_35[mode][ep] = _float(r, "sem_coop_rate")
            n_seeds_map[mode] = _int(r, "n_seeds")
        elif fv == "5.000":
            by_mode_50[mode][ep] = _float(r, "mean_coop_rate")
            sem_50[mode][ep] = _float(r, "sem_coop_rate")

    # Load per-seed raw data for paired bootstrap (learned vs each control)
    raw_csv = SAMECKPT_DIR / "report" / "channel_control_raw.csv"
    per_seed: Dict[str, Dict[Tuple[int, str], float]] = defaultdict(dict)
    if raw_csv.exists():
        raw_rows = _read_csv(raw_csv)
        for r in raw_rows:
            mode = r["mode"]
            seed = _int(r, "train_seed")
            ep = _int(r, "checkpoint_episode")
            fv = r.get("f_value", "")
            per_seed[mode][(seed, ep, fv)] = _float(r, "coop_rate")

    out = []
    mode_order = ["learned", "indep_random", "public_random", "always_zero"]
    for mode in mode_order:
        if mode not in by_mode_35:
            continue
        row = Table3Row(
            mode=mode,
            ep50k=by_mode_35[mode].get(50_000, float("nan")),
            ep100k=by_mode_35[mode].get(100_000, None),
            ep150k=by_mode_35[mode].get(150_000, float("nan")),
            n_seeds=n_seeds_map.get(mode, 0),
            ep50k_sem=sem_35.get(mode, {}).get(50_000, float("nan")),
            ep100k_sem=sem_35.get(mode, {}).get(100_000, float("nan")),
            ep150k_sem=sem_35.get(mode, {}).get(150_000, float("nan")),
            ep50k_f50=by_mode_50.get(mode, {}).get(50_000, float("nan")),
            ep150k_f50=by_mode_50.get(mode, {}).get(150_000, float("nan")),
            ep50k_f50_sem=sem_50.get(mode, {}).get(50_000, float("nan")),
            ep150k_f50_sem=sem_50.get(mode, {}).get(150_000, float("nan")),
        )
        # Paired gap: learned - control at 150k f=3.5
        if mode != "learned" and per_seed:
            learned_data = per_seed.get("learned", {})
            control_data = per_seed.get(mode, {})
            seeds = sorted({s for (s, ep, fv) in learned_data
                          if ep == 150_000 and fv == "3.500"})
            deltas = []
            for s in seeds:
                lv = learned_data.get((s, 150_000, "3.500"))
                cv = control_data.get((s, 150_000, "3.500"))
                if lv is not None and cv is not None:
                    deltas.append(lv - cv)
            if deltas:
                mean_d, lo, hi, p = _paired_delta_summary(deltas)
                row.gap_vs_learned_150k = mean_d
                row.gap_vs_learned_150k_ci_lo = lo
                row.gap_vs_learned_150k_ci_hi = hi
                row.gap_vs_learned_150k_p = p
        out.append(row)
    return out


def table3_is_sameckpt() -> bool:
    """True if Table 3 data comes from the regenerated same-checkpoint summary."""
    sameckpt_csv = SAMECKPT_DIR / "report" / "channel_control_summary.csv"
    return sameckpt_csv.exists()

# ---------------------------------------------------------------------------
# Table 4: frozen-checkpoint interventions
# ---------------------------------------------------------------------------

@dataclass
class FrozenInterventionRow:
    intervention: str
    delta_50k: float
    delta_150k: float
    # Per-seed values for bootstrap CI
    delta_50k_sem: float = float("nan")
    delta_150k_sem: float = float("nan")
    delta_50k_ci_lo: float = float("nan")
    delta_50k_ci_hi: float = float("nan")
    delta_150k_ci_lo: float = float("nan")
    delta_150k_ci_hi: float = float("nan")
    delta_50k_p: float = float("nan")
    delta_150k_p: float = float("nan")
    n_seeds_50k: int = 0
    n_seeds_150k: int = 0
    # Count of seeds where natural > intervention (for sign consistency)
    n_positive_150k: int = 0


def load_table4_frozen() -> List[FrozenInterventionRow]:
    """Load frozen-checkpoint interventions from multiple sources.

    This table is intentionally stitched:

    - 50k comes from the reduced frozen 50k suite for
      ``zeros``, ``indep_random``, ``public_random``, ``sender_shuffle``,
      and ``permute_slots``.
    - 150k comes from the reduced frozen 150k suite for that same set.
    - ``fixed0`` / ``fixed1`` are taken from the base 15-seed evaluation
      report, where those interventions were already present.
    """
    wanted = ["zeros", "fixed0", "public_random", "indep_random",
              "sender_shuffle", "permute_slots"]
    by_intervention: Dict[str, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    def _ingest(rows: List[Dict[str, str]], allowed: Optional[Sequence[str]] = None) -> None:
        allowed_set = set(allowed) if allowed is not None else None
        for r in rows:
            if r.get("condition") != "cond1":
                continue
            if r.get("f_value") != "3.500":
                continue
            intervention = str(r.get("intervention", ""))
            if intervention not in wanted:
                continue
            if allowed_set is not None and intervention not in allowed_set:
                continue
            ep = _int(r, "checkpoint_episode")
            by_intervention[intervention][ep].append(
                _float(r, "delta_coop_none_minus_intervention")
            )

    # Primary frozen suites (zeros, indep_random, public_random,
    # sender_shuffle, permute_slots)
    _ingest(_read_csv(FROZEN_50K_CSV))
    _ingest(_read_csv(FROZEN_150K_CSV))
    # fixed0 comes from the base ext150k eval report.
    _ingest(_read_csv(INTERVENTION_CSV), allowed=("fixed0",))

    out = []
    for intervention in wanted:
        ep50_vals = by_intervention[intervention].get(50_000, [])
        ep150_vals = by_intervention[intervention].get(150_000, [])

        row = FrozenInterventionRow(
            intervention=intervention,
            delta_50k=_mean(ep50_vals) if ep50_vals else float("nan"),
            delta_150k=_mean(ep150_vals) if ep150_vals else float("nan"),
            n_seeds_50k=len(ep50_vals),
            n_seeds_150k=len(ep150_vals),
        )

        # Bootstrap CI and sign-flip p for 50k
        if len(ep50_vals) >= 2:
            row.delta_50k_sem = _sem(ep50_vals)
            mean_d, lo, hi, p = _paired_delta_summary(ep50_vals)
            row.delta_50k_ci_lo = lo
            row.delta_50k_ci_hi = hi
            row.delta_50k_p = p

        # Bootstrap CI and sign-flip p for 150k
        if len(ep150_vals) >= 2:
            row.delta_150k_sem = _sem(ep150_vals)
            mean_d, lo, hi, p = _paired_delta_summary(ep150_vals)
            row.delta_150k_ci_lo = lo
            row.delta_150k_ci_hi = hi
            row.delta_150k_p = p
            row.n_positive_150k = sum(1 for v in ep150_vals if v > 0)

        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Endpoint sender-causal probe
# ---------------------------------------------------------------------------

@dataclass
class SenderCausalRow:
    f_value: str
    n_seeds: int
    mean_delta: float
    mean_delta_ci_lo: float
    mean_delta_ci_hi: float
    mean_abs_delta: float
    mean_abs_delta_ci_lo: float
    mean_abs_delta_ci_hi: float
    flip0: float
    flip0_ci_lo: float
    flip0_ci_hi: float
    flip1: float
    flip1_ci_lo: float
    flip1_ci_hi: float


def load_sender_causal() -> List[SenderCausalRow]:
    rows = _read_csv(SENDER_CAUSAL_MATRIX_CSV)
    by_f_seed: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        if r.get("condition") != "cond1":
            continue
        if r.get("ablation") not in ("", "none"):
            continue
        if r.get("cross_play") not in ("", "none"):
            continue
        if r.get("sender_remap") not in ("", "none"):
            continue
        if _int(r, "checkpoint_episode") != 150_000:
            continue
        if _int(r, "receiver_is_sender") != 0:
            continue
        f_value = r.get("true_f", "")
        seed = _int(r, "train_seed")
        bucket = by_f_seed[(f_value, seed)]
        delta = _float(r, "delta_p_cooperate_1_minus_0")
        bucket["delta"].append(delta)
        bucket["abs_delta"].append(abs(delta))
        bucket["flip0"].append(_float(r, "action_flip_rate_force0_vs_natural"))
        bucket["flip1"].append(_float(r, "action_flip_rate_force1_vs_natural"))

    out: List[SenderCausalRow] = []
    for f_value in sorted({fv for fv, _seed in by_f_seed}, key=float):
        seeds = sorted({seed for fv, seed in by_f_seed if fv == f_value})
        delta_vals = [_mean(by_f_seed[(f_value, seed)]["delta"]) for seed in seeds]
        abs_vals = [_mean(by_f_seed[(f_value, seed)]["abs_delta"]) for seed in seeds]
        flip0_vals = [_mean(by_f_seed[(f_value, seed)]["flip0"]) for seed in seeds]
        flip1_vals = [_mean(by_f_seed[(f_value, seed)]["flip1"]) for seed in seeds]

        delta_mean, delta_lo, delta_hi = _bootstrap_mean(delta_vals)
        abs_mean, abs_lo, abs_hi = _bootstrap_mean(abs_vals)
        flip0_mean, flip0_lo, flip0_hi = _bootstrap_mean(flip0_vals)
        flip1_mean, flip1_lo, flip1_hi = _bootstrap_mean(flip1_vals)

        out.append(
            SenderCausalRow(
                f_value=f_value,
                n_seeds=len(seeds),
                mean_delta=delta_mean,
                mean_delta_ci_lo=delta_lo,
                mean_delta_ci_hi=delta_hi,
                mean_abs_delta=abs_mean,
                mean_abs_delta_ci_lo=abs_lo,
                mean_abs_delta_ci_hi=abs_hi,
                flip0=flip0_mean,
                flip0_ci_lo=flip0_lo,
                flip0_ci_hi=flip0_hi,
                flip1=flip1_mean,
                flip1_ci_lo=flip1_lo,
                flip1_ci_hi=flip1_hi,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Table 5:  mute experiment
# ---------------------------------------------------------------------------

@dataclass
class Table4:
    comm_coop: float
    comm_welfare: float
    no_comm_coop: float
    no_comm_welfare: float
    mute_coop: float
    mute_welfare: float
    mute_n_seeds: int
    # Bootstrap on mute vs no_comm difference
    mute_vs_nocomm_delta: float = 0.0


def load_table4() -> Table4:
    # comm and no_comm from main experiment
    t1 = load_table1()
    comm_cell = t1.comm[("3.500", 150_000)]
    nc_cell = t1.no_comm[("3.500", 150_000)]

    # Mute from mute experiment
    mute_rows = _read_csv(MUTE_INTERVENTION_CSV)
    mute_coops = []
    mute_welfares = []
    for r in mute_rows:
        if _int(r, "checkpoint_episode") != 150_000:
            continue
        if r.get("f_value") != "3.500":
            continue
        if r.get("intervention") != "fixed0":
            continue
        mute_coops.append(_float(r, "coop_none"))
        mute_welfares.append(_float(r, "welfare_none"))

    # Also get comm welfare and no_comm welfare from main experiment
    main_rows = _read_csv(MAIN_CSV)
    comm_welfares = []
    nc_welfares = []
    for r in main_rows:
        if r.get("ablation") != "none" or r.get("scope") != "f_value":
            continue
        if r.get("key") != "3.500" or _int(r, "checkpoint_episode") != 150_000:
            continue
        if r.get("condition") == "cond1":
            comm_welfares.append(_float(r, "avg_welfare"))
        elif r.get("condition") == "cond2":
            nc_welfares.append(_float(r, "avg_welfare"))

    return Table4(
        comm_coop=comm_cell.mean,
        comm_welfare=_mean(comm_welfares),
        no_comm_coop=nc_cell.mean,
        no_comm_welfare=_mean(nc_welfares),
        mute_coop=_mean(mute_coops),
        mute_welfare=_mean(mute_welfares),
        mute_n_seeds=len(mute_coops),
        mute_vs_nocomm_delta=_mean(mute_coops) - nc_cell.mean,
    )

# ---------------------------------------------------------------------------
# Table 6:  aggregate vs per-sender token effects
# ---------------------------------------------------------------------------

@dataclass
class Table5Row:
    episode: int
    agg_effect: float
    per_sender_effect: float
    ratio: Optional[float]  # None if agg is near zero or negative
    agg_ci_lo: float = float("nan")
    agg_ci_hi: float = float("nan")
    per_sender_ci_lo: float = float("nan")
    per_sender_ci_hi: float = float("nan")


def load_table5() -> List[Table5Row]:
    """Compute aggregate and per-sender token effects at f=3.5.

    Aggregate token effect:
        For each (seed, fhat_bin), we have delta = P(C|any_m=1) - P(C|any_m=0).
        We restrict to fhat_bins in [2.5, 4.5) (the bins most relevant to f=3.5).
        We compute the mean absolute delta across bins within each seed,
        then average across seeds.

    Per-sender token effect:
        For each (seed, sender, receiver, fhat_bin), we have
        delta = P(C|sender_m=1) - P(C|sender_m=0).
        Same fhat restriction. Mean |delta| across all sender-receiver-bin
        combinations within each seed, then average across seeds.
    """
    # Aggregate: use ALL fhat bins (the receiver_semantics_summary already
    # conditions on the observation bin; averaging across all bins gives the
    # overall aggregate token effect).
    agg_rows = _read_csv(RECV_SEMANTICS_CSV)
    episodes = [50_000, 100_000, 150_000]

    # Group by (episode, seed) -> list of deltas (one per fhat_bin)
    agg_by_ep_seed: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for r in agg_rows:
        ep = _int(r, "checkpoint_episode")
        seed = _int(r, "train_seed")
        agg_by_ep_seed[(ep, seed)].append(_float(r, "delta_m1_minus_m0"))

    # Per-sender: use ALL fhat bins, individual receiver-sender pairs
    sender_rows = _read_csv(RECV_BY_SENDER_CSV)
    ps_by_ep_seed: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for r in sender_rows:
        # Only cross-agent effects (not self)
        if r.get("receiver_id") == r.get("sender_id"):
            continue
        # Use individual receiver-sender pairs (not "all_agents" aggregate)
        if r.get("receiver_id") == "all_agents":
            continue
        ep = _int(r, "checkpoint_episode")
        seed = _int(r, "train_seed")
        ps_by_ep_seed[(ep, seed)].append(abs(_float(r, "delta_m1_minus_m0")))

    out = []
    for ep in episodes:
        # Aggregate: mean delta (signed) per seed, then mean across seeds
        seed_agg_means = []
        for seed in sorted({k[1] for k in agg_by_ep_seed if k[0] == ep}):
            vals = agg_by_ep_seed[(ep, seed)]
            if vals:
                seed_agg_means.append(_mean(vals))

        # Per-sender: mean |delta| per seed, then mean across seeds
        seed_ps_means = []
        for seed in sorted({k[1] for k in ps_by_ep_seed if k[0] == ep}):
            vals = ps_by_ep_seed[(ep, seed)]
            if vals:
                seed_ps_means.append(_mean(vals))

        agg_mean = _mean(seed_agg_means) if seed_agg_means else float("nan")
        ps_mean = _mean(seed_ps_means) if seed_ps_means else float("nan")

        # Bootstrap CIs
        agg_ci = _bootstrap_mean(seed_agg_means) if len(seed_agg_means) >= 2 else (agg_mean, float("nan"), float("nan"))
        ps_ci = _bootstrap_mean(seed_ps_means) if len(seed_ps_means) >= 2 else (ps_mean, float("nan"), float("nan"))

        ratio = None
        if abs(agg_mean) > 0.005:
            ratio = ps_mean / abs(agg_mean)

        out.append(Table5Row(
            episode=ep,
            agg_effect=agg_mean,
            per_sender_effect=ps_mean,
            ratio=ratio,
            agg_ci_lo=agg_ci[1],
            agg_ci_hi=agg_ci[2],
            per_sender_ci_lo=ps_ci[1],
            per_sender_ci_hi=ps_ci[2],
        ))
    return out

# ---------------------------------------------------------------------------
# Table 7:  polarity alignment
# ---------------------------------------------------------------------------

@dataclass
class AlignmentRow:
    seed: int
    ep50k_pos: int
    ep50k_neg: int
    ep150k_pos: int
    ep150k_neg: int
    trajectory: str


def load_table6() -> List[AlignmentRow]:
    rows = _read_csv(ALIGNMENT_CSV)
    by_seed: Dict[int, Dict[int, Tuple[int, int]]] = defaultdict(dict)
    for r in rows:
        seed = _int(r, "train_seed")
        ep = _int(r, "checkpoint_episode")
        n_pos = _int(r, "n_regime_positive")
        n_neg = _int(r, "n_regime_negative")
        by_seed[seed][ep] = (n_pos, n_neg)

    out = []
    for seed in sorted(by_seed):
        p50_pos, p50_neg = by_seed[seed].get(50_000, (0, 0))
        p150_pos, p150_neg = by_seed[seed].get(150_000, (0, 0))

        # Determine trajectory description
        if p50_pos == p150_pos and p50_neg == p150_neg:
            traj = "Stable"
        elif p150_pos + p150_neg == max(p150_pos, p150_neg):
            if p50_pos + p50_neg != max(p50_pos, p50_neg):
                traj = "Achieved full alignment"
            elif p150_pos > p50_pos or p150_neg > p50_neg:
                traj = "Improved"
            else:
                traj = "Stable"
        elif (p50_pos + p50_neg == max(p50_pos, p50_neg)) and \
             (p150_pos + p150_neg != max(p150_pos, p150_neg)):
            traj = "Lost alignment"
        elif (p50_pos > p50_neg and p150_pos < p150_neg) or \
             (p50_pos < p50_neg and p150_pos > p150_neg):
            traj = "Polarity flipped"
        elif abs(p150_pos - p150_neg) > abs(p50_pos - p50_neg):
            traj = "Improved"
        elif abs(p150_pos - p150_neg) < abs(p50_pos - p50_neg):
            traj = "Lost alignment"
        else:
            traj = "Changed"

        out.append(AlignmentRow(
            seed=seed,
            ep50k_pos=p50_pos, ep50k_neg=p50_neg,
            ep150k_pos=p150_pos, ep150k_neg=p150_neg,
            trajectory=traj,
        ))
    return out

# ---------------------------------------------------------------------------
# Derived prose claims
# ---------------------------------------------------------------------------

@dataclass
class ProseClaims:
    """Computed values for inline references in the paper text."""
    # Abstract / Section 3
    comm_gap_35_50k_pp: str = ""           # e.g. "30.4"
    comm_gap_35_50k_pval: str = ""         # e.g. "0.002"
    comm_gap_35_50k_stars: str = ""        # e.g. "**"
    comm_gap_35_150k_pp: str = ""
    gap_decline_pct: str = ""              # e.g. "37"
    gap_range_lo: str = ""                 # per-seed min at 50k
    gap_range_hi: str = ""                 # per-seed max at 50k
    n_positive_150k: str = ""              # "4"
    negative_seed: str = ""                # "101"
    comm_gap_50_50k_pp: str = ""           # f=5.0 gap at 50k
    comm_gap_50_150k_pp: str = ""

    # Section 4.1
    ctrl_advantage_lo: str = ""            # "14"
    ctrl_advantage_hi: str = ""            # "19"

    # Section 4.3
    mute_below_nocomm_pp: str = ""         # "6"

    # Section 5
    ratio_50k: str = ""                    # "8.3"
    ratio_150k: str = ""                   # "8.7"
    agg_50k_pp: str = ""
    ps_50k_pp: str = ""
    agg_150k_pp: str = ""
    ps_150k_pp: str = ""


def compute_prose_claims(t1: Table1, t2: List[ControlRow],
                         t4: Table4, t5: List[Table5Row]) -> ProseClaims:
    p = ProseClaims()

    # Comm gap at f=3.5, 50k
    g35_50 = t1.gap[("3.500", 50_000)]
    p.comm_gap_35_50k_pp = f"{g35_50.mean * 100:.1f}"
    p.comm_gap_35_50k_pval = f"{g35_50.p_value:.4f}"
    p.comm_gap_35_50k_stars = ""

    g35_150 = t1.gap[("3.500", 150_000)]
    p.comm_gap_35_150k_pp = f"{g35_150.mean * 100:.1f}"

    # Decline %
    decline = (g35_50.mean - g35_150.mean) / g35_50.mean * 100
    p.gap_decline_pct = f"{decline:.0f}"

    # Per-seed range at 50k
    seed_gaps_50 = [t1.per_seed_gap[(s, "3.500", 50_000)]
                    for s in t1.comm[("3.500", 50_000)].seeds]
    p.gap_range_lo = f"{min(seed_gaps_50) * 100:.1f}"
    p.gap_range_hi = f"{max(seed_gaps_50) * 100:.1f}"

    # 150k: how many positive
    seed_gaps_150 = [(s, t1.per_seed_gap[(s, "3.500", 150_000)])
                     for s in t1.comm[("3.500", 150_000)].seeds]
    n_pos = sum(1 for _, d in seed_gaps_150 if d > 0)
    p.n_positive_150k = str(n_pos)
    neg_seeds = [s for s, d in seed_gaps_150 if d <= 0]
    p.negative_seed = str(neg_seeds[0]) if neg_seeds else "none"

    # f=5.0 gaps
    g50_50 = t1.gap[("5.000", 50_000)]
    g50_150 = t1.gap[("5.000", 150_000)]
    p.comm_gap_50_50k_pp = f"{g50_50.mean * 100:.1f}"
    p.comm_gap_50_150k_pp = f"{g50_150.mean * 100:.1f}"

    # Control advantages (learned - worst, learned - best)
    learned = next(c for c in t2 if c.mode == "learned")
    others = [c for c in t2 if c.mode != "learned"]
    advantages = [learned.mean_coop - c.mean_coop for c in others]
    p.ctrl_advantage_lo = f"{min(advantages) * 100:.0f}"
    p.ctrl_advantage_hi = f"{max(advantages) * 100:.0f}"

    # Mute below no-comm
    p.mute_below_nocomm_pp = f"{abs(t4.mute_vs_nocomm_delta) * 100:.0f}"

    # Table 5 ratios and pp values
    for row in t5:
        if row.episode == 50_000:
            p.agg_50k_pp = f"{row.agg_effect * 100:.1f}"
            p.ps_50k_pp = f"{row.per_sender_effect * 100:.1f}"
            if row.ratio is not None:
                p.ratio_50k = f"{row.ratio:.0f}"
            else:
                # Aggregate near zero: report per-sender vs "near-zero"
                p.ratio_50k = f">{row.per_sender_effect / 0.005:.0f}"
        elif row.episode == 150_000:
            p.agg_150k_pp = f"{abs(row.agg_effect) * 100:.1f}"
            p.ps_150k_pp = f"{row.per_sender_effect * 100:.1f}"
            if row.ratio is not None:
                p.ratio_150k = f"{row.ratio:.0f}"

    return p

# ---------------------------------------------------------------------------
# Master loader
# ---------------------------------------------------------------------------

@dataclass
class PaperData:
    """All data needed by the paper, loaded once."""
    t1: Table1
    t2: List[ControlRow]
    t3: List[Table3Row]
    t3_is_sameckpt: bool
    t4_frozen: List[FrozenInterventionRow]
    sender_causal: List[SenderCausalRow]
    t4: Table4
    t5: List[Table5Row]
    t6: List[AlignmentRow]
    prose: ProseClaims
    sender_shuffle_150k_f35_mean: float = float("nan")
    sender_shuffle_150k_f35_sem: float = float("nan")
    sender_shuffle_150k_f50_mean: float = float("nan")
    sender_shuffle_150k_f50_sem: float = float("nan")
    sender_shuffle_gap_150k_f35: float = float("nan")
    sender_shuffle_gap_150k_f35_ci_lo: float = float("nan")
    sender_shuffle_gap_150k_f35_ci_hi: float = float("nan")
    sender_shuffle_gap_150k_f35_sign: str = ""
    sender_shuffle_gap_150k_f50: float = float("nan")
    sender_shuffle_gap_150k_f50_ci_lo: float = float("nan")
    sender_shuffle_gap_150k_f50_ci_hi: float = float("nan")
    sender_shuffle_gap_150k_f50_sign: str = ""


def _load_sender_shuffle_sameckpt_50k() -> Dict[str, object]:
    """Load the matched-stat sender-shuffle continuation summary at 150k.

    This branch starts from the exact same 50k learned checkpoints as the
    learned comparator in the base 15-seed suite, so gaps are paired by seed.
    """
    if not SAMECKPT_SENDER_SHUFFLE_50K_CSV.exists():
        return {}

    learned_rows = _read_csv(MAIN_CSV)
    shuffle_rows = _read_csv(SAMECKPT_SENDER_SHUFFLE_50K_CSV)

    def _collect(rows: List[Dict[str, str]], f_value: str) -> Dict[int, Tuple[float, float, float]]:
        out: Dict[int, Tuple[float, float, float]] = {}
        for r in rows:
            if r.get("condition") != "cond1":
                continue
            if r.get("ablation") != "none":
                continue
            if r.get("scope") != "f_value":
                continue
            if _int(r, "checkpoint_episode") != 150_000:
                continue
            if f"{float(r.get('key', 'nan')):.1f}" != f_value:
                continue
            seed = _int(r, "train_seed")
            out[seed] = (
                _float(r, "coop_rate"),
                _float(r, "avg_reward"),
                _float(r, "avg_welfare"),
            )
        return out

    result: Dict[str, object] = {}
    for f_value, tag in (("3.5", "f35"), ("5.0", "f50")):
        learned = _collect(learned_rows, f_value)
        shuffle = _collect(shuffle_rows, f_value)
        seeds = sorted(set(learned) & set(shuffle))
        if not seeds:
            continue
        shuffle_coop = [shuffle[s][0] for s in seeds]
        result[f"mean_{tag}"] = _mean(shuffle_coop)
        result[f"sem_{tag}"] = _sem(shuffle_coop)
        deltas = [learned[s][0] - shuffle[s][0] for s in seeds]
        mean_d, lo, hi, _p = _paired_delta_summary(deltas)
        result[f"gap_{tag}"] = mean_d
        result[f"gap_{tag}_ci_lo"] = lo
        result[f"gap_{tag}_ci_hi"] = hi
        result[f"gap_{tag}_sign"] = f"{sum(d > 0 for d in deltas)}/{len(deltas)}"
    return result


def load_all() -> PaperData:
    t1 = load_table1()
    t2 = load_table2()
    t3 = load_table3()
    t4_frozen = load_table4_frozen()
    sender_causal = load_sender_causal()
    t4 = load_table4()
    t5 = load_table5()
    t6 = load_table6()
    prose = compute_prose_claims(t1, t2, t4, t5)
    sender_shuffle = _load_sender_shuffle_sameckpt_50k()
    return PaperData(
        t1=t1, t2=t2, t3=t3,
        t3_is_sameckpt=table3_is_sameckpt(),
        t4_frozen=t4_frozen,
        sender_causal=sender_causal,
        t4=t4, t5=t5, t6=t6,
        prose=prose,
        sender_shuffle_150k_f35_mean=sender_shuffle.get("mean_f35", float("nan")),
        sender_shuffle_150k_f35_sem=sender_shuffle.get("sem_f35", float("nan")),
        sender_shuffle_150k_f50_mean=sender_shuffle.get("mean_f50", float("nan")),
        sender_shuffle_150k_f50_sem=sender_shuffle.get("sem_f50", float("nan")),
        sender_shuffle_gap_150k_f35=sender_shuffle.get("gap_f35", float("nan")),
        sender_shuffle_gap_150k_f35_ci_lo=sender_shuffle.get("gap_f35_ci_lo", float("nan")),
        sender_shuffle_gap_150k_f35_ci_hi=sender_shuffle.get("gap_f35_ci_hi", float("nan")),
        sender_shuffle_gap_150k_f35_sign=sender_shuffle.get("gap_f35_sign", ""),
        sender_shuffle_gap_150k_f50=sender_shuffle.get("gap_f50", float("nan")),
        sender_shuffle_gap_150k_f50_ci_lo=sender_shuffle.get("gap_f50_ci_lo", float("nan")),
        sender_shuffle_gap_150k_f50_ci_hi=sender_shuffle.get("gap_f50_ci_hi", float("nan")),
        sender_shuffle_gap_150k_f50_sign=sender_shuffle.get("gap_f50_sign", ""),
    )

# ---------------------------------------------------------------------------
# LaTeX \def generation
# ---------------------------------------------------------------------------

def generate_defs_tex(d: PaperData) -> str:
    r"""Generate \def commands for every data value used in the paper.

    These are included via include-before-body in the Quarto build, placing
    them right after \begin{document} where \def works fine.
    """
    lines = [
        "% Auto-generated by paper_data.py — do not edit manually",
        "% Run: python paper_data.py --gen-defs",
        "",
    ]

    def D(name: str, value: str, comment: str = "") -> None:
        """Append a \\def line."""
        c = f"  % {comment}" if comment else ""
        lines.append(f"\\def\\{name}{{{value}}}{c}")

    # --- Table 1: comm vs no-comm (18 cells) ---
    lines.append("% Table 1: comm vs no-comm cooperation rates")
    fv_labels = [("3.500", "TF"), ("5.000", "FV")]
    ep_labels = [(50_000, "Fk"), (100_000, "Hk"), (150_000, "Ofk")]

    for fv, fl in fv_labels:
        for ep, el in ep_labels:
            c = d.t1.comm[(fv, ep)]
            nc = d.t1.no_comm[(fv, ep)]
            g = d.t1.gap[(fv, ep)]
            tag = f"{fl}{el}"  # e.g. TFFk, TFHk, FVOfk

            D(f"TIcomm{tag}", fmt_pm(c.mean, c.sem),
              f"comm f={fv} {ep//1000}k")
            D(f"TInc{tag}", fmt_pm(nc.mean, nc.sem),
              f"no_comm f={fv} {ep//1000}k")

            if fv == "3.500" and ep == 50_000:
                gap_str = f"\\mathbf{{{fmt_delta(g.mean)}}}"
            else:
                gap_str = f"{fmt_delta(g.mean)}"
            D(f"TIgap{tag}", gap_str,
              f"gap f={fv} {ep//1000}k  p={g.p_value:.4f}")

    # --- Table 2: separate-training controls (4 cells) ---
    lines.append("")
    lines.append("% Table 2: channel controls at 50k")
    mode_labels = {"learned": "Learned", "indep_random": "Indep",
                   "always_zero": "Zero", "public_random": "Pub"}
    for row in d.t2:
        ml = mode_labels.get(row.mode, row.mode)
        D(f"TII{ml}", fmt(row.mean_coop),
          f"{row.mode} (n={row.n_seeds})")

    # --- Table 3: continuation controls (4 modes × 3 eps) ---
    lines.append("")
    lines.append("% Table 3: continuation controls")
    t3_n = d.t3[0].n_seeds if d.t3 else 0
    D("TIIInSeeds", str(t3_n), "number of seeds")
    D("TIIIsource", "same-checkpoint" if d.t3_is_sameckpt else "continuation",
      "data source type")
    t3_pending = "" if d.t3_is_sameckpt else \
        " [Pending: same-checkpoint replication.]"
    D("TIIIpending", t3_pending, "pending note")

    cont_mode_labels = {"learned": "Learned", "indep_random": "Indep",
                        "public_random": "Pub", "always_zero": "Zero"}
    for row in d.t3:
        ml = cont_mode_labels.get(row.mode, row.mode)
        # f=3.5: mean ± SEM
        D(f"TIII{ml}Fk", fmt_pm(row.ep50k, row.ep50k_sem) if not math.isnan(row.ep50k_sem) else fmt(row.ep50k),
          f"{row.mode} 50k f=3.5")
        if row.ep100k is not None:
            hk_val = fmt_pm(row.ep100k, row.ep100k_sem) if not math.isnan(row.ep100k_sem) else fmt(row.ep100k)
        else:
            hk_val = "---"
        D(f"TIII{ml}Hk", hk_val, f"{row.mode} 100k f=3.5")
        D(f"TIII{ml}Ofk", fmt_pm(row.ep150k, row.ep150k_sem) if not math.isnan(row.ep150k_sem) else fmt(row.ep150k),
          f"{row.mode} 150k f=3.5")
        # f=5.0 values for difficulty-gating table
        if not math.isnan(row.ep50k_f50):
            D(f"TIII{ml}FkFV", fmt_pm(row.ep50k_f50, row.ep50k_f50_sem) if not math.isnan(row.ep50k_f50_sem) else fmt(row.ep50k_f50),
              f"{row.mode} 50k f=5.0")
        if not math.isnan(row.ep150k_f50):
            D(f"TIII{ml}OfkFV", fmt_pm(row.ep150k_f50, row.ep150k_f50_sem) if not math.isnan(row.ep150k_f50_sem) else fmt(row.ep150k_f50),
              f"{row.mode} 150k f=5.0")
        # Paired gap vs learned
        if not math.isnan(row.gap_vs_learned_150k):
            D(f"TIII{ml}GapOfk",
              f"{fmt_delta(row.gap_vs_learned_150k * 100, 1)}",
              f"learned - {row.mode} gap at 150k (pp)")
            D(f"TIII{ml}GapOfkCI",
              f"[{row.gap_vs_learned_150k_ci_lo * 100:+.1f},\\,{row.gap_vs_learned_150k_ci_hi * 100:+.1f}]",
              f"95% CI for gap (pp)")
            D(f"TIII{ml}GapOfkP",
              f"={row.gap_vs_learned_150k_p:.4f}" if row.gap_vs_learned_150k_p >= 0.0001 else "<0.0001",
              f"p-value for gap")

    # --- Table 3b: matched-stat sender-shuffle continuation (150k only) ---
    lines.append("")
    lines.append("% Same-checkpoint sender-shuffle continuation (matched-stat control)")
    if not math.isnan(d.sender_shuffle_150k_f35_mean):
        D("TSSshufCoopTFOfk", fmt(d.sender_shuffle_150k_f35_mean),
          "sender-shuffle coop f=3.5 150k")
        D("TSSshufCoopTFOfkSEM", f"{d.sender_shuffle_150k_f35_sem:.3f}",
          "sender-shuffle coop f=3.5 150k SEM")
        D("TSSshufCoopTFOfkPM", fmt_pm(d.sender_shuffle_150k_f35_mean, d.sender_shuffle_150k_f35_sem),
          "sender-shuffle coop f=3.5 150k mean ± SEM")
        D("TSSgapTFOfk", fmt_delta(d.sender_shuffle_gap_150k_f35 * 100.0, 1),
          "learned - shuffle gap f=3.5 150k (pp)")
        D("TSSgapTFOfkCI",
          f"[{d.sender_shuffle_gap_150k_f35_ci_lo * 100:+.1f},\\,{d.sender_shuffle_gap_150k_f35_ci_hi * 100:+.1f}]",
          "95% CI for gap f=3.5 150k (pp)")
        D("TSSgapTFOfkSign", d.sender_shuffle_gap_150k_f35_sign,
          "seeds favoring learned at f=3.5 150k")
    if not math.isnan(d.sender_shuffle_150k_f50_mean):
        D("TSSshufCoopFVOfk", fmt(d.sender_shuffle_150k_f50_mean),
          "sender-shuffle coop f=5.0 150k")
        D("TSSshufCoopFVOfkSEM", f"{d.sender_shuffle_150k_f50_sem:.3f}",
          "sender-shuffle coop f=5.0 150k SEM")
        D("TSSshufCoopFVOfkPM", fmt_pm(d.sender_shuffle_150k_f50_mean, d.sender_shuffle_150k_f50_sem),
          "sender-shuffle coop f=5.0 150k mean ± SEM")
        D("TSSgapFVOfk", fmt_delta(d.sender_shuffle_gap_150k_f50 * 100.0, 1),
          "learned - shuffle gap f=5.0 150k (pp)")
        D("TSSgapFVOfkCI",
          f"[{d.sender_shuffle_gap_150k_f50_ci_lo * 100:+.1f},\\,{d.sender_shuffle_gap_150k_f50_ci_hi * 100:+.1f}]",
          "95% CI for gap f=5.0 150k (pp)")
        D("TSSgapFVOfkSign", d.sender_shuffle_gap_150k_f50_sign,
          "seeds favoring learned at f=5.0 150k")

    # Min/max at 150k for prose
    t3_150k_vals = [r.ep150k for r in d.t3 if not math.isnan(r.ep150k)]
    if t3_150k_vals:
        D("TIIIminOfk", fmt(min(t3_150k_vals)), "min 150k across modes")
        D("TIIImaxOfk", fmt(max(t3_150k_vals)), "max 150k across modes")

    # --- Table 4: frozen-checkpoint interventions ---
    lines.append("")
    lines.append("% Table 4: frozen-checkpoint interventions")
    frozen_labels = {
        "zeros": "Zeros",
        "fixed0": "FixZero",
        "public_random": "Pub",
        "indep_random": "Indep",
        "sender_shuffle": "Shuffle",
        "permute_slots": "Permute",
    }
    for row in d.t4_frozen:
        rl = frozen_labels.get(row.intervention, row.intervention)
        D(f"TFR{rl}Fk", fmt_delta(row.delta_50k * 100.0, 1), f"{row.intervention} 50k (pp)")
        D(f"TFR{rl}Ofk", fmt_delta(row.delta_150k * 100.0, 1), f"{row.intervention} 150k (pp)")
        # SEM
        if not math.isnan(row.delta_50k_sem):
            D(f"TFR{rl}FkSEM", f"{row.delta_50k_sem * 100.0:.1f}", f"{row.intervention} 50k SEM (pp)")
        if not math.isnan(row.delta_150k_sem):
            D(f"TFR{rl}OfkSEM", f"{row.delta_150k_sem * 100.0:.1f}", f"{row.intervention} 150k SEM (pp)")
        # 95% CI
        if not math.isnan(row.delta_150k_ci_lo):
            D(f"TFR{rl}OfkCI",
              f"[{row.delta_150k_ci_lo * 100:+.1f},\\,{row.delta_150k_ci_hi * 100:+.1f}]",
              f"{row.intervention} 150k 95% CI (pp)")
        # p-value
        if not math.isnan(row.delta_150k_p):
            p_str = f"={row.delta_150k_p:.4f}" if row.delta_150k_p >= 0.0001 else "<0.0001"
            D(f"TFR{rl}OfkP", p_str, f"{row.intervention} 150k p-value")
        # Sign consistency
        if row.n_seeds_150k > 0:
            D(f"TFR{rl}OfkSign", f"{row.n_positive_150k}/{row.n_seeds_150k}",
              f"{row.intervention} 150k seeds with natural > intervention")

    # --- Table 5: endpoint sender-causal probe ---
    lines.append("")
    lines.append("% Table 5: endpoint sender-causal probe")
    causal_labels = {
        "0.5": "ZeroFive",
        "1.5": "OneFive",
        "2.5": "TwoFive",
        "3.5": "ThreeFive",
        "5.0": "FiveZero",
    }
    for row in d.sender_causal:
        tag = causal_labels.get(row.f_value, row.f_value.replace(".", ""))
        D(f"TSCDelta{tag}", fmt_delta(row.mean_delta * 100.0, 1),
          f"sender-causal mean delta at f={row.f_value} (pp)")
        D(f"TSCAbs{tag}", f"{row.mean_abs_delta * 100.0:.1f}",
          f"sender-causal mean abs delta at f={row.f_value} (pp)")
        D(f"TSCFlipZero{tag}", f"{row.flip0 * 100.0:.1f}",
          f"sender-causal flip0 at f={row.f_value} (%)")
        D(f"TSCFlipOne{tag}", f"{row.flip1 * 100.0:.1f}",
          f"sender-causal flip1 at f={row.f_value} (%)")
    lines.append("")
    lines.append("% Table 6: mute experiment")
    D("TIVcommCoop", fmt(d.t4.comm_coop), "comm coop at 150k")
    D("TIVcommWelf", fmt(d.t4.comm_welfare, 2), "comm welfare")
    D("TIVncCoop", fmt(d.t4.no_comm_coop), "no-comm coop at 150k")
    D("TIVncWelf", fmt(d.t4.no_comm_welfare, 2), "no-comm welfare")
    D("TIVmuteCoop", fmt(d.t4.mute_coop), "mute coop at 150k")
    D("TIVmuteWelf", fmt(d.t4.mute_welfare, 2), "mute welfare")
    # Formatted percentages for prose
    D("TIVmuteCoopPct", f"{d.t4.mute_coop*100:.1f}", "mute coop %")
    D("TIVncCoopPct", f"{d.t4.no_comm_coop*100:.1f}", "no-comm coop %")

    # --- Table 7: aggregate vs per-sender (9 cells + 3 ratios) ---
    lines.append("")
    lines.append("% Table 7: aggregate vs per-sender token effects")
    t5_ep_labels = {50_000: "Fk", 100_000: "Hk", 150_000: "Ofk"}
    for row in d.t5:
        el = t5_ep_labels[row.episode]
        D(f"TVagg{el}", fmt(row.agg_effect), f"aggregate {row.episode//1000}k")
        D(f"TVps{el}", fmt(row.per_sender_effect), f"per-sender {row.episode//1000}k")
        if row.ratio is not None:
            D(f"TVratio{el}", f"{row.ratio:.1f}\\times", f"ratio {row.episode//1000}k")
        else:
            D(f"TVratio{el}", "---", f"ratio {row.episode//1000}k (agg~0)")

    # --- Table 8: polarity alignment (15 seeds — summary) ---
    lines.append("")
    lines.append("% Table 8: polarity alignment (summary over 15 seeds)")
    D("TVInSeeds", str(len(d.t6)), "number of seeds in alignment table")
    # Summarize: count trajectories
    traj_counts: Dict[str, int] = defaultdict(int)
    for row in d.t6:
        traj_counts[row.trajectory] += 1
    for traj, count in sorted(traj_counts.items()):
        tag = traj.replace(" ", "").replace("/", "")[:12]
        D(f"TVItraj{tag}", str(count), f"n seeds with trajectory '{traj}'")

    # --- Prose claims ---
    lines.append("")
    lines.append("% Prose claims")
    p = d.prose
    D("pvGapTFFkPP", p.comm_gap_35_50k_pp, "comm gap f=3.5 50k (pp)")
    D("pvGapTFOfkPP", p.comm_gap_35_150k_pp, "comm gap f=3.5 150k (pp)")
    D("pvGapTFFkPval",
      f"={d.t1.gap[('3.500', 50_000)].p_value:.4f}",
      "p-value for gap at f=3.5 50k")
    D("pvGapDeclinePct", p.gap_decline_pct, "gap decline %")
    D("pvGapRangeLo", p.gap_range_lo, "per-seed gap min at 50k (pp)")
    D("pvGapRangeHi", p.gap_range_hi, "per-seed gap max at 50k (pp)")
    D("pvNposOfk", p.n_positive_150k, "n seeds with positive gap at 150k")
    D("pvGapFVFkPP", p.comm_gap_50_50k_pp, "comm gap f=5.0 50k (pp)")
    D("pvGapFVOfkPP", p.comm_gap_50_150k_pp, "comm gap f=5.0 150k (pp)")
    D("pvCtrlAdvLo", p.ctrl_advantage_lo, "control advantage low (pp)")
    D("pvCtrlAdvHi", p.ctrl_advantage_hi, "control advantage high (pp)")
    D("pvMuteBelowPP", p.mute_below_nocomm_pp, "mute below no-comm (pp)")
    D("pvRatioFk", p.ratio_50k, "per-sender/agg ratio 50k")
    D("pvRatioOfk", p.ratio_150k if p.ratio_150k else "---",
      "per-sender/agg ratio 150k")
    D("pvAggFkPP", p.agg_50k_pp, "aggregate token effect 50k (pp)")
    D("pvPsFkPP", p.ps_50k_pp, "per-sender effect 50k (pp)")
    D("pvAggOfkPP", p.agg_150k_pp, "aggregate effect 150k (pp)")
    D("pvPsOfkPP", p.ps_150k_pp, "per-sender effect 150k (pp)")

    # Formatted cooperation percentages for prose
    c35_50 = d.t1.comm[("3.500", 50_000)]
    nc35_50 = d.t1.no_comm[("3.500", 50_000)]
    D("pvCommCoopTFFkPct", f"{c35_50.mean*100:.1f}",
      "comm coop % at f=3.5 50k")
    D("pvNcCoopTFFkPct", f"{nc35_50.mean*100:.0f}",
      "no-comm coop % at f=3.5 50k")
    D("pvNSeeds", str(len(c35_50.seeds)), "total number of seeds")

    # --- Appendix: summary stats over all seeds at f=3.5 ---
    lines.append("")
    lines.append("% Appendix: summary stats for cooperation and gaps at f=3.5")
    for ep, el in ep_labels:
        c = d.t1.comm[("3.500", ep)]
        nc = d.t1.no_comm[("3.500", ep)]
        g = d.t1.gap[("3.500", ep)]
        n = len(c.values)
        D(f"pvPSnSeeds{el}", str(n), f"n seeds at {ep//1000}k")
        D(f"pvPScommMean{el}", fmt(c.mean), f"mean comm 3.5 {ep//1000}k")
        D(f"pvPScommSD{el}", fmt(_sem(c.values) * math.sqrt(n)) if n > 1 else "---",
          f"SD comm 3.5 {ep//1000}k")
        D(f"pvPScommMin{el}", fmt(min(c.values)), f"min comm 3.5 {ep//1000}k")
        D(f"pvPScommMax{el}", fmt(max(c.values)), f"max comm 3.5 {ep//1000}k")
        D(f"pvPSgapMean{el}", fmt_delta(g.mean), f"mean gap 3.5 {ep//1000}k")
        D(f"pvPSgapSD{el}", fmt(stdev(g.deltas)) if len(g.deltas) > 1 else "---",
          f"SD gap 3.5 {ep//1000}k")
        D(f"pvPSgapMin{el}", fmt_delta(min(g.deltas)), f"min gap 3.5 {ep//1000}k")
        D(f"pvPSgapMax{el}", fmt_delta(max(g.deltas)), f"max gap 3.5 {ep//1000}k")

    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _check():
    """Run basic sanity checks on loaded data."""
    d = load_all()

    print("=== Source map ===")
    for entry in build_source_map():
        path_list = ", ".join(_path_label(path) for path in entry.paths)
        print(f"  {entry.component}: {entry.new_source} [{path_list}]")

    print("=== Table 1: comm vs no-comm ===")
    for fv in ("3.500", "5.000"):
        for ep in (50_000, 100_000, 150_000):
            c = d.t1.comm.get((fv, ep))
            nc = d.t1.no_comm.get((fv, ep))
            g = d.t1.gap.get((fv, ep))
            if c and nc and g:
                print(f"  f={fv} ep={ep//1000}k: "
                      f"comm={fmt(c.mean)} ± {fmt(c.sem)} "
                      f"no_comm={fmt(nc.mean)} ± {fmt(nc.sem)} "
                      f"gap={fmt_delta(g.mean)} p={g.p_value:.4f}")

    print("\n=== Table 2: separate controls at 50k ===")
    for row in d.t2:
        print(f"  {row.mode}: {fmt(row.mean_coop)} (n={row.n_seeds})")

    print("\n=== Table 3: continuation controls ===")
    print(f"  Using same-checkpoint data: {d.t3_is_sameckpt}")
    for row in d.t3:
        ep100_str = fmt(row.ep100k) if row.ep100k is not None else "---"
        print(f"  {row.mode}: 50k={fmt(row.ep50k)} 100k={ep100_str} 150k={fmt(row.ep150k)} (n={row.n_seeds})")

    print("\n=== Table 4: frozen checkpoint interventions ===")
    for row in d.t4_frozen:
        print(
            f"  {row.intervention}: 50k={fmt_delta(row.delta_50k * 100.0, 1)}pp "
            f"150k={fmt_delta(row.delta_150k * 100.0, 1)}pp"
        )

    print("\n=== Table 5: endpoint sender-causal probe ===")
    for row in d.sender_causal:
        print(
            f"  f={row.f_value}: delta={fmt_delta(row.mean_delta * 100.0, 1)}pp "
            f"|delta|={row.mean_abs_delta * 100.0:.1f}pp "
            f"flip0={row.flip0 * 100.0:.1f}% flip1={row.flip1 * 100.0:.1f}%"
        )

    print("\n=== Table 5: mute ===")
    print(f"  comm={fmt(d.t4.comm_coop)} welfare={fmt(d.t4.comm_welfare, 2)}")
    print(f"  no_comm={fmt(d.t4.no_comm_coop)} welfare={fmt(d.t4.no_comm_welfare, 2)}")
    print(f"  mute={fmt(d.t4.mute_coop)} welfare={fmt(d.t4.mute_welfare, 2)} (n={d.t4.mute_n_seeds})")

    print("\n=== Table 6: aggregate vs per-sender ===")
    for row in d.t5:
        ratio_str = f"{row.ratio:.1f}x" if row.ratio else "---"
        print(f"  ep={row.episode//1000}k: agg={fmt(row.agg_effect)} "
              f"per_sender={fmt(row.per_sender_effect)} ratio={ratio_str}")

    print("\n=== Table 7: alignment ===")
    for row in d.t6:
        print(f"  seed {row.seed}: {row.ep50k_pos}:{row.ep50k_neg} -> "
              f"{row.ep150k_pos}:{row.ep150k_neg}  ({row.trajectory})")

    print("\n=== Prose claims ===")
    p = d.prose
    print(f"  Comm gap f=3.5 50k: +{p.comm_gap_35_50k_pp}pp "
          f"(p={p.comm_gap_35_50k_pval})")
    print(f"  Gap range at 50k: +{p.gap_range_lo} to +{p.gap_range_hi}pp")
    print(f"  At 150k: {p.n_positive_150k}/15 positive, seed {p.negative_seed} negative")
    print(f"  Gap decline: {p.gap_decline_pct}%")
    print(f"  Control advantage: {p.ctrl_advantage_lo}--{p.ctrl_advantage_hi}pp")
    print(f"  Mute below no-comm: {p.mute_below_nocomm_pp}pp")
    print(f"  Ratio 50k: {p.ratio_50k}x, 150k: {p.ratio_150k}x")


if __name__ == "__main__":
    import sys
    if "--gen-defs" in sys.argv:
        d = load_all()
        tex = generate_defs_tex(d)
        out_path = _HERE / "paper_values.tex"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"Wrote {len(tex.splitlines())} lines to {out_path}")
    elif "--gen-source-map" in sys.argv:
        text = generate_source_map_markdown()
        with open(SOURCE_MAP_MD, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote source map to {SOURCE_MAP_MD}")
    else:
        _check()
