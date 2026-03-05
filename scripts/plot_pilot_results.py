#!/usr/bin/env python3
"""
Plot Phase 2b pilot results for supervisor presentation.

Reads training JSONL metrics and greedy eval CSVs.
Produces annotated multi-panel figures with interpretive legends,
saved individually and as a combined multi-page PDF.

Usage:
    python3 scripts/plot_pilot_results.py
"""

import csv
import json
import os
import sys
import textwrap
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────

BASE = os.path.join(os.path.dirname(__file__), "..", "outputs", "train", "phase2b")
METRICS_DIR = os.path.join(BASE, "metrics")
EVAL_DIR = os.path.join(BASE, "eval")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [101, 202]
F_VALS = ["0.500", "1.500", "2.500", "3.500", "5.000"]
F_LABELS = {"0.500": "f=0.5", "1.500": "f=1.5", "2.500": "f=2.5", "3.500": "f=3.5", "5.000": "f=5.0"}
MILESTONES = [50000, 100000, 150000, 200000]

# Colors
C_COMM = "#2176AE"       # blue for comm
C_NOCOMM = "#D7263D"     # red for no-comm
C_COMM_LIGHT = "#89C4F4"
C_NOCOMM_LIGHT = "#F1948A"


# ── Data loading ───────────────────────────────────────────────────────────

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_training_metrics():
    """Load per-f cooperation and MI from training JSONL files."""
    data = {}  # (cond, seed) -> list of rows
    for fname in os.listdir(METRICS_DIR):
        if not fname.endswith(".jsonl"):
            continue
        if "smoke" in fname or "tmp" in fname:
            continue
        # Parse cond and seed from filename like cond1_seed101.jsonl
        parts = fname.replace(".jsonl", "").split("_")
        cond = parts[0]  # cond1 or cond2
        seed = int(parts[1].replace("seed", ""))
        rows = load_jsonl(os.path.join(METRICS_DIR, fname))
        data[(cond, seed)] = rows
    return data


def extract_f_coop_series(rows, f_key, window="window"):
    """Extract (episodes, coop_rates) for a given f-value from training JSONL."""
    episodes = []
    coop_rates = []
    for r in rows:
        if r.get("scope") == "f_value" and r.get("key") == f_key and r.get("window") == window:
            episodes.append(int(r["episode"]))
            coop_rates.append(float(r["coop_rate"]))
    return np.array(episodes), np.array(coop_rates)


def extract_mi_series(rows, metric="mi_message_f", key="all_senders"):
    """Extract (episodes, MI) from comm rows."""
    episodes = []
    mi_vals = []
    for r in rows:
        if (r.get("scope") == "comm" and r.get("metric") == metric
                and r.get("key") == key and r.get("window") == "window"):
            episodes.append(int(r["episode"]))
            mi_vals.append(float(r["mi"]))
    return np.array(episodes), np.array(mi_vals)


def extract_regime_coop_series(rows, regime, window="window"):
    """Extract (episodes, coop_rates) for a given regime."""
    episodes = []
    coop_rates = []
    for r in rows:
        if r.get("scope") == "regime" and r.get("key") == regime and r.get("window") == window:
            episodes.append(int(r["episode"]))
            coop_rates.append(float(r["coop_rate"]))
    return np.array(episodes), np.array(coop_rates)


def load_greedy_eval_data():
    """Load greedy eval CSVs and return per-(cond, seed, milestone, f_key) -> coop_rate."""
    data = {}
    for fname in os.listdir(EVAL_DIR):
        if not fname.endswith("_greedy.csv") or "summary" in fname:
            continue
        # Parse: cond1_seed101_ep50000_greedy.csv
        parts = fname.replace("_greedy.csv", "").split("_")
        cond = parts[0]
        seed = int(parts[1].replace("seed", ""))
        milestone = int(parts[2].replace("ep", ""))
        with open(os.path.join(EVAL_DIR, fname), encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["scope"] == "f_value":
                    f_key = row["key"]
                    data[(cond, seed, milestone, f_key)] = float(row["coop_rate"])
    return data


def smooth(y, window=5):
    """Centered moving average with shrinking window at boundaries.

    The previous implementation used np.convolve(..., mode="same") which
    zero-pads at both ends, creating artificial dips near the first and
    last data points.  This version uses a shrinking window so every
    output value is a true average of only real data.
    """
    if len(y) < window:
        return y.copy()
    out = np.empty_like(y)
    half = window // 2
    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        out[i] = np.mean(y[lo:hi])
    return out


def add_interpretation_box(fig, text, y_pos=0.0, fontsize=11, alpha=0.92):
    """Add an interpretation text box below the plot area."""
    wrapped = textwrap.fill(text, width=105)
    fig.text(
        0.5, y_pos, wrapped,
        ha="center", va="top", fontsize=fontsize,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFFFDD", edgecolor="#CCCC88", alpha=alpha),
        transform=fig.transFigure,
        wrap=True,
    )


# ── Figure 1: Per-f cooperation over training (seed 101) ──────────────────

def plot_per_f_training(data, seed=101):
    """5-panel figure: one per f-value, comm vs no-comm training curves."""
    fig, axes = plt.subplots(1, 5, figsize=(24, 8), sharey=True)
    fig.suptitle(
        f"Fig 1. P(Cooperate) During Training by Multiplication Factor (seed {seed})",
        fontsize=16, fontweight="bold", y=0.97,
    )

    for i, f_key in enumerate(F_VALS):
        ax = axes[i]
        for cond, color, label in [("cond1", C_COMM, "Comm (cond1)"), ("cond2", C_NOCOMM, "No-comm (cond2)")]:
            rows = data.get((cond, seed), [])
            ep, coop = extract_f_coop_series(rows, f_key)
            if len(ep) > 0:
                order = np.argsort(ep)
                ep, coop = ep[order], coop[order]
                ax.plot(ep / 1000, smooth(coop, 10), color=color, alpha=0.85,
                        linewidth=1.5, label=label)

        ax.set_title(F_LABELS[f_key], fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode (\u00d71k)", fontsize=10)
        if i == 0:
            ax.set_ylabel("P(Cooperate)", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
        ax.grid(True, alpha=0.2)

    # Shared legend at top-right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=12,
               framealpha=0.9, edgecolor="gray", bbox_to_anchor=(0.98, 0.96))

    interp = (
        "Interpretation: Training-window P(C) is noisy (due to \u03b5-tremble=0.05 and entropy exploration) "
        "and shows no clear communication advantage. Both conditions oscillate without converging. "
        "At f=5.0 (cooperation-dominant), both conditions reach ~55%, well below rational. "
        "At f=0.5 (defection-dominant), both correctly learn to defect. "
        "Lesson: exploration noise masks learned policy quality \u2014 greedy evaluation is needed (see Fig 2)."
    )
    fig.subplots_adjust(bottom=0.25)
    add_interpretation_box(fig, interp, y_pos=0.17)

    path = os.path.join(FIG_DIR, f"fig1_per_f_training_seed{seed}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    return fig


# ── Figure 2: Greedy eval across milestones (mean ± individual seeds) ─────

def plot_greedy_milestones(eval_data):
    """Bar chart: greedy P(C) at f=3.5 and f=5.0 across milestones."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle(
        "Fig 2. Greedy Evaluation: P(Cooperate) Across Training Milestones",
        fontsize=16, fontweight="bold", y=0.97,
    )

    for ax_idx, f_key in enumerate(["3.500", "5.000"]):
        ax = axes[ax_idx]
        x = np.arange(len(MILESTONES))
        width = 0.35

        for j, (cond, color, label) in enumerate([
            ("cond1", C_COMM, "Comm (cond1)"),
            ("cond2", C_NOCOMM, "No-comm (cond2)"),
        ]):
            means = []
            seed_vals = []
            for m in MILESTONES:
                vals = [eval_data.get((cond, s, m, f_key), np.nan) for s in SEEDS]
                seed_vals.append(vals)
                means.append(np.nanmean(vals))

            offset = (j - 0.5) * width
            bars = ax.bar(x + offset, means, width, color=color, alpha=0.75, label=label)

            # Individual seeds as dots
            for k, m in enumerate(MILESTONES):
                for s_idx, s in enumerate(SEEDS):
                    val = eval_data.get((cond, s, m, f_key), np.nan)
                    if not np.isnan(val):
                        marker = "o" if s == 101 else "s"
                        ax.scatter(x[k] + offset, val, color="black", s=25, zorder=5,
                                   alpha=0.6, marker=marker)

        ax.set_title(F_LABELS[f_key], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m//1000}k" for m in MILESTONES])
        ax.set_xlabel("Training episode", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Greedy P(Cooperate)", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
        ax.grid(True, axis="y", alpha=0.2)

    # Combined legend
    handles, labels = axes[0].get_legend_handles_labels()
    import matplotlib.lines as mlines
    dot_s101 = mlines.Line2D([], [], color="black", marker="o", linestyle="None",
                              markersize=5, label="seed 101", alpha=0.6)
    dot_s202 = mlines.Line2D([], [], color="black", marker="s", linestyle="None",
                              markersize=5, label="seed 202", alpha=0.6)
    fig.legend(handles + [dot_s101, dot_s202], labels + ["seed 101", "seed 202"],
               loc="upper right", fontsize=11, framealpha=0.9, edgecolor="gray",
               bbox_to_anchor=(0.98, 0.96))

    interp = (
        "KEY FINDING: Greedy evaluation reveals the communication effect masked by exploration noise. "
        "At f=5.0, comm agents reach P(C)=0.777 vs 0.535 for no-comm at 200k (\u039424pp). "
        "At f=3.5 (ambiguous regime), the gap is smaller (+10pp). "
        "Learning is non-monotonic: at 150k, no-comm briefly outperforms comm (P(C)=0.669 vs 0.406). "
        "This oscillation suggests policies cycle rather than converge \u2014 likely due to fixed entropy coefficient "
        "(ent_coeff=0.01, never annealed) preventing policy commitment. "
        "Note: cond2_seed202 is stuck at P(C)=0.500 at f=5.0 (2 agents always C, 2 always D \u2014 PPO local minimum)."
    )
    fig.subplots_adjust(bottom=0.24)
    add_interpretation_box(fig, interp, y_pos=0.16)

    path = os.path.join(FIG_DIR, "fig2_greedy_milestones.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    return fig


# ── Figure 3: MI(m;f) trajectory for both seeds ──────────────────────────

def plot_mi_trajectory(data):
    """MI(m;f) over training for both cond1 seeds."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        "Fig 3. Mutual Information MI(message; f) During Training \u2014 Comm Only",
        fontsize=16, fontweight="bold",
    )

    colors_seed = {101: C_COMM, 202: C_COMM_LIGHT}
    for seed in SEEDS:
        rows = data.get(("cond1", seed), [])
        ep, mi = extract_mi_series(rows, metric="mi_message_f")
        if len(ep) > 0:
            order = np.argsort(ep)
            ep, mi = ep[order], mi[order]
            ax.plot(ep / 1000, mi, color=colors_seed[seed], linewidth=1.8,
                    label=f"cond1 seed {seed}", alpha=0.85)

    # Annotate key phases
    ax.annotate("Peak MI\n(~5k ep)", xy=(5, 0.15), fontsize=11,
                ha="center", color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999"),
                xytext=(25, 0.20))
    ax.annotate("Collapse\n(\u22480 by 100k)", xy=(100, 0.001), fontsize=11,
                ha="center", color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999"),
                xytext=(120, 0.08))

    ax.set_xlabel("Episode (\u00d71k)", fontsize=13)
    ax.set_ylabel("MI(m; f)  [bits]", fontsize=13)
    ax.legend(fontsize=12, loc="upper right", framealpha=0.9, edgecolor="gray",
              title="Run", title_fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=-0.005)

    interp = (
        "KEY FINDING: MI(m;f) peaks early (~0.158 bits at 5k episodes) then collapses to \u22480 by 100k. "
        "Seed 202 starts near zero but catches up at ~40k before collapsing. "
        "Despite MI collapse, the cooperation advantage GROWS at 200k (see Fig 2). Three competing hypotheses: "
        "(1) Convention drift \u2014 message meanings wander but still carry instant-by-instant signal that "
        "window-averaged MI misses. (2) Internalization \u2014 agents learned cooperation policies that no longer "
        "need messages. (3) PPO coupling \u2014 joint updates create implicit coordination without explicit "
        "communication. Diagnosing which mechanism dominates is a key next step (cross-play + ablation tests)."
    )
    fig.subplots_adjust(bottom=0.22)
    add_interpretation_box(fig, interp, y_pos=0.15)

    path = os.path.join(FIG_DIR, "fig3_mi_trajectory.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    return fig


# ── Figure 4: Communication advantage Δ = cond1 − cond2 (greedy) ─────────

def plot_comm_advantage(eval_data):
    """Line plot: communication advantage ΔP(C) across milestones for all f."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        "Fig 4. Communication Advantage \u0394P(C) = Comm \u2212 NoComm  (greedy eval, mean of 2 seeds)",
        fontsize=16, fontweight="bold",
    )

    cmap = plt.cm.viridis
    f_colors = {f: cmap(i / (len(F_VALS) - 1)) for i, f in enumerate(F_VALS)}

    for f_key in F_VALS:
        deltas = []
        for m in MILESTONES:
            c1_vals = [eval_data.get(("cond1", s, m, f_key), np.nan) for s in SEEDS]
            c2_vals = [eval_data.get(("cond2", s, m, f_key), np.nan) for s in SEEDS]
            delta = np.nanmean(c1_vals) - np.nanmean(c2_vals)
            deltas.append(delta)

        ms_k = [m / 1000 for m in MILESTONES]
        ax.plot(ms_k, deltas, "o-", color=f_colors[f_key], linewidth=2,
                markersize=7, label=F_LABELS[f_key])

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.fill_between([40, 210], 0, 0.5, color="#d4edda", alpha=0.15, label="_nolegend_")
    ax.fill_between([40, 210], -0.5, 0, color="#f8d7da", alpha=0.15, label="_nolegend_")
    ax.text(195, 0.03, "Comm helps \u2191", fontsize=11, color="#155724", ha="right", alpha=0.7)
    ax.text(195, -0.08, "Comm hurts \u2193", fontsize=11, color="#721c24", ha="right", alpha=0.7)

    ax.set_xlabel("Training episode (\u00d71k)", fontsize=13)
    ax.set_ylabel("\u0394P(Cooperate)", fontsize=13)
    ax.legend(fontsize=12, title="Multiplication factor (f)", title_fontsize=12,
              ncol=2, loc="upper left", framealpha=0.9, edgecolor="gray")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(40, 210)

    interp = (
        "Interpretation: The communication advantage is strongly non-monotonic. At 50k, comm helps at f=5.0 "
        "(+17.5pp) and f=3.5. At 150k, the advantage reverses: no-comm briefly outperforms at f=5.0 (\u039426pp "
        "in favour of no-comm). By 200k, the advantage rebounds to +24pp. Low-f regimes (f=0.5, 1.5) show minimal "
        "effect \u2014 expected, since defection is dominant regardless of communication. The oscillation highlights "
        "that 2-seed results are inherently noisy and that policy cycling (likely due to constant entropy + "
        "multi-agent nonstationarity) creates phase-dependent conclusions. Expanding to 5 seeds is essential."
    )
    fig.subplots_adjust(bottom=0.22)
    add_interpretation_box(fig, interp, y_pos=0.15)

    path = os.path.join(FIG_DIR, "fig4_comm_advantage.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    return fig


# ── Figure 5: Regime cooperation over training (cooperative regime) ───────

def plot_regime_training(data):
    """Cooperative regime P(C) over training for all 4 runs."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        "Fig 5. P(Cooperate) in Cooperative Regime (f > N) During Training \u2014 All 4 Runs",
        fontsize=16, fontweight="bold",
    )

    styles = {
        ("cond1", 101): (C_COMM, "-", "Comm seed 101"),
        ("cond1", 202): (C_COMM, "--", "Comm seed 202"),
        ("cond2", 101): (C_NOCOMM, "-", "NoComm seed 101"),
        ("cond2", 202): (C_NOCOMM, "--", "NoComm seed 202"),
    }

    for (cond, seed), (color, ls, label) in styles.items():
        rows = data.get((cond, seed), [])
        ep, coop = extract_regime_coop_series(rows, "cooperative")
        if len(ep) > 0:
            order = np.argsort(ep)
            ep, coop = ep[order], coop[order]
            ax.plot(ep / 1000, smooth(coop, 10), color=color, linestyle=ls,
                    linewidth=1.5, label=label, alpha=0.85)

    ax.set_xlabel("Episode (\u00d71k)", fontsize=13)
    ax.set_ylabel("P(Cooperate)", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.legend(fontsize=12, loc="lower right", framealpha=0.9, edgecolor="gray",
              title="Condition / Seed", title_fontsize=12)
    ax.grid(True, alpha=0.2)

    interp = (
        "Interpretation: All four runs show high variance and non-monotonic learning in the cooperative regime. "
        "Seed-to-seed variability within the same condition is almost as large as the between-condition difference, "
        "confirming that 2 seeds are insufficient for reliable inference. "
        "NoComm seed 202 shows the weakest trajectory \u2014 it gets stuck near 0.50 (confirmed as PPO local minimum "
        "in greedy eval, where 2/4 agents always cooperate and 2/4 always defect). "
        "The training-window metric includes tremble (\u03b5=0.05) and entropy exploration, so absolute P(C) values "
        "understate the true greedy cooperation rate. Use Fig 2 and Fig 6 for policy-quality assessment."
    )
    fig.subplots_adjust(bottom=0.22)
    add_interpretation_box(fig, interp, y_pos=0.15)

    path = os.path.join(FIG_DIR, "fig5_cooperative_regime_training.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    return fig


# ── Figure 6: Greedy heatmap — all f × all milestones (seed 101) ─────────

def plot_greedy_heatmap(eval_data, seed=101):
    """2-panel heatmap: greedy P(C) for cond1 and cond2 across f and milestone."""
    fig = plt.figure(figsize=(16, 9))
    # Use gridspec to give heatmaps most of the width, colorbar a narrow column
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])
    axes = [ax_left, ax_right]

    fig.suptitle(
        f"Fig 6. Greedy P(Cooperate) Heatmap \u2014 Seed {seed} (rows=f, cols=milestone)",
        fontsize=16, fontweight="bold", y=0.97,
    )

    im = None
    for ax_idx, (cond, title) in enumerate([("cond1", "Comm (cond1)"), ("cond2", "No-comm (cond2)")]):
        ax = axes[ax_idx]
        mat = np.full((len(F_VALS), len(MILESTONES)), np.nan)
        for i, f_key in enumerate(F_VALS):
            for j, m in enumerate(MILESTONES):
                val = eval_data.get((cond, seed, m, f_key), np.nan)
                mat[i, j] = val

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(MILESTONES)))
        ax.set_xticklabels([f"{m//1000}k" for m in MILESTONES], fontsize=11)
        ax.set_yticks(range(len(F_VALS)))
        ax.set_yticklabels([F_LABELS[f] for f in F_VALS], fontsize=11)
        ax.set_xlabel("Training episode", fontsize=13)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Annotate cells
        for i in range(len(F_VALS)):
            for j in range(len(MILESTONES)):
                val = mat[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.3 or val > 0.8 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=12, color=color, fontweight="bold")

    if im is not None:
        fig.colorbar(im, cax=cbar_ax, label="P(Cooperate)")

    interp = (
        "Interpretation: The heatmap reveals the full f \u00d7 milestone landscape. "
        "LOW f (0.5, 1.5): both conditions correctly learn to defect (green=cooperate is absent, red cells). "
        "MEDIUM f (2.5): borderline regime, cooperation is low and inconsistent. "
        "HIGH f (3.5, 5.0): communication (left) produces higher cooperation than no-comm (right), especially "
        "at 200k where comm reaches 0.78 at f=5.0 vs 0.54 for no-comm. "
        "Non-monotonicity is visible: comm P(C) at f=5.0 drops from 0.72 at 50k to 0.22 at 150k before "
        "rebounding to 0.78. This oscillation is a signature of policy cycling under constant entropy + PPO."
    )
    fig.subplots_adjust(bottom=0.22)
    add_interpretation_box(fig, interp, y_pos=0.15)

    path = os.path.join(FIG_DIR, f"fig6_greedy_heatmap_seed{seed}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    return fig


# ── Figure 7: MI + cooperation joint trajectory (cond1 seed 101) ─────────

def plot_mi_coop_joint(data, seed=101):
    """Dual-axis plot: P(C|f=5.0) and MI(m;f) on same timeline."""
    fig, ax1 = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        f"Fig 7. Cooperation at f=5.0 vs MI(m;f) Joint Trajectory \u2014 Cond1 Seed {seed}",
        fontsize=16, fontweight="bold",
    )

    rows = data.get(("cond1", seed), [])

    # P(C|f=5.0)
    ep_c, coop_c = extract_f_coop_series(rows, "5.000")
    if len(ep_c) > 0:
        order = np.argsort(ep_c)
        ep_c, coop_c = ep_c[order], coop_c[order]
        ax1.plot(ep_c / 1000, smooth(coop_c, 10), color=C_COMM, linewidth=2.2,
                 label="P(C | f=5.0)")
        ax1.fill_between(ep_c / 1000, smooth(coop_c, 10), alpha=0.08, color=C_COMM)

    ax1.set_xlabel("Episode (\u00d71k)", fontsize=13)
    ax1.set_ylabel("P(Cooperate | f=5.0)", fontsize=13, color=C_COMM)
    ax1.tick_params(axis="y", labelcolor=C_COMM)
    ax1.set_ylim(-0.05, 1.05)

    # MI on second axis
    ax2 = ax1.twinx()
    ep_mi, mi = extract_mi_series(rows, metric="mi_message_f")
    if len(ep_mi) > 0:
        order = np.argsort(ep_mi)
        ep_mi, mi = ep_mi[order], mi[order]
        ax2.plot(ep_mi / 1000, mi, color="#F39C12", linewidth=2.2,
                 linestyle="--", label="MI(m; f)")
        ax2.fill_between(ep_mi / 1000, mi, alpha=0.08, color="#F39C12")

    ax2.set_ylabel("MI(m; f)  [bits]", fontsize=13, color="#F39C12")
    ax2.tick_params(axis="y", labelcolor="#F39C12")
    ax2.set_ylim(bottom=-0.005)

    # Phase annotations
    ax1.axvspan(0, 20, alpha=0.06, color="green", label="_nolegend_")
    ax1.axvspan(60, 120, alpha=0.06, color="red", label="_nolegend_")
    ax1.text(10, 1.0, "Phase 1:\nMI active", fontsize=11, ha="center", color="#155724", alpha=0.8)
    ax1.text(90, 1.0, "Phase 2:\nMI collapsed", fontsize=11, ha="center", color="#721c24", alpha=0.8)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="center right",
               framealpha=0.9, edgecolor="gray")
    ax1.grid(True, alpha=0.2)

    interp = (
        "KEY FINDING: MI and cooperation are temporally decoupled. MI peaks in the first ~20k episodes then "
        "collapses, but cooperation continues to grow after MI reaches zero. This is the \"transient communication\" "
        "signature: messages initially scaffold coordination, but the learned behaviour persists after the "
        "communication channel degenerates. This could reflect internalization (agents no longer need messages) "
        "or convention drift (messages still carry instant signal but window-averaged MI averages out). "
        "Distinguishing these requires: (a) message ablation at 200k, (b) cross-play sender@50k/receiver@200k, "
        "and (c) message responsiveness KL tracking."
    )
    fig.subplots_adjust(bottom=0.22)
    add_interpretation_box(fig, interp, y_pos=0.15)

    path = os.path.join(FIG_DIR, f"fig7_mi_coop_joint_seed{seed}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    return fig


# ── Title page ─────────────────────────────────────────────────────────────

def make_title_page():
    """Create a title/summary page for the combined PDF."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.82, "DSC-EPGG Phase 2b Pilot Results",
             ha="center", fontsize=22, fontweight="bold")
    fig.text(0.5, 0.76, "2 seeds \u00d7 2 conditions \u00d7 200k episodes",
             ha="center", fontsize=14, color="#555555")
    fig.text(0.5, 0.72, "March 2026",
             ha="center", fontsize=12, color="#888888")

    summary = textwrap.dedent("""\
    Executive Summary

    Pilot training (4 runs: cond1=comm, cond2=no-comm, seeds 101 & 202) reveals:

    1. Communication effect is real but masked by stochastic evaluation.
       Greedy eval shows a 24pp advantage at f=5.0 (P(C)=0.777 vs 0.535 at 200k).

    2. MI(m;f) collapses to ~0 by 100k episodes, yet the cooperation advantage
       grows. This is a "transient communication" signature.

    3. Learning is non-monotonic. At 150k, no-comm briefly outperforms comm,
       before the advantage reverses by 200k. Policies cycle rather than converge.

    4. PPO failure mode: cond2_seed202 gets stuck at P(C)=0.500 exactly at f=5.0
       (2 agents always C, 2 always D).

    Likely cause: entropy_coeff=0.01 is fixed throughout all 200k episodes (never
    annealed), preventing policy commitment and causing convention drift.

    Next steps: (a) entropy/LR annealing ablations, (b) 5-seed expansion,
    (c) cross-play and message ablation diagnostics.
    """)

    fig.text(0.12, 0.60, summary, ha="left", va="top", fontsize=11,
             fontfamily="monospace", linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8F9FA",
                       edgecolor="#DEE2E6", alpha=0.95))

    fig.text(0.5, 0.04,
             "Settings: PPO (GAE \u03b3=0.99, \u03bb=0.95, clip=0.2) | ent_coeff=0.01 (fixed) | "
             "lr=3e-4 (fixed) | hidden=64 | V=2 tokens | msg_dropout=0.1 | "
             "\u03c3=0.5 | \u03b5_tremble=0.05",
             ha="center", fontsize=8.5, color="#888888", style="italic")

    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading training metrics...")
    data = load_training_metrics()
    print(f"  Loaded {len(data)} run(s): {sorted(data.keys())}")

    print("Loading greedy eval data...")
    eval_data = load_greedy_eval_data()
    print(f"  Loaded {len(eval_data)} eval entries")

    print("\nGenerating annotated figures...")

    figures = []

    print("\n[Title] Summary page")
    figures.append(make_title_page())

    print("[Fig 1] Per-f cooperation during training (seed 101)")
    figures.append(plot_per_f_training(data, seed=101))

    print("[Fig 2] Greedy eval bar chart at f=3.5 and f=5.0")
    figures.append(plot_greedy_milestones(eval_data))

    print("[Fig 3] MI(m;f) trajectory (both seeds)")
    figures.append(plot_mi_trajectory(data))

    print("[Fig 4] Communication advantage \u0394P(C) across milestones")
    figures.append(plot_comm_advantage(eval_data))

    print("[Fig 5] Cooperative regime P(C) training curves (all 4 runs)")
    figures.append(plot_regime_training(data))

    print("[Fig 6] Greedy heatmap (seed 101)")
    figures.append(plot_greedy_heatmap(eval_data, seed=101))

    print("[Fig 7] MI + cooperation joint trajectory (seed 101)")
    figures.append(plot_mi_coop_joint(data, seed=101))

    # Combine all into one multi-page PDF
    combined_path = os.path.join(FIG_DIR, "pilot_results_all.pdf")
    print(f"\nCombining all figures into: {combined_path}")
    with PdfPages(combined_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"\nAll figures saved to: {os.path.abspath(FIG_DIR)}")
    print(f"Combined PDF: {os.path.abspath(combined_path)}")


if __name__ == "__main__":
    main()
