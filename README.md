# MARL-EmeCom: Multi-Agent RL with Emergent Communication in Mixed-Motive Settings

**Paper:** *Learning in Public Goods Games: The Effects of Uncertainty and Communication on Cooperation* (Orzan et al. 2025)  
[📄 Read on SpringerLink](https://link.springer.com/article/10.1007/s00521-024-10530-6)

---

## Overview

This project studies **emergent communication in multi-agent reinforcement learning (MARL)** under **mixed incentives** and **uncertainty**.  
We extend the Public Goods Game into an **Extended Public Goods Game (EPGG)**, spanning cooperative, mixed, and competitive settings. The code reproduces the experiments from our paper.

Example Outcome:

[W&B - 2-agent Experiments with Uncertainty and Communication](https://wandb.ai/nicoleorzan/2agents_comm[1,%200]_list[0,%201]_noGmm_unc[0.0,%202.0]_mfact[0.5,%201.5,%202.5,%203.5]_algo_reinforce_BEST/reports/Extended-Public-Goods-Games-Communication-and-Uncertainty--VmlldzoxNDU1NTkzOQ).

**Key findings:**  
- Communication supports cooperation under **symmetric uncertainty**.  
- Under **asymmetric uncertainty**, agents may exploit communication.  
- Agents trained across multiple incentive environments learn richer strategies that **generalize** better to unseen settings.

## Current Data Location Guide

For the current local run roots, checkpoint/eval paths, and legacy-path traps, start with:

- [`DATA_MAP.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/DATA_MAP.md)
- [`outputs/data_catalog/DATA_CATALOG.md`](/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/data_catalog/DATA_CATALOG.md)

## Project Features & Repository Structure

- **Environments**: Extended Public Goods Game (EPGG) with cooperative/mixed/competitive incentives.    
- **Uncertainty**: noisy observations of the incentive factor (Gaussian).
- **Emergent communication**: discrete (“cheap talk”) messages before acting.  
- **Algorithms**:  
  - **REINFORCE** (policy gradient)
  - **DQN** (deep Q-learning)  
- **Uncertainty modelling**: agents can optionally maintain a **Gaussian Mixture Model (GMM)** to infer hidden incentive structure.

**Code structure**:
- [`/envs`](envs): Extended Public Goods Game (EPGG) environments.
- [`/agents`](agents): Implementations of REINFORCE and DQN agents.
- [`/comm`](comm): Modules for emergent communication channels.
- [`/analysis`](analysis): Scripts for metrics (mutual information, speaker consistency, coordination).
- [`/experiments`](experiments): Configurations and training scripts to reproduce paper results.


## Getting Started / Implementation

### 1. Clone & Dependencies

```bash
git clone https://github.com/nicoleorzan/marl-emecom.git
cd marl-emecom
pip install -r requirements.txt
```
(The use of a virtual environment is suggested)

### 2. Training Agents

You can train agents either:
- Without communication
- With communication (a subset of agents sends discrete messages before action)

Example usage:

The launcher sets parameters inside `src/experiments_pgg_v0/caller_given_params.py`; you can edit them there, or pass them as input:
```
python caller_given_params.py --n_agents 2 --mult_fact 0.5 1.5 2.5 --uncertainties 0 0 --communicating_agents 1 1 --listening_agents 1 1 --gmm_ 0 --algorithm reinforce
```

Base run:
```
python src/experiments_pgg_v0/caller_given_params.py
```

---

## Manuscript Assembly Tool

A zero-dependency Python script that merges markdown section files from different workflows (deep research, methods drafts, results, etc.) into a single submission-ready manuscript.

### Quick Start

```bash
python3 assemble_manuscript.py manuscript_manifest.yaml \
  --input-dir /path/to/your/md/files \
  --output-dir ./manuscript
```

### What It Produces

| File | Description |
|------|-------------|
| `manuscript_merged.md` | Assembled manuscript with sequential section numbering and reference stubs |
| `bibliography.md` | Deduplicated citations extracted from inline references, with known entries auto-filled |
| `notation_report.md` | Canonical symbol table, notation inconsistencies, and cross-reference warnings |

### How It Works

1. **Manifest-driven assembly** — `manuscript_manifest.yaml` defines which `.md` files to include, their order, roles (`intro`, `methods`, `results`, etc.), and heading level offsets
2. **Section renumbering** — all `##`/`###`/`####` headings are renumbered sequentially across files (e.g., intro sections 1–3 + methods sections 1–5 become sections 1–8)
3. **Citation extraction** — finds parenthetical `(Author et al., 2020)`, textual `Author et al. (2020)`, and semicolon-separated citations; deduplicates by author+year
4. **Bibliography auto-fill** — known project references (Durstewitz 2017, Schulman 2016/2017, Orzan 2024) are filled in automatically with title and venue
5. **Notation consistency** — checks math blocks against the canonical symbol table in the manifest; flags bare-vs-subscripted conflicts (e.g., `f` vs `f_t`), missing symbols, and coverage gaps
6. **Cross-reference validation** — verifies that `Section X.Y` references point to actual headings
7. **Statistics** — reports word count, section counts, and equation counts

### Adding Sections

As new sections arrive from different workflows, add entries to the manifest:

```yaml
sections:
  - file: intro_from_deep_research.md
    role: intro
    title: "Introduction"
    level: 0

  - file: DSC_EPGG_Methods_Section_Draft.md
    role: methods
    title: "Methods"
    level: 0

  - file: results.md
    role: results
    title: "Results"
    level: 0
```

Re-run the tool — sections are assembled in manifest order.

### Validate Without Writing

```bash
python3 assemble_manuscript.py manuscript_manifest.yaml --validate-only
```

Runs all checks (renumbering, citations, notation, cross-refs) and prints the summary without writing output files.
