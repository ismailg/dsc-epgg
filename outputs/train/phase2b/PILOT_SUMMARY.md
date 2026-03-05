# Phase 2b Pilot Summary

Date: 2026-03-05

## Completion Status

- Training runs completed: `cond1_seed101`, `cond1_seed202`, `cond2_seed101`, `cond2_seed202` (all at 200k episodes).
- Eval completion: complete (32/32).

## Greedy Per-f Cooperation (Primary Comparison; mean across seeds)

| Milestone | Cond | f=0.5 | f=1.5 | f=2.5 | f=3.5 | f=5.0 |
|---:|---|---:|---:|---:|---:|---:|
| 50000 | cond1 | 0.008 | 0.043 | 0.234 | 0.583 | 0.688 |
| 50000 | cond2 | 0.000 | 0.001 | 0.067 | 0.253 | 0.513 |
| 100000 | cond1 | 0.000 | 0.004 | 0.061 | 0.303 | 0.471 |
| 100000 | cond2 | 0.018 | 0.026 | 0.151 | 0.300 | 0.440 |
| 150000 | cond1 | 0.003 | 0.018 | 0.184 | 0.423 | 0.406 |
| 150000 | cond2 | 0.010 | 0.022 | 0.213 | 0.469 | 0.669 |
| 200000 | cond1 | 0.008 | 0.044 | 0.212 | 0.507 | 0.777 |
| 200000 | cond2 | 0.019 | 0.067 | 0.177 | 0.407 | 0.535 |

## Greedy Regime Cooperation (mean across seeds)

| Milestone | Cond | competitive | mixed | cooperative |
|---:|---|---:|---:|---:|
| 50000 | cond1 | 0.008 | 0.299 | 0.688 |
| 50000 | cond2 | 0.000 | 0.111 | 0.513 |
| 100000 | cond1 | 0.000 | 0.129 | 0.471 |
| 100000 | cond2 | 0.018 | 0.163 | 0.440 |
| 150000 | cond1 | 0.003 | 0.217 | 0.406 |
| 150000 | cond2 | 0.010 | 0.242 | 0.669 |
| 200000 | cond1 | 0.008 | 0.265 | 0.777 |
| 200000 | cond2 | 0.019 | 0.223 | 0.535 |

## Key Comparisons: Cond1 - Cond2

### Greedy
| Milestone | ΔP(C|f=3.5) | ΔP(C|f=5.0) |
|---:|---:|---:|
| 50000 | 0.330 | 0.175 |
| 100000 | 0.003 | 0.031 |
| 150000 | -0.046 | -0.262 |
| 200000 | 0.100 | 0.242 |

### Sample
| Milestone | ΔP(C|f=3.5) | ΔP(C|f=5.0) |
|---:|---:|---:|
| 50000 | 0.189 | 0.135 |
| 100000 | 0.017 | 0.012 |
| 150000 | 0.028 | -0.018 |
| 200000 | 0.038 | 0.069 |

## Communication Diagnostics (Cond1, all_senders, window MI)

| Milestone | Seed | MI(m;f) | MI(m;a) |
|---:|---:|---:|---:|
| 50000 | 101 | 0.078 | 0.035 |
| 50000 | 202 | 0.002 | 0.000 |
| 100000 | 101 | 0.001 | 0.002 |
| 100000 | 202 | 0.001 | 0.000 |
| 150000 | 101 | 0.024 | 0.006 |
| 150000 | 202 | 0.000 | 0.000 |
| 200000 | 101 | 0.000 | 0.001 |
| 200000 | 202 | 0.000 | 0.001 |

## Decision

- Final greedy deltas (mean across seeds): ΔP(C|f=3.5)=0.100, ΔP(C|f=5.0)=0.242.
- Final MI check at 200k: MI>0 condition met = True.
- Recommendation: **Partial**.
- Interpretation: communication has measurable effects but direction is mixed across milestones/seeds; extend to 5 seeds before final claim.

## Notes

- Primary comparison uses greedy evaluation as specified.
- Sample-mode deltas included to assess exploration/noise effects.