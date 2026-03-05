# Interim Cond1 vs Cond2 Comparison

Generated from completed eval checkpoints only.

## Per-f Comparison (Cond1 - Cond2)

| Mode | Seed | Episode | Cond1 P(C|f=3.5) | Cond2 P(C|f=3.5) | Delta | Cond1 P(C|f=5.0) | Cond2 P(C|f=5.0) | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 101 | 50000 | 0.678 | 0.270 | +0.408 | 0.705 | 0.382 | +0.322 |
| sample | 101 | 50000 | 0.599 | 0.339 | +0.260 | 0.617 | 0.436 | +0.181 |
| greedy | 101 | 100000 | 0.272 | 0.256 | +0.016 | 0.252 | 0.476 | -0.225 |
| sample | 101 | 100000 | 0.340 | 0.332 | +0.008 | 0.353 | 0.488 | -0.135 |
| greedy | 202 | 50000 | 0.488 | 0.235 | +0.253 | 0.671 | 0.643 | +0.028 |
| sample | 202 | 50000 | 0.469 | 0.350 | +0.119 | 0.632 | 0.544 | +0.088 |

## Latest Cond1 Message MI (all_senders, window)

| Seed | Latest Episode | MI(m;f) | MI(m;a) |
|---:|---:|---:|---:|
| 101 | 116000 | 0.025505 | 0.010047 |\n| 202 | 101000 | 0.020726 | 0.006722 |\n