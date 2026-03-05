# Phase 2b Progress Log

## 2026-03-04
- Added Phase 2b execution checklist: `/Users/mbp17/POSTDOC/NPS26/dsc-epgg/PHASE2B_EXECUTION_TODO.md`.
- Implemented comm MI metrics in trainer JSONL (`scope="comm"`, metrics: `mi_message_f`, `mi_message_action`).
- Implemented `--checkpoint_interval` in trainer with `_ep{N}` intermediate checkpoint naming.
- Implemented evaluator `--greedy` mode with argmax policy and tremble disabled.
- Smoke-validated:
  - Comm MI rows emitted to JSONL.
  - Intermediate checkpoint saved (`_ep2.pt` in smoke run).
  - Greedy evaluator produced CSV with `eval_policy=greedy`.
- Full `pytest` currently blocked in this shell by torch import abort during collection.
- Validation update:
  - `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 -m pytest tests/ -v` -> 28 passed.
- Launched Phase 2b Cond2 pilot runs (2 concurrent PTY sessions):
  - seed 101 session: `36766`
  - seed 202 session: `70320`
- First milestone (`episode=1000`) observed:
  - seed 101: coop=0.385, avg_reward=6.610, regime coop(comp/mixed/cooperative)=0.621/0.632/0.677
  - seed 202: coop=0.250, avg_reward=7.865, regime coop(comp/mixed/cooperative)=0.476/0.482/0.507
- Additional milestones:
  - episode 2000
    - seed 101: coop=0.345, avg_reward=9.315, regime=0.078/0.118/0.346
    - seed 202: coop=0.147, avg_reward=5.820, regime=0.064/0.089/0.268
  - episode 3000
    - seed 101: coop=0.190, avg_reward=5.450, regime=0.051/0.157/0.563
    - seed 202: coop=0.098, avg_reward=4.700, regime=0.058/0.081/0.329
  - episode 4000
    - seed 101: coop=0.460, avg_reward=10.985, regime=0.052/0.157/0.546
    - seed 202: coop=0.145, avg_reward=4.900, regime=0.049/0.131/0.432
  - episode 5000
    - seed 101: coop=0.245, avg_reward=6.700, regime=0.051/0.188/0.561
    - seed 202: coop=0.182, avg_reward=5.475, regime=0.051/0.161/0.492
- Per-f cumulative snapshot at episode 4000:
  - seed 101: f=0.5:0.204, f=1.5:0.202, f=2.5:0.232, f=3.5:0.367, f=5.0:0.532
  - seed 202: f=0.5:0.163, f=1.5:0.161, f=2.5:0.178, f=3.5:0.246, f=5.0:0.383
- Later milestones observed:
  - episode 24000
    - seed 101: coop=0.547, avg_reward=10.360, regime=0.057/0.278/0.539
    - seed 202: coop=0.135, avg_reward=5.265, regime=0.054/0.281/0.550
  - episode 25000
    - seed 101: coop=0.310, avg_reward=6.700, regime=0.065/0.327/0.545
- Task 1 launch update (4 concurrent runs active):
  - cond2 seed 101: PID 82100 (session 36766)
  - cond2 seed 202: PID 82106 (session 70320)
  - cond1 seed 101: PID 87938 (session 44127)
  - cond1 seed 202: PID 87940 (session 9160)
- Cond1 early sanity check (~5k episodes):
  - cond1 seed 101 @ ep5000: coop=0.120, avg_reward=5.160, regime=0.050/0.204/0.600, mi(m;f)=0.158, mi(m;a)=0.031
  - cond1 seed 202 @ ep5000: coop=0.302, avg_reward=7.640, regime=0.050/0.206/0.557, mi(m;f)=0.030, mi(m;a)=0.006
  - JSONL confirmation: both `outputs/train/phase2b/metrics/cond1_seed*.jsonl` emit `scope="comm"` rows with `mi_message_f` and `mi_message_action`.
- Cond2 checkpoints reached:
  - cond2 seed 101: `outputs/train/phase2b/cond2_seed101_ep50000.pt`
  - cond2 seed 202: `outputs/train/phase2b/cond2_seed202_ep50000.pt`
- Task 2 health check:
  - `ps aux | grep train_ppo`: 4 active training PIDs (82100, 82106, 87938, 87940)
  - metrics files are growing (`wc -l`): cond1_seed101=156, cond1_seed202=130, cond2_seed101=864, cond2_seed202=848
  - crash scan (`grep -iE 'nan|error|traceback' outputs/train/phase2b/logs/*.log`): no matches
- Task 3 evals completed for Cond2 @ 50k (sample + greedy):
  - cond2 seed 101 sample summary: comp=0.052, mixed=0.193, coop=0.436
  - cond2 seed 101 greedy summary: comp=0.000, mixed=0.133, coop=0.382
  - cond2 seed 202 sample summary: comp=0.052, mixed=0.172, coop=0.544
  - cond2 seed 202 greedy summary: comp=0.001, mixed=0.089, coop=0.643
  - outputs:
    - `outputs/train/phase2b/eval/cond2_seed101_ep50000_{sample,greedy}.csv`
    - `outputs/train/phase2b/eval/cond2_seed101_ep50000_{sample,greedy}_summary.csv`
    - `outputs/train/phase2b/eval/cond2_seed202_ep50000_{sample,greedy}.csv`
    - `outputs/train/phase2b/eval/cond2_seed202_ep50000_{sample,greedy}_summary.csv`
- Cond1 comm-on milestones:
  - seed 101:
    - ep7000: coop=0.100, avg_reward=4.320, regime=0.050/0.231/0.626, mi(m;f)=0.047, mi(m;a)=0.009
    - ep8000: coop=0.410, avg_reward=7.970, regime=0.050/0.239/0.622, mi(m;f)=0.013, mi(m;a)=0.001
    - ep9000: coop=0.095, avg_reward=4.270, regime=0.052/0.282/0.564, mi(m;f)=0.003, mi(m;a)=0.001
    - ep10000: coop=0.328, avg_reward=8.535, regime=0.054/0.237/0.569, mi(m;f)=0.009, mi(m;a)=0.003
  - seed 202:
    - ep7000: coop=0.168, avg_reward=5.175, regime=0.051/0.210/0.534, mi(m;f)=0.000, mi(m;a)=0.000
    - ep8000: coop=0.250, avg_reward=7.580, regime=0.049/0.249/0.570, mi(m;f)=0.001, mi(m;a)=0.000
    - ep9000: coop=0.165, avg_reward=5.610, regime=0.050/0.212/0.606, mi(m;f)=0.000, mi(m;a)=0.000
    - ep10000: coop=0.300, avg_reward=7.870, regime=0.051/0.199/0.536, mi(m;f)=0.000, mi(m;a)=0.000
  - watchpoint: Cond1 seed 202 MI is near-zero through 10k; monitor trend, not a failure.
- Cond2 additional milestones (comm-off):
  - seed 101:
    - ep60000: coop=0.287, avg_reward=7.710, regime=0.063/0.216/0.500
  - seed 202:
    - ep60000: coop=0.237, avg_reward=6.855, regime=0.076/0.276/0.556

- [2026-03-04 14:06:34] watchdog health: active_train_ppo=0
```text
286 outputs/train/phase2b/metrics/cond1_seed101.jsonl
     260 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    1024 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    1008 outputs/train/phase2b/metrics/cond2_seed202.jsonl
    2578 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log
- note: the watchdog's first process-count sample is unreliable due process-listing permission context; manual `ps aux | grep train_ppo` checks confirmed 4 active training processes immediately before/after.

- [2026-03-04 14:06:34] milestone cond1 seed 101 ep10000: regime_coop(comp/mixed/cooperative)=0.054/0.237/0.569, avg_reward(comp/mixed/cooperative)=3.893/5.931/13.109, mi(m;f)=0.009, mi(m;a)=0.003
- [2026-03-04 14:06:34] milestone cond1 seed 202 ep10000: regime_coop(comp/mixed/cooperative)=0.051/0.199/0.536, avg_reward(comp/mixed/cooperative)=3.898/5.616/12.580, mi(m;f)=0.000, mi(m;a)=0.000
- [2026-03-04 14:06:34] milestone cond2 seed 101 ep10000: regime_coop(comp/mixed/cooperative)=0.051/0.237/0.504, avg_reward(comp/mixed/cooperative)=3.898/5.931/12.069
- [2026-03-04 14:06:34] milestone cond2 seed 101 ep20000: regime_coop(comp/mixed/cooperative)=0.082/0.258/0.448, avg_reward(comp/mixed/cooperative)=3.836/5.895/11.164
- [2026-03-04 14:06:34] milestone cond2 seed 101 ep30000: regime_coop(comp/mixed/cooperative)=0.059/0.225/0.531, avg_reward(comp/mixed/cooperative)=3.883/5.788/12.492
- [2026-03-04 14:06:34] milestone cond2 seed 101 ep40000: regime_coop(comp/mixed/cooperative)=0.060/0.241/0.505, avg_reward(comp/mixed/cooperative)=3.881/5.905/12.082
- [2026-03-04 14:06:34] milestone cond2 seed 101 ep50000: regime_coop(comp/mixed/cooperative)=0.057/0.240/0.480, avg_reward(comp/mixed/cooperative)=3.885/5.866/11.679
- [2026-03-04 14:06:34] milestone cond2 seed 101 ep60000: regime_coop(comp/mixed/cooperative)=0.063/0.216/0.500, avg_reward(comp/mixed/cooperative)=3.873/5.724/12.007
- [2026-03-04 14:06:34] milestone cond2 seed 202 ep10000: regime_coop(comp/mixed/cooperative)=0.054/0.204/0.479, avg_reward(comp/mixed/cooperative)=3.893/5.663/11.658
- [2026-03-04 14:06:34] milestone cond2 seed 202 ep20000: regime_coop(comp/mixed/cooperative)=0.054/0.281/0.556, avg_reward(comp/mixed/cooperative)=3.892/6.261/12.892
- [2026-03-04 14:06:34] milestone cond2 seed 202 ep30000: regime_coop(comp/mixed/cooperative)=0.072/0.286/0.520, avg_reward(comp/mixed/cooperative)=3.857/6.186/12.327
- [2026-03-04 14:06:34] milestone cond2 seed 202 ep40000: regime_coop(comp/mixed/cooperative)=0.072/0.233/0.528, avg_reward(comp/mixed/cooperative)=3.856/5.787/12.448
- [2026-03-04 14:06:34] milestone cond2 seed 202 ep50000: regime_coop(comp/mixed/cooperative)=0.069/0.238/0.539, avg_reward(comp/mixed/cooperative)=3.862/5.853/12.620
- [2026-03-04 14:06:34] milestone cond2 seed 202 ep60000: regime_coop(comp/mixed/cooperative)=0.076/0.276/0.556, avg_reward(comp/mixed/cooperative)=3.847/6.091/12.900

- [2026-03-04 14:22:23] milestone cond2 seed 101 ep70000: regime_coop(comp/mixed/cooperative)=0.066/0.258/0.555, avg_reward(comp/mixed/cooperative)=3.869/6.008/12.885

- [2026-03-04 14:32:23] milestone cond2 seed 202 ep70000: regime_coop(comp/mixed/cooperative)=0.081/0.276/0.549, avg_reward(comp/mixed/cooperative)=3.839/6.036/12.791

- [2026-03-04 14:37:23] watchdog health: active_train_ppo=UNAVAILABLE
```text
442 outputs/train/phase2b/metrics/cond1_seed101.jsonl
     364 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    1200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    1168 outputs/train/phase2b/metrics/cond2_seed202.jsonl
    3174 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 14:52:23] milestone cond2 seed 101 ep80000: regime_coop(comp/mixed/cooperative)=0.066/0.215/0.506, avg_reward(comp/mixed/cooperative)=3.868/5.714/12.092

- [2026-03-04 14:57:23] milestone cond1 seed 101 ep20000: regime_coop(comp/mixed/cooperative)=0.050/0.254/0.602, avg_reward(comp/mixed/cooperative)=3.900/6.087/13.638, mi(m;f)=0.091, mi(m;a)=0.036

- [2026-03-04 15:02:23] milestone cond2 seed 202 ep80000: regime_coop(comp/mixed/cooperative)=0.064/0.280/0.564, avg_reward(comp/mixed/cooperative)=3.871/6.159/13.020

- [2026-03-04 15:07:24] watchdog health: active_train_ppo=UNAVAILABLE
```text
572 outputs/train/phase2b/metrics/cond1_seed101.jsonl
     494 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    1360 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    1328 outputs/train/phase2b/metrics/cond2_seed202.jsonl
    3754 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 15:12:24] milestone cond1 seed 202 ep20000: regime_coop(comp/mixed/cooperative)=0.050/0.253/0.520, avg_reward(comp/mixed/cooperative)=3.900/6.058/12.320, mi(m;f)=0.025, mi(m;a)=0.005

- [2026-03-04 15:22:24] milestone cond2 seed 101 ep90000: regime_coop(comp/mixed/cooperative)=0.069/0.265/0.539, avg_reward(comp/mixed/cooperative)=3.862/6.069/12.616

- [2026-03-04 15:27:24] milestone cond2 seed 202 ep90000: regime_coop(comp/mixed/cooperative)=0.124/0.286/0.563, avg_reward(comp/mixed/cooperative)=3.752/6.076/13.011

- [2026-03-04 15:37:24] watchdog health: active_train_ppo=UNAVAILABLE
```text
728 outputs/train/phase2b/metrics/cond1_seed101.jsonl
     624 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    1520 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    1488 outputs/train/phase2b/metrics/cond2_seed202.jsonl
    4360 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 15:52:24] milestone cond1 seed 101 ep30000: regime_coop(comp/mixed/cooperative)=0.052/0.213/0.515, avg_reward(comp/mixed/cooperative)=3.896/5.746/12.237, mi(m;f)=0.099, mi(m;a)=0.022
- [2026-03-04 15:52:24] milestone cond2 seed 101 ep100000: regime_coop(comp/mixed/cooperative)=0.065/0.260/0.549, avg_reward(comp/mixed/cooperative)=3.869/6.029/12.776

- [2026-03-04 15:52:48] eval done: cond2 seed 101 ep100000 sample
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond2_seed101_ep100000_sample_summary.csv
[summary] unknown competitive [sample] coop=0.055 reward=3.889 n_rounds=5309
[summary] unknown cooperative [sample] coop=0.488 reward=11.808 n_rounds=6363
[summary] unknown mixed [sample] coop=0.218 reward=5.671 n_rounds=18328
```

- [2026-03-04 15:53:03] eval done: cond2 seed 101 ep100000 greedy
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond2_seed101_ep100000_greedy_summary.csv
[summary] unknown competitive [greedy] coop=0.002 reward=3.996 n_rounds=5309
[summary] unknown cooperative [greedy] coop=0.476 reward=11.619 n_rounds=6363
[summary] unknown mixed [greedy] coop=0.147 reward=5.220 n_rounds=18328
```

- [2026-03-04 15:58:08] milestone cond2 seed 202 ep100000: regime_coop(comp/mixed/cooperative)=0.082/0.276/0.503, avg_reward(comp/mixed/cooperative)=3.836/6.123/12.054

- [2026-03-04 15:58:32] eval done: cond2 seed 202 ep100000 sample
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond2_seed202_ep100000_sample_summary.csv
[summary] unknown competitive [sample] coop=0.117 reward=3.767 n_rounds=5309
[summary] unknown cooperative [sample] coop=0.465 reward=11.446 n_rounds=6363
[summary] unknown mixed [sample] coop=0.258 reward=5.954 n_rounds=18328
```

- [2026-03-04 15:58:47] eval done: cond2 seed 202 ep100000 greedy
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond2_seed202_ep100000_greedy_summary.csv
[summary] unknown competitive [greedy] coop=0.034 reward=3.933 n_rounds=5309
[summary] unknown cooperative [greedy] coop=0.403 reward=10.452 n_rounds=6363
[summary] unknown mixed [greedy] coop=0.180 reward=5.515 n_rounds=18328
```

- [2026-03-04 16:08:52] watchdog health: active_train_ppo=UNAVAILABLE
```text
858 outputs/train/phase2b/metrics/cond1_seed101.jsonl
     754 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    1712 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    1664 outputs/train/phase2b/metrics/cond2_seed202.jsonl
    4988 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 16:13:52] milestone cond1 seed 202 ep30000: regime_coop(comp/mixed/cooperative)=0.054/0.252/0.579, avg_reward(comp/mixed/cooperative)=3.892/6.023/13.265, mi(m;f)=0.018, mi(m;a)=0.007

- [2026-03-04 16:18:52] milestone cond2 seed 101 ep110000: regime_coop(comp/mixed/cooperative)=0.086/0.239/0.537, avg_reward(comp/mixed/cooperative)=3.828/5.815/12.593

- [2026-03-04 16:28:52] milestone cond2 seed 202 ep110000: regime_coop(comp/mixed/cooperative)=0.105/0.265/0.535, avg_reward(comp/mixed/cooperative)=3.790/5.944/12.554

- [2026-03-04 16:38:52] watchdog health: active_train_ppo=UNAVAILABLE
```text
1014 outputs/train/phase2b/metrics/cond1_seed101.jsonl
     884 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    1872 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    1824 outputs/train/phase2b/metrics/cond2_seed202.jsonl
    5594 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 16:43:52] milestone cond1 seed 101 ep40000: regime_coop(comp/mixed/cooperative)=0.052/0.303/0.589, avg_reward(comp/mixed/cooperative)=3.896/6.365/13.418, mi(m;f)=0.025, mi(m;a)=0.007

- [2026-03-04 16:48:52] milestone cond2 seed 101 ep120000: regime_coop(comp/mixed/cooperative)=0.070/0.269/0.506, avg_reward(comp/mixed/cooperative)=3.860/5.982/12.092

- [2026-03-04 16:58:52] milestone cond2 seed 202 ep120000: regime_coop(comp/mixed/cooperative)=0.076/0.274/0.534, avg_reward(comp/mixed/cooperative)=3.848/6.023/12.542

- [2026-03-04 17:08:52] watchdog health: active_train_ppo=UNAVAILABLE
```text
1144 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    1014 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    2032 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    1984 outputs/train/phase2b/metrics/cond2_seed202.jsonl
    6174 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 17:13:52] milestone cond1 seed 202 ep40000: regime_coop(comp/mixed/cooperative)=0.051/0.255/0.499, avg_reward(comp/mixed/cooperative)=3.897/6.058/11.980, mi(m;f)=0.024, mi(m;a)=0.004

- [2026-03-04 17:18:52] milestone cond2 seed 101 ep130000: regime_coop(comp/mixed/cooperative)=0.088/0.264/0.522, avg_reward(comp/mixed/cooperative)=3.824/6.001/12.359

- [2026-03-04 17:28:53] milestone cond2 seed 202 ep130000: regime_coop(comp/mixed/cooperative)=0.079/0.262/0.573, avg_reward(comp/mixed/cooperative)=3.841/5.970/13.163

- [2026-03-04 17:38:53] watchdog health: active_train_ppo=UNAVAILABLE
```text
1300 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    1118 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    2208 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    2144 outputs/train/phase2b/metrics/cond2_seed202.jsonl
    6770 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 17:38:53] milestone cond1 seed 101 ep50000: regime_coop(comp/mixed/cooperative)=0.066/0.357/0.615, avg_reward(comp/mixed/cooperative)=3.869/6.685/13.845, mi(m;f)=0.078, mi(m;a)=0.035

- [2026-03-04 17:39:42] eval done: cond1 seed 101 ep50000 sample
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed101_ep50000_sample_summary.csv
[summary] unknown competitive [sample] coop=0.057 reward=3.887 n_rounds=5668
[summary] unknown cooperative [sample] coop=0.617 reward=13.872 n_rounds=6512
[summary] unknown mixed [sample] coop=0.370 reward=6.975 n_rounds=17820
```

- [2026-03-04 17:40:02] eval done: cond1 seed 101 ep50000 greedy
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed101_ep50000_greedy_summary.csv
[summary] unknown competitive [greedy] coop=0.016 reward=3.968 n_rounds=5288
[summary] unknown cooperative [greedy] coop=0.705 reward=15.277 n_rounds=5917
[summary] unknown mixed [greedy] coop=0.377 reward=7.124 n_rounds=18795
```

- [2026-03-04 17:45:07] milestone cond2 seed 101 ep140000: regime_coop(comp/mixed/cooperative)=0.058/0.269/0.528, avg_reward(comp/mixed/cooperative)=3.883/6.113/12.445

- [2026-03-04 17:55:08] milestone cond2 seed 202 ep140000: regime_coop(comp/mixed/cooperative)=0.056/0.252/0.512, avg_reward(comp/mixed/cooperative)=3.889/5.934/12.187

- [2026-03-04 18:10:08] watchdog health: active_train_ppo=UNAVAILABLE
```text
1430 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    1248 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    2368 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    2320 outputs/train/phase2b/metrics/cond2_seed202.jsonl
    7366 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 18:15:08] milestone cond2 seed 101 ep150000: regime_coop(comp/mixed/cooperative)=0.057/0.285/0.498, avg_reward(comp/mixed/cooperative)=3.886/6.212/11.974

- [2026-03-04 18:15:33] eval done: cond2 seed 101 ep150000 sample
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond2_seed101_ep150000_sample_summary.csv
[summary] unknown competitive [sample] coop=0.056 reward=3.889 n_rounds=5309
[summary] unknown cooperative [sample] coop=0.501 reward=12.016 n_rounds=6363
[summary] unknown mixed [sample] coop=0.292 reward=6.239 n_rounds=18328
```

- [2026-03-04 18:15:48] eval done: cond2 seed 101 ep150000 greedy
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond2_seed101_ep150000_greedy_summary.csv
[summary] unknown competitive [greedy] coop=0.004 reward=3.993 n_rounds=5309
[summary] unknown cooperative [greedy] coop=0.549 reward=12.784 n_rounds=6363
[summary] unknown mixed [greedy] coop=0.234 reward=5.914 n_rounds=18328
```

- [2026-03-04 18:20:53] milestone cond1 seed 202 ep50000: regime_coop(comp/mixed/cooperative)=0.052/0.264/0.539, avg_reward(comp/mixed/cooperative)=3.897/6.079/12.623, mi(m;f)=0.002, mi(m;a)=0.000

- [2026-03-04 18:21:30] eval done: cond1 seed 202 ep50000 sample
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed202_ep50000_sample_summary.csv
[summary] unknown competitive [sample] coop=0.050 reward=3.901 n_rounds=5668
[summary] unknown cooperative [sample] coop=0.632 reward=14.107 n_rounds=6512
[summary] unknown mixed [sample] coop=0.263 reward=6.187 n_rounds=17820
```

- [2026-03-04 18:21:50] eval done: cond1 seed 202 ep50000 greedy
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed202_ep50000_greedy_summary.csv
[summary] unknown competitive [greedy] coop=0.000 reward=4.000 n_rounds=5288
[summary] unknown cooperative [greedy] coop=0.671 reward=14.733 n_rounds=5917
[summary] unknown mixed [greedy] coop=0.220 reward=6.009 n_rounds=18795
```

- [2026-03-04 18:26:55] milestone cond2 seed 202 ep150000: regime_coop(comp/mixed/cooperative)=0.060/0.259/0.520, avg_reward(comp/mixed/cooperative)=3.880/5.958/12.325

- [2026-03-04 18:27:19] eval done: cond2 seed 202 ep150000 sample
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond2_seed202_ep150000_sample_summary.csv
[summary] unknown competitive [sample] coop=0.065 reward=3.870 n_rounds=5309
[summary] unknown cooperative [sample] coop=0.571 reward=13.142 n_rounds=6363
[summary] unknown mixed [sample] coop=0.314 reward=6.481 n_rounds=18328
```

- [2026-03-04 18:27:34] eval done: cond2 seed 202 ep150000 greedy
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond2_seed202_ep150000_greedy_summary.csv
[summary] unknown competitive [greedy] coop=0.016 reward=3.969 n_rounds=5309
[summary] unknown cooperative [greedy] coop=0.788 reward=16.614 n_rounds=6363
[summary] unknown mixed [greedy] coop=0.249 reward=6.249 n_rounds=18328
```

- [2026-03-04 18:32:39] milestone cond1 seed 101 ep60000: regime_coop(comp/mixed/cooperative)=0.055/0.258/0.512, avg_reward(comp/mixed/cooperative)=3.889/6.023/12.196, mi(m;f)=0.075, mi(m;a)=0.026

- [2026-03-04 18:42:39] watchdog health: active_train_ppo=UNAVAILABLE
```text
1586 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    1404 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    2560 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    2496 outputs/train/phase2b/metrics/cond2_seed202.jsonl
    8046 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 18:42:39] milestone cond2 seed 101 ep160000: regime_coop(comp/mixed/cooperative)=0.083/0.309/0.534, avg_reward(comp/mixed/cooperative)=3.834/6.218/12.542

- [2026-03-04 18:52:39] milestone cond2 seed 202 ep160000: regime_coop(comp/mixed/cooperative)=0.078/0.254/0.518, avg_reward(comp/mixed/cooperative)=3.845/5.975/12.290

- [2026-03-04 19:12:39] watchdog health: active_train_ppo=UNAVAILABLE
```text
1742 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    1534 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    2720 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    2672 outputs/train/phase2b/metrics/cond2_seed202.jsonl
    8668 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 19:12:39] milestone cond2 seed 101 ep170000: regime_coop(comp/mixed/cooperative)=0.057/0.299/0.522, avg_reward(comp/mixed/cooperative)=3.886/6.254/12.358

- [2026-03-04 19:17:39] milestone cond1 seed 202 ep60000: regime_coop(comp/mixed/cooperative)=0.053/0.273/0.568, avg_reward(comp/mixed/cooperative)=3.894/6.171/13.089, mi(m;f)=0.000, mi(m;a)=0.001

- [2026-03-04 19:22:39] milestone cond2 seed 202 ep170000: regime_coop(comp/mixed/cooperative)=0.081/0.298/0.554, avg_reward(comp/mixed/cooperative)=3.838/6.261/12.872

- [2026-03-04 19:27:39] milestone cond1 seed 101 ep70000: regime_coop(comp/mixed/cooperative)=0.055/0.321/0.592, avg_reward(comp/mixed/cooperative)=3.890/6.541/13.472, mi(m;f)=0.017, mi(m;a)=0.007

- [2026-03-04 19:42:40] watchdog health: active_train_ppo=UNAVAILABLE
```text
1898 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    1664 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    2880 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    2832 outputs/train/phase2b/metrics/cond2_seed202.jsonl
    9274 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 19:42:40] milestone cond2 seed 101 ep180000: regime_coop(comp/mixed/cooperative)=0.112/0.314/0.515, avg_reward(comp/mixed/cooperative)=3.777/6.301/12.240

- [2026-03-04 19:52:40] milestone cond2 seed 202 ep180000: regime_coop(comp/mixed/cooperative)=0.071/0.258/0.491, avg_reward(comp/mixed/cooperative)=3.859/5.946/11.863

- [2026-03-04 20:12:40] watchdog health: active_train_ppo=UNAVAILABLE
```text
2028 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    1768 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3040 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    2992 outputs/train/phase2b/metrics/cond2_seed202.jsonl
    9828 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 20:12:40] milestone cond2 seed 101 ep190000: regime_coop(comp/mixed/cooperative)=0.072/0.290/0.534, avg_reward(comp/mixed/cooperative)=3.857/6.185/12.540

- [2026-03-04 20:22:40] milestone cond1 seed 101 ep80000: regime_coop(comp/mixed/cooperative)=0.067/0.362/0.568, avg_reward(comp/mixed/cooperative)=3.865/6.656/13.081, mi(m;f)=0.090, mi(m;a)=0.015
- [2026-03-04 20:22:40] milestone cond1 seed 202 ep70000: regime_coop(comp/mixed/cooperative)=0.053/0.268/0.509, avg_reward(comp/mixed/cooperative)=3.895/6.159/12.142, mi(m;f)=0.022, mi(m;a)=0.008
- [2026-03-04 20:22:40] milestone cond2 seed 202 ep190000: regime_coop(comp/mixed/cooperative)=0.089/0.261/0.514, avg_reward(comp/mixed/cooperative)=3.822/5.984/12.228

- [2026-03-04 20:42:40] watchdog health: active_train_ppo=UNAVAILABLE
```text
2184 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    1898 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3152 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   10434 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 20:42:40] milestone cond2 seed 101 ep200000: regime_coop(comp/mixed/cooperative)=0.084/0.276/0.511, avg_reward(comp/mixed/cooperative)=3.832/6.026/12.175

- [2026-03-04 20:43:06] eval done: cond2 seed 101 ep200000 sample
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond2_seed101_ep200000_sample_summary.csv
[summary] cond2 competitive [sample] coop=0.107 reward=3.786 n_rounds=5309
[summary] cond2 cooperative [sample] coop=0.533 reward=12.533 n_rounds=6363
[summary] cond2 mixed [sample] coop=0.295 reward=6.111 n_rounds=18328
```

- [2026-03-04 20:43:21] eval done: cond2 seed 101 ep200000 greedy
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond2_seed101_ep200000_greedy_summary.csv
[summary] cond2 competitive [greedy] coop=0.035 reward=3.930 n_rounds=5309
[summary] cond2 cooperative [greedy] coop=0.571 reward=13.135 n_rounds=6363
[summary] cond2 mixed [greedy] coop=0.250 reward=5.920 n_rounds=18328
```

- [2026-03-04 20:53:26] milestone cond2 seed 202 ep200000: regime_coop(comp/mixed/cooperative)=0.067/0.291/0.505, avg_reward(comp/mixed/cooperative)=3.867/6.237/12.075

- [2026-03-04 20:53:49] eval done: cond2 seed 202 ep200000 sample
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond2_seed202_ep200000_sample_summary.csv
[summary] cond2 competitive [sample] coop=0.061 reward=3.878 n_rounds=5309
[summary] cond2 cooperative [sample] coop=0.515 reward=12.236 n_rounds=6363
[summary] cond2 mixed [sample] coop=0.294 reward=6.333 n_rounds=18328
```

- [2026-03-04 20:54:03] eval done: cond2 seed 202 ep200000 greedy
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond2_seed202_ep200000_greedy_summary.csv
[summary] cond2 competitive [greedy] coop=0.003 reward=3.994 n_rounds=5309
[summary] cond2 cooperative [greedy] coop=0.500 reward=12.000 n_rounds=6363
[summary] cond2 mixed [greedy] coop=0.196 reward=5.725 n_rounds=18328
```

- [2026-03-04 21:14:09] watchdog health: active_train_ppo=UNAVAILABLE
```text
2340 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    2054 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   10794 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 21:14:09] milestone cond1 seed 101 ep90000: regime_coop(comp/mixed/cooperative)=0.075/0.341/0.600, avg_reward(comp/mixed/cooperative)=3.850/6.525/13.593, mi(m;f)=0.006, mi(m;a)=0.005

- [2026-03-04 21:24:09] milestone cond1 seed 202 ep80000: regime_coop(comp/mixed/cooperative)=0.060/0.307/0.578, avg_reward(comp/mixed/cooperative)=3.880/6.378/13.246, mi(m;f)=0.023, mi(m;a)=0.006

- [2026-03-04 21:44:09] watchdog health: active_train_ppo=UNAVAILABLE
```text
2496 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    2184 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   11080 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 22:04:09] milestone cond1 seed 101 ep100000: regime_coop(comp/mixed/cooperative)=0.058/0.315/0.471, avg_reward(comp/mixed/cooperative)=3.883/6.398/11.543, mi(m;f)=0.001, mi(m;a)=0.002

- [2026-03-04 22:04:48] eval done: cond1 seed 101 ep100000 sample
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed101_ep100000_sample_summary.csv
[summary] unknown competitive [sample] coop=0.051 reward=3.897 n_rounds=5668
[summary] unknown cooperative [sample] coop=0.353 reward=9.645 n_rounds=6512
[summary] unknown mixed [sample] coop=0.224 reward=5.752 n_rounds=17820
```

- [2026-03-04 22:05:08] eval done: cond1 seed 101 ep100000 greedy
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed101_ep100000_greedy_summary.csv
[summary] unknown competitive [greedy] coop=0.000 reward=4.000 n_rounds=5288
[summary] unknown cooperative [greedy] coop=0.252 reward=8.026 n_rounds=5917
[summary] unknown mixed [greedy] coop=0.118 reward=5.095 n_rounds=18795
```

- [2026-03-04 22:15:13] watchdog health: active_train_ppo=UNAVAILABLE
```text
2652 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    2314 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   11366 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 22:20:13] milestone cond1 seed 202 ep90000: regime_coop(comp/mixed/cooperative)=0.054/0.281/0.474, avg_reward(comp/mixed/cooperative)=3.893/6.114/11.587, mi(m;f)=0.020, mi(m;a)=0.019

- [2026-03-04 22:45:14] watchdog health: active_train_ppo=UNAVAILABLE
```text
2808 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    2444 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   11652 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 22:55:14] milestone cond1 seed 101 ep110000: regime_coop(comp/mixed/cooperative)=0.058/0.319/0.525, avg_reward(comp/mixed/cooperative)=3.885/6.389/12.401, mi(m;f)=0.001, mi(m;a)=0.000

- [2026-03-04 23:15:14] watchdog health: active_train_ppo=UNAVAILABLE
```text
2964 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    2600 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   11964 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-04 23:15:14] milestone cond1 seed 202 ep100000: regime_coop(comp/mixed/cooperative)=0.050/0.229/0.552, avg_reward(comp/mixed/cooperative)=3.900/5.854/12.832, mi(m;f)=0.001, mi(m;a)=0.000

- [2026-03-04 23:15:52] eval done: cond1 seed 202 ep100000 sample
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed202_ep100000_sample_summary.csv
[summary] unknown competitive [sample] coop=0.050 reward=3.900 n_rounds=5668
[summary] unknown cooperative [sample] coop=0.625 reward=13.999 n_rounds=6512
[summary] unknown mixed [sample] coop=0.225 reward=5.905 n_rounds=17820
```

- [2026-03-04 23:16:11] eval done: cond1 seed 202 ep100000 greedy
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed202_ep100000_greedy_summary.csv
[summary] unknown competitive [greedy] coop=0.001 reward=3.999 n_rounds=5288
[summary] unknown cooperative [greedy] coop=0.690 reward=15.033 n_rounds=5917
[summary] unknown mixed [greedy] coop=0.140 reward=5.301 n_rounds=18795
```

- [2026-03-04 23:41:16] milestone cond1 seed 101 ep120000: regime_coop(comp/mixed/cooperative)=0.051/0.248/0.460, avg_reward(comp/mixed/cooperative)=3.897/5.931/11.353, mi(m;f)=0.001, mi(m;a)=0.000

- [2026-03-04 23:46:17] watchdog health: active_train_ppo=UNAVAILABLE
```text
3146 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    2730 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   12276 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 00:11:17] milestone cond1 seed 202 ep110000: regime_coop(comp/mixed/cooperative)=0.056/0.284/0.553, avg_reward(comp/mixed/cooperative)=3.888/6.234/12.841, mi(m;f)=0.016, mi(m;a)=0.007

- [2026-03-05 00:16:17] watchdog health: active_train_ppo=UNAVAILABLE
```text
3302 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    2886 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   12588 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 00:26:17] milestone cond1 seed 101 ep130000: regime_coop(comp/mixed/cooperative)=0.090/0.317/0.564, avg_reward(comp/mixed/cooperative)=3.819/6.332/13.030, mi(m;f)=0.021, mi(m;a)=0.000

- [2026-03-05 00:46:17] watchdog health: active_train_ppo=UNAVAILABLE
```text
3484 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    3042 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   12926 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 01:01:17] milestone cond1 seed 202 ep120000: regime_coop(comp/mixed/cooperative)=0.068/0.348/0.588, avg_reward(comp/mixed/cooperative)=3.864/6.607/13.404, mi(m;f)=0.000, mi(m;a)=0.002

- [2026-03-05 01:11:17] milestone cond1 seed 101 ep140000: regime_coop(comp/mixed/cooperative)=0.062/0.331/0.546, avg_reward(comp/mixed/cooperative)=3.875/6.575/12.735, mi(m;f)=0.016, mi(m;a)=0.008

- [2026-03-05 01:16:17] watchdog health: active_train_ppo=UNAVAILABLE
```text
3666 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    3198 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   13264 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 01:46:18] watchdog health: active_train_ppo=UNAVAILABLE
```text
3822 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    3328 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   13550 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 01:56:18] milestone cond1 seed 101 ep150000: regime_coop(comp/mixed/cooperative)=0.060/0.288/0.571, avg_reward(comp/mixed/cooperative)=3.880/6.257/13.134, mi(m;f)=0.024, mi(m;a)=0.006
- [2026-03-05 01:56:18] milestone cond1 seed 202 ep130000: regime_coop(comp/mixed/cooperative)=0.059/0.334/0.556, avg_reward(comp/mixed/cooperative)=3.881/6.554/12.893, mi(m;f)=0.009, mi(m;a)=0.000

- [2026-03-05 01:56:54] eval done: cond1 seed 101 ep150000 sample
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed101_ep150000_sample_summary.csv
[summary] unknown competitive [sample] coop=0.055 reward=3.890 n_rounds=5668
[summary] unknown cooperative [sample] coop=0.680 reward=14.875 n_rounds=6512
[summary] unknown mixed [sample] coop=0.351 reward=6.890 n_rounds=17820
```

- [2026-03-05 01:57:12] eval done: cond1 seed 101 ep150000 greedy
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed101_ep150000_greedy_summary.csv
[summary] unknown competitive [greedy] coop=0.007 reward=3.987 n_rounds=5288
[summary] unknown cooperative [greedy] coop=0.741 reward=15.849 n_rounds=5917
[summary] unknown mixed [greedy] coop=0.322 reward=6.824 n_rounds=18795
```

- [2026-03-05 02:17:18] watchdog health: active_train_ppo=UNAVAILABLE
```text
4004 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    3484 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   13888 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 02:42:18] milestone cond1 seed 101 ep160000: regime_coop(comp/mixed/cooperative)=0.090/0.405/0.501, avg_reward(comp/mixed/cooperative)=3.821/6.803/12.023, mi(m;f)=0.028, mi(m;a)=0.006

- [2026-03-05 02:47:18] watchdog health: active_train_ppo=UNAVAILABLE
```text
4186 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    3640 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   14226 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 02:47:18] milestone cond1 seed 202 ep140000: regime_coop(comp/mixed/cooperative)=0.061/0.290/0.510, avg_reward(comp/mixed/cooperative)=3.878/6.231/12.152, mi(m;f)=0.018, mi(m;a)=0.000

- [2026-03-05 03:17:18] watchdog health: active_train_ppo=UNAVAILABLE
```text
4368 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    3796 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   14564 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 03:27:18] milestone cond1 seed 101 ep170000: regime_coop(comp/mixed/cooperative)=0.095/0.325/0.558, avg_reward(comp/mixed/cooperative)=3.810/6.458/12.930, mi(m;f)=0.001, mi(m;a)=0.001

- [2026-03-05 03:37:19] milestone cond1 seed 202 ep150000: regime_coop(comp/mixed/cooperative)=0.063/0.323/0.456, avg_reward(comp/mixed/cooperative)=3.874/6.399/11.299, mi(m;f)=0.000, mi(m;a)=0.000

- [2026-03-05 03:37:55] eval done: cond1 seed 202 ep150000 sample
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed202_ep150000_sample_summary.csv
[summary] unknown competitive [sample] coop=0.055 reward=3.890 n_rounds=5668
[summary] unknown cooperative [sample] coop=0.356 reward=9.693 n_rounds=6512
[summary] unknown mixed [sample] coop=0.271 reward=6.060 n_rounds=17820
```

- [2026-03-05 03:38:13] eval done: cond1 seed 202 ep150000 greedy
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed202_ep150000_greedy_summary.csv
[summary] unknown competitive [greedy] coop=0.000 reward=4.000 n_rounds=5288
[summary] unknown cooperative [greedy] coop=0.072 reward=5.149 n_rounds=5917
[summary] unknown mixed [greedy] coop=0.113 reward=4.941 n_rounds=18795
```

- [2026-03-05 03:48:19] watchdog health: active_train_ppo=UNAVAILABLE
```text
4524 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    3952 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   14876 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 04:13:19] milestone cond1 seed 101 ep180000: regime_coop(comp/mixed/cooperative)=0.087/0.373/0.670, avg_reward(comp/mixed/cooperative)=3.826/6.712/14.715, mi(m;f)=0.000, mi(m;a)=0.011

- [2026-03-05 04:18:19] watchdog health: active_train_ppo=UNAVAILABLE
```text
4706 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    4108 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   15214 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 04:33:19] milestone cond1 seed 202 ep160000: regime_coop(comp/mixed/cooperative)=0.058/0.276/0.574, avg_reward(comp/mixed/cooperative)=3.885/6.189/13.190, mi(m;f)=0.001, mi(m;a)=0.000

- [2026-03-05 04:48:19] watchdog health: active_train_ppo=UNAVAILABLE
```text
4888 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    4238 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   15526 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 04:58:19] milestone cond1 seed 101 ep190000: regime_coop(comp/mixed/cooperative)=0.059/0.289/0.490, avg_reward(comp/mixed/cooperative)=3.882/6.259/11.840, mi(m;f)=0.000, mi(m;a)=0.003

- [2026-03-05 05:18:20] watchdog health: active_train_ppo=UNAVAILABLE
```text
5044 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    4394 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   15838 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 05:23:20] milestone cond1 seed 202 ep170000: regime_coop(comp/mixed/cooperative)=0.064/0.297/0.554, avg_reward(comp/mixed/cooperative)=3.873/6.226/12.860, mi(m;f)=0.024, mi(m;a)=0.007

- [2026-03-05 05:43:20] milestone cond1 seed 101 ep200000: regime_coop(comp/mixed/cooperative)=0.057/0.251/0.580, avg_reward(comp/mixed/cooperative)=3.887/6.008/13.279, mi(m;f)=0.000, mi(m;a)=0.001

- [2026-03-05 05:43:55] eval done: cond1 seed 101 ep200000 sample
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed101_ep200000_sample_summary.csv
[summary] cond1 competitive [sample] coop=0.051 reward=3.897 n_rounds=5668
[summary] cond1 cooperative [sample] coop=0.561 reward=12.975 n_rounds=6512
[summary] cond1 mixed [sample] coop=0.261 reward=6.172 n_rounds=17820
```

- [2026-03-05 05:44:13] eval done: cond1 seed 101 ep200000 greedy
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed101_ep200000_greedy_summary.csv
[summary] cond1 competitive [greedy] coop=0.006 reward=3.989 n_rounds=5288
[summary] cond1 cooperative [greedy] coop=0.696 reward=15.133 n_rounds=5917
[summary] cond1 mixed [greedy] coop=0.320 reward=6.813 n_rounds=18795
```

- [2026-03-05 05:49:18] watchdog health: active_train_ppo=UNAVAILABLE
```text
5200 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    4550 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   16150 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 06:19:18] watchdog health: active_train_ppo=UNAVAILABLE
```text
5200 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    4680 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   16280 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 06:19:18] milestone cond1 seed 202 ep180000: regime_coop(comp/mixed/cooperative)=0.053/0.277/0.572, avg_reward(comp/mixed/cooperative)=3.893/6.215/13.147, mi(m;f)=0.020, mi(m;a)=0.006

- [2026-03-05 06:49:19] watchdog health: active_train_ppo=UNAVAILABLE
```text
5200 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    4836 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   16436 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 07:09:19] milestone cond1 seed 202 ep190000: regime_coop(comp/mixed/cooperative)=0.052/0.309/0.638, avg_reward(comp/mixed/cooperative)=3.895/6.447/14.204, mi(m;f)=0.001, mi(m;a)=0.004

- [2026-03-05 07:19:19] watchdog health: active_train_ppo=UNAVAILABLE
```text
5200 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    4992 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   16592 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 07:49:19] watchdog health: active_train_ppo=UNAVAILABLE
```text
5200 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    5148 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   16748 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 08:04:20] milestone cond1 seed 202 ep200000: regime_coop(comp/mixed/cooperative)=0.054/0.232/0.513, avg_reward(comp/mixed/cooperative)=3.893/5.837/12.207, mi(m;f)=0.000, mi(m;a)=0.001

- [2026-03-05 08:04:53] eval done: cond1 seed 202 ep200000 sample
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed202_ep200000_sample_summary.csv
[summary] cond1 competitive [sample] coop=0.054 reward=3.891 n_rounds=5668
[summary] cond1 cooperative [sample] coop=0.625 reward=13.994 n_rounds=6512
[summary] cond1 mixed [sample] coop=0.310 reward=6.486 n_rounds=17820
```

- [2026-03-05 08:05:11] eval done: cond1 seed 202 ep200000 greedy
```text
[eval] condition_summary_rows=3 out=/Users/mbp17/POSTDOC/NPS26/dsc-epgg/outputs/train/phase2b/eval/cond1_seed202_ep200000_greedy_summary.csv
[summary] cond1 competitive [greedy] coop=0.010 reward=3.980 n_rounds=5288
[summary] cond1 cooperative [greedy] coop=0.858 reward=17.731 n_rounds=5917
[summary] cond1 mixed [greedy] coop=0.210 reward=5.695 n_rounds=18795
```

- [2026-03-05 08:20:16] watchdog health: active_train_ppo=UNAVAILABLE
```text
5200 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    5200 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   16800 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log

- [2026-03-05 08:50:16] watchdog health: active_train_ppo=UNAVAILABLE
```text
5200 outputs/train/phase2b/metrics/cond1_seed101.jsonl
    5200 outputs/train/phase2b/metrics/cond1_seed202.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed101.jsonl
    3200 outputs/train/phase2b/metrics/cond2_seed202.jsonl
   16800 total
```
- crash scan: no matches in outputs/train/phase2b/logs/*.log
