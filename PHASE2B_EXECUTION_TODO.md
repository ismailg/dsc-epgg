# Phase 2b Execution TODO

Last updated: 2026-03-04 (local)

## 1) Diagnostic/tooling prerequisites (must complete first)
- [x] Verify per-`f` cooperation is already streamed in `train_ppo.py` JSONL (`scope="f_value"`; window + cumulative).
- [x] Verify evaluator already reports per-`f` breakdown (`scope="f_value"` rows in CSV).
- [ ] Add comm metrics to trainer JSONL (`scope="comm"`):
  - [x] `mi_message_f` over latest logging window
  - [x] `mi_message_action` over latest logging window
- [x] Add `--greedy` mode to `evaluate_regime_conditional.py`:
  - [x] Action/message argmax instead of sampling
  - [x] Force eval tremble off (`epsilon_tremble=0.0`) in greedy mode
- [x] Add intermediate checkpoint saving to `train_ppo.py`:
  - [x] New flag `--checkpoint_interval` (default `0`)
  - [x] Save at interval episodes and at final episode

## 2) Validation after code changes
- [ ] Run tests: `python -m pytest tests/ -v` (currently blocked in this shell by torch import abort)
- [x] Run quick smoke eval of `--greedy` mode on an existing checkpoint
- [x] Confirm comm JSONL records are emitted for a short comm-enabled run

## 3) Phase 2b pilot training (2 seeds)
- [ ] Condition 2 pilot (`seed=101,202`, 200k, warm start from fixed-f `f=5` checkpoints)
- [ ] Condition 1 pilot (`seed=101,202`, 200k, comm enabled, warm start from fixed-f `f=5` checkpoints)
- [ ] Enforce constraints:
  - [ ] Max 2 training processes in parallel
  - [ ] Always `--gamma 0.99 --reward_scale 20.0`
  - [ ] Always set `--metrics_jsonl_path`
  - [ ] Use `--checkpoint_interval 50000`

## 4) Milestone evaluation + summary
- [ ] Evaluate pilot checkpoints at 50k / 100k / 150k / 200k with regime + per-`f` outputs
- [ ] Extract comm MI trends from JSONL for comm-on runs
- [ ] Write summary: `outputs/train/phase2b/PILOT_SUMMARY.md`
- [ ] Decide go/no-go for full 5-seed 200k grid
