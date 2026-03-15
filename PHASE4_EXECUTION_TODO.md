# Phase 4 Execution TODO

## Status
- [x] Implement same-checkpoint continuation runner
- [x] Extend channel-control summary tooling
- [x] Implement bootstrap CI and paired-delta analysis
- [x] Implement sender-slot permutation summary
- [x] Create paper package scaffold
- [x] Validate new analysis scripts on existing Phase 3 outputs
- [ ] Launch same-checkpoint continuation training (`fixed0`, `uniform`, `public_random`)
  - `fixed0` training live on seeds `101/202/303`
  - seeds `404/505` queued behind `max_workers=3`
  - `uniform` and `public_random` queued after `fixed0`
- [ ] Evaluate same-checkpoint continuation bundle
- [ ] Regenerate bootstrap / paired-delta outputs with same-checkpoint bundle
- [ ] Regenerate permutation summary with same-checkpoint bundle
- [ ] Update paper package memo with same-checkpoint results

## Key commands
- Training + eval bundle:
  - `./scripts/run_phase3_sameckpt_suite.sh`
- Stats only:
  - `python3 -m src.analysis.compute_bootstrap_cis ...`
- Permutation summary only:
  - `python3 -m src.analysis.summarize_sender_slot_permutation ...`
