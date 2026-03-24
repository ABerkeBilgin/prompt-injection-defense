# File-Based Task List

## src/model

- [ ] `src/model/setup.py`: add argument support for token count and output directory naming.
- [ ] `src/model/demo.py`: separate trusted/untrusted input assembly into reusable function.
- [ ] `src/model/README.md`: document exact run command for baseline and defended runs.

## src/data

- [x] `src/data/build_dataset.py`: create initial JSONL dataset builder.
- [ ] `src/data/build_dataset.py`: add CLI options for dataset size and attack mix.
- [ ] `src/data/README.md`: document data schema and source assumptions.

## src/evaluation

- [x] `src/evaluation/compute_metrics.py`: compute initial ASR/utility from JSONL.
- [x] `src/evaluation/README.md`: define expected input/output format.
- [ ] `src/evaluation/compute_metrics.py`: replace heuristic scoring with stricter evaluation logic.

## data/processed

- [ ] `data/processed/eval_set.jsonl`: generate using `make prepare`.
- [ ] `data/processed/predictions_baseline.jsonl`: save baseline outputs.
- [ ] `data/processed/predictions_defensivetoken.jsonl`: save defended outputs.

## docs/raporlar

- [x] `docs/raporlar/TEZ_SPRINT_PLANI_V1.md`: 4-week thesis execution plan.
- [x] `docs/raporlar/DOSYA_BAZLI_GOREV_LISTESI_V1.md`: file-level tasks and ownership hints.
- [ ] `docs/raporlar/metrics_baseline.json`: first baseline metric artifact.
- [ ] `docs/raporlar/metrics_defensivetoken.json`: first defended metric artifact.
- [ ] `docs/raporlar/final_findings_v1.md`: thesis-ready summary draft.

## scripts and automation

- [ ] `Makefile`: add `run-baseline` and `run-defended` targets after inference pipeline is finalized.
- [ ] `scripts/`: add reproducible run scripts for local and server environments.

## Immediate order (next 3 actions)

1. Run `make prepare` and verify `data/processed/eval_set.jsonl`.
2. Create `data/processed/predictions.jsonl` from one model run.
3. Run `make eval` and save output as baseline artifact.
