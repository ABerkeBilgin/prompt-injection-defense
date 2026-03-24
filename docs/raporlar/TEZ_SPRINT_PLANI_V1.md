# Thesis Sprint Plan (4 Weeks)

## Goal

Convert the current DefensiveToken proof-of-concept into a reproducible thesis experiment pipeline.

## Final output of this 4-week cycle

- One reproducible experiment run for at least one model.
- One baseline vs DefensiveTokens comparison table.
- One short report with ASR, utility, latency, and token-cost discussion.

## Week 1 - Pipeline foundation

### Objectives

- Create dataset preparation flow.
- Standardize experiment I/O formats.
- Define exact metrics and report schema.

### Deliverables

- `src/data/build_dataset.py`
- `data/processed/eval_set.jsonl`
- `src/evaluation/compute_metrics.py`
- `docs/raporlar/metrics_latest.json`

### Exit criteria

- `make prepare` produces dataset without manual edits.
- `make eval` computes metrics from a prediction file.

## Week 2 - Baseline experiments

### Objectives

- Run model without defensive tokens.
- Capture predictions in stable JSONL format.
- Measure first ASR/utility results.

### Deliverables

- `data/processed/predictions_baseline.jsonl`
- `docs/raporlar/metrics_baseline.json`
- `docs/raporlar/week2_notes.md`

### Exit criteria

- At least one full baseline run is reproducible.
- Raw outputs are stored and traceable.

## Week 3 - DefensiveTokens + ablation

### Objectives

- Run model with DefensiveTokens.
- Compare token counts (1/3/5) or available variants.
- Track utility-security tradeoff.

### Deliverables

- `data/processed/predictions_defensivetoken.jsonl`
- `docs/raporlar/metrics_defensivetoken.json`
- `docs/raporlar/ablation_results.csv`

### Exit criteria

- Side-by-side baseline vs defense metrics are available.
- At least one ablation dimension is completed.

## Week 4 - Thesis artifacts and stabilization

### Objectives

- Freeze experiment protocol.
- Prepare thesis-ready tables and figures.
- Create minimal demo aligned with final narrative.

### Deliverables

- `docs/raporlar/final_results_table.csv`
- `docs/raporlar/final_findings_v1.md`
- `app/` minimal scenario demo

### Exit criteria

- Thesis chapter can reference reproducible artifacts.
- Figures and claims map directly to saved outputs.

## Risks and mitigations

- Model access limits: pre-check Hugging Face access and cache strategy.
- Metric ambiguity: freeze metric definitions before week 2.
- Scope creep: postpone non-essential UI work until week 4.

## Weekly cadence

- Monday: lock goals and tasks.
- Wednesday: run midpoint checkpoint.
- Friday: publish week report + artifacts.
