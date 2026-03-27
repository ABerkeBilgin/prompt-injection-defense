# Evaluation Module

This folder contains metric scripts for thesis experiments.

## Input format

`compute_metrics.py` expects:

1. Dataset JSONL (`--dataset`) with fields:
- `id`
- `benchmark`
- `instruction`
- `untrusted_data`
- `injection`
- `attack_type`
- `expected_task`
- `judge_hint` (optional)
- `metadata` (optional object)

2. Prediction JSONL (`--predictions`) with fields:
- `id`
- `output`

## Benchmark-aware metric behavior

- `asr`: ratio of security-eligible samples whose output appears to follow the attack target.
- `utility`: ratio of utility-eligible samples whose output appears to satisfy the expected task.
- `by_benchmark`: per-benchmark metric breakdown.

Utility-eligible benchmarks in this version:
- `manual`
- `alpaca_farm`
- `sep`

Security-eligible rows are those where:
- `attack_type != "none"`
- `injection` is not empty

## Notes

- This implementation is benchmark-aware but still heuristic.
- `metadata.reference_output` improves utility estimation when available.
- `metadata.attack_targets` improves ASR estimation when available.
- `judge_hint` is preserved for future LLM-as-a-judge integration, especially for `cyberseceval2`.
