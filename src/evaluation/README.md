# Evaluation Module

This folder contains metric scripts for thesis experiments.

## Input format

`compute_metrics.py` expects two JSONL files:

1. Dataset file (`--dataset`) with fields:
- `id`
- `instruction`
- `untrusted_data`
- `injection`
- `attack_type`
- `expected_task`

2. Prediction file (`--predictions`) with fields:
- `id`
- `output`

## Example prediction row

```json
{"id": "sample-001", "output": "The Witcher is a dark fantasy series..."}
```

## Current metric behavior

- `asr`: ratio of outputs that look like they followed injection.
- `utility`: ratio of outputs that look like the expected task.

The current implementation is a lightweight baseline. Replace heuristic checks with stricter judges before final thesis reporting.
