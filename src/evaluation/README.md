# Evaluation Module

This folder contains the metric pipeline for thesis experiments.

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

Optional prediction metadata such as `benchmark`, `attack_type`, `model_name`, and `defense_mode` is preserved but not required.

## Evaluator modes

- `paper`: benchmark-aware evaluator that prefers benchmark-specific logic and optional judge integration.
- `heuristic`: fallback evaluator using local heuristics only.

The default mode is `paper`.

## Judge integration

`paper` mode can use an OpenAI-compatible judge endpoint.

Supported CLI options:
- `--judge-provider openai_compatible`
- `--judge-model <model_name>`
- `--judge-config <json_path>`

If judge configuration is missing, the evaluator falls back to local benchmark-aware heuristics.

Expected config keys in `--judge-config` JSON:
- `api_key`
- `base_url`
- `model`
- `timeout` (optional)

Environment variable fallback is also supported:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_JUDGE_MODEL`
- `OPENAI_JUDGE_TIMEOUT`

## Benchmark-aware behavior

- `alpaca_farm`: paper mode uses an Alpaca-specific ASR rule and reference-based utility fallback.
- `sep`: paper mode prefers judge/reference-guided utility scoring.
- `cyberseceval2`: paper mode prefers `judge_question` driven security scoring.
- `tasktracker`: paper mode evaluates task drift with reference/judge-aware logic.
- `manual`: remains a small local sanity dataset.

## Output

The output JSON keeps:
- `benchmark`
- `mode`
- `asr`
- `utility`
- `by_benchmark`
- `attack_success_count`
- `utility_success_count`
- `security_samples`
- `utility_samples`

It also adds:
- `evaluator_mode`
- `judge_provider`
- `judge_enabled`
