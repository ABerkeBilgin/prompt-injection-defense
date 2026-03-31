# Evaluation Module

This folder contains the official metric pipeline for thesis experiments.

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

## Evaluator mode

- `paper`: official benchmark-aware evaluator for the first thesis wave.

## Judge integration

`paper` mode can use an OpenAI-compatible judge endpoint.

Supported CLI options:
- `--judge-provider openai_compatible`
- `--judge-model <model_name>`
- `--judge-config <json_path>`

Judge configuration is required in official mode. The evaluator does not silently fall back.

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

- `alpaca_farm`: official mode reports `ASR` and `win_rate`.
- `sep`: official mode reports judge-based security and utility.
- `cyberseceval2`: official mode reports judge-question-driven security.

Unsupported in the first official wave:
- `tasktracker`
- `manual`

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

For `alpaca_farm`, the output also includes:
- `win_rate`
