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

`paper` mode requires a configured judge provider.

Supported CLI options:
- `--judge-provider openai_compatible`
- `--judge-provider local_transformers`
- `--judge-model <model_name>`
- `--judge-config <json_path>`

Judge configuration is required in official mode. The evaluator does not silently fall back.

Expected config keys in `--judge-config` JSON:
- `model`
- `api_key` (`openai_compatible` only)
- `base_url` (`openai_compatible` only)
- `timeout` (optional)
- `max_new_tokens` (`local_transformers` only, optional)
- `trust_remote_code` (`local_transformers` only, optional)

Environment variable fallback is also supported:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_JUDGE_MODEL`
- `OPENAI_JUDGE_TIMEOUT`
- `LOCAL_JUDGE_MAX_NEW_TOKENS`
- `LOCAL_JUDGE_TRUST_REMOTE_CODE`

Free temporary thesis path:
- `local_transformers` can run a local judge model such as `meta-llama/Meta-Llama-3-8B-Instruct`.
- This is a cost-saving temporary setup.
- Before final thesis runs, replace it with the intended API-based judge and record that switch in methodology notes.

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
