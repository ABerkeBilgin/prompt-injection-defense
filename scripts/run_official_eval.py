import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from src.official_stacks.defensivetoken import SUPPORTED_MODELS as DEFENSIVE_MODELS
from src.official_stacks.defensivetoken import prepare_defended_model, resolve_defended_model_path
from src.official_stacks.defensivetoken.core import STACK_ROOT as DEFENSIVE_TOKEN_ROOT
from src.official_stacks.meta_secalign.paths import DATA_DIR as META_DATA_DIR
from src.official_stacks.meta_secalign.paths import STACK_ROOT as META_SECALIGN_ROOT


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENAI_CONFIG_PATH = META_DATA_DIR / 'openai_configs.yaml'
GEMINI_CONFIG_PATH = META_DATA_DIR / 'gemini_configs.yaml'
OPEN_WEIGHT_MODELS = set(DEFENSIVE_MODELS)
SUMMARY_FILENAME = 'summary.tsv'

REQUIRED_DATA_FILES = {
    'instruction': [
        'davinci_003_outputs.json',
        'SEP_dataset_test.json',
        'CySE_prompt_injections.json',
        'TaskTracker_dataset_test.json',
    ],
    'utility': [
        'davinci_003_outputs.json',
        'SEP_dataset_test.json',
        'SEP_dataset_test_Meta-Llama-3-8B-Instruct.json',
    ],
    'agentic': [
        'tools.json',
        'attacker_simulated_responses.json',
        'test_cases_dh_base.json',
        'test_cases_ds_base.json',
        'test_cases_dh_enhanced.json',
        'test_cases_ds_enhanced.json',
    ],
}

def normalize_model_id(model_name: str) -> str:
    return model_name.replace('/', '__')


def is_supported_defensive_model(model_name: str) -> bool:
    return model_name in OPEN_WEIGHT_MODELS


def model_output_path(model_name: str, mode: str) -> str:
    if mode == 'baseline':
        return model_name
    if model_name.endswith('-5DefensiveTokens'):
        return model_name
    return str(resolve_defended_model_path(model_name))


def ensure_required_data_files(suite: str) -> None:
    if suite == 'all':
        required = set(REQUIRED_DATA_FILES['instruction'] + REQUIRED_DATA_FILES['utility'] + REQUIRED_DATA_FILES['agentic'])
    else:
        required = set(REQUIRED_DATA_FILES.get(suite, []))
    missing = [name for name in sorted(required) if not (META_DATA_DIR / name).exists()]
    if missing:
        missing_text = ', '.join(missing)
        raise FileNotFoundError(
            'Missing official Meta_SecAlign data files: '
            + missing_text
            + '. Populate src/official_stacks/meta_secalign/data first. '
            + 'Recommended command: python scripts/bootstrap_official_data.py'
        )

def validate_judge_mode(judge: str) -> None:
    if judge == 'local_dev':
        return
    if not OPENAI_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f'official_api mode requires {OPENAI_CONFIG_PATH}. '
            'Create the Meta_SecAlign OpenAI config before running official evaluations.'
        )


def ensure_defensive_model(model_name: str, mode: str, dry_run: bool) -> str:
    output_path = model_output_path(model_name, mode)
    if mode == 'baseline':
        return output_path
    if not is_supported_defensive_model(model_name):
        supported = ', '.join(sorted(OPEN_WEIGHT_MODELS))
        raise ValueError(
            f'Defense mode only supports the DefensiveToken paper models. Unsupported model: {model_name}. '
            f'Supported models: {supported}'
        )
    expected_path = Path(output_path)
    if expected_path.exists() or dry_run:
        return output_path
    return str(prepare_defended_model(model_name))


def build_commands(model_name_or_path: str, suite: str, judge_mode: str, judge_model: str | None) -> list[list[str]]:
    commands: list[list[str]] = []
    if suite in {'instruction', 'all'}:
        commands.extend(
            [
                [sys.executable, '-m', 'src.official_stacks.meta_secalign.test', '--attack', 'none', 'ignore', 'completion', 'completion_ignore', '--defense', 'none', '--test_data', 'data/davinci_003_outputs.json', '--lora_alpha', '8.0', '-m', model_name_or_path],
                [sys.executable, '-m', 'src.official_stacks.meta_secalign.test', '--attack', 'none', 'ignore', 'ignore_before', '--defense', 'none', '--test_data', 'data/SEP_dataset_test.json', '--lora_alpha', '8.0', '-m', model_name_or_path],
                [sys.executable, '-m', 'src.official_stacks.meta_secalign.test', '--attack', 'straightforward', '--defense', 'none', '--test_data', 'data/CySE_prompt_injections.json', '--lora_alpha', '8.0', '-m', model_name_or_path],
                [sys.executable, '-m', 'src.official_stacks.meta_secalign.test', '--attack', 'straightforward', '--defense', 'none', '--test_data', 'data/TaskTracker_dataset_test.json', '--lora_alpha', '8.0', '-m', model_name_or_path],
            ]
        )
    if suite in {'utility', 'all'}:
        commands.extend(
            [
                [sys.executable, '-m', 'src.official_stacks.meta_secalign.test', '--attack', 'none', '--defense', 'none', '--test_data', 'data/davinci_003_outputs.json', '--lora_alpha', '8.0', '-m', model_name_or_path],
                [sys.executable, '-m', 'src.official_stacks.meta_secalign.test', '--attack', 'none', '--defense', 'none', '--test_data', 'data/SEP_dataset_test.json', '--lora_alpha', '8.0', '-m', model_name_or_path],
                [sys.executable, '-m', 'src.official_stacks.meta_secalign.test_lm_eval', '--lora_alpha', '8.0', '-m', model_name_or_path],
            ]
        )
    if suite in {'agentic', 'all'}:
        commands.extend(
            [
                [sys.executable, '-m', 'src.official_stacks.meta_secalign.test_injecagent', '--defense', 'sandwich', '--lora_alpha', '8.0', '-m', model_name_or_path],
                [sys.executable, '-m', 'src.official_stacks.meta_secalign.test_agentdojo', '-a', 'none', '-d', 'repeat_user_prompt', '--lora_alpha', '8.0', '-m', model_name_or_path],
                [sys.executable, '-m', 'src.official_stacks.meta_secalign.test_agentdojo', '-a', 'important_instructions', '-d', 'repeat_user_prompt', '--lora_alpha', '8.0', '-m', model_name_or_path],
            ]
        )
    if judge_mode == 'official_api' and judge_model:
        for command in commands:
            if 'src.official_stacks.meta_secalign.test' in command or 'src.official_stacks.meta_secalign.test_injecagent' in command or 'src.official_stacks.meta_secalign.test_agentdojo' in command:
                command.extend(['--judge_model', judge_model])
                if command[2] == 'src.official_stacks.meta_secalign.test':
                    command.extend(['--alpacaeval_judge_model', judge_model])
    return commands


def read_summary_rows(model_name_or_path: str) -> list[dict]:
    log_dir = Path(model_name_or_path if Path(model_name_or_path).exists() else f'{model_name_or_path}-log')
    summary_path = log_dir / SUMMARY_FILENAME
    if not summary_path.exists():
        return []
    with summary_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        return [dict(row) for row in reader]


def parse_percent(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = str(value).strip().replace('%', '')
    if not cleaned:
        return None
    try:
        return round(float(cleaned) / 100.0, 4)
    except ValueError:
        return None


def filter_rows(rows: list[dict], test_data_suffix: str) -> list[dict]:
    return [row for row in rows if str(row.get('test_data', '')).endswith(test_data_suffix)]


def row_metric(row: dict) -> float | None:
    return parse_percent(row.get('ASR/Utility'))


def max_metric(rows: list[dict], attacks: set[str]) -> float | None:
    metrics = [row_metric(row) for row in rows if row.get('attack') in attacks]
    metrics = [metric for metric in metrics if metric is not None]
    return max(metrics) if metrics else None


def none_metric(rows: list[dict]) -> float | None:
    for row in rows:
        if row.get('attack') == 'none':
            return row_metric(row)
    return None


def collect_official_metrics(model_name_or_path: str, suite: str) -> dict:
    rows = read_summary_rows(model_name_or_path)
    metrics: dict[str, float | None] = {}

    alpaca_rows = filter_rows(rows, 'davinci_003_outputs.json')
    sep_rows = filter_rows(rows, 'SEP_dataset_test.json')
    cyse_rows = filter_rows(rows, 'CySE_prompt_injections.json')
    tasktracker_rows = filter_rows(rows, 'TaskTracker_dataset_test.json')

    if suite in {'instruction', 'all', 'utility'}:
        metrics['alpaca_asr'] = max_metric(alpaca_rows, {'ignore', 'completion', 'completion_ignore'})
        metrics['alpaca_win_rate'] = none_metric(alpaca_rows)
        metrics['sep_asr'] = max_metric(sep_rows, {'ignore', 'ignore_before'})
        metrics['sep_win_rate'] = none_metric(sep_rows)
        metrics['cyse_asr'] = max_metric(cyse_rows, {'straightforward'})
        metrics['tasktracker_asr'] = max_metric(tasktracker_rows, {'straightforward'})

    if suite in {'utility', 'all'}:
        for key in ['mmlu', 'mmlu_pro', 'bbh', 'ifeval', 'gpqa_diamond', 'alpacaeval2', 'agentdojo_utility']:
            metrics.setdefault(key, None)
        metrics['alpacaeval2'] = metrics.get('alpaca_win_rate')

    if suite in {'agentic', 'all'}:
        injecagent_rows = [row for row in rows if str(row.get('attack', '')).startswith('InjecAgent')]
        agentdojo_rows = filter_rows(rows, 'AgentDojo')
        metrics['injecagent_asr'] = max((row_metric(row) for row in injecagent_rows if row_metric(row) is not None), default=None)
        metrics['agentdojo_asr'] = max_metric(agentdojo_rows, {'AgentDojo-important_instructions'})
        metrics['agentdojo_utility'] = max_metric(agentdojo_rows, {'AgentDojo-none'})

    return metrics


def run_commands(commands: list[list[str]], dry_run: bool) -> None:
    for command in commands:
        print(' '.join(command))
        if dry_run:
            continue
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def write_report(args, model_name_or_path: str, commands: list[list[str]], metrics: dict) -> Path:
    normalized_model = normalize_model_id(args.model)
    out_path = REPO_ROOT / 'docs' / 'raporlar' / 'official' / normalized_model / args.suite / f'{args.mode}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(model_name_or_path if Path(model_name_or_path).exists() else f'{model_name_or_path}-log') / SUMMARY_FILENAME
    payload = {
        'model': args.model,
        'resolved_model_path': model_name_or_path,
        'suite': args.suite,
        'mode': args.mode,
        'judge': args.judge,
        'judge_model': args.judge_model,
        'development_only': args.judge == 'local_dev',
        'supported_defensive_model': is_supported_defensive_model(args.model),
        'commands': [' '.join(command) for command in commands],
        'metrics': metrics,
        'artifacts': {
            'meta_secalign_root': str(META_SECALIGN_ROOT),
            'defensive_token_root': str(DEFENSIVE_TOKEN_ROOT),
            'summary_tsv': str(summary_path),
            'openai_config_path': str(OPENAI_CONFIG_PATH),
            'gemini_config_path': str(GEMINI_CONFIG_PATH),
        },
        'notes': [
            'Official paper path is the first-party vendorized DefensiveToken + Meta_SecAlign stack in this repository.',
            'Development-only local judge mode must not be used for final thesis tables.',
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--suite', choices=['instruction', 'utility', 'agentic', 'all'], default='all')
    parser.add_argument('--mode', choices=['baseline', 'defense'], default='baseline')
    parser.add_argument('--defense-only', action='store_true', help='Alias for --mode defense')
    parser.add_argument('--judge', choices=['official_api', 'local_dev'], default='official_api')
    parser.add_argument('--judge-model', default='o4-mini-2025-04-16', help='Judge/evaluator model name used in official_api mode')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.defense_only:
        args.mode = 'defense'

    validate_judge_mode(args.judge)
    ensure_required_data_files(args.suite)
    model_name_or_path = ensure_defensive_model(args.model, args.mode, dry_run=args.dry_run)
    commands = build_commands(model_name_or_path, args.suite, args.judge, args.judge_model)
    run_commands(commands, dry_run=args.dry_run)
    metrics = collect_official_metrics(model_name_or_path, args.suite)
    report_path = write_report(args, model_name_or_path, commands, metrics)
    print(json.dumps({'report': str(report_path), 'metrics': metrics}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
