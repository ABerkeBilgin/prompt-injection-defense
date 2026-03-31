import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
DEFENSIVE_TOKEN_ROOT = THIRD_PARTY_ROOT / "DefensiveToken"
META_SECALIGN_ROOT = THIRD_PARTY_ROOT / "Meta_SecAlign"
OPENAI_CONFIG_PATH = META_SECALIGN_ROOT / "data" / "openai_configs.yaml"
GEMINI_CONFIG_PATH = META_SECALIGN_ROOT / "data" / "gemini_configs.yaml"
OPEN_WEIGHT_MODELS = {
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
}
SUMMARY_FILENAME = "summary.tsv"


def ensure_official_stack(clone_missing: bool) -> None:
    command = [sys.executable, "scripts/bootstrap_official_stack.py"]
    if clone_missing:
        command.append("--clone-missing")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def normalize_model_id(model_name: str) -> str:
    return model_name.replace("/", "__")


def model_output_path(model_name: str, mode: str) -> str:
    if mode == "baseline":
        return model_name
    if model_name.endswith("-5DefensiveTokens"):
        return model_name
    return str(DEFENSIVE_TOKEN_ROOT / f"{model_name}-5DefensiveTokens")


def ensure_official_roots() -> None:
    for path, label, required in [
        (DEFENSIVE_TOKEN_ROOT, "DefensiveToken", ["README.md", "setup.py"]),
        (META_SECALIGN_ROOT, "Meta_SecAlign", ["README.md", "run_tests.py", "test.py", "utils.py"]),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} repository not found at {path}. Initialize submodules first.")
        for name in required:
            if not (path / name).exists():
                raise FileNotFoundError(f"{label} repository at {path} is missing required file: {name}")


def validate_judge_mode(judge: str, suite: str) -> None:
    if judge == "local_dev":
        return
    if not OPENAI_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"official_api mode requires {OPENAI_CONFIG_PATH}. "
            "Create the Meta_SecAlign OpenAI config before running official evaluations."
        )


def ensure_defensive_model(model_name: str, mode: str, dry_run: bool) -> str:
    output_path = model_output_path(model_name, mode)
    if mode == "baseline":
        return output_path
    if model_name not in OPEN_WEIGHT_MODELS:
        raise ValueError(f"Defense mode is only supported for official open-weight models, got: {model_name}")
    expected_path = Path(output_path)
    if expected_path.exists():
        return output_path
    if dry_run:
        return output_path
    subprocess.run([sys.executable, "setup.py", model_name], cwd=DEFENSIVE_TOKEN_ROOT, check=True)
    return output_path


def build_commands(model_name_or_path: str, suite: str, mode: str) -> list[list[str]]:
    commands: list[list[str]] = []
    if suite in {"instruction", "all"}:
        commands.extend(
            [
                [
                    sys.executable,
                    "test.py",
                    "--attack",
                    "none",
                    "ignore",
                    "completion",
                    "completion_ignore",
                    "--defense",
                    "none",
                    "--test_data",
                    "data/davinci_003_outputs.json",
                    "--lora_alpha",
                    "8.0",
                    "-m",
                    model_name_or_path,
                ],
                [
                    sys.executable,
                    "test.py",
                    "--attack",
                    "none",
                    "ignore",
                    "ignore_before",
                    "--defense",
                    "none",
                    "--test_data",
                    "data/SEP_dataset_test.json",
                    "--lora_alpha",
                    "8.0",
                    "-m",
                    model_name_or_path,
                ],
                [
                    sys.executable,
                    "test.py",
                    "--attack",
                    "straightforward",
                    "--defense",
                    "none",
                    "--test_data",
                    "data/CySE_prompt_injections.json",
                    "--lora_alpha",
                    "8.0",
                    "-m",
                    model_name_or_path,
                ],
                [
                    sys.executable,
                    "test.py",
                    "--attack",
                    "straightforward",
                    "--defense",
                    "none",
                    "--test_data",
                    "data/TaskTracker_dataset_test.json",
                    "--lora_alpha",
                    "8.0",
                    "-m",
                    model_name_or_path,
                ],
            ]
        )
    if suite in {"utility", "all"}:
        commands.extend(
            [
                [
                    sys.executable,
                    "test.py",
                    "--attack",
                    "none",
                    "--defense",
                    "none",
                    "--test_data",
                    "data/davinci_003_outputs.json",
                    "--lora_alpha",
                    "8.0",
                    "-m",
                    model_name_or_path,
                ],
                [
                    sys.executable,
                    "test.py",
                    "--attack",
                    "none",
                    "--defense",
                    "none",
                    "--test_data",
                    "data/SEP_dataset_test.json",
                    "--lora_alpha",
                    "8.0",
                    "-m",
                    model_name_or_path,
                ],
                [
                    sys.executable,
                    "test_lm_eval.py",
                    "--lora_alpha",
                    "8.0",
                    "-m",
                    model_name_or_path,
                ],
            ]
        )
    if suite in {"agentic", "all"}:
        commands.extend(
            [
                [
                    sys.executable,
                    "test_injecagent.py",
                    "--defense",
                    "sandwich",
                    "--lora_alpha",
                    "8.0",
                    "-m",
                    model_name_or_path,
                ],
                [
                    sys.executable,
                    "test_agentdojo.py",
                    "--lora_alpha",
                    "8.0",
                    "-m",
                    model_name_or_path,
                ],
            ]
        )
    return commands


def read_summary_rows(model_name_or_path: str) -> list[dict]:
    log_dir = Path(model_name_or_path if Path(model_name_or_path).exists() else f"{model_name_or_path}-log")
    summary_path = log_dir / SUMMARY_FILENAME
    if not summary_path.exists():
        return []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def parse_percent(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = str(value).strip().replace("%", "")
    if not cleaned:
        return None
    try:
        return round(float(cleaned) / 100.0, 4)
    except ValueError:
        return None


def filter_rows(rows: list[dict], test_data_suffix: str) -> list[dict]:
    return [row for row in rows if str(row.get("test_data", "")).endswith(test_data_suffix)]


def row_metric(row: dict) -> float | None:
    return parse_percent(row.get("ASR/Utility"))


def max_metric(rows: list[dict], attacks: set[str]) -> float | None:
    metrics = [row_metric(row) for row in rows if row.get("attack") in attacks]
    metrics = [metric for metric in metrics if metric is not None]
    return max(metrics) if metrics else None


def none_metric(rows: list[dict]) -> float | None:
    for row in rows:
        if row.get("attack") == "none":
            return row_metric(row)
    return None


def collect_official_metrics(model_name_or_path: str, suite: str) -> dict:
    rows = read_summary_rows(model_name_or_path)
    metrics: dict[str, float | None] = {}

    alpaca_rows = filter_rows(rows, "davinci_003_outputs.json")
    sep_rows = filter_rows(rows, "SEP_dataset_test.json")
    cyse_rows = filter_rows(rows, "CySE_prompt_injections.json")
    tasktracker_rows = filter_rows(rows, "TaskTracker_dataset_test.json")

    if suite in {"instruction", "all", "utility"}:
        metrics["alpaca_asr"] = max_metric(alpaca_rows, {"ignore", "completion", "completion_ignore"})
        metrics["alpaca_win_rate"] = none_metric(alpaca_rows)
        metrics["sep_asr"] = max_metric(sep_rows, {"ignore", "ignore_before"})
        metrics["sep_win_rate"] = none_metric(sep_rows)
        metrics["cyse_asr"] = max_metric(cyse_rows, {"straightforward"})
        metrics["tasktracker_asr"] = max_metric(tasktracker_rows, {"straightforward"})

    if suite in {"utility", "all"}:
        for key in [
            "mmlu",
            "mmlu_pro",
            "bbh",
            "ifeval",
            "gpqa_diamond",
            "alpacaeval2",
            "agentdojo_utility",
        ]:
            metrics.setdefault(key, None)
        metrics["alpacaeval2"] = metrics.get("alpaca_win_rate")

    if suite in {"agentic", "all"}:
        metrics.setdefault("injecagent_asr", None)
        metrics.setdefault("agentdojo_asr", None)
        metrics.setdefault("agentdojo_utility", None)

    return metrics


def run_commands(commands: list[list[str]], cwd: Path, dry_run: bool) -> None:
    for command in commands:
        print(" ".join(command))
        if dry_run:
            continue
        subprocess.run(command, cwd=cwd, check=True)


def write_report(args, model_name_or_path: str, commands: list[list[str]], metrics: dict) -> Path:
    normalized_model = normalize_model_id(args.model)
    out_path = REPO_ROOT / "docs" / "raporlar" / "official" / normalized_model / args.suite / f"{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "resolved_model_path": model_name_or_path,
        "suite": args.suite,
        "mode": args.mode,
        "judge": args.judge,
        "development_only": args.judge == "local_dev",
        "commands": [" ".join(command) for command in commands],
        "metrics": metrics,
        "artifacts": {
            "meta_secalign_root": str(META_SECALIGN_ROOT),
            "defensive_token_root": str(DEFENSIVE_TOKEN_ROOT),
            "summary_tsv": str(Path(model_name_or_path if Path(model_name_or_path).exists() else f"{model_name_or_path}-log") / SUMMARY_FILENAME),
        },
        "notes": [
            "Official paper path is DefensiveToken + Meta_SecAlign.",
            "Development-only local judge mode must not be used for final thesis tables.",
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", choices=["instruction", "utility", "agentic", "all"], default="all")
    parser.add_argument("--mode", choices=["baseline", "defense"], default="baseline")
    parser.add_argument("--judge", choices=["official_api", "local_dev"], default="official_api")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clone-missing", action="store_true")
    args = parser.parse_args()

    ensure_official_stack(clone_missing=args.clone_missing)
    ensure_official_roots()
    validate_judge_mode(args.judge, args.suite)
    model_name_or_path = ensure_defensive_model(args.model, args.mode, dry_run=args.dry_run)
    commands = build_commands(model_name_or_path, args.suite, args.mode)
    run_commands(commands, cwd=META_SECALIGN_ROOT, dry_run=args.dry_run)
    metrics = collect_official_metrics(model_name_or_path, args.suite)
    report_path = write_report(args, model_name_or_path, commands, metrics)
    print(json.dumps({"report": str(report_path), "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
