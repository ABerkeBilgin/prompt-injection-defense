import argparse
import json
from pathlib import Path

from benchmark_evaluators import (
    heuristic_attack_success,
    heuristic_utility_success,
    is_security_sample,
    is_utility_sample,
    paper_attack_success,
    paper_utility_success,
)
from judges import create_judge


SUPPORTED_BENCHMARKS = {"alpaca_farm", "sep", "cyberseceval2", "tasktracker", "manual"}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_mode(rows: list[dict]) -> str:
    attack_rows = 0
    benign_rows = 0
    for row in rows:
        if row.get("attack_type") == "none" or not row.get("injection"):
            benign_rows += 1
        else:
            attack_rows += 1
    if attack_rows and benign_rows:
        return "combined"
    if attack_rows:
        return "security"
    return "utility"


def evaluate_sample(sample: dict, output: str, evaluator: str, judge) -> tuple[bool, bool]:
    if evaluator == "paper":
        attack_success = paper_attack_success(sample, output, judge)
        utility_success = paper_utility_success(sample, output, judge)
        return attack_success, utility_success

    attack_success = heuristic_attack_success(sample, output)
    utility_success = heuristic_utility_success(sample, output)
    return attack_success, utility_success


def evaluate_group(
    dataset_rows: list[dict],
    prediction_rows: list[dict],
    benchmark: str,
    evaluator: str,
    judge,
) -> dict:
    pred_map = {row["id"]: row.get("output", "") for row in prediction_rows}
    total = len(dataset_rows)
    matched_predictions = 0
    attack_success = 0
    utility_success = 0
    security_samples = 0
    utility_samples = 0

    for sample in dataset_rows:
        output = pred_map.get(sample["id"])
        if output is None:
            continue

        matched_predictions += 1
        sample_attack_success, sample_utility_success = evaluate_sample(sample, output, evaluator=evaluator, judge=judge)

        if is_security_sample(sample):
            security_samples += 1
            if sample_attack_success:
                attack_success += 1

        if is_utility_sample(sample):
            utility_samples += 1
            if sample_utility_success:
                utility_success += 1

    asr = round(attack_success / security_samples, 4) if security_samples else None
    utility = round(utility_success / utility_samples, 4) if utility_samples else None

    notes = []
    if benchmark not in SUPPORTED_BENCHMARKS:
        notes.append("Unsupported benchmark.")
    if not security_samples:
        notes.append("No security-eligible samples in this slice.")
    if not utility_samples:
        notes.append("No utility-eligible samples in this slice.")

    return {
        "benchmark": benchmark,
        "mode": infer_mode(dataset_rows),
        "total": total,
        "matched_predictions": matched_predictions,
        "asr": asr,
        "utility": utility,
        "attack_success_count": attack_success,
        "utility_success_count": utility_success,
        "security_samples": security_samples,
        "utility_samples": utility_samples,
        "notes": " ".join(notes),
    }


def infer_mode_from_groups(group_results: list[dict]) -> str:
    has_security = any(group["security_samples"] for group in group_results)
    has_utility = any(group["utility_samples"] for group in group_results)
    if has_security and has_utility:
        return "combined"
    if has_security:
        return "security"
    return "utility"


def aggregate_results(group_results: list[dict], total_rows: int, matched_predictions: int) -> dict:
    attack_success = sum(group["attack_success_count"] for group in group_results)
    utility_success = sum(group["utility_success_count"] for group in group_results)
    security_samples = sum(group["security_samples"] for group in group_results)
    utility_samples = sum(group["utility_samples"] for group in group_results)

    return {
        "benchmark": "all" if len(group_results) > 1 else group_results[0]["benchmark"],
        "mode": infer_mode_from_groups(group_results),
        "total": total_rows,
        "matched_predictions": matched_predictions,
        "asr": round(attack_success / security_samples, 4) if security_samples else None,
        "utility": round(utility_success / utility_samples, 4) if utility_samples else None,
        "attack_success_count": attack_success,
        "utility_success_count": utility_success,
        "security_samples": security_samples,
        "utility_samples": utility_samples,
        "notes": "",
    }


def evaluate(dataset_rows: list[dict], prediction_rows: list[dict], evaluator: str, judge) -> dict:
    if not dataset_rows:
        return {
            "benchmark": "all",
            "mode": "utility",
            "total": 0,
            "matched_predictions": 0,
            "asr": None,
            "utility": None,
            "attack_success_count": 0,
            "utility_success_count": 0,
            "security_samples": 0,
            "utility_samples": 0,
            "notes": "Empty dataset",
            "by_benchmark": [],
        }

    grouped_rows: dict[str, list[dict]] = {}
    for row in dataset_rows:
        benchmark = row.get("benchmark", "unknown")
        if benchmark not in SUPPORTED_BENCHMARKS:
            raise ValueError(f"Unsupported benchmark for evaluator: {benchmark}")
        grouped_rows.setdefault(benchmark, []).append(row)

    group_results = [
        evaluate_group(rows, prediction_rows, benchmark, evaluator=evaluator, judge=judge)
        for benchmark, rows in sorted(grouped_rows.items())
    ]

    matched_predictions = sum(group["matched_predictions"] for group in group_results)
    result = aggregate_results(group_results, total_rows=len(dataset_rows), matched_predictions=matched_predictions)
    result["by_benchmark"] = group_results
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluator", choices=["paper", "heuristic"], default="paper")
    parser.add_argument("--judge-provider", default="openai_compatible")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-config", default=None)
    args = parser.parse_args()

    dataset_rows = read_jsonl(Path(args.dataset))
    prediction_rows = read_jsonl(Path(args.predictions))
    judge = create_judge(
        provider=args.judge_provider,
        config_path=args.judge_config,
        model_override=args.judge_model,
    )
    result = evaluate(dataset_rows, prediction_rows, evaluator=args.evaluator, judge=judge)
    result["evaluator_mode"] = args.evaluator
    result["judge_provider"] = judge.mode
    result["judge_enabled"] = judge.available()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
