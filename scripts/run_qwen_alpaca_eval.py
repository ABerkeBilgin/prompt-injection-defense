import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.official_stacks.defensivetoken import OUTPUT_SUFFIX as DEFENSIVE_SUFFIX, prepare_defended_model
from src.official_stacks.utilitytokens.core import OUTPUT_SUFFIX as UTILITY_SUFFIX, prepare_utility_model
from src.official_stacks.meta_secalign.config import JUDGE_MODEL, TARGET_MODEL
from src.official_stacks.meta_secalign.paths import DATA_DIR

OPENAI_CONFIG_PATH = DATA_DIR / "openai_configs.yaml"
DEFENDED_MODEL_PATH = REPO_ROOT / "src" / "official_stacks" / "defensivetoken" / f"{TARGET_MODEL}{DEFENSIVE_SUFFIX}"
UTILITY_MODEL_PATH  = DEFENDED_MODEL_PATH.parent / f"{TARGET_MODEL}{DEFENSIVE_SUFFIX}{UTILITY_SUFFIX}"


def resolve_model_path(mode: str, dry_run: bool) -> str:
    if mode == "baseline":
        return TARGET_MODEL
    if mode == "defense":
        if DEFENDED_MODEL_PATH.exists() or dry_run:
            return str(DEFENDED_MODEL_PATH)
        return str(prepare_defended_model(TARGET_MODEL))
    # mode == "utility"
    if UTILITY_MODEL_PATH.exists() or dry_run:
        return str(UTILITY_MODEL_PATH)
    # Önce defense modelinin hazır olduğundan emin ol
    defended_path = DEFENDED_MODEL_PATH if DEFENDED_MODEL_PATH.exists() else Path(prepare_defended_model(TARGET_MODEL))
    return str(prepare_utility_model(str(defended_path)))


def write_report(mode: str, model_name_or_path: str, metrics: dict) -> Path:
    report_path = REPO_ROOT / "docs" / "raporlar" / "qwen_alpaca" / f"{mode}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metric_payload = {
        "win_rate": metrics["win_rate"],
        "asr": metrics["asr"],
    }
    if "gcg_asr" in metrics:
        metric_payload["gcg_asr"] = metrics["gcg_asr"]
    payload = {
        "model": TARGET_MODEL,
        "resolved_model_path": model_name_or_path,
        "mode": mode,
        "judge_model": JUDGE_MODEL,
        "metrics": metric_payload,
        "artifacts": metrics["artifacts"],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "defense", "utility"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-gcg", action="store_true")
    args = parser.parse_args()

    if not OPENAI_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing OpenAI config: {OPENAI_CONFIG_PATH}")

    model_name_or_path = resolve_model_path(args.mode, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps({
            "model": TARGET_MODEL,
            "mode": args.mode,
            "resolved_model_path": model_name_or_path,
            "judge_model": JUDGE_MODEL,
            "openai_config_path": str(OPENAI_CONFIG_PATH),
        }, ensure_ascii=False, indent=2))
        return

    from src.official_stacks.meta_secalign.qwen_alpaca import run_qwen_alpaca_eval

    metrics = run_qwen_alpaca_eval(args.mode, model_name_or_path, str(OPENAI_CONFIG_PATH), include_gcg=not args.skip_gcg)
    report_path = write_report(args.mode, model_name_or_path, metrics)
    print(json.dumps({"report": str(report_path), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
