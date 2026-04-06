import argparse
import json
from pathlib import Path

from src.official_stacks.defensivetoken import prepare_defended_model, resolve_defended_model_path
from src.official_stacks.meta_secalign.config import JUDGE_MODEL, TARGET_MODEL
from src.official_stacks.meta_secalign.paths import DATA_DIR
from src.official_stacks.meta_secalign.qwen_alpaca import run_qwen_alpaca_eval


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENAI_CONFIG_PATH = DATA_DIR / "openai_configs.yaml"


def resolve_model_path(mode: str, dry_run: bool) -> str:
    if mode == "baseline":
        return TARGET_MODEL
    defended_path = resolve_defended_model_path(TARGET_MODEL)
    if defended_path.exists() or dry_run:
        return str(defended_path)
    return str(prepare_defended_model(TARGET_MODEL))


def write_report(mode: str, model_name_or_path: str, metrics: dict) -> Path:
    report_path = REPO_ROOT / "docs" / "raporlar" / "qwen_alpaca" / f"{mode}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": TARGET_MODEL,
        "resolved_model_path": model_name_or_path,
        "mode": mode,
        "judge_model": JUDGE_MODEL,
        "metrics": {
            "win_rate": metrics["win_rate"],
            "asr": metrics["asr"],
            "gcg_asr": metrics["gcg_asr"],
        },
        "artifacts": metrics["artifacts"],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "defense"], required=True)
    parser.add_argument("--dry-run", action="store_true")
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

    metrics = run_qwen_alpaca_eval(args.mode, model_name_or_path, str(OPENAI_CONFIG_PATH))
    report_path = write_report(args.mode, model_name_or_path, metrics)
    print(json.dumps({"report": str(report_path), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()