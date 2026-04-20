import argparse
import json
import ssl
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DATA_DIR = REPO_ROOT / "src" / "official_stacks" / "meta_secalign" / "data"
ALPACA_FARM_URL = (
    "https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/"
    "resolve/main/davinci_003_outputs.json"
)


def read_url_text(url: str) -> str:
    context = ssl.create_default_context()
    with urllib.request.urlopen(url, context=context) as response:
        return response.read().decode("utf-8")


def download_alpaca_farm(target_dir: Path) -> Path:
    payload = json.loads(read_url_text(ALPACA_FARM_URL))
    out_path = target_dir / "davinci_003_outputs.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"dataset": "alpaca_farm", "rows": len(payload), "output": str(out_path)}, ensure_ascii=False, indent=2))
    return out_path


def download_alpaca_eval_instructions(target_dir: Path) -> Path:
    out_path = target_dir / "alpaca_eval_instructions.json"
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca_eval", trust_remote_code=True)
        split = ds.get("eval") or ds.get("train")
        instructions = [{"instruction": item["instruction"]} for item in split]
    except Exception:
        import alpaca_eval as _ae
        pkg_dir = Path(_ae.__file__).parent
        for candidate in [
            pkg_dir / "evaluators_configs" / "alpaca_eval_gpt4_turbo" / "alpaca_eval_gpt4_turbo.json",
            pkg_dir / "data" / "alpaca_eval.json",
        ]:
            if candidate.exists():
                data = json.loads(candidate.read_text(encoding="utf-8"))
                instructions = [{"instruction": d["instruction"]} for d in data]
                break
        else:
            raise RuntimeError("Could not load AlpacaEval2 instructions from datasets or alpaca_eval package.")
    out_path.write_text(json.dumps(instructions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"dataset": "alpaca_eval2", "rows": len(instructions), "output": str(out_path)}, ensure_ascii=False, indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", default=str(TARGET_DATA_DIR))
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    download_alpaca_farm(target_dir)
    download_alpaca_eval_instructions(target_dir)


if __name__ == "__main__":
    main()
