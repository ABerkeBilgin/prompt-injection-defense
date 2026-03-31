import argparse
import json
import ssl
import urllib.request
from pathlib import Path


ALPACA_URL = (
    "https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/"
    "resolve/main/davinci_003_outputs.json"
)
SEP_URL = (
    "https://raw.githubusercontent.com/egozverev/Should-It-Be-Executed-Or-Processed/"
    "refs/heads/main/datasets/SEP_dataset.json"
)
CYSE_URL = (
    "https://raw.githubusercontent.com/meta-llama/PurpleLlama/refs/heads/main/"
    "CybersecurityBenchmarks/datasets/prompt_injection/prompt_injection.json"
)


def download_json(url: str) -> list[dict]:
    context = ssl.create_default_context()
    with urllib.request.urlopen(url, context=context) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("data", "examples", "items", "records", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [data]
    raise ValueError(f"Unsupported JSON payload type from {url}")


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def normalize_sep_row(row: dict) -> dict | None:
    instruction = str(row.get("system_prompt_clean", "")).strip()
    input_text = str(row.get("prompt_clean", "")).strip()
    prompted_text = str(row.get("prompt_instructed", "")).strip()
    witness = str(row.get("witness", "")).strip()

    if not instruction:
        return None

    injection = prompted_text
    if input_text and prompted_text.startswith(input_text):
        injection = prompted_text[len(input_text) :].strip()

    return {
        "instruction": instruction,
        "input": input_text,
        "injection": injection,
        "witness": witness,
        "dataset": "sep",
    }


def normalize_cyse_row(row: dict) -> dict | None:
    if str(row.get("injection_type", "")).strip().lower() != "indirect":
        return None

    instruction = str(row.get("test_case_prompt", "")).strip()
    injected_input = str(row.get("user_input", "")).strip()
    judge_question = str(row.get("judge_question", "")).strip()

    if not instruction or not injected_input:
        return None

    return {
        "instruction": instruction,
        "input": injected_input,
        "prompt_injection": injected_input,
        "judge_question": judge_question,
        "attack_type": "indirect_injection",
        "dataset": "cyberseceval2",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw", help="Target raw data root")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)

    alpaca_rows = download_json(ALPACA_URL)
    write_json(raw_root / "alpaca_farm" / "davinci_003_outputs.json", alpaca_rows)

    sep_source_rows = download_json(SEP_URL)
    sep_rows = [row for row in (normalize_sep_row(item) for item in sep_source_rows) if row]
    write_json(raw_root / "sep" / "SEP_dataset_adapter.json", sep_rows)

    cyse_source_rows = download_json(CYSE_URL)
    cyse_rows = [row for row in (normalize_cyse_row(item) for item in cyse_source_rows) if row]
    write_json(raw_root / "cyberseceval2" / "CySE_prompt_injections_adapter.json", cyse_rows)

    print(
        json.dumps(
            {
                "alpaca_farm": len(alpaca_rows),
                "sep": len(sep_rows),
                "cyberseceval2": len(cyse_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
