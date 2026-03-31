import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def infer_defense_from_name(path: Path, payload: dict) -> str:
    stem = path.stem.lower()
    if "baseline" in stem:
        return "off"
    if "defense" in stem or "defended" in stem:
        return "on"
    return str(payload.get("defense_mode", "unknown"))


def infer_model_from_name(path: Path) -> str:
    stem = path.stem.lower()
    if "qwen25_7b" in stem or "qwen2.5-7b" in stem:
        return "Qwen/Qwen2.5-7B-Instruct"
    return "unknown"


def build_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        payload = read_json(path)
        model_name = infer_model_from_name(path)
        defense_mode = infer_defense_from_name(path, payload)
        for benchmark_result in payload.get("by_benchmark", []):
            rows.append(
                {
                    "model": model_name,
                    "benchmark": benchmark_result.get("benchmark"),
                    "defense_on": defense_mode,
                    "asr": benchmark_result.get("asr"),
                    "utility": benchmark_result.get("utility"),
                    "evaluator_mode": payload.get("evaluator_mode", "unknown"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Metric json files")
    parser.add_argument("--output", required=True, help="Aggregate output json path")
    args = parser.parse_args()

    rows = build_rows([Path(item) for item in args.inputs])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
