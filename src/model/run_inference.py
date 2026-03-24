"""
Batch inference: eval_set.jsonl dosyasındaki tüm örnekleri çalıştırır,
predictions.jsonl olarak kaydeder.

Kullanım:
  # Baseline (DefensiveToken YOK)
  python src/model/run_inference.py -m meta-llama/Llama-3.1-8B-Instruct --no-defense \
      --dataset data/processed/eval_set.jsonl \
      --output data/processed/predictions_baseline.jsonl

  # DefensiveTokens İLE
  python src/model/run_inference.py -m meta-llama/Llama-3.1-8B-Instruct \
      --dataset data/processed/eval_set.jsonl \
      --output data/processed/predictions_defense.jsonl
"""

import argparse
import json
from pathlib import Path

import transformers


def recursive_filter(s, filters):
    orig = s
    for f in filters:
        s = s.replace(f, "")
    if s != orig:
        return recursive_filter(s, filters)
    return s


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_model_name(base_name: str, use_defense: bool) -> str:
    if use_defense:
        return base_name + "-5DefensiveTokens"
    return base_name


def run(args):
    model_name = build_model_name(args.model_name, not args.no_defense)
    print(f"Model: {model_name}")
    print(f"DefensiveTokens: {'KAPALI' if args.no_defense else 'AÇIK'}\n")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto"
    )

    dataset = load_jsonl(Path(args.dataset))
    results = []

    for i, sample in enumerate(dataset):
        print(f"[{i+1}/{len(dataset)}] id={sample['id']}")

        raw_input = sample["untrusted_data"] + " " + sample["injection"]
        clean_input = recursive_filter(raw_input, tokenizer.all_special_tokens)

        conversation = [
            {"role": "system", "content": sample["instruction"]},
            {"role": "user", "content": clean_input},
        ]

        input_string = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            add_defensive_tokens=(not args.no_defense),
        )

        inputs = tokenizer(input_string, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=model.generation_config,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=256,
        )
        output_text = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        print(f"  -> {output_text[:100]}")
        results.append({"id": sample["id"], "output": output_text})

    write_jsonl(Path(args.output), results)
    print(f"\nKaydedildi: {args.output} ({len(results)} satır)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_name", required=True,
        choices=[
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "tiiuae/Falcon3-7B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
    )
    parser.add_argument("--no-defense", action="store_true",
                        help="DefensiveToken kullanma (baseline modu)")
    parser.add_argument("--dataset", default="data/processed/eval_set.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
