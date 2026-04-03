"""
Batch inference for benchmark datasets.

Usage:
  python src/model/run_inference.py -m Qwen/Qwen2.5-7B-Instruct --no-defense \
      --dataset data/processed/eval_set.jsonl \
      --output data/processed/predictions_baseline.jsonl

  python src/model/run_inference.py -m Qwen/Qwen2.5-7B-Instruct \
      --dataset data/processed/eval_set.jsonl \
      --output data/processed/predictions_defense.jsonl
"""

import argparse
import json
from pathlib import Path

import transformers

from src.official_stacks.defensivetoken import SUPPORTED_MODELS, resolve_defended_model_path


def recursive_filter(value: str, filters: list[str]) -> str:
    original = value
    for item in filters:
        value = value.replace(item, "")
    if value != original:
        return recursive_filter(value, filters)
    return value


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_model_name(base_name: str, use_defense: bool) -> str:
    if use_defense:
        return str(resolve_defended_model_path(base_name))
    return base_name


def build_prompts(dataset: list[dict], tokenizer, use_defense: bool) -> list[str]:
    prompts = []
    for sample in dataset:
        raw_input = sample["untrusted_data"] + " " + sample["injection"]
        clean_input = recursive_filter(raw_input, tokenizer.all_special_tokens)
        conversation = [
            {"role": "system", "content": sample["instruction"]},
            {"role": "user", "content": clean_input},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                add_defensive_tokens=use_defense,
            )
        )
    return prompts


def build_result_row(sample: dict, output_text: str, args) -> dict:
    return {
        "id": sample["id"],
        "output": output_text,
        "benchmark": sample.get("benchmark", ""),
        "attack_type": sample.get("attack_type", ""),
        "model_name": args.model_name,
        "defense_mode": "off" if args.no_defense else "on",
    }


def run_with_transformers(args, dataset: list[dict], tokenizer, model_name: str) -> list[dict]:
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    results = []
    for index, sample in enumerate(dataset, start=1):
        print(f"[{index}/{len(dataset)}] id={sample['id']}")
        prompt = build_prompts([sample], tokenizer, use_defense=(not args.no_defense))[0]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=model.generation_config,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"  -> {output_text[:100]}")
        results.append(build_result_row(sample, output_text, args))
    return results


def run_with_vllm(args, dataset: list[dict], tokenizer, model_name: str) -> list[dict]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed. Install it before using --engine vllm.") from exc

    prompts = build_prompts(dataset, tokenizer, use_defense=(not args.no_defense))
    print(f"Prepared {len(prompts)} prompts for vLLM.")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for sample, output in zip(dataset, outputs):
        generated_text = output.outputs[0].text if output.outputs else ""
        results.append(build_result_row(sample, generated_text, args))
    return results


def run(args) -> None:
    model_name = build_model_name(args.model_name, not args.no_defense)
    print(f"Model: {model_name}")
    print(f"DefensiveTokens: {'KAPALI' if args.no_defense else 'ACIK'}")
    print(f"Engine: {args.engine}\n")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    dataset = load_jsonl(Path(args.dataset))

    if args.engine == "vllm":
        results = run_with_vllm(args, dataset, tokenizer, model_name)
    else:
        results = run_with_transformers(args, dataset, tokenizer, model_name)

    write_jsonl(Path(args.output), results)
    print(f"\nKaydedildi: {args.output} ({len(results)} satir)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--no-defense", action="store_true", help="DefensiveToken kullanma (baseline modu)")
    parser.add_argument("--dataset", default="data/processed/eval_set.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--engine", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
