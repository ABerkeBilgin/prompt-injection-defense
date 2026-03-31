"""
Batch inference with vLLM for ultra-fast generation.
"""

import argparse
import json
from pathlib import Path
from vllm import LLM, SamplingParams
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

    # Tokenizer'ı formatlama için kullanmaya devam ediyoruz
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    dataset = load_jsonl(Path(args.dataset))
    
    print("Promptlar hazırlanıyor...")
    prompts = []
    for sample in dataset:
        if sample["untrusted_data"] == sample["injection"]:
            raw_input = sample["untrusted_data"]
        else:
            raw_input = (sample["untrusted_data"] + " " + sample["injection"]).strip()
            
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
        prompts.append(input_string)

    # --- vLLM MOTORUNU BAŞLAT ---
    print("vLLM Motoru A100 üzerinde başlatılıyor...")
    # gpu_memory_utilization=0.9 -> A100 VRAM'inin %90'ını vLLM'e ayırır
    llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9)
    
    # Parametreleri makaleye uygun olarak rastgelelik kapalı (temperature=0.0) ayarlıyoruz
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    # --- TOPLU (BATCH) INFERENCE ---
    print(f"{len(prompts)} adet örnek işleniyor. Lütfen bekleyin...")
    outputs = llm.generate(prompts, sampling_params)

    # Sonuçları kaydet
    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        results.append({"id": dataset[i]["id"], "output": generated_text})

    write_jsonl(Path(args.output), results)
    print(f"\nKaydedildi: {args.output} ({len(results)} satır)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", required=True)
    parser.add_argument("--no-defense", action="store_true")
    parser.add_argument("--dataset", default="data/processed/eval_set.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()