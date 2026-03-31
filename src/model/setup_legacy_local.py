import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transformers


BASE_DIR = Path(__file__).resolve().parent
DEFENSIVE_TOKENS_PATH = BASE_DIR / "defensivetokens.json"


CHAT_TEMPLATES = {
    "meta-llama/Meta-Llama-3-8B-Instruct":
    """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n
    {{- bos_token }}\n
    {%- for message in messages %}\n
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n'+ message['content'] | trim + '\\n\\n' + '<|eot_id|>' }}\n
    {%- endfor %}\n
    {%- if add_generation_prompt %}\n{{- '<|start_header_id|>assistant<|end_header_id|>\\n' }}\n{%- endif %}\n""",
    "meta-llama/Llama-3.1-8B-Instruct":
    """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n
    {{- bos_token }}\n
    {%- for message in messages %}\n
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n'+ message['content'] | trim + '\\n\\n' + '<|eot_id|>' }}\n
    {%- endfor %}\n
    {%- if add_generation_prompt %}\n{{- '<|start_header_id|>assistant<|end_header_id|>\\n' }}\n{%- endif %}\n""",
    "tiiuae/Falcon3-7B-Instruct":
    """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n
    {%- for message in messages %}\n
        {{- '<|' + message['role'] + '|>\\n' + message['content'] | trim + '\\n\\n' }}\n
    {%- endfor %}\n
    {%- if add_generation_prompt %}\n{{- '<|assistant|>\\n' }}\n{%- endif %}\n""",
    "Qwen/Qwen2.5-7B-Instruct":
    """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n
    {%- for message in messages %}\n
        {{- '<|im_start|>' + message['role'] + '\\n' + message['content'] | trim + '\\n\\n<|im_end|>\\n' }}\n
    {%- endfor %}\n
    {%- if add_generation_prompt %}\n{{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n""",
}


def load_defensive_tokens() -> dict:
    return json.loads(DEFENSIVE_TOKENS_PATH.read_text(encoding="utf-8"))


def process_model(model_name: str, defensive_vector: list[list[float]]) -> str:
    output_dir = BASE_DIR / f"{model_name}-5DefensiveTokens"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    defensive_tensor = torch.tensor(np.array(defensive_vector)).to(model.device)
    additional_special_tokens = {}
    for i in range(len(defensive_tensor)):
        additional_special_tokens[f"[DefensiveToken{i}]"] = defensive_tensor[i:i + 1, :]
    tokenizer.add_special_tokens({"additional_special_tokens": list(additional_special_tokens.keys())})
    model.resize_token_embeddings(len(tokenizer))
    for i in range(len(defensive_tensor)):
        model.get_input_embeddings().weight.data[-len(defensive_tensor) + i] = additional_special_tokens[f"[DefensiveToken{i}]"]
    model.save_pretrained(output_dir)
    tokenizer.chat_template = CHAT_TEMPLATES[model_name]
    tokenizer.save_pretrained(output_dir)
    return str(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", choices=list(CHAT_TEMPLATES.keys()), nargs="?", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()
    defensive_tokens = load_defensive_tokens()
    if args.model_name not in defensive_tokens:
        raise ValueError(f"No defensive token vector found for model: {args.model_name}")
    print(process_model(args.model_name, defensive_tokens[args.model_name]))


if __name__ == "__main__":
    main()
