import json
from pathlib import Path

import torch
import transformers


OUTPUT_SUFFIX = "-5DefensiveTokens"
STACK_ROOT = Path(__file__).resolve().parent
ASSET_PATH = STACK_ROOT / "defensivetokens.json"
SUPPORTED_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
CHAT_TEMPLATES = {
    "meta-llama/Meta-Llama-3-8B-Instruct": """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n{{- bos_token }}\n{%- for message in messages %}\n{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n' + message['content'] | trim + '\\n\\n' + '<|eot_id|>' }}\n{%- endfor %}\n{%- if add_generation_prompt %}\n{{- '<|start_header_id|>assistant<|end_header_id|>\\n' }}\n{%- endif %}\n""",
    "meta-llama/Llama-3.1-8B-Instruct": """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n{{- bos_token }}\n{%- for message in messages %}\n{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n' + message['content'] | trim + '\\n\\n' + '<|eot_id|>' }}\n{%- endfor %}\n{%- if add_generation_prompt %}\n{{- '<|start_header_id|>assistant<|end_header_id|>\\n' }}\n{%- endif %}\n""",
    "tiiuae/Falcon3-7B-Instruct": """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n{%- for message in messages %}\n{{- '<|' + message['role'] + '|>\\n' + message['content'] | trim + '\\n\\n' }}\n{%- endfor %}\n{%- if add_generation_prompt %}\n{{- '<|assistant|>\\n' }}\n{%- endif %}\n""",
    "Qwen/Qwen2.5-7B-Instruct": """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n{%- for message in messages %}\n{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] | trim + '\\n\\n<|im_end|>\\n' }}\n{%- endfor %}\n{%- if add_generation_prompt %}\n{{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n""",
}


def load_defensive_tokens() -> dict:
    with ASSET_PATH.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_defended_model_path(model_name: str, output_root: Path | None = None) -> Path:
    root = output_root or STACK_ROOT
    return root / f"{model_name}{OUTPUT_SUFFIX}"


def prepare_defended_model(model_name: str, output_root: Path | None = None) -> Path:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model for DefensiveToken preparation: {model_name}")

    defensive_tokens = load_defensive_tokens()
    if model_name not in defensive_tokens:
        raise ValueError(f"No defensive token vectors found for model: {model_name}")

    output_dir = resolve_defended_model_path(model_name, output_root=output_root)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    defensive_tensor = torch.tensor(defensive_tokens[model_name], device=model.device)
    additional_special_tokens = [f"[DefensiveToken{i}]" for i in range(len(defensive_tensor))]
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    model.resize_token_embeddings(len(tokenizer))
    for index in range(len(additional_special_tokens)):
        model.get_input_embeddings().weight.data[-len(defensive_tensor) + index] = defensive_tensor[index : index + 1]

    model.save_pretrained(output_dir)
    tokenizer.chat_template = CHAT_TEMPLATES[model_name]
    tokenizer.save_pretrained(output_dir)
    return output_dir
