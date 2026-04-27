import json
from pathlib import Path

import torch
import transformers

STACK_ROOT = Path(__file__).resolve().parent
ASSET_PATH = STACK_ROOT / "utilitytokens.json"
OUTPUT_SUFFIX = "-5UtilityTokens"
SUPPORTED_MODELS = ["Qwen/Qwen2.5-7B-Instruct"]

# DefensiveToken chat template'ini genişlet: defensive prefix'in hemen arkasına utility prefix ekle
_DEFENSIVE_PREFIX = "".join(f"[DefensiveToken{i}]" for i in range(5))
_UTILITY_PLACEHOLDER = "{{UTILITY_PREFIX}}"

CHAT_TEMPLATES = {
    "Qwen/Qwen2.5-7B-Instruct": (
        "{%- if add_defensive_tokens %}\n"
        "{{- '" + _DEFENSIVE_PREFIX + "{UTILITY_PREFIX}' }}\n"
        "{%- endif %}\n"
        "{%- for message in messages %}\n"
        "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] | trim + '\\n\\n<|im_end|>\\n' }}\n"
        "{%- endfor %}\n"
        "{%- if add_generation_prompt %}\n"
        "{{- '<|im_start|>assistant\\n' }}\n"
        "{%- endif %}\n"
    ),
}


def load_utility_tokens() -> dict:
    with ASSET_PATH.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_utility_model_path(defended_model_path: str, output_root: Path | None = None) -> Path:
    base = Path(defended_model_path)
    root = output_root or base.parent
    return root / f"{base.name}{OUTPUT_SUFFIX}"


def prepare_utility_model(
    defended_model_path: str,
    output_root: Path | None = None,
    n_defensive: int = 5,
) -> Path:
    """
    defended_model_path: DefensiveToken uygulanmış model dizini
                         (örn. .../Qwen2.5-7B-Instruct-5DefensiveTokens)
    n_defensive        : mevcut defensive token sayısı (varsayılan 5)

    Çıktı: defended_model_path + "-5UtilityTokens" dizini
    """
    base_name = Path(defended_model_path).name
    # base_name içinden orijinal model adını çıkar
    original_model = None
    for supported in SUPPORTED_MODELS:
        short = supported.split("/")[-1]
        if short in base_name:
            original_model = supported
            break
    if original_model is None:
        raise ValueError(
            f"Desteklenmeyen model: {defended_model_path}. "
            f"Desteklenenler: {SUPPORTED_MODELS}"
        )

    utility_tokens_data = load_utility_tokens()
    if original_model not in utility_tokens_data:
        raise ValueError(f"{original_model} için utility token vektörleri bulunamadı.")

    n_utility = len(utility_tokens_data[original_model])

    output_dir = resolve_utility_model_path(defended_model_path, output_root=output_root)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(defended_model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(defended_model_path)

    # Utility token'ları tokenizer'a ekle
    utility_token_names = [f"[UtilityToken{i}]" for i in range(n_utility)]
    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    new_tokens = [t for t in utility_token_names if t not in existing]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": existing + new_tokens})

    model.resize_token_embeddings(len(tokenizer))

    # Utility embedding vektörlerini modele yaz
    utility_tensor = torch.tensor(utility_tokens_data[original_model], device=model.device)
    utility_ids = [tokenizer.convert_tokens_to_ids(t) for t in utility_token_names]
    for idx, tid in enumerate(utility_ids):
        model.get_input_embeddings().weight.data[tid] = utility_tensor[idx].to(
            model.get_input_embeddings().weight.dtype
        )

    # Chat template'i güncelle: defensive + utility prefix
    utility_prefix = "".join(utility_token_names)
    template = CHAT_TEMPLATES[original_model].replace("{UTILITY_PREFIX}", utility_prefix)
    tokenizer.chat_template = template

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir
