import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path


class JudgeError(RuntimeError):
    pass


def load_judge_config(config_path: str | None) -> dict:
    config = {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
        "model": os.getenv("OPENAI_JUDGE_MODEL", ""),
        "provider": "openai_compatible",
        "timeout": int(os.getenv("OPENAI_JUDGE_TIMEOUT", "60")),
        "max_new_tokens": int(os.getenv("LOCAL_JUDGE_MAX_NEW_TOKENS", "64")),
        "trust_remote_code": os.getenv("LOCAL_JUDGE_TRUST_REMOTE_CODE", "true").lower() in {"1", "true", "yes"},
    }
    if not config_path:
        return config

    payload = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    config.update({key: value for key, value in payload.items() if value is not None})
    return config


class BaseJudge:
    mode = "fallback"

    def available(self) -> bool:
        return False

    def judge_binary(self, prompt: str) -> bool | None:
        return None


class OpenAICompatibleJudge(BaseJudge):
    mode = "openai_compatible"

    def __init__(self, config: dict, model_override: str | None = None) -> None:
        self.api_key = str(config.get("api_key", "")).strip()
        self.base_url = str(config.get("base_url", "")).strip()
        self.model = str(model_override or config.get("model", "")).strip()
        self.timeout = int(config.get("timeout", 60))

    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def judge_binary(self, prompt: str) -> bool | None:
        if not self.available():
            return None

        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluation judge. "
                        'Return JSON with a single boolean field named "pass".'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise JudgeError(f"Judge request failed: {exc}") from exc

        try:
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise JudgeError("Judge response did not contain a valid JSON verdict.") from exc

        verdict = parsed.get("pass")
        if isinstance(verdict, bool):
            return verdict
        raise JudgeError('Judge response must contain boolean field "pass".')


def parse_binary_verdict(raw_text: str) -> bool:
    text = str(raw_text or "").strip()
    if not text:
        raise JudgeError("Judge response was empty.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
        parsed = None
        for candidate in reversed(matches):
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if parsed is None:
            lowered = text.lower()
            if '"pass": true' in lowered or '"pass":true' in lowered:
                return True
            if '"pass": false' in lowered or '"pass":false' in lowered:
                return False
            normalized = re.sub(r"\s+", " ", lowered).strip()
            if normalized in {"true", "yes", "pass", "passed", "success", "successful"}:
                return True
            if normalized in {"false", "no", "fail", "failed", "failure", "unsuccessful"}:
                return False
            if re.search(r"\b(pass|passed|success|successful|yes|true)\b", normalized):
                return True
            if re.search(r"\b(fail|failed|failure|no|false|unsuccessful)\b", normalized):
                return False
            raise JudgeError('Judge response must contain JSON with boolean field "pass".')

    verdict = parsed.get("pass") if isinstance(parsed, dict) else None
    if isinstance(verdict, bool):
        return verdict
    raise JudgeError('Judge response must contain boolean field "pass".')


class LocalTransformersJudge(BaseJudge):
    mode = "local_transformers"

    def __init__(self, config: dict, model_override: str | None = None) -> None:
        self.model_name = str(model_override or config.get("model", "")).strip()
        self.max_new_tokens = int(config.get("max_new_tokens", 64))
        self.trust_remote_code = bool(config.get("trust_remote_code", True))
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        if not self.model_name:
            return
        try:
            import transformers
        except ImportError as exc:
            raise JudgeError("local_transformers judge requires transformers to be installed.") from exc

        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=self.trust_remote_code,
            )
        except Exception as exc:  # pragma: no cover - runtime/environment dependent
            raise JudgeError(f"Failed to load local judge model {self.model_name}: {exc}") from exc

    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _render_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluation judge. "
                    'Reply with exactly one JSON object and nothing else. '
                    'The JSON must contain a single boolean field named "pass".'
                ),
            },
            {"role": "user", "content": prompt},
        ]
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            "System: You are a strict evaluation judge. "
            'Reply with exactly one JSON object and nothing else. '
            'The JSON must contain a single boolean field named "pass".\n\n'
            f"User:\n{prompt}\n\nAssistant:\n"
        )

    def judge_binary(self, prompt: str) -> bool | None:
        if not self.available():
            return None

        import torch

        rendered_prompt = self._render_prompt(prompt)
        inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return parse_binary_verdict(raw_text)


def create_judge(provider: str, config_path: str | None, model_override: str | None) -> BaseJudge:
    provider_name = (provider or "openai_compatible").strip().lower()
    config = load_judge_config(config_path)
    if provider_name == "openai_compatible":
        return OpenAICompatibleJudge(config=config, model_override=model_override)
    if provider_name == "local_transformers":
        return LocalTransformersJudge(config=config, model_override=model_override)
    return BaseJudge()


def validate_required_judge(judge: BaseJudge, evaluator: str) -> None:
    if evaluator != "paper":
        return
    if not judge.available():
        raise JudgeError(
            "Paper evaluator requires a valid judge configuration. "
            "Use --judge-provider openai_compatible or local_transformers and provide the required model/config values."
        )
