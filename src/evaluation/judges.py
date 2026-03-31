import json
import os
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


def create_judge(provider: str, config_path: str | None, model_override: str | None) -> BaseJudge:
    provider_name = (provider or "openai_compatible").strip().lower()
    if provider_name == "openai_compatible":
        config = load_judge_config(config_path)
        return OpenAICompatibleJudge(config=config, model_override=model_override)
    return BaseJudge()


def validate_required_judge(judge: BaseJudge, evaluator: str) -> None:
    if evaluator != "paper":
        return
    if not judge.available():
        raise JudgeError(
            "Paper evaluator requires a valid external judge configuration. "
            "Provide --judge-config or OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_JUDGE_MODEL."
        )
