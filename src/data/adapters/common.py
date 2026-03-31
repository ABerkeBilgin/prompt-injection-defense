import json
import re
from pathlib import Path

TEXT_KEYS = {
    "instruction",
    "input",
    "output",
    "prompt",
    "data",
    "dataset",
    "datasplit",
    "task",
    "category",
    "judge_question",
    "judge_hint",
    "success_criteria",
    "witness",
    "reference_output",
    "probe",
    "attack",
    "injection",
    "attack_type",
    "task_type",
    "source_split",
    "source_dataset",
    "injection_type",
}


def load_records(root: Path) -> list[dict]:
    if not root.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {root}")

    records: list[dict] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".json", ".jsonl"} or not path.is_file():
            continue
        records.extend(load_file(path))

    if not records:
        raise FileNotFoundError(f"No .json or .jsonl files found under {root}")
    return records


def load_file(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return [row for row in rows if isinstance(row, dict)]

    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        for key in ("data", "examples", "rows", "items", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]

    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    return []


def normalize_text(value) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return re.sub(r"\s+", " ", text).strip()


def first_text(record: dict, keys: list[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and normalize_text(value):
            return normalize_text(value)
    return ""


def infer_task_name(text: str, fallback: str = "general_instruction_following") -> str:
    lowered = normalize_text(text).lower()
    if not lowered:
        return fallback
    if "summar" in lowered:
        return "summarization"
    if "sentiment" in lowered or "positive" in lowered or "negative" in lowered:
        return "sentiment_classification"
    if "translate" in lowered:
        return "translation"
    if "classif" in lowered:
        return "classification"
    if "email" in lowered:
        return "email_writing"
    if "math" in lowered or "calculate" in lowered:
        return "reasoning"
    if "code" in lowered or "python" in lowered:
        return "code_generation"
    return fallback


def extract_attack_targets(injection: str) -> list[str]:
    lowered = normalize_text(injection).lower()
    patterns = [
        r"output exactly:\s*\"([^\"]+)\"",
        r"output exactly:\s*'([^']+)'",
        r"output exactly:\s*([a-z0-9 :_\-!?.]+)",
        r"reply only with\s*\"([^\"]+)\"",
        r"reply only with\s*'([^']+)'",
        r"reply only with\s*([a-z0-9 :_\-!?.]+)",
        r"start with\s*\"([^\"]+)\"",
        r"start with\s*'([^']+)'",
        r"start with\s*([a-z0-9 :_\-!?.]+)",
    ]
    targets = []
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            target = normalize_text(match.group(1)).lower().rstrip(".")
            if target:
                targets.append(target)
    return dedupe_preserve_order(targets)


def sanitize_metadata(record: dict) -> dict:
    cleaned = {}
    for key, value in record.items():
        if key not in TEXT_KEYS:
            continue
        text = normalize_text(value)
        if text:
            cleaned[key] = text
    return cleaned


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
