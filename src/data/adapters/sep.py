from pathlib import Path

from .common import (
    dedupe_preserve_order,
    extract_attack_targets,
    first_text,
    infer_task_name,
    load_records,
    normalize_text,
    sanitize_metadata,
)


def load_sep_rows(raw_root: Path, mode: str, seed: int) -> list[dict]:
    del seed
    records = load_records(raw_root / "sep")
    rows = []

    for index, record in enumerate(records):
        instruction = first_text(
            record,
            ["instruction", "instruction_prompt", "prompt", "system_prompt", "task_prompt"],
        )
        untrusted_data = first_text(
            record,
            ["data", "data_prompt", "input", "content", "document", "text"],
        )
        injection = first_text(
            record,
            ["injection", "probe", "attack", "malicious_prompt", "trigger"],
        )
        witness = first_text(record, ["witness", "expected_output", "reference_output"])
        task_name = first_text(record, ["task", "task_type", "dataset"])

        if not instruction:
            continue

        metadata = sanitize_metadata(record)
        if witness:
            metadata["reference_output"] = witness

        base_row = {
            "id": f"sep-{index:06d}",
            "benchmark": "sep",
            "instruction": instruction,
            "untrusted_data": untrusted_data,
            "expected_task": infer_task_name(task_name or instruction),
            "judge_hint": witness,
            "metadata": metadata,
        }

        if mode in {"utility", "combined"}:
            utility_row = dict(base_row)
            utility_row["id"] = base_row["id"] + "-utility"
            utility_row["injection"] = ""
            utility_row["attack_type"] = "none"
            utility_row["metadata"] = dict(metadata)
            utility_row["metadata"]["attack_targets"] = []
            rows.append(utility_row)

        if injection and mode in {"security", "combined"}:
            security_row = dict(base_row)
            security_row["id"] = base_row["id"] + "-security"
            security_row["injection"] = injection
            security_row["attack_type"] = normalize_text(record.get("attack_type")) or "indirect_injection"
            security_row["metadata"] = dict(metadata)
            security_row["metadata"]["attack_targets"] = dedupe_preserve_order(
                extract_attack_targets(injection) + ([witness.lower()] if witness else [])
            )
            rows.append(security_row)

    return rows
