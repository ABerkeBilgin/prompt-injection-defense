from alpaca_farm_official import prepare_asr as prepare_alpaca_asr
from alpaca_farm_official import prepare_winrate as prepare_alpaca_winrate
from cyberseceval2_official import prepare_security as prepare_cyse_security
from judges import BaseJudge
from sep_official import prepare_security as prepare_sep_security
from sep_official import prepare_utility as prepare_sep_utility


SUPPORTED_PAPER_BENCHMARKS = {"alpaca_farm", "sep", "cyberseceval2"}


def is_security_sample(sample: dict) -> bool:
    return sample.get("attack_type") != "none" and bool(sample.get("injection"))


def is_utility_sample(sample: dict) -> bool:
    benchmark = sample.get("benchmark")
    return benchmark in {"alpaca_farm", "sep"} or sample.get("attack_type") == "none"


def _resolve_prepared(entries: list[dict], judge: BaseJudge) -> list[bool]:
    prompts = [entry["prompt"] for entry in entries if entry["prompt"]]
    verdicts = judge.judge_many_binary(prompts) if prompts else []
    prompt_index = 0
    resolved: list[bool] = []
    for entry in entries:
        preset = entry["preset"]
        if preset is not None:
            resolved.append(bool(preset))
            continue
        resolved.append(bool(verdicts[prompt_index]))
        prompt_index += 1
    return resolved


def evaluate_paper_samples(samples: list[dict], outputs: list[str], judge: BaseJudge) -> list[tuple[bool, bool]]:
    if len(samples) != len(outputs):
        raise ValueError("Sample and output counts must match.")

    security_entries: list[dict] = []
    utility_entries: list[dict] = []

    for sample, output in zip(samples, outputs):
        benchmark = sample.get("benchmark")
        if benchmark == "alpaca_farm":
            security_preset, security_prompt = prepare_alpaca_asr(sample, output)
            utility_preset, utility_prompt = prepare_alpaca_winrate(sample, output)
        elif benchmark == "sep":
            security_preset, security_prompt = prepare_sep_security(sample, output)
            utility_preset, utility_prompt = prepare_sep_utility(sample, output)
        elif benchmark == "cyberseceval2":
            security_preset, security_prompt = prepare_cyse_security(sample, output)
            utility_preset, utility_prompt = False, None
        else:
            raise ValueError(f"Unsupported benchmark for paper evaluator: {benchmark}")

        security_entries.append({"preset": security_preset, "prompt": security_prompt})
        utility_entries.append({"preset": utility_preset, "prompt": utility_prompt})

    security_results = _resolve_prepared(security_entries, judge)
    utility_results = _resolve_prepared(utility_entries, judge)
    return list(zip(security_results, utility_results))
