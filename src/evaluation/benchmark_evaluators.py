from alpaca_farm_official import evaluate_asr as evaluate_alpaca_asr
from alpaca_farm_official import evaluate_winrate as evaluate_alpaca_winrate
from cyberseceval2_official import evaluate_security as evaluate_cyse_security
from judges import BaseJudge
from sep_official import evaluate_security as evaluate_sep_security
from sep_official import evaluate_utility as evaluate_sep_utility


SUPPORTED_PAPER_BENCHMARKS = {"alpaca_farm", "sep", "cyberseceval2"}


def is_security_sample(sample: dict) -> bool:
    return sample.get("attack_type") != "none" and bool(sample.get("injection"))


def is_utility_sample(sample: dict) -> bool:
    benchmark = sample.get("benchmark")
    return benchmark in {"alpaca_farm", "sep"} or sample.get("attack_type") == "none"


def evaluate_paper_sample(sample: dict, output: str, judge: BaseJudge) -> tuple[bool, bool]:
    benchmark = sample.get("benchmark")
    if benchmark == "alpaca_farm":
        return evaluate_alpaca_asr(sample, output, judge), evaluate_alpaca_winrate(sample, output, judge)
    if benchmark == "sep":
        return evaluate_sep_security(sample, output, judge), evaluate_sep_utility(sample, output, judge)
    if benchmark == "cyberseceval2":
        return evaluate_cyse_security(sample, output, judge), False
    raise ValueError(f"Unsupported benchmark for paper evaluator: {benchmark}")
