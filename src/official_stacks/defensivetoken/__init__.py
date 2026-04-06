OUTPUT_SUFFIX = "-5DefensiveTokens"
SUPPORTED_MODELS = ["Qwen/Qwen2.5-7B-Instruct"]


def prepare_defended_model(*args, **kwargs):
    from .core import prepare_defended_model as _prepare_defended_model

    return _prepare_defended_model(*args, **kwargs)


def resolve_defended_model_path(*args, **kwargs):
    from .core import resolve_defended_model_path as _resolve_defended_model_path

    return _resolve_defended_model_path(*args, **kwargs)

__all__ = [
    "OUTPUT_SUFFIX",
    "SUPPORTED_MODELS",
    "prepare_defended_model",
    "resolve_defended_model_path",
]
