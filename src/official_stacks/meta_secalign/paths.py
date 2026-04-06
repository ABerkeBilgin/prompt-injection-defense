import sys
from pathlib import Path


STACK_ROOT = Path(__file__).resolve().parent
DATA_DIR = STACK_ROOT / "data"
GCG_VENDOR_DIR = STACK_ROOT / "gcg_vendor"


def add_vendor_paths() -> None:
    for path in (STACK_ROOT, GCG_VENDOR_DIR):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def resolve_data_path(value: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    if candidate.exists():
        return str(candidate)
    stack_candidate = STACK_ROOT / candidate
    if stack_candidate.exists():
        return str(stack_candidate)
    data_candidate = DATA_DIR / candidate.name
    if data_candidate.exists():
        return str(data_candidate)
    return str(stack_candidate)