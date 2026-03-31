import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = REPO_ROOT / "third_party" / "DefensiveToken"
LEGACY_SCRIPT = Path(__file__).resolve().with_name("setup_legacy_local.py")
SUPPORTED_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]


def run_official_setup(model_name: str) -> str:
    if not OFFICIAL_ROOT.exists():
        raise FileNotFoundError(
            f"Official DefensiveToken repository not found at {OFFICIAL_ROOT}. "
            "Initialize submodules and run scripts/bootstrap_official_stack.py first."
        )
    subprocess.run([sys.executable, "setup.py", model_name], cwd=OFFICIAL_ROOT, check=True)
    return str(OFFICIAL_ROOT / f"{model_name}-5DefensiveTokens")


def run_legacy_setup(model_name: str) -> str:
    if not LEGACY_SCRIPT.exists():
        raise FileNotFoundError(f"Legacy local setup script not found: {LEGACY_SCRIPT}")
    subprocess.run([sys.executable, str(LEGACY_SCRIPT), model_name], cwd=Path(__file__).resolve().parent, check=True)
    return str(Path(__file__).resolve().parent / f"{model_name}-5DefensiveTokens")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", choices=SUPPORTED_MODELS, nargs="?", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--legacy-local",
        action="store_true",
        help="Use the old in-repo embedding patch path instead of the official DefensiveToken repository.",
    )
    args = parser.parse_args()

    output_dir = run_legacy_setup(args.model_name) if args.legacy_local else run_official_setup(args.model_name)
    print(output_dir)


if __name__ == "__main__":
    main()
