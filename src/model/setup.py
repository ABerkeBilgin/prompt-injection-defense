import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.official_stacks.meta_secalign.config import TARGET_MODEL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional directory for defended model artifacts.",
    )
    args = parser.parse_args()

    from src.official_stacks.defensivetoken import prepare_defended_model

    output_root = Path(args.output_root) if args.output_root else None
    print(prepare_defended_model(TARGET_MODEL, output_root=output_root))


if __name__ == "__main__":
    main()
