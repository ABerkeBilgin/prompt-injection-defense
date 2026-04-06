import argparse
from pathlib import Path

from src.official_stacks.defensivetoken import prepare_defended_model
from src.official_stacks.meta_secalign.config import TARGET_MODEL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional directory for defended model artifacts.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else None
    print(prepare_defended_model(TARGET_MODEL, output_root=output_root))


if __name__ == "__main__":
    main()