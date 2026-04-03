import argparse
from pathlib import Path

from src.official_stacks.defensivetoken import SUPPORTED_MODELS, prepare_defended_model


SUPPORTED_MODELS_TEXT = ', '.join(SUPPORTED_MODELS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', nargs='?', default='Qwen/Qwen2.5-7B-Instruct', help=f'Supported models: {SUPPORTED_MODELS_TEXT}')
    parser.add_argument('--output-root', default=None, help='Optional directory for defended model artifacts.')
    args = parser.parse_args()

    if args.model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f'Unsupported model for DefensiveToken setup: {args.model_name}. '
            f'Supported models: {SUPPORTED_MODELS_TEXT}'
        )

    output_root = Path(args.output_root) if args.output_root else None
    print(prepare_defended_model(args.model_name, output_root=output_root))


if __name__ == '__main__':
    main()
