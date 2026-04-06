import logging
import sys


def setup_logger(verbose: bool) -> None:
    logging.basicConfig(
        stream=sys.stdout,
        format="[%(asctime)s - %(name)s - %(levelname)s]: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        force=True,
    )