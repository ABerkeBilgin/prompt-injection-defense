import argparse
import json
import ssl
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DATA_DIR = REPO_ROOT / "src" / "official_stacks" / "meta_secalign" / "data"
ALPACA_URL = (
    "https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/"
    "resolve/main/davinci_003_outputs.json"
)


def read_url_text(url: str) -> str:
    context = ssl.create_default_context()
    with urllib.request.urlopen(url, context=context) as response:
        return response.read().decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", default=str(TARGET_DATA_DIR))
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(read_url_text(ALPACA_URL))
    out_path = target_dir / "davinci_003_outputs.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"rows": len(payload), "output": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()