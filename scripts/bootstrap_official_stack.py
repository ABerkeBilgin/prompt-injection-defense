import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
DEFENSIVE_TOKEN_ROOT = THIRD_PARTY_ROOT / "DefensiveToken"
META_SECALIGN_ROOT = THIRD_PARTY_ROOT / "Meta_SecAlign"
PATCH_STATE_PATH = REPO_ROOT / "patches" / "meta_secalign" / "bootstrap_state.json"
DEFENSIVE_TOKEN_URL = "https://github.com/Sizhe-Chen/DefensiveToken.git"
META_SECALIGN_URL = "https://github.com/facebookresearch/Meta_SecAlign.git"


def run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def clone_repo(url: str, dest: Path, recurse_submodules: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "clone"]
    if recurse_submodules:
        command.append("--recurse-submodules")
    command.extend([url, str(dest)])
    subprocess.run(command, check=True)


def ensure_repo(path: Path, required_files: list[str], label: str, clone_url: str | None = None, recurse_submodules: bool = False) -> None:
    if not path.exists():
        if clone_url:
            clone_repo(clone_url, path, recurse_submodules=recurse_submodules)
        else:
            raise FileNotFoundError(
                f"{label} repository not found at {path}. Run `git submodule update --init --recursive` first."
            )
    for name in required_files:
        if not (path / name).exists():
            raise FileNotFoundError(f"{label} repository at {path} is missing required file: {name}")


def maybe_checkout(repo_root: Path, ref: str | None) -> None:
    if not ref:
        return
    run_git(["fetch", "--all", "--tags"], cwd=repo_root)
    run_git(["checkout", ref], cwd=repo_root)


def patch_metasecalign_utils(meta_root: Path) -> tuple[bool, bool]:
    utils_path = meta_root / "utils.py"
    original = utils_path.read_text(encoding="utf-8-sig")
    updated = original

    patterns = [
        (
            r"""apply_chat_template\(\s*\[\s*\{"role":\s*"user",\s*"content":\s*d_item\['instruction'\]\}\s*,\s*\{"role":\s*"input",\s*"content":\s*d_item\['input'\]\}\s*\]\s*,\s*tokenize=False\s*,\s*add_generation_prompt=True(?:\s*,\s*add_defensive_tokens=False)?\s*\)""",
            """apply_chat_template([{"role": "system", "content": d_item['instruction']}, {"role": "user", "content": d_item['input']}], tokenize=False, add_generation_prompt=True, add_defensive_tokens=False)""",
        ),
        (
            r"""apply_chat_template\(\s*\[\s*\{"role":\s*"user",\s*"content":\s*d_item\['instruction'\]\s*\+\s*'\\n\\n'\s*\+\s*d_item\['input'\]\}\s*\]\s*,\s*tokenize=False\s*,\s*add_generation_prompt=True(?:\s*,\s*add_defensive_tokens=False)?\s*\)""",
            """apply_chat_template([{"role": "system", "content": d_item['instruction'] + '\\n\\n' + d_item['input']}], tokenize=False, add_generation_prompt=True, add_defensive_tokens=False)""",
        ),
        (
            r"""apply_chat_template\(\s*\[\s*\{"role":\s*"user",\s*"content":\s*d_item\['instruction'\]\}\s*\]\s*,\s*tokenize=False\s*,\s*add_generation_prompt=True(?:\s*,\s*add_defensive_tokens=False)?\s*\)""",
            """apply_chat_template([{"role": "system", "content": d_item['instruction']}], tokenize=False, add_generation_prompt=True, add_defensive_tokens=False)""",
        ),
    ]

    replacement_count = 0
    for pattern, replacement in patterns:
        updated, count = re.subn(pattern, replacement, updated)
        replacement_count += count

    already_patched = all(
        snippet in updated
        for snippet in [
            """{"role": "system", "content": d_item['instruction']}, {"role": "user", "content": d_item['input']}""",
            """{"role": "system", "content": d_item['instruction'] + '\\n\\n' + d_item['input']}""",
            """{"role": "system", "content": d_item['instruction']}""",
            "add_defensive_tokens=False",
        ]
    )

    if updated == original:
        return False, already_patched

    utils_path.write_text(updated.rstrip("\n") + "\n", encoding="utf-8", newline="\n")
    return True, True


def write_patch_state(meta_ref: str | None, dt_ref: str | None, patched: bool, already_patched: bool) -> None:
    PATCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta_secalign_ref": meta_ref or "current",
        "defensive_token_ref": dt_ref or "current",
        "metasecalign_utils_patched": patched,
        "metasecalign_utils_already_patched": already_patched,
    }
    PATCH_STATE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--third-party-root", default=str(THIRD_PARTY_ROOT))
    parser.add_argument("--metasecalign-ref", default=None)
    parser.add_argument("--defensivetoken-ref", default=None)
    parser.add_argument("--apply-patches", action="store_true")
    parser.add_argument("--checkout-refs", action="store_true")
    parser.add_argument("--clone-missing", action="store_true")
    args = parser.parse_args()

    third_party_root = Path(args.third_party_root)
    defensive_root = third_party_root / "DefensiveToken"
    meta_root = third_party_root / "Meta_SecAlign"

    ensure_repo(
        defensive_root,
        ["README.md", "setup.py"],
        "DefensiveToken",
        clone_url=DEFENSIVE_TOKEN_URL if args.clone_missing else None,
    )
    ensure_repo(
        meta_root,
        ["README.md", "run_tests.py", "test.py", "utils.py"],
        "Meta_SecAlign",
        clone_url=META_SECALIGN_URL if args.clone_missing else None,
        recurse_submodules=True,
    )

    if args.checkout_refs:
        maybe_checkout(defensive_root, args.defensivetoken_ref)
        maybe_checkout(meta_root, args.metasecalign_ref)

    patched = False
    already_patched = False
    if args.apply_patches:
        patched, already_patched = patch_metasecalign_utils(meta_root)

    write_patch_state(args.metasecalign_ref, args.defensivetoken_ref, patched, already_patched)
    print(
        json.dumps(
            {
                "defensive_token_root": str(defensive_root),
                "meta_secalign_root": str(meta_root),
                "patched": patched,
                "already_patched": already_patched,
                "patch_state": str(PATCH_STATE_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
