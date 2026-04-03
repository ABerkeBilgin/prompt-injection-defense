import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DATA_DIR = REPO_ROOT / 'src' / 'official_stacks' / 'meta_secalign' / 'data'
META_SECALIGN_REPO = 'https://github.com/facebookresearch/Meta_SecAlign.git'
TORCHTUNE_INDEX = 'https://download.pytorch.org/whl/cu126'


def run(command: list[str], cwd: Path | None = None) -> None:
    print(' '.join(command))
    subprocess.run(command, cwd=str(cwd) if cwd is not None else None, check=True)


def copy_tree_files(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', default=None, help='Optional existing directory to use instead of a temporary clone dir')
    parser.add_argument('--skip-install', action='store_true', help='Skip pip dependency installation in the official repo')
    args = parser.parse_args()

    if args.workdir is None:
        with tempfile.TemporaryDirectory(prefix='meta_secalign_') as tmpdir:
            repo_dir = Path(tmpdir) / 'Meta_SecAlign'
            run(['git', 'clone', '--recurse-submodules', META_SECALIGN_REPO, str(repo_dir)])
            if not args.skip_install:
                run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], cwd=repo_dir)
                run([sys.executable, '-m', 'pip', 'install', 'torchtune==0.6.0', '--index-url', TORCHTUNE_INDEX], cwd=repo_dir)
            run([sys.executable, 'setup.py'], cwd=repo_dir)
            copy_tree_files(repo_dir / 'data', TARGET_DATA_DIR)
    else:
        repo_dir = Path(args.workdir).resolve()
        if not repo_dir.exists():
            raise FileNotFoundError(f'workdir not found: {repo_dir}')
        run([sys.executable, 'setup.py'], cwd=repo_dir)
        copy_tree_files(repo_dir / 'data', TARGET_DATA_DIR)

    print(f'Official Meta_SecAlign data copied to: {TARGET_DATA_DIR}')


if __name__ == '__main__':
    main()
