# Model Module

This folder now serves two roles:

1. Official wrapper entrypoints
- `setup.py`: calls the official `third_party/DefensiveToken/setup.py` by default

2. Legacy local implementation
- `setup_legacy_local.py`: the old in-repo embedding patch path, kept only for debugging or fallback
- `run_inference.py`: legacy/custom local inference helper

Official paper-aligned flow:
- prepare defended model via `src/model/setup.py`
- run benchmark evaluation through `scripts/run_official_eval.py`
- do not treat `run_inference.py` as the primary official evaluation path
