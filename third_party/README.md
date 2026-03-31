# Third-Party Official Stack

This directory is reserved for the official upstream repositories used by the paper-aligned workflow:

- `third_party/DefensiveToken`
- `third_party/Meta_SecAlign`

Initialize them with:

```bash
git submodule update --init --recursive
```

Then apply the local orchestration patches with:

```bash
python scripts/bootstrap_official_stack.py --apply-patches
```
