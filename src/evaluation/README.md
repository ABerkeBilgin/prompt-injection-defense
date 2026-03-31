# Evaluation Module

This folder now contains the legacy in-repo evaluation pipeline.

Important:
- The official paper-aligned path is no longer this folder.
- Final thesis tables must be produced through `third_party/Meta_SecAlign` and `scripts/run_official_eval.py`.

What remains here:
- custom evaluator experiments
- local debug utilities
- temporary development-only judge helpers

What is official now:
- benchmark execution: `third_party/Meta_SecAlign`
- DefensiveToken model preparation: `third_party/DefensiveToken`
- orchestration and report normalization: `scripts/run_official_eval.py`

If you need paper-equivalent results, do not use `src/evaluation/compute_metrics.py` as the primary path.
