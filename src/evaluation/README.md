# Evaluation Module

Bu dizin artik yardimci evaluator ve yerel analiz kodlarini tutar.

Resmi paper-aligned benchmark yolu:
- harness: `src/official_stacks/meta_secalign`
- DefensiveToken hazirlama: `src/official_stacks/defensivetoken`
- orchestration ve raporlama: `scripts/run_official_eval.py`

`src/evaluation/compute_metrics.py` gelistirme ve ek analiz icin kalir; final tablo uretiminde birincil yol degildir.
