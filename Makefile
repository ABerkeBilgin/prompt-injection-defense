PYTHON ?= python
MODEL  ?= meta-llama/Llama-3.1-8B-Instruct

.PHONY: prepare setup baseline defense eval-baseline eval-defense

# ──────────────────────────────────────────────
# YERELDE çalışır (GPU gerekmez)
# ──────────────────────────────────────────────

# 1. Veri setini oluştur
prepare:
	$(PYTHON) src/data/build_dataset.py --output data/processed/eval_set.jsonl

# 5. Metrik hesapla — baseline
eval-baseline:
	$(PYTHON) src/evaluation/compute_metrics.py \
		--dataset data/processed/eval_set.jsonl \
		--predictions data/processed/predictions_baseline.jsonl \
		--output docs/raporlar/metrics_baseline.json

# 6. Metrik hesapla — defense
eval-defense:
	$(PYTHON) src/evaluation/compute_metrics.py \
		--dataset data/processed/eval_set.jsonl \
		--predictions data/processed/predictions_defense.jsonl \
		--output docs/raporlar/metrics_defense.json

# ──────────────────────────────────────────────
# COLAB'DA çalışır (GPU gerektirir)
# ──────────────────────────────────────────────

# 2. DefensiveToken'lı model versiyonunu oluştur (bir kez çalıştır)
setup:
	$(PYTHON) src/model/setup.py

# 3. Baseline: DefensiveToken OLMADAN çıkarım
baseline:
	$(PYTHON) src/model/run_inference.py -m $(MODEL) --no-defense \
		--dataset data/processed/eval_set.jsonl \
		--output data/processed/predictions_baseline.jsonl

# 4. Defense: DefensiveToken İLE çıkarım
defense:
	$(PYTHON) src/model/run_inference.py -m $(MODEL) \
		--dataset data/processed/eval_set.jsonl \
		--output data/processed/predictions_defense.jsonl
