# Colab Kurulum ve Calistirma Adimlari

Bu belge, resmi ilk dalga tez evaluator akisini `Qwen/Qwen2.5-7B-Instruct` icin Colab uzerinde calistirmak icindir.

Ilk resmi kapsam:
- `alpaca_farm`
- `sep`
- `cyberseceval2`

Bu akista:
- `alpaca_farm` icin `ASR` ve `win_rate`
- `sep` icin security + utility
- `cyberseceval2` icin security
raporlanir.

Kritik fark:
- `paper` evaluator icin external judge config zorunludur.
- `judge_config.json` olmadan resmi metric akisi calismaz.

## 1. Runtime

- `Runtime -> Change runtime type -> GPU`
- Onerilen:
  - en hizli: `H100`
  - cok iyi: `A100`
  - iyi: `L4`
  - kabul edilebilir: `T4`

## 2. Paketleri kur

```bash
!pip install -U pip
!pip uninstall -y torch torchvision torchaudio
!pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
!pip install -U transformers==4.57.1 accelerate sentencepiece huggingface_hub
!pip install vllm
```

Kontrol:

```bash
!python -m pip show torch torchvision torchaudio vllm
```

## 3. Runtime restart

Eger Colab restart isterse:

1. `Restart runtime` de
2. Runtime geri acildiginda repo'yu yeniden klonla
3. Once paketlerin durdugunu kontrol et
4. Paket eksikse veya surumler kaymissa kurulum adimini tekrar calistir

Not:
- Runtime restart sonrasi `/content` altindaki repo kaybolabilir.
- En guvenli yol: repo'yu her zaman paket kurulumundan sonra klonla.

Kontrol komutu:

```bash
!python -m pip show torch torchvision torchaudio vllm
```

Eger burada paketlerden biri gorunmuyorsa veya surumler beklenen gibi degilse, `Paketleri kur` adimini tekrar calistir.

## 4. Repo'yu cek

Varsayilan branch:

```bash
%cd /content
!rm -rf prompt-injection-defense
!git clone https://github.com/ABerkeBilgin/prompt-injection-defense.git
%cd /content/prompt-injection-defense
```

`salih` branch'i icin:

```bash
%cd /content
!rm -rf prompt-injection-defense
!git clone --branch salih --single-branch https://github.com/ABerkeBilgin/prompt-injection-defense.git
%cd /content/prompt-injection-defense
```

## 5. Hugging Face login

```bash
!hf auth login
!hf auth whoami
```

## 6. Ham verileri indir

```bash
!python scripts/fetch_defensivetokens_datasets.py
```

Kontrol:

```bash
!find data/raw -maxdepth 2 -type f | sort
```

## 7. Datasetleri uret

Resmi ilk dalga benchmarklari:

```bash
!python src/data/build_dataset.py --source alpaca_farm --mode combined --output data/processed/eval_alpaca_farm_qwen25.jsonl
!python src/data/build_dataset.py --source sep --mode combined --output data/processed/eval_sep_qwen25.jsonl
!python src/data/build_dataset.py --source cyberseceval2 --mode security --output data/processed/eval_cyberseceval2_qwen25.jsonl
```

Kontrol:

```bash
!wc -l data/processed/eval_alpaca_farm_qwen25.jsonl
!wc -l data/processed/eval_sep_qwen25.jsonl
!wc -l data/processed/eval_cyberseceval2_qwen25.jsonl
```

## 8. Calisma klasoru kurali

- `src/model` klasorunde:
  - `python setup.py ...`
  - `python run_inference.py ...`
- repo kokunde:
  - `python scripts/fetch_defensivetokens_datasets.py`
  - `python src/data/build_dataset.py ...`
  - `python src/evaluation/compute_metrics.py ...`
  - `python scripts/build_metrics_table.py ...`

Model tarafina gecerken:

```bash
%cd /content/prompt-injection-defense/src/model
```

Metric veya tablo tarafina donerken:

```bash
%cd /content/prompt-injection-defense
```

## 9. DefensiveTokens modelini olustur

Baseline icin gerekmez.

Defense inference'dan once:

```bash
%cd /content/prompt-injection-defense/src/model
!python setup.py Qwen/Qwen2.5-7B-Instruct
```

Kontrol:

```bash
!find /content/prompt-injection-defense/src/model/Qwen -maxdepth 2 -type f | sort
```

Beklenen klasor:

- `/content/prompt-injection-defense/src/model/Qwen/Qwen2.5-7B-Instruct-5DefensiveTokens`

## 10. Inference motoru secimi

Iki secenek var:

- `transformers`
  - daha guvenli
  - daha yavas
- `vllm`
  - daha hizli
  - `A100`, `H100`, `L4` icin tavsiye edilir

Varsayilan motor `transformers`tir.
Hizli yol icin `--engine vllm` kullan.

## 11. Baseline inference

Calisma klasoru:

```bash
%cd /content/prompt-injection-defense/src/model
```

### AlpacaFarm baseline

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_alpaca_farm_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_alpaca_farm_baseline.jsonl
```

### SEP baseline

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_sep_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_sep_baseline.jsonl
```

### CyberSecEval2 baseline

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_cyberseceval2_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_cyberseceval2_baseline.jsonl
```

## 12. Defense inference

### AlpacaFarm defense

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --dataset /content/prompt-injection-defense/data/processed/eval_alpaca_farm_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_alpaca_farm_defense.jsonl
```

### SEP defense

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --dataset /content/prompt-injection-defense/data/processed/eval_sep_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_sep_defense.jsonl
```

### CyberSecEval2 defense

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --dataset /content/prompt-injection-defense/data/processed/eval_cyberseceval2_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_cyberseceval2_defense.jsonl
```

## 13. Judge config olustur

Bu adim zorunludur. `paper` evaluator judge config olmadan calismaz.

```python
import json
from pathlib import Path

cfg = {
    "api_key": "YOUR_API_KEY",
    "base_url": "https://api.openai.com/v1/chat/completions",
    "model": "gpt-4o-mini",
    "timeout": 60,
}
Path("/content/prompt-injection-defense/judge_config.json").write_text(
    json.dumps(cfg, indent=2) + "\n",
    encoding="utf-8",
)
print("Wrote judge_config.json")
```

## 14. Resmi metrikleri hesapla

Calisma klasoru:

```bash
%cd /content/prompt-injection-defense
```

### AlpacaFarm

`alpaca_farm` ciktilarinda hem `asr` hem `win_rate` gorunur.
Buradaki `win_rate`, tezde Alpaca utility kolonu olarak kullanilacak alandir.

```bash
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_alpaca_farm_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_alpaca_farm_baseline.jsonl --output docs/raporlar/metrics_qwen25_7b_alpaca_farm_baseline.json --evaluator paper --judge-config judge_config.json
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_alpaca_farm_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_alpaca_farm_defense.jsonl --output docs/raporlar/metrics_qwen25_7b_alpaca_farm_defense.json --evaluator paper --judge-config judge_config.json
```

### SEP

```bash
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_sep_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_sep_baseline.jsonl --output docs/raporlar/metrics_qwen25_7b_sep_baseline.json --evaluator paper --judge-config judge_config.json
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_sep_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_sep_defense.jsonl --output docs/raporlar/metrics_qwen25_7b_sep_defense.json --evaluator paper --judge-config judge_config.json
```

### CyberSecEval2

```bash
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_cyberseceval2_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_cyberseceval2_baseline.jsonl --output docs/raporlar/metrics_qwen25_7b_cyberseceval2_baseline.json --evaluator paper --judge-config judge_config.json
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_cyberseceval2_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_cyberseceval2_defense.jsonl --output docs/raporlar/metrics_qwen25_7b_cyberseceval2_defense.json --evaluator paper --judge-config judge_config.json
```

## 15. Aggregate tablo dosyasi uret

Yalnizca mevcut dosyalari verebilirsin; olmayan dosyalar atlanir.

```bash
!python scripts/build_metrics_table.py \
  --inputs \
  docs/raporlar/metrics_qwen25_7b_alpaca_farm_baseline.json \
  docs/raporlar/metrics_qwen25_7b_alpaca_farm_defense.json \
  docs/raporlar/metrics_qwen25_7b_sep_baseline.json \
  docs/raporlar/metrics_qwen25_7b_sep_defense.json \
  docs/raporlar/metrics_qwen25_7b_cyberseceval2_baseline.json \
  docs/raporlar/metrics_qwen25_7b_cyberseceval2_defense.json \
  --output docs/raporlar/metrics_qwen25_7b_table.json
```

Kontrol:

```bash
!cat docs/raporlar/metrics_qwen25_7b_table.json
```

## En hizli pratik yol

Eger once tek benchmark ile ilerlemek istiyorsan:

1. `vllm` kur
2. restart et
3. repo'yu yeniden klonla
4. sadece ilgili benchmark dataset'ini uret
5. Qwen setup
6. baseline + defense inference
7. judge config olustur
8. metric hesapla

Onerilen ilk benchmark:
- `alpaca_farm`
- veya `cyberseceval2`

## Kisa Notlar

- Bu resmi ilk dalga yalnizca `alpaca_farm`, `sep`, `cyberseceval2` icindir.
- `TaskTracker`, `InjecAgent`, `AgentDojo` sonraki fazdadir.
- `paper` evaluator judge config olmadan calismaz.
- `alpaca_farm` utility kolonu icin `win_rate` alanini kullan.
- `run_inference.py` varsayilan olarak `transformers` motoru kullanir; hiz icin `--engine vllm` desteklenir.
- Colab runtime reset olursa repo ve `/content` altindaki dosyalari yeniden olusturman gerekir.
