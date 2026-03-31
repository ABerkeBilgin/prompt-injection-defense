# Colab Kurulum ve Calistirma Adimlari

Bu belge, `Qwen/Qwen2.5-7B-Instruct` icin makale-bazli tez pipeline'ini Colab uzerinde calistirmak icindir.

## 1. Runtime

- `Runtime -> Change runtime type -> T4 GPU` sec.
- Daha hizli sonuc istiyorsan `L4`, `A100`, veya `H100` kullanabilirsin.

## 2. Repo'yu cek

```bash
%cd /content
!rm -rf prompt-injection-defense
!git clone https://github.com/ABerkeBilgin/prompt-injection-defense.git
veya
!git clone --branch salih --single-branch https://github.com/ABerkeBilgin/prompt-injection-defense.git
%cd /content/prompt-injection-defense
```

## 3. Paketleri kur

```bash
!pip install -U pip
!pip uninstall -y torch torchvision torchaudio
!pip install -r requirements.txt
!pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
!pip install -U transformers==4.57.1 accelerate sentencepiece huggingface_hub
```

Kurulumdan sonra surumleri kontrol et:

```bash
!python -m pip show torch torchvision torchaudio
```

`A100`, `L4` veya `H100` ile hizli toplu inference yapmak istersen opsiyonel olarak `vllm` de kurabilirsin:

```bash
!pip install vllm
```

## 4. Hugging Face login

```bash
!hf auth login
!hf auth whoami
```

## 5. Benchmark ham verilerini indir

```bash
!python scripts/fetch_defensivetokens_datasets.py
```

Bu adim su benchmarklari indirir veya uretir:

- `alpaca_farm`
- `sep`
- `cyberseceval2`
- `tasktracker`

Kontrol:

```bash
!find data/raw -maxdepth 2 -type f | sort
```

## 6. Benchmark-bazli datasetleri uret

```bash
!python src/data/build_dataset.py --source alpaca_farm --mode combined --output data/processed/eval_alpaca_farm_qwen25.jsonl
!python src/data/build_dataset.py --source sep --mode combined --output data/processed/eval_sep_qwen25.jsonl
!python src/data/build_dataset.py --source cyberseceval2 --mode security --output data/processed/eval_cyberseceval2_qwen25.jsonl
!python src/data/build_dataset.py --source tasktracker --mode combined --output data/processed/eval_tasktracker_qwen25.jsonl
```

Kontrol:

```bash
!wc -l data/processed/eval_alpaca_farm_qwen25.jsonl
!wc -l data/processed/eval_sep_qwen25.jsonl
!wc -l data/processed/eval_cyberseceval2_qwen25.jsonl
!wc -l data/processed/eval_tasktracker_qwen25.jsonl
```

## 7. Baseline inference

```bash
%cd /content/prompt-injection-defense/src/model
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_alpaca_farm_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_alpaca_farm_baseline.jsonl
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_sep_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_sep_baseline.jsonl
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_cyberseceval2_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_cyberseceval2_baseline.jsonl
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_tasktracker_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_tasktracker_baseline.jsonl
```

GPU'n gucluyse ayni komutlara `--engine vllm` ekleyerek cok daha hizli inference alabilirsin:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_sep_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_sep_baseline.jsonl
```

## 8. Defense inference

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --dataset /content/prompt-injection-defense/data/processed/eval_alpaca_farm_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_alpaca_farm_defense.jsonl
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --dataset /content/prompt-injection-defense/data/processed/eval_sep_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_sep_defense.jsonl
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --dataset /content/prompt-injection-defense/data/processed/eval_cyberseceval2_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_cyberseceval2_defense.jsonl
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --dataset /content/prompt-injection-defense/data/processed/eval_tasktracker_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_tasktracker_defense.jsonl
```

## 9. Judge konfigurasyonu

`paper` evaluator OpenAI-compatible judge kullanabilir. Bir config dosyasi olustur:

```python
import json
from pathlib import Path

cfg = {
    "api_key": "YOUR_API_KEY",
    "base_url": "https://api.openai.com/v1/chat/completions",
    "model": "gpt-4o-mini",
    "timeout": 60,
}
Path("judge_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
print("Wrote judge_config.json")
```

Judge kullanmak istemezsen `paper` evaluator local fallback ile devam eder.

## 10. Metrikleri hesapla

```bash
%cd /content/prompt-injection-defense
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_alpaca_farm_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_alpaca_farm_baseline.jsonl --output docs/raporlar/metrics_qwen25_7b_alpaca_farm_baseline.json --evaluator paper --judge-config judge_config.json
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_alpaca_farm_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_alpaca_farm_defense.jsonl --output docs/raporlar/metrics_qwen25_7b_alpaca_farm_defense.json --evaluator paper --judge-config judge_config.json
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_sep_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_sep_baseline.jsonl --output docs/raporlar/metrics_qwen25_7b_sep_baseline.json --evaluator paper --judge-config judge_config.json
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_sep_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_sep_defense.jsonl --output docs/raporlar/metrics_qwen25_7b_sep_defense.json --evaluator paper --judge-config judge_config.json
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_cyberseceval2_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_cyberseceval2_baseline.jsonl --output docs/raporlar/metrics_qwen25_7b_cyberseceval2_baseline.json --evaluator paper --judge-config judge_config.json
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_cyberseceval2_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_cyberseceval2_defense.jsonl --output docs/raporlar/metrics_qwen25_7b_cyberseceval2_defense.json --evaluator paper --judge-config judge_config.json
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_tasktracker_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_tasktracker_baseline.jsonl --output docs/raporlar/metrics_qwen25_7b_tasktracker_baseline.json --evaluator paper --judge-config judge_config.json
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_tasktracker_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_tasktracker_defense.jsonl --output docs/raporlar/metrics_qwen25_7b_tasktracker_defense.json --evaluator paper --judge-config judge_config.json
```

## 11. Tez tablosu icin aggregate dosya uret

```bash
!python scripts/build_metrics_table.py \
  --inputs \
  docs/raporlar/metrics_qwen25_7b_alpaca_farm_baseline.json \
  docs/raporlar/metrics_qwen25_7b_alpaca_farm_defense.json \
  docs/raporlar/metrics_qwen25_7b_sep_baseline.json \
  docs/raporlar/metrics_qwen25_7b_sep_defense.json \
  docs/raporlar/metrics_qwen25_7b_cyberseceval2_baseline.json \
  docs/raporlar/metrics_qwen25_7b_cyberseceval2_defense.json \
  docs/raporlar/metrics_qwen25_7b_tasktracker_baseline.json \
  docs/raporlar/metrics_qwen25_7b_tasktracker_defense.json \
  --output docs/raporlar/metrics_qwen25_7b_table.json
```

Kontrol:

```bash
!cat docs/raporlar/metrics_qwen25_7b_table.json
```

## Kisa Notlar

- Bu pipeline makaleye daha yakindir, ama tum tablo degerlerini birebir garanti etmez.
- `paper` evaluator judge config olmadan da calisir; bu durumda local fallback kullanir.
- `tasktracker` verisi bu repoda makale-benzeri uyarlanmis benchmark seklinde uretilebilir.
- `run_inference.py` deterministic calisir; ayni input ile ayni ciktiyi hedefler.
- `run_inference.py` varsayilan olarak `transformers` motoru kullanir; hiz icin opsiyonel `--engine vllm` desteklenir.
- Colab runtime reset olursa tum adimlari bastan tekrarlaman gerekir.
