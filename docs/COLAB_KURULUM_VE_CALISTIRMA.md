# Colab Kurulum ve Calistirma Adimlari

Bu belge, `Qwen/Qwen2.5-7B-Instruct` icin makale-bazli tez pipeline'ini Colab uzerinde calistirmak icindir.

Kritik nokta: `vllm` kurduktan sonra Colab runtime restart isteyebilir. Bu nedenle en guvenli sira:

1. GPU sec
2. Paketleri kur
3. Gerekirse runtime restart et
4. Repo'yu tekrar klonla
5. Veri cek
6. Dataset uret
7. Inference calistir
8. Metrikleri hesapla

## 1. Runtime

- `Runtime -> Change runtime type -> GPU`
- Onerilen:
  - en hizli: `H100`
  - cok iyi: `A100`
  - iyi: `L4`
  - kabul edilebilir: `T4`

## 2. Paketleri kur

Bu adimi yeni runtime acildiginda ilk yap.

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

Eger Colab su uyariyi verirse:

`You must restart the runtime in order to use newly installed versions.`

sunlari yap:

1. `Restart runtime` de
2. Runtime geri acildiginda paket kurulumunu tekrar etme
3. Dogrudan repo klonlama adimina gec

Not:
- Runtime restart, yerel degiskenleri siler.
- `/content` altindaki repo veya uretilen dosyalar kaybolabilir.
- Bu yuzden repo klonlamayi paket kurulumundan sonra yap.

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

## 6. Benchmark ham verilerini indir

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

## 7. Dataset uret

Tum benchmarklar:

```bash
!python src/data/build_dataset.py --source alpaca_farm --mode combined --output data/processed/eval_alpaca_farm_qwen25.jsonl
!python src/data/build_dataset.py --source sep --mode combined --output data/processed/eval_sep_qwen25.jsonl
!python src/data/build_dataset.py --source cyberseceval2 --mode security --output data/processed/eval_cyberseceval2_qwen25.jsonl
!python src/data/build_dataset.py --source tasktracker --mode combined --output data/processed/eval_tasktracker_qwen25.jsonl
```

Tek benchmark calistirmak istersen:

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

## 8. Inference motoru secimi

Iki secenek var:

- `transformers`
  - daha guvenli
  - daha yavas
- `vllm`
  - cok daha hizli
  - ozellikle `A100`, `H100`, `L4` icin tavsiye edilir

`run_inference.py` varsayilan olarak `transformers` kullanir.
Hizlandirmak icin `--engine vllm` ekle.

## 9. DefensiveTokens modelini olustur

Baseline icin bu adim gerekmez.

Defense inference'dan once, `-5DefensiveTokens` uzantili local modeli olusturman gerekir.
Qwen icin:

```bash
%cd /content/prompt-injection-defense/src/model
!python setup.py Qwen/Qwen2.5-7B-Instruct
```

Kontrol:

```bash
!find /content/prompt-injection-defense/src/model/Qwen -maxdepth 2 -type f | sort
```

Bu adimdan sonra localde su klasor olusmali:

- `/content/prompt-injection-defense/src/model/Qwen/Qwen2.5-7B-Instruct-5DefensiveTokens`

Eger bu adim calistirilmazsa defense inference asamasinda
`Qwen/Qwen2.5-7B-Instruct-5DefensiveTokens is not a valid model identifier`
hatası alırsın.

## 10. Baseline inference komutlari

Calisma klasoru:

```bash
%cd /content/prompt-injection-defense/src/model
```

### AlpacaFarm baseline

Transformers:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_alpaca_farm_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_alpaca_farm_baseline.jsonl
```

vLLM:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_alpaca_farm_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_alpaca_farm_baseline.jsonl
```

### SEP baseline

Transformers:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_sep_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_sep_baseline.jsonl
```

vLLM:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_sep_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_sep_baseline.jsonl
```

### CyberSecEval2 baseline

Transformers:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_cyberseceval2_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_cyberseceval2_baseline.jsonl
```

vLLM:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_cyberseceval2_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_cyberseceval2_baseline.jsonl
```

### TaskTracker baseline

Transformers:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_tasktracker_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_tasktracker_baseline.jsonl
```

vLLM:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_tasktracker_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_tasktracker_baseline.jsonl
```

## 11. Defense inference komutlari

### AlpacaFarm defense

Transformers:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --dataset /content/prompt-injection-defense/data/processed/eval_alpaca_farm_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_alpaca_farm_defense.jsonl
```

vLLM:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --dataset /content/prompt-injection-defense/data/processed/eval_alpaca_farm_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_alpaca_farm_defense.jsonl
```

### SEP defense

Transformers:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --dataset /content/prompt-injection-defense/data/processed/eval_sep_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_sep_defense.jsonl
```

vLLM:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --dataset /content/prompt-injection-defense/data/processed/eval_sep_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_sep_defense.jsonl
```

### CyberSecEval2 defense

Transformers:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --dataset /content/prompt-injection-defense/data/processed/eval_cyberseceval2_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_cyberseceval2_defense.jsonl
```

vLLM:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --dataset /content/prompt-injection-defense/data/processed/eval_cyberseceval2_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_cyberseceval2_defense.jsonl
```

### TaskTracker defense

Transformers:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --dataset /content/prompt-injection-defense/data/processed/eval_tasktracker_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_tasktracker_defense.jsonl
```

vLLM:

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --engine vllm --dataset /content/prompt-injection-defense/data/processed/eval_tasktracker_qwen25.jsonl --output /content/prompt-injection-defense/data/processed/predictions_qwen25_7b_tasktracker_defense.jsonl
```

## 12. Judge konfigurasyonu

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
Path("/content/prompt-injection-defense/judge_config.json").write_text(
    json.dumps(cfg, indent=2) + "\n",
    encoding="utf-8",
)
print("Wrote judge_config.json")
```

Judge kullanmak istemezsen `paper` evaluator local fallback ile devam eder.

## 13. Metrikleri hesapla

Calisma klasoru:

```bash
%cd /content/prompt-injection-defense
```

### AlpacaFarm

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

### TaskTracker

```bash
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_tasktracker_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_tasktracker_baseline.jsonl --output docs/raporlar/metrics_qwen25_7b_tasktracker_baseline.json --evaluator paper --judge-config judge_config.json
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_tasktracker_qwen25.jsonl --predictions data/processed/predictions_qwen25_7b_tasktracker_defense.jsonl --output docs/raporlar/metrics_qwen25_7b_tasktracker_defense.json --evaluator paper --judge-config judge_config.json
```

## 14. Tez tablosu icin aggregate dosya uret

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

## En hizli pratik calisma

Eger sadece tek benchmark icin hizli sonuc almak istiyorsan:

1. `vllm` kur
2. runtime restart et
3. repo'yu tekrar klonla
4. sadece ilgili benchmark dataset'ini uret
5. `--engine vllm` ile baseline ve defense calistir
6. metric hesapla

Onerilen ilk benchmark:
- `cyberseceval2`
- veya `sep`

## Kisa Notlar

- Bu pipeline makaleye daha yakindir, ama tum tablo degerlerini birebir garanti etmez.
- `paper` evaluator judge config olmadan da calisir; bu durumda local fallback kullanir.
- `tasktracker` verisi bu repoda makale-benzeri uyarlanmis benchmark seklinde uretilebilir.
- `run_inference.py` deterministic calisir; ayni input ile ayni ciktiyi hedefler.
- `run_inference.py` varsayilan olarak `transformers` motoru kullanir; hiz icin opsiyonel `--engine vllm` desteklenir.
- Colab runtime reset olursa repo ve `/content` altindaki dosyalari tekrar olusturman gerekebilir.
