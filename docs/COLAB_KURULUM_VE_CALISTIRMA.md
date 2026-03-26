# Colab Kurulum ve Calistirma Adimlari

1. Colab ac, `Runtime -> Change runtime type -> T4 GPU` sec.
2. Repo'yu cek:

```bash
%cd /content
!rm -rf prompt-injection-defense
!git clone https://github.com/ABerkeBilgin/prompt-injection-defense.git
%cd /content/prompt-injection-defense
```

3. Paket kur:

```bash
!pip install -r requirements.txt
!pip install -U transformers==4.57.1 accelerate sentencepiece huggingface_hub
```

4. Hugging Face login:

```bash
!hf auth login
!hf auth whoami
```

5. Dataset uret:

```bash
!python src/data/build_dataset.py --output data/processed/eval_set.jsonl
```

6. Baseline inference (savunma kapali):

```bash
%cd /content/prompt-injection-defense/src/model
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --no-defense --dataset /content/prompt-injection-defense/data/processed/eval_set.jsonl --output /content/prompt-injection-defense/data/processed/predictions_baseline.jsonl
```

7. Defense inference (savunma acik):

```bash
!python run_inference.py -m Qwen/Qwen2.5-7B-Instruct --dataset /content/prompt-injection-defense/data/processed/eval_set.jsonl --output /content/prompt-injection-defense/data/processed/predictions_defense.jsonl
```

8. Metrikleri hesapla:

```bash
%cd /content/prompt-injection-defense
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_set.jsonl --predictions data/processed/predictions_baseline.jsonl --output docs/raporlar/metrics_baseline.json
!python src/evaluation/compute_metrics.py --dataset data/processed/eval_set.jsonl --predictions data/processed/predictions_defense.jsonl --output docs/raporlar/metrics_defense.json
```

9. Sonuclari gor:

```bash
!cat docs/raporlar/metrics_baseline.json
!cat docs/raporlar/metrics_defense.json
```

## Kisa Notlar

- Llama icin ayrica HF model erisim onayi gerekir; Qwen ile bu engel yok.
- Colab runtime reset olursa 2. adimdan itibaren tekrar etmen gerekir.
