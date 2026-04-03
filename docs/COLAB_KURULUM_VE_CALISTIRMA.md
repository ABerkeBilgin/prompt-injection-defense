# Colab Kurulum ve Calistirma

Bu belge artik tek repo akisini hedefler. Ayri clone veya bootstrap adimi gerekmiyor.

## 1. Repo
```bash
%cd /content
!rm -rf prompt-injection-defense
!git clone https://github.com/ABerkeBilgin/prompt-injection-defense.git
%cd /content/prompt-injection-defense
```

## 2. Paketler
```bash
!pip install -U pip
!pip uninstall -y torch torchvision torchaudio
!pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
!pip install -U transformers==4.57.1 accelerate sentencepiece huggingface_hub vllm
```

## 3. API config
OpenAI veya Gemini kullanacaksan ilgili config dosyalarini doldur:
- `src/official_stacks/meta_secalign/data/openai_configs.yaml`
- `src/official_stacks/meta_secalign/data/gemini_configs.yaml`

## 4. DefensiveToken modelini olustur
```bash
!python src/model/setup.py Qwen/Qwen2.5-7B-Instruct
```

Beklenen artifact:
- `src/official_stacks/defensivetoken/Qwen/Qwen2.5-7B-Instruct-5DefensiveTokens`

Desteklenen DefensiveToken modelleri:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `tiiuae/Falcon3-7B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

## 5. Dry-run ile komut dogrulama
```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --mode baseline --judge local_dev --dry-run
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --mode defense --judge local_dev --dry-run
```

Sadece defensive path'i dogrulamak istersen:
```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --defense-only --suite instruction --judge local_dev --dry-run
```

## 6. Resmi benchmark
```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --mode baseline --judge official_api
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --mode defense --judge official_api
```

Sadece defensive modeli test edip cikti almak istersen:
```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --defense-only --suite instruction --judge official_api
```

Judge modeli secmek istersen:
```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --defense-only --suite instruction --judge official_api --judge-model o4-mini-2025-04-16
```

Dogrudan test scripti calistirirken judge modeli secmek istersen:
```bash
!python -m src.official_stacks.meta_secalign.test -m o4-mini-2025-04-16 -a ignore --judge_model o4-mini-2025-04-16
```

LM eval tarafinda model artik zorunlu verilir:
```bash
!python -m src.official_stacks.meta_secalign.test_lm_eval -m /content/my-model-path
```

## 7. Notlar
- Recovery sirasinda yeniden bootstrap veya ek patch uygulama adimi yok.
- Tum orchestration artik `src/official_stacks/meta_secalign` icindeki first-party kod ile calisir.
- Final raporlar `docs/raporlar/official/` altina yazilir.
- Test tarafinda judge modeli secilebilir hale geldi.
- DefensiveToken hazirlama tarafinda ise hala sadece bu repoda defensive token vektoru bulunan modeller desteklenir.
- `--defense-only` kullandiginda script sadece defensive path ile ilerler; baseline cagrisi uretmez.
