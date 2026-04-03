# Colab Kurulum ve Calistirma

Bu belge artik tek repo akisini hedefler. Ayri clone veya notebook-ici hotfix gerektirmemelidir.

## 1. Repo
```bash
%cd /content
!rm -rf prompt-injection-defense
!git clone -b salihv2 --single-branch https://github.com/ABerkeBilgin/prompt-injection-defense.git
%cd /content/prompt-injection-defense
%env PYTHONPATH=/content/prompt-injection-defense
```

## 2. Paketler
```bash
!pip install -U pip
!pip uninstall -y torch torchvision torchaudio
!pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
!pip install -U transformers==4.57.1 accelerate sentencepiece huggingface_hub vllm
```

Runtime yeniden baslatma istenirse yeniden baslat ve su iki satiri tekrar calistir:
```bash
%cd /content/prompt-injection-defense
%env PYTHONPATH=/content/prompt-injection-defense
```

## 3. Official data bootstrap
Paper-aligned evaluation icin gerekli benchmark veri dosyalari repo icinde gelmez. Resmi Meta_SecAlign `setup.py` akisi ile alinmalidir.

```bash
!python scripts/bootstrap_official_data.py
```

Data klasoru doldu mu kontrol et:
```bash
!find src/official_stacks/meta_secalign/data -maxdepth 1 -type f | sort
```

## 4. API config
OpenAI veya Gemini kullanacaksan ilgili config dosyalarini doldur.

OpenAI ornegi:
```bash
%%writefile src/official_stacks/meta_secalign/data/openai_configs.yaml
default:
  - client_class: "openai.OpenAI"
    api_key: "BURAYA_OPENAI_API_KEY"
```

## 5. DefensiveToken modelini olustur
```bash
!python src/model/setup.py Qwen/Qwen2.5-7B-Instruct
```

Beklenen artifact:
- `src/official_stacks/defensivetoken/Qwen/Qwen2.5-7B-Instruct-5DefensiveTokens`

## 6. Syntax ve dry-run kontrolu
```bash
!python -m py_compile \
  src/official_stacks/meta_secalign/utils.py \
  src/official_stacks/meta_secalign/test.py \
  src/official_stacks/meta_secalign/test_injecagent.py \
  src/official_stacks/meta_secalign/test_lm_eval.py \
  src/official_stacks/meta_secalign/test_agentdojo.py \
  scripts/run_official_eval.py \
  scripts/bootstrap_official_data.py
```

```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --defense-only --judge local_dev --dry-run
```

## 7. Resmi benchmark
Sadece defensive modeli test etmek icin:
```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --defense-only --suite instruction --judge official_api
```

Agentic taraf:
```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --defense-only --suite agentic --judge official_api
```

## 8. Notlar
- `run_official_eval.py` eksik official data dosyalarini preflight asamasinda kontrol eder.
- Judge varsayilani `o4-mini-2025-04-16` olarak ayarlidir.
- Judge tarafinda rate-limit riskini azaltmak icin istemci batch boyutu dusurulmustur.
- `--defense-only` kullandiginda script sadece defensive path ile ilerler; baseline cagrisi uretmez.
