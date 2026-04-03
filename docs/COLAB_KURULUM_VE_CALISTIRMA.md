# Colab Kurulum ve Calistirma

Bu belge guncel tek-repo Colab akisidir. Official `Meta_SecAlign/setup.py` kullanilmaz; cunku Colab'da buyuk gated model indirmelerine ve disk sorunlarina yol acar.

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

## 3. Official benchmark data bootstrap
Paper-aligned evaluation icin gerekli benchmark veri dosyalari repo icinde gelmez. Bu repodaki `scripts/bootstrap_official_data.py` gerekli public benchmark dosyalarini indirip `src/official_stacks/meta_secalign/data/` altina yazar.

```bash
!python scripts/bootstrap_official_data.py
```

Data klasoru doldu mu kontrol et:
```bash
!find src/official_stacks/meta_secalign/data -maxdepth 1 -type f | sort
```

Beklenen instruction ve agentic veri dosyalari:
- `davinci_003_outputs.json`
- `SEP_dataset_test.json`
- `CySE_prompt_injections.json`
- `TaskTracker_dataset_test.json`
- `tools.json`
- `attacker_simulated_responses.json`
- `test_cases_dh_base.json`
- `test_cases_ds_base.json`
- `test_cases_dh_enhanced.json`
- `test_cases_ds_enhanced.json`

## 4. API config
OpenAI veya Gemini kullanacaksan ilgili config dosyalarini doldur.

OpenAI ornegi:
```bash
%%writefile src/official_stacks/meta_secalign/data/openai_configs.yaml
default:
  - client_class: "openai.OpenAI"
    api_key: "BURAYA_OPENAI_API_KEY"
```

Not:
- `official_api` judge icin bu dosya zorunludur.
- Normal OpenAI kullaniyorsan ekstra Hugging Face login gerekmez.
- Sadece gated HF modellerine erismen gerekiyorsa `huggingface_hub` login kullan.
- `client_class` alani API key degil, istemci sinifidir; `openai.OpenAI` olarak kalmalidir.

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
Sadece defensive modeli test etmek icin instruction tarafi:
```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --defense-only --suite instruction --judge official_api
```

Agentic tarafi:
```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --defense-only --suite agentic --judge official_api
```

## 8. Agentic Notlari
Colab debug sirasinda agentic tarafta iki zorunlu uyumluluk duzeltmesi ortaya cikti:
- `test_injecagent.py` veri dosyalarini eski `data/...` relative path ile aciyordu. Bu repo benchmark dosyalarini `src/official_stacks/meta_secalign/data/` altinda tuttugu icin kod `DATA_DIR` tabanli hale getirildi.
- `test_injecagent.py` icinde `resolve_base_model_path(...)` kullanimi vardi ama import eksikti. Bu nedenle Colab'da `NameError` olusuyordu.

Bu degisiklikler benchmark metodolojisini degistirmez. Yalnizca vendorized official stack ile bu repo klasor yapisini uyumlu hale getirir.

Agentic akisi sessiz gorunurse en olasi nedenler:
- vLLM model acilisi zaman aliyor olabilir.
- `test_injecagent` ve `test_agentdojo` alt komutlari ilk logu gec veriyor olabilir.
- `python -u` ile calistirmak ciktiyi daha gorunur yapar.

Gerekirse alt benchmarklari tek tek calistir:
```bash
!python -u -m src.official_stacks.meta_secalign.test_injecagent --defense sandwich --lora_alpha 8.0 -m /content/prompt-injection-defense/src/official_stacks/defensivetoken/Qwen/Qwen2.5-7B-Instruct-5DefensiveTokens --judge_model o4-mini-2025-04-16
!python -u -m src.official_stacks.meta_secalign.test_agentdojo -a none -d repeat_user_prompt --lora_alpha 8.0 -m /content/prompt-injection-defense/src/official_stacks/defensivetoken/Qwen/Qwen2.5-7B-Instruct-5DefensiveTokens --judge_model o4-mini-2025-04-16
!python -u -m src.official_stacks.meta_secalign.test_agentdojo -a important_instructions -d repeat_user_prompt --lora_alpha 8.0 -m /content/prompt-injection-defense/src/official_stacks/defensivetoken/Qwen/Qwen2.5-7B-Instruct-5DefensiveTokens --judge_model o4-mini-2025-04-16
```

## 9. AlpacaEval Notu
AlpacaFarm utility degerlendirmesi alpaca_eval uzerinden calisir. Varsayilan evaluator modeli `gpt-4-1106-preview` her OpenAI project'inde erisilebilir olmayabilir.

Bu repo `--alpacaeval_judge_model` ile runtime annotator config uretir ve evaluator modelini degistirebilir. Paper evaluator modeli yerine farkli bir judge modeli kullanilirsa utility sonucu adapted reproduction olarak yorumlanmalidir.

## 10. Notlar
- `run_official_eval.py` eksik official data dosyalarini preflight asamasinda kontrol eder.
- Judge varsayilani `o4-mini-2025-04-16` olarak ayarlidir.
- Judge tarafinda rate-limit riskini azaltmak icin istemci batch boyutu dusurulmustur.
- `--defense-only` kullandiginda script sadece defensive path ile ilerler; baseline cagrisi uretmez.
- `utility` veya `all` suite icin `SEP_dataset_test_Meta-Llama-3-8B-Instruct.json` ayrica gereklidir. Bu dosya bootstrap scripti tarafindan uretilmez veya indirilmez.
- Bu belge instruction ve agentic benchmarklarini Colab'da gereksiz official model indirmesi olmadan calistirmaya odaklanir.
