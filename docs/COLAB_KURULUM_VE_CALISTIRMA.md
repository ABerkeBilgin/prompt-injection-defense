# Colab Kurulum ve Calistirma Adimlari

Bu belge, `DefensiveToken + Meta_SecAlign` resmi stack'i Colab uzerinde calistirmak icin tek dogru sirayi verir.

Bu akista:
- `DefensiveToken` sadece savunmali model varyantini uretir
- benchmark ve metrik mantigi `Meta_SecAlign` tarafinda calisir
- bu repo yalnizca wrapper, bootstrap, raporlama ve Colab orkestrasyon katmanidir

Onemli:
- final tez tablolari icin `official_api` kullan
- `local_dev` sadece gelistirme / ara dogrulama icindir
- eski `src/data/*`, `src/evaluation/*`, `run_inference.py` akisi resmi yol degildir

## 1. Runtime secimi

Colab:
- `Runtime -> Change runtime type -> GPU`

Onerilen GPU sirasi:
- en hizli: `H100`
- cok iyi: `A100`
- iyi: `L4`
- kabul edilebilir: `T4`

Not:
- resmi benchmark inference tarafinda `Meta_SecAlign` vLLM kullanir
- 70B testleri bu belge kapsamina dahil degildir; ilk resmi denemeyi 8B / 7B modellerle yap

## 2. Paket kurulumu

Colab bash/terminal hucrenizde:

```bash
!pip install -U pip
!pip install -U uv
!pip uninstall -y torch torchvision torchaudio
!pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
!pip install -U transformers==4.57.1 accelerate sentencepiece huggingface_hub
!pip install -U vllm
```

Kurulum kontrolu:

```bash
!python -m pip show torch torchvision torchaudio transformers vllm uv
```

Beklenen:
- `transformers 4.57.1`
- `vllm` kurulu
- `torch/torchvision/torchaudio` kurulu

## 3. Runtime restart

Eger Colab runtime restart isterse:

1. `Restart runtime` de
2. repo'yu yeniden klonlamadan once paketleri kontrol et
3. eksik paket varsa sadece kurulum hucrelerini tekrar calistir

Kontrol:

```bash
!python -m pip show torch torchvision torchaudio transformers vllm uv
```

Not:
- restart sonrasi `/content` altindaki repo ve tum uretilen dosyalar kaybolabilir
- paketler bazen kalir, bazen kayar; kontrol etmeden varsayim yapma

## 4. Repo'yu cek

Varsayilan branch:

```bash
%cd /content
!rm -rf prompt-injection-defense
!git clone --recurse-submodules https://github.com/ABerkeBilgin/prompt-injection-defense.git
%cd /content/prompt-injection-defense
```

`salih` branch'i icin:

```bash
%cd /content
!rm -rf prompt-injection-defense
!git clone --branch salih --single-branch --recurse-submodules https://github.com/ABerkeBilgin/prompt-injection-defense.git
%cd /content/prompt-injection-defense
```

Eger submodule'lar eksik geldiyse:

```bash
!git submodule update --init --recursive
```

Ilk kontrol:

```bash
!find third_party -maxdepth 2 -type f | sort | head -50
```

Beklenen:
- `third_party/DefensiveToken/setup.py`
- `third_party/Meta_SecAlign/run_tests.py`
- `third_party/Meta_SecAlign/test.py`
- `third_party/Meta_SecAlign/utils.py`

## 5. Hugging Face login

Model ve gated repo erisimi icin:

```bash
!hf auth login
!hf auth whoami
```

Eger Llama gated repo kullanacaksan:
- Hugging Face hesabinda erisim verilmis olmali

## 6. Resmi stack bootstrap

Bu repo `Meta_SecAlign` uzerine makalede `DefensiveToken` README'de tarif edilen patch'leri uygular.

Calistir:

```bash
!python scripts/bootstrap_official_stack.py --apply-patches
```

State kontrolu:

```bash
!cat patches/meta_secalign/bootstrap_state.json
```

Patch kontrolu:

```bash
!grep -n "add_defensive_tokens=False" third_party/Meta_SecAlign/utils.py
!grep -n '"role": "system"' third_party/Meta_SecAlign/utils.py
!grep -n '"role": "user"' third_party/Meta_SecAlign/utils.py
```

Beklenen:
- trusted instruction `system`
- untrusted data `user`
- `add_defensive_tokens=False`

## 7. Meta_SecAlign veri ve benchmark setup

Bu adim benchmark veri bagimliliklarini indirir:

```bash
%cd /content/prompt-injection-defense/third_party/Meta_SecAlign
!python setup.py
```

Kontrol:

```bash
!find data -maxdepth 2 -type f | sort | head -100
```

Beklenen:
- benchmark veri dosyalari
- test script'lerinin kullandigi resmi data dosyalari

## 8. Judge config mantigi

Iki mod vardir:

### A. Final resmi yol: `official_api`

Final tez tablolari icin bunu kullan.

Gerekli dosya:
- `third_party/Meta_SecAlign/data/openai_configs.yaml`

Opsiyonel:
- `third_party/Meta_SecAlign/data/gemini_configs.yaml`

Kontrol:

```bash
!ls third_party/Meta_SecAlign/data/openai_configs.yaml
```

Bu dosya yoksa `official_api` modunda wrapper bilincli olarak hata verir.

### B. Gecici gelistirme yolu: `local_dev`

Bu mod:
- ara dogrulama
- komut sirasi denetimi
- sinirli denemeler
icin vardir

Final tez tablosu icin kullanma.

Metodoloji notu:
- TODO: final raporlamadan once tum son kosular `official_api` ile tekrar alinacak

## 9. DefensiveTokens modelini olustur

Savunmasiz (`baseline`) kosu icin gerekmez.

Defense kosusu icin:

```bash
%cd /content/prompt-injection-defense/src/model
!python setup.py Qwen/Qwen2.5-7B-Instruct
```

Bu komut artik resmi script'i cagirir:
- `third_party/DefensiveToken/setup.py`

Kontrol:

```bash
!find /content/prompt-injection-defense/third_party/DefensiveToken/Qwen -maxdepth 2 -type f | sort
```

Beklenen klasor:
- `/content/prompt-injection-defense/third_party/DefensiveToken/Qwen/Qwen2.5-7B-Instruct-5DefensiveTokens`

## 10. Calisma klasoru kurali

Wrapper komutlari repo kokunde calistirilir:

```bash
%cd /content/prompt-injection-defense
```

Model setup icin:

```bash
%cd /content/prompt-injection-defense/src/model
```

Benchmark setup icin:

```bash
%cd /content/prompt-injection-defense/third_party/Meta_SecAlign
```

## 11. Wrapper dry-run dogrulamasi

Gercek kosudan once komutlari sadece uretip kontrol et.

### Instruction baseline dry-run

```bash
%cd /content/prompt-injection-defense
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --mode baseline --judge local_dev --dry-run
```

### Instruction defense dry-run

```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --mode defense --judge local_dev --dry-run
```

Dry-run'da kontrol edeceginiz seyler:
- `Meta_SecAlign/test.py` komutlari basiliyor mu
- Alpaca / SEP / CySE / TaskTracker komutlari gorunuyor mu
- defense modunda model yolu `-5DefensiveTokens` olarak cozuluyor mu
- JSON rapor yolu olusuyor mu

## 12. Ilk gercek resmi kosu

Ilk gercek denemeyi en kucuk resmi kapsamla yap:

### Instruction suite baseline

```bash
%cd /content/prompt-injection-defense
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --mode baseline --judge official_api
```

### Instruction suite defense

```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --mode defense --judge official_api
```

Bu komutlar sirasiyla:
- gerekli modeli/cozumu kontrol eder
- `Meta_SecAlign` benchmark komutlarini cagirir
- `summary.tsv` ve ilgili artefact'lari parse eder
- resmi JSON raporu yazar

## 13. VLLM kullanimi ne zaman var

Bu yeni resmi akista `vLLM` su sekilde kullanilir:

- `Meta_SecAlign` tarafindaki open-weight model inference yolu
- benchmark testleri ve utility/instruction kosulari

Bu repo icindeki eski `run_inference.py --engine vllm` yolu:
- sadece legacy / debug yardimcisidir
- resmi tez tablosu icin birincil yol degildir

Yani Colab'da ayrica bu repodan `run_inference.py` calistirmaniz gerekmez.

## 14. Ilk cikti dogrulamasi

Kosudan sonra:

```bash
!find docs/raporlar/official -type f | sort
```

Ornek rapor:

```bash
!cat docs/raporlar/official/Qwen__Qwen2.5-7B-Instruct/instruction/baseline.json
```

JSON icinde sunlar olmali:
- `model`
- `suite`
- `mode`
- `judge`
- `commands`
- `metrics`
- `artifacts.summary_tsv`

## 15. Summary.tsv dogrulamasi

Wrapper parse'inin dogru alanlari okuyup okumadigini kontrol et:

```bash
!find third_party -name summary.tsv
```

Bulunan ilk dosyayi incele:

```bash
!cat $(find third_party -name summary.tsv | head -1)
```

Kontrol edilecek kolonlar:
- `attack`
- `ASR/Utility`
- `test_data`

Not:
- wrapper su an `summary.tsv` merkezli ilk normalize edici katmandir
- upstream log formatinda fark cikarsa burasi ince ayar ister

## 16. Utility suite kosulari

Instruction suite oturduktan sonra utility:

```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite utility --mode baseline --judge official_api
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite utility --mode defense --judge official_api
```

Beklenen utility alanlari:
- `alpacaeval2`
- `sep_win_rate`
- `mmlu`
- `mmlu_pro`
- `bbh`
- `ifeval`
- `gpqa_diamond`

Not:
- `lm_eval` parse tarafi ilk sahada dogrulanacak kisimlardan biridir

## 17. Agentic suite kosulari

Sonra agentic:

```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite agentic --mode baseline --judge official_api
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite agentic --mode defense --judge official_api
```

Beklenen alanlar:
- `injecagent_asr`
- `agentdojo_asr`
- `agentdojo_utility`

## 18. Tum suite'ler

Hepsi tek seferde:

```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite all --mode baseline --judge official_api
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite all --mode defense --judge official_api
```

## 19. Aggregate tablo dosyasi

Olusan resmi JSON'lari tek tabloda topla:

```bash
!python scripts/build_metrics_table.py \
  --inputs \
  docs/raporlar/official/Qwen__Qwen2.5-7B-Instruct/instruction/baseline.json \
  docs/raporlar/official/Qwen__Qwen2.5-7B-Instruct/instruction/defense.json \
  docs/raporlar/official/Qwen__Qwen2.5-7B-Instruct/utility/baseline.json \
  docs/raporlar/official/Qwen__Qwen2.5-7B-Instruct/utility/defense.json \
  docs/raporlar/official/Qwen__Qwen2.5-7B-Instruct/agentic/baseline.json \
  docs/raporlar/official/Qwen__Qwen2.5-7B-Instruct/agentic/defense.json \
  --output docs/raporlar/official/Qwen__Qwen2.5-7B-Instruct/table.json
```

Kontrol:

```bash
!cat docs/raporlar/official/Qwen__Qwen2.5-7B-Instruct/table.json
```

## 20. Minimum dogrulama checklist

Asagidakiler tamamsa iskelet dogru calisiyor deriz:

1. submodule'lar geldi
2. bootstrap patch uygulandi
3. `Meta_SecAlign/setup.py` veri indirdi
4. `src/model/setup.py` resmi savunmali modeli uretti
5. `run_official_eval.py --dry-run` dogru komutlari basti
6. ilk gercek kosu sonunda resmi JSON rapor olustu
7. `summary.tsv` parse edilebildi
8. aggregate tablo script'i resmi JSON'lari okuyabildi

## 21. Kisa notlar

- Resmi paper yolu `DefensiveToken + Meta_SecAlign` stack'idir.
- `src/data/build_dataset.py`, `src/evaluation/compute_metrics.py`, `src/model/run_inference.py` resmi yol olarak kullanilmaz.
- Bu eski yol sadece `legacy` / debug amaclidir.
- `local_dev` judge veya local custom evaluator ile alinmis sonuclar final tez tablosu sayilmaz.
- Final tablolar icin `official_api` zorunlu kabul edilir.
