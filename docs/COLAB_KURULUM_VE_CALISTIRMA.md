# Colab Kurulum ve Calistirma Adimlari

Bu belge, `DefensiveToken + Meta_SecAlign` resmi stack'ini Colab uzerinde calistirmak icin duzenlenmistir.

Bu akista:
- `DefensiveToken` savunmali model varyantini uretir
- benchmark ve metrik mantigi `Meta_SecAlign` tarafinda calisir
- bu repo wrapper, bootstrap, raporlama ve Colab orkestrasyon katmanidir

Onemli durust not:
- Bu belge su an `tamamen risksiz ve tek seferde sorunsuz` bir akisi garanti etmez
- En buyuk iki risk:
  - `Meta_SecAlign/setup.py` icindeki gated model/tokenizer erisimleri
  - upstream patch/agentdojo patch uyumsuzluklari
- Final tez tablolari icin hedef yol `official_api`'dir
- `local_dev` final sonuc yolu degil, yalnizca sinirli gelistirme/komut dogrulamasi notasyonudur

## 1. Runtime secimi

Colab:
- `Runtime -> Change runtime type -> GPU`

Onerilen GPU sirasi:
- en hizli: `H100`
- cok iyi: `A100`
- iyi: `L4`
- kabul edilebilir: `T4`

Not:
- Resmi inference/benchmark tarafi `Meta_SecAlign` tarafinda calisir
- 70B modeller bu akisin pratik ilk hedefi degildir

## 2. Paket kurulumu

```bash
!pip install -U pip
!pip install -U uv
!pip uninstall -y torch torchvision torchaudio
!pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
!pip install -U transformers==4.57.1 accelerate sentencepiece huggingface_hub
!pip install -U vllm
```

Kontrol:

```bash
!python -m pip show torch torchvision torchaudio transformers vllm uv
```

## 3. Runtime restart

Eger Colab restart isterse:

1. `Restart runtime`
2. Paketleri kontrol et
3. `/content` altinda repo kalmis mi bak
4. Repo yoksa yeniden klonla
5. `third_party` ve patch katmanini yeniden kur
6. Kaldigin asamadan devam et

Recovery sequence:

```bash
!python -m pip show torch torchvision torchaudio transformers vllm uv
!ls /content
```

Repo yoksa:

```bash
%cd /content
!git clone --branch salih --single-branch https://github.com/ABerkeBilgin/prompt-injection-defense.git
%cd /content/prompt-injection-defense
!python scripts/bootstrap_official_stack.py --clone-missing --apply-patches
```

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

Ilk kontrol:

```bash
!find third_party -maxdepth 2 -type f | sort | head -50
```

Eger `third_party/DefensiveToken` ve `third_party/Meta_SecAlign` gelmediyse:

```bash
!python scripts/bootstrap_official_stack.py --clone-missing
```

Beklenen:
- `third_party/DefensiveToken/setup.py`
- `third_party/Meta_SecAlign/setup.py`
- `third_party/Meta_SecAlign/test.py`
- `third_party/Meta_SecAlign/utils.py`

## 5. Hugging Face login ve erisim kontrolu

Login:

```bash
!hf auth login
!hf auth whoami
```

Kritik gercek:
- `Meta_SecAlign/setup.py` su an gated Llama tokenizer/model erisimi isteyebilir
- sadece login yetmez; ilgili gated repo erisimi de gerekir

Ozellikle kontrol etmeniz gereken erisim:
- `meta-llama/Llama-3.1-8B-Instruct`

Eger bu erisim yoksa:
- `Meta_SecAlign/setup.py` burada bloklanir
- resmi akisin bu adimi devam etmez

Yani bu asamada karar:
- erisim varsa devam edin
- erisim yoksa burada durup erisim bekleyin veya upstream setup'a patch gerektigini not edin

## 6. Resmi stack bootstrap

Bu repo, `DefensiveToken` README'de istenen `Meta_SecAlign/utils.py` patch'ini uygular.

```bash
%cd /content/prompt-injection-defense
!python scripts/bootstrap_official_stack.py --clone-missing --apply-patches
```

State:

```bash
!cat patches/meta_secalign/bootstrap_state.json
```

Beklenen:
- `patched: true`
- veya `patched: false` ama `already_patched: true`

Kod kontrolu:

```bash
!grep -n "add_defensive_tokens=False" third_party/Meta_SecAlign/utils.py
!grep -n '"role": "system"' third_party/Meta_SecAlign/utils.py
!grep -n '"role": "user"' third_party/Meta_SecAlign/utils.py
```

Beklenen:
- trusted instruction `system`
- untrusted data `user`
- `add_defensive_tokens=False`

Eger:
- `patched: false`
- ve `already_patched: false`
ise, patch script'i upstream dosya formatini yakalayamamis demektir. Bu durumda burada durup patch script'ini guncelleyin.

## 7. Meta_SecAlign bagimliliklari

Bu adimi `setup.py`'dan once yapin.

Minimum:

```bash
!pip install wget
```

Onerilen:

```bash
%cd /content/prompt-injection-defense/third_party/Meta_SecAlign
!pip install -r requirements.txt
```

Durust not:
- `requirements.txt` kurmak runtime'i yeniden baslatmanizi gerektirebilir
- bu durumda `Recovery sequence`e donersiniz

## 8. Meta_SecAlign setup

Bu adim ancak su durumda calistirilmali:
- HF login yapildi
- gated Llama erisimi var
- requirements tamamlandi
- bootstrap patch dogrulandi

Sonra:

```bash
%cd /content/prompt-injection-defense/third_party/Meta_SecAlign
!python setup.py
```

Basariliysa kontrol:

```bash
!find data -maxdepth 2 -type f | sort | head -100
```

Durust not:
- `agentdojo.patch does not apply` gibi upstream uyumsuzluklar cikabilir
- bu durumda resmi akisin bu kurulumu upstream surum farki nedeniyle yarim kalir
- bu noktada wrapper'a gecmeden once setup sorununu ayri ele almak gerekir

## 9. Judge config mantigi

Iki mod vardir:

### A. `official_api`

Final tez tablolari icin hedef yol budur.

Gerekli dosya:
- `third_party/Meta_SecAlign/data/openai_configs.yaml`

Opsiyonel:
- `third_party/Meta_SecAlign/data/gemini_configs.yaml`

Kontrol:

```bash
!ls third_party/Meta_SecAlign/data/openai_configs.yaml
```

Yoksa:
- wrapper bilincli olarak hata verir

### B. `local_dev`

Durust tanim:
- Bu tam bir resmi local-judge modu degildir
- Bu repo wrapper seviyesinde bir gelistirme/iskele secenegidir
- Final tez tablosu yolu olarak kabul edilmez

Yani:
- komut uretimi
- wrapper davranisi
- dosya yolu dogrulamasi
icin kullanin

Ama final sonuc icin kullanmayin.

## 10. DefensiveTokens modelini olustur

Savunmasiz kosu icin gerekmez.
Defense icin:

```bash
%cd /content/prompt-injection-defense/src/model
!python setup.py Qwen/Qwen2.5-7B-Instruct
```

Bu wrapper resmi script'i cagirir:
- `third_party/DefensiveToken/setup.py`

Kontrol:

```bash
!find /content/prompt-injection-defense/third_party/DefensiveToken/Qwen -maxdepth 2 -type f | sort
```

## 11. Wrapper dry-run

Gercek benchmark kosusundan once:

```bash
%cd /content/prompt-injection-defense
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --mode baseline --judge local_dev --dry-run --clone-missing
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --mode defense --judge local_dev --dry-run --clone-missing
```

Burada kontrol edeceginiz seyler:
- `Meta_SecAlign/test.py` komutlari dogru mu
- Alpaca / SEP / CySE / TaskTracker gorunuyor mu
- defense model yolu `-5DefensiveTokens` olarak cozuluyor mu
- JSON rapor yolu uretiliyor mu

Bu adim, resmi setup tamamlanmasa bile kismen yararlidir; en azindan wrapper mantigini dogrular.

## 12. Ilk gercek resmi kosu

Bu adima ancak sunlar tamamsa gecin:
- `Meta_SecAlign/setup.py` basariyla bitti
- `openai_configs.yaml` var
- savunmali model uretildi
- dry-run dogru komutlari basti

O zaman:

```bash
%cd /content/prompt-injection-defense
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --mode baseline --judge official_api --clone-missing
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --mode defense --judge official_api --clone-missing
```

## 13. Cikti dogrulamasi

```bash
!find docs/raporlar/official -type f | sort
!cat docs/raporlar/official/Qwen__Qwen2.5-7B-Instruct/instruction/baseline.json
```

JSON icinde beklenen alanlar:
- `model`
- `suite`
- `mode`
- `judge`
- `commands`
- `metrics`
- `artifacts.summary_tsv`

## 14. Summary.tsv dogrulamasi

```bash
!find third_party -name summary.tsv
!cat $(find third_party -name summary.tsv | head -1)
```

Kontrol:
- `attack`
- `ASR/Utility`
- `test_data`

Durust not:
- wrapper su an `summary.tsv` merkezli ilk normalize katmanidir
- upstream artefact formatinda fark olursa burada ek patch gerekir

## 15. Sonraki kosular

Instruction suite oturduktan sonra:

### Utility

```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite utility --mode baseline --judge official_api --clone-missing
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite utility --mode defense --judge official_api --clone-missing
```

### Agentic

```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite agentic --mode baseline --judge official_api --clone-missing
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite agentic --mode defense --judge official_api --clone-missing
```

### Tum suite'ler

```bash
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite all --mode baseline --judge official_api --clone-missing
!python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite all --mode defense --judge official_api --clone-missing
```

## 16. Aggregate tablo dosyasi

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

## 17. Minimum dogrulama checklist

Asagidakiler tamamsa iskelet dogru calisiyor deriz:

1. `third_party` repolari var
2. bootstrap patch dogru uygulandi
3. gated HF erisimi dogrulandi
4. `Meta_SecAlign/setup.py` basariyla bitti
5. `src/model/setup.py` savunmali modeli uretti
6. dry-run dogru komutlari basti
7. ilk resmi kosu JSON rapor uretti
8. `summary.tsv` parse edildi
9. aggregate tablo script'i resmi JSON'lari okuyabildi

## 18. Kisa notlar

- Resmi paper yolu `DefensiveToken + Meta_SecAlign` stack'idir.
- `src/data/build_dataset.py`, `src/evaluation/compute_metrics.py`, `src/model/run_inference.py` resmi yol olarak kullanilmaz.
- Bu eski yol sadece `legacy` / debug amaclidir.
- `local_dev` final tez sonuclari icin uygun degildir.
- Final tablolar icin `official_api` ve gercek upstream setup basarisi zorunlu kabul edilmelidir.
