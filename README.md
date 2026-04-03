# Prompt Injection Defense

Bu repo artik `DefensiveToken` ve `Meta_SecAlign` akisini dogrudan kendi icinde barindirir. Dis repo clone, patch-script ve ayri bagimlilik akisi kaldirilmistir.

## Ana akis
- Savunmali model hazirlama: `python src/model/setup.py Qwen/Qwen2.5-7B-Instruct`
- Resmi benchmark orkestrasyonu: `python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --mode defense --judge official_api`
- Yerel inference: `python src/model/run_inference.py -m Qwen/Qwen2.5-7B-Instruct --output data/processed/predictions_defense.jsonl`

## Dizinler
- `src/official_stacks/defensivetoken`: first-party DefensiveToken varliklari ve setup mantigi
- `src/official_stacks/meta_secalign`: first-party Meta_SecAlign harness ve vendor yardimcilari
- `src/model`: model hazirlama, demo ve yerel inference girisleri
- `src/evaluation`: ek evaluator notlari ve yerel analiz yardimcilari
- `docs`: Colab ve benchmark notlari

## Notlar
- Final benchmark yolu artik bu repo icindeki vendorized stack'tir.
- `openai_configs.yaml` ve `gemini_configs.yaml` dosyalari `src/official_stacks/meta_secalign/data/` altindadir.
- Bu repo icinde build/test otomasyonu bu degisiklikte calistirilmadi.
