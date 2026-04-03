# Benchmark Veri Notlari

Bu repo artik resmi benchmark akisinin first-party kopyasini kullanir.

## Resmi harness
- instruction ve utility benchmark scriptleri: `src/official_stacks/meta_secalign/test.py`
- lm-eval utility scripti: `src/official_stacks/meta_secalign/test_lm_eval.py`
- agentic benchmark scriptleri: `src/official_stacks/meta_secalign/test_injecagent.py` ve `src/official_stacks/meta_secalign/test_agentdojo.py`
- orchestration: `scripts/run_official_eval.py`
- data bootstrap: `scripts/bootstrap_official_data.py`

## DefensiveToken entegrasyonu
- trusted instruction rolu: `system`
- untrusted data rolu: `user`
- savunmali tokenizer/model kullanildiginda official harness `add_defensive_tokens=True` ile tokenlari prompt basina ekler

## Colab uyumluluk duzeltmeleri
Colab uzerinde official stacki bu repo klasor yapisiyla calistirirken iki pratik uyumluluk duzeltmesi gerekti:
- `test_injecagent.py` eski `data/...` relative path'lerini kullaniyordu. Bu repo benchmark verilerini `src/official_stacks/meta_secalign/data/` altinda tuttugu icin kod `DATA_DIR` tabanli hale getirildi.
- `test_injecagent.py` icinde `resolve_base_model_path(...)` kullanimina ragmen import eksikti. Bu durum agentic calisma sirasinda `NameError` uretiyordu.

Bu degisiklikler evaluation mantigini degistirmez. Yalnizca vendorized benchmark kodunu repo yerlesimiyle uyumlu hale getirir.

## Veri kapsam notu
- instruction ve agentic benchmarklari icin gerekli public veri dosyalari `scripts/bootstrap_official_data.py` ile indirilebilir.
- `utility` ve tam SEP win-rate reproduksiyonu icin `SEP_dataset_test_Meta-Llama-3-8B-Instruct.json` ayrica gereklidir.
- `scripts/bootstrap_official_data.py` bilincli olarak official `Meta_SecAlign/setup.py` cagirmiyor; cunku o akis Colab'da buyuk gated model indirmelerine ve disk sorunlarina yol acabiliyor.

## AlpacaEval utility notu
- `alpaca_eval` varsayilan olarak eski ve artik erisilemeyebilen `gpt-4-1106-preview` evaluator'una bagli kalabiliyor.
- Bu repo `test.py` icinde runtime annotator config uretip `--alpacaeval_judge_model` ile evaluator modelini degistirebilir.
- Bu degisiklik AlpacaFarm utility calismasini yeni OpenAI project'lerinde tekrar calistirabilmek icindir.
- Repo, AlpacaEval judge config'inde `num_procs: 1` ve dusuk `max_tokens` kullanir; amac TPM rate limitlerini azaltmaktir.
- Paper'daki evaluator modeli yerine farkli bir judge modeli kullanilirsa utility sonucu tam paper reproduction degil, adapted reproduction olarak yorumlanmalidir.

## Konfig dosyalari
- `src/official_stacks/meta_secalign/data/openai_configs.yaml`
- `src/official_stacks/meta_secalign/data/gemini_configs.yaml`
