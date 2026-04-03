# Benchmark Veri Notlari

Bu repo artik resmi benchmark akisinin first-party kopyasini kullanir.

## Resmi harness
- instruction ve utility benchmark scriptleri: `src/official_stacks/meta_secalign/test.py`
- lm-eval utility scripti: `src/official_stacks/meta_secalign/test_lm_eval.py`
- agentic benchmark scriptleri: `src/official_stacks/meta_secalign/test_injecagent.py` ve `src/official_stacks/meta_secalign/test_agentdojo.py`
- orchestration: `scripts/run_official_eval.py`

## DefensiveToken entegrasyonu
- trusted instruction rolu: `system`
- untrusted data rolu: `user`
- savunmali tokenizer/model kullanildiginda official harness `add_defensive_tokens=True` ile tokenlari prompt basina ekler

## Konfig dosyalari
- `src/official_stacks/meta_secalign/data/openai_configs.yaml`
- `src/official_stacks/meta_secalign/data/gemini_configs.yaml`
