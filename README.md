# Prompt Injection Defense

Bu repo artik tek amacli bir Qwen degerlendirme hattidir.

- Hedef model: `Qwen/Qwen2.5-7B-Instruct`
- Deney modlari: `baseline`, `defense`
- Paper-facing metrikler: `WinRate`, `ASR`, `GCG-ASR`
- Judge modeli: `gpt-4o-mini`

## Kurulum
- AlpacaFarm veri dosyasini indir: `python scripts/bootstrap_qwen_alpaca_data.py`
- OpenAI config ornegini duzenle: `src/official_stacks/meta_secalign/data/openai_configs.yaml`
- Defended model hazirla: `python src/model/setup.py`

## Calistirma
- Baseline: `python scripts/run_qwen_alpaca_eval.py --mode baseline`
- Defense: `python scripts/run_qwen_alpaca_eval.py --mode defense`

## Cikti
- Raporlar: `docs/raporlar/qwen_alpaca/`
- Ham artifactler: hedef model klasoru veya `-log` klasoru altinda

Repo yalnizca bu Qwen + AlpacaFarm + GCG degerlendirme yolunu tasir.