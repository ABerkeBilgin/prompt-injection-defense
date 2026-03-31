# Benchmark Veri Notlari

Bu repo ilk benchmark entegrasyonunda su kaynaklari hedefler:

- AlpacaFarm
- SEP
- CyberSecEval2

DefensiveTokens makalesinde bunlara ek olarak `TaskTracker`, `InjecAgent` ve `AgentDojo` da kullanilir. Bu repo su anda bu uc ek benchmark icin adapter icermemektedir.

## Beklenen ham veri klasorleri

Ham veri dosyalarini manuel olarak su klasorlere yerlestir:

- `data/raw/alpaca_farm/`
- `data/raw/sep/`
- `data/raw/cyberseceval2/`

Adapterlar bu klasorler altinda recursive olarak `.json` ve `.jsonl` dosyalari arar.

Makaledeki desteklenen mevcut uc benchmark icin ham veriyi otomatik indirmek istersen:

```bash
python scripts/fetch_defensivetokens_datasets.py
```

## Beklenen alanlar

### AlpacaFarm

Adapter bu alanlari okumayi dener:
- `instruction`
- `input`
- `output`
- `dataset`
- `datasplit`

### SEP

Adapter su alias alanlardan uygun olanlari secmeye calisir:
- instruction: `instruction`, `instruction_prompt`, `prompt`, `system_prompt`, `task_prompt`
- data: `data`, `data_prompt`, `input`, `content`, `document`, `text`
- injection: `injection`, `probe`, `attack`, `malicious_prompt`, `trigger`
- witness/reference: `witness`, `expected_output`, `reference_output`

### CyberSecEval2

Adapter su alias alanlardan uygun olanlari secmeye calisir:
- instruction: `instruction`, `prompt`, `system_prompt`, `user_instruction`, `question`
- data: `data_with_injection`, `data`, `context`, `prompt_injection`, `input`, `content`
- judge: `judge_question`, `judge_hint`, `success_criteria`

## Lisans ve kullanim notlari

- AlpacaFarm README, verinin arastirma amacli ve `CC BY-NC 4.0` oldugunu belirtir.
- SEP ve CyberSecEval2 verilerini projeye eklemeden once kendi lisans ve kullanim kosullarini kontrol et.
- Bu repo benchmark verisini varsayilan olarak commit etmeye zorlamaz; `data/raw` altina kullanici tarafinda yerlestirilmesi beklenir.
