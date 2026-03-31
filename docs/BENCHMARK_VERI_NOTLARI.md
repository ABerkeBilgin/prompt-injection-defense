# Benchmark Veri Notlari

Bu repo, tez odakli makale-benzeri benchmark entegrasyonunda su kaynaklari hedefler:

- AlpacaFarm
- SEP
- CyberSecEval2
- TaskTracker

Ilk refactor dalgasinda `InjecAgent` ve `AgentDojo` desteklenmez.

## Beklenen ham veri klasorleri

Ham veri dosyalari su klasorlerde tutulur:

- `data/raw/alpaca_farm/`
- `data/raw/sep/`
- `data/raw/cyberseceval2/`
- `data/raw/tasktracker/`

Makaledeki desteklenen benchmarklar icin ham veriyi otomatik indirmek istersen:

```bash
python scripts/fetch_defensivetokens_datasets.py
```

## Beklenen alanlar

### AlpacaFarm

Adapter su alanlari kullanir:
- `instruction`
- `input`
- `output`
- `dataset`
- `datasplit`

Security satirlari makale-benzeri Alpaca varyantlari ile uretilir.

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

### TaskTracker

Bu repoda TaskTracker, resmi injection kaynaklari ve tez pipeline'i icin ortak satir semasina uyarlanmis bir adapter dosyasi ile kullanilir.

Adapter su alanlari okumayi dener:
- instruction: `instruction`, `user_prompt`, `primary_task_prompt`, `prompt`
- data: `input`, `text`, `final_text_paragraph`, `data`, `content`
- injection: `injection`, `secondary_task_prompt`, `attack`, `probe`, `trigger`
- reference: `reference_output`, `primary_task_answer`, `expected_output`, `witness`
- judge: `judge_hint`, `judge_question`, `success_criteria`

## Lisans ve kullanim notlari

- AlpacaFarm README, verinin arastirma amacli ve `CC BY-NC 4.0` oldugunu belirtir.
- SEP, CyberSecEval2 ve TaskTracker verilerini projeye eklemeden once kendi lisans ve kullanim kosullarini kontrol et.
- Bu repo benchmark verisini varsayilan olarak commit etmeye zorlamaz; `data/raw` altina runtime sirasinda yerlestirilmesi beklenir.
