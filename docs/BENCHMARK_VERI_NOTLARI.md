# Benchmark Veri Notlari

Bu belge resmi paper-aligned benchmark kaynaklarini ve hangi script ile calistirildiklarini ozetler.

## Resmi kaynak ilke

Bu repo benchmark mantigini yeniden tanimlamaz.
Resmi benchmark dosyalari ve evaluator akisi `third_party/Meta_SecAlign` tarafindan yonetilir.

## Instruction-following benchmarklari

- `AlpacaFarm-Hacked`
  - resmi script: `third_party/Meta_SecAlign/test.py`
  - attack seti: `none`, `ignore`, `completion`, `completion_ignore`
  - resmi utility kolonu: `AlpacaEval2 WinRate`
- `SEP`
  - resmi script: `third_party/Meta_SecAlign/test.py`
  - attack seti: `none`, `ignore`, `ignore_before`
  - utility: resmi reference + AlpacaEval2 prompting yolu
- `TaskTracker`
  - resmi script: `third_party/Meta_SecAlign/test.py`
  - attack seti: `straightforward`
- `CyberSecEval2`
  - resmi script: `third_party/Meta_SecAlign/test.py`
  - attack seti: `straightforward`

## Utility benchmarklari

- `AlpacaEval2`
- `SEP utility`
- `MMLU`
- `MMLU-Pro`
- `BBH`
- `IFEval`
- `GPQA Diamond`

Resmi utility script'i:

- `third_party/Meta_SecAlign/test_lm_eval.py`

## Agentic benchmarklar

- `InjecAgent`
  - resmi script: `third_party/Meta_SecAlign/test_injecagent.py`
- `AgentDojo`
  - resmi script: `third_party/Meta_SecAlign/test_agentdojo.py`

## Bu repodaki rol

Bu repo su gorevleri yapar:
- official submodule bootstrap
- DefensiveToken model hazirlama wrapper'i
- resmi komutlari orkestre etme
- resmi ciktilari JSON rapora normalize etme
- Colab ve tez raporlama akisini duzenleme

Bu repo artik su gorevleri resmi yol olarak yapmaz:
- benchmark datasetlerini kendi adapter'lari ile yeniden kurma
- custom `ASR` / `utility` heuristigi tanimlama
- resmi metrikleri repo icinde yeniden yorumlama

## Legacy notu

`src/data/*` ve `src/evaluation/*` altindaki custom veri/evaluator kodlari korunur, ancak yalnizca `legacy` / debug amacli kabul edilir.
