# Colab Runbook

Bu rehber, projeyi Google Colab notebook hucrelerinde dogrudan calistirmak icin hazirlanmistir.

## Varsayimlar

- Repo Colab icinde `/content/prompt-injection-defense` altina klonlanacak.
- Klonlama `tez_salih` branch'i uzerinden yapilacak.
- Hedef model yalnizca `Qwen/Qwen2.5-7B-Instruct`.
- Judge modeli yalnizca `gpt-4o-mini` (AlpacaEval2 protokolu; referans model GPT-4 Turbo, 805 ornek).
- Calistirilacak deneyler: `baseline`, `defense`, `utility`.

## Deney Akisi Genel Bakis

```
baseline  →  defense  →  utility
   ↓             ↓           ↓
win_rate      win_rate    win_rate   ← karsilastir
  asr           asr         asr     ← sabit kalmali
```

`utility` modu: DefensiveToken uzerine eklenen 5 adet UtilityToken embedding vektoru,
model agirliklarina dokunmadan win-rate'i geri kazandirmak icin egitilmistir.

---

## 1. Repo'yu Klonla

```python
%cd /content
!git clone --branch tez_salih https://github.com/ABerkeBilgin/prompt-injection-defense.git prompt-injection-defense
%cd /content/prompt-injection-defense
!pwd
```

Repo zaten varsa:

```python
%cd /content/prompt-injection-defense
!git pull
!git branch --show-current
```

## 2. Paketleri Kur

```python
%cd /content/prompt-injection-defense
!pip install -r requirements.txt
```

## 3. OpenAI Config Dosyasini Doldur

Dosya yolu: `src/official_stacks/meta_secalign/data/openai_configs.yaml`

```python
%%writefile /content/prompt-injection-defense/src/official_stacks/meta_secalign/data/openai_configs.yaml
default:
  - client_class: "openai.OpenAI"
    api_key: "YOUR_OPENAI_API_KEY"
    model: "gpt-4o-mini"
    min_interval_seconds: 1.5
    max_retries: 8
    backoff_seconds: 10.0
```

## 4. Alpaca Verisini Indir

```python
%cd /content/prompt-injection-defense
!python scripts/bootstrap_qwen_alpaca_data.py
```

Olusacak dosya:

- `src/official_stacks/meta_secalign/data/davinci_003_outputs.json`

Kontrol:

```python
!ls -lh src/official_stacks/meta_secalign/data/
```

## 5. Dry-Run ile Yol Kontrolu

```python
%cd /content/prompt-injection-defense
!python scripts/run_qwen_alpaca_eval.py --mode baseline --dry-run
!python scripts/run_qwen_alpaca_eval.py --mode defense  --dry-run
!python scripts/run_qwen_alpaca_eval.py --mode utility  --dry-run
```

Beklenen ciktilar:

- `baseline` → model: `Qwen/Qwen2.5-7B-Instruct`
- `defense`  → model: `.../Qwen2.5-7B-Instruct-5DefensiveTokens`
- `utility`  → model: `.../Qwen2.5-7B-Instruct-5DefensiveTokens-5UtilityTokens`

## 6. Defended Modeli Hazirla

```python
%cd /content/prompt-injection-defense
!python src/model/setup.py
```

Olusacak klasor:

- `src/official_stacks/defensivetoken/Qwen/Qwen2.5-7B-Instruct-5DefensiveTokens/`

Kontrol:

```python
!ls -lh src/official_stacks/defensivetoken/Qwen/
```

## 7. Utility Token Egitimi (ayri notebook)

> **Bu adim `notebooks/train_utility_tokens.ipynb` icerisinde yapilir.**
> Ayni Colab oturumunda yeni bir sekme acip notebook'u calistirin.

Notebook icinde yapilacaklar (otomatik):

1. `defensivetokens.json` + `davinci_003_outputs.json` yuklenir
2. Qwen2.5-7B + 5 DefensiveToken frozen yuklenir
3. 5 adet `[UtilityToken]` embedding vektoru SFT loss ile egitilir (3 epoch, A100 ~30 dk)
4. `utilitytokens.json` indirilir

Indirilen dosyayi projeye kopyalayin:

```python
# utilitytokens.json'u Drive'a kaydettiyseniz:
import shutil
shutil.copy(
    "/content/drive/MyDrive/utilitytokens.json",
    "/content/prompt-injection-defense/src/official_stacks/utilitytokens/utilitytokens.json"
)
```

Ya da dosyayi Colab'a manuel yukleyin:

```python
from google.colab import files
uploaded = files.upload()  # utilitytokens.json secin
import shutil
shutil.move(
    "utilitytokens.json",
    "src/official_stacks/utilitytokens/utilitytokens.json"
)
```

Kontrol:

```python
%cd /content/prompt-injection-defense
!ls -lh src/official_stacks/utilitytokens/
```

## 8. Utility Modelini Hazirla

UtilityToken'lari defended model uzerine uygular, yeni model dizinini olusturur.

```python
%cd /content/prompt-injection-defense
!python - <<'PY'
import sys
sys.path.insert(0, ".")
from src.official_stacks.utilitytokens.core import prepare_utility_model

defended_path = "src/official_stacks/defensivetoken/Qwen/Qwen2.5-7B-Instruct-5DefensiveTokens"
out = prepare_utility_model(defended_path)
print(f"Utility model hazir: {out}")
PY
```

Olusacak klasor:

- `src/official_stacks/defensivetoken/Qwen/Qwen2.5-7B-Instruct-5DefensiveTokens-5UtilityTokens/`

## 9. Baseline Evaluation

```python
%cd /content/prompt-injection-defense
!python scripts/run_qwen_alpaca_eval.py --mode baseline --skip-gcg
```

Rapor: `docs/raporlar/qwen_alpaca/baseline.json`

## 10. Defense Evaluation

```python
%cd /content/prompt-injection-defense
!python scripts/run_qwen_alpaca_eval.py --mode defense --skip-gcg
```

GCG dahil:

```python
!python scripts/run_qwen_alpaca_eval.py --mode defense
```

Rapor: `docs/raporlar/qwen_alpaca/defense.json`

## 11. Utility Evaluation

```python
%cd /content/prompt-injection-defense
!python scripts/run_qwen_alpaca_eval.py --mode utility --skip-gcg
```

GCG dahil:

```python
!python scripts/run_qwen_alpaca_eval.py --mode utility
```

Rapor: `docs/raporlar/qwen_alpaca/utility.json`

## 12. Sonuclari Karsilastir

```python
%cd /content/prompt-injection-defense
!python - <<'PY'
import json
from pathlib import Path

print(f"{'Mode':<10} {'win_rate':>10} {'asr':>8}")
print("-" * 32)
for name in ["baseline", "defense", "utility"]:
    path = Path("docs/raporlar/qwen_alpaca") / f"{name}.json"
    if not path.exists():
        print(f"{name:<10}  (henuz calistirilmadi)")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    m = data["metrics"]
    print(f"{name:<10} {m['win_rate']:>10.4f} {m['asr']:>8.4f}")
PY
```

Beklenen sonuc:

```
Mode        win_rate      asr
--------------------------------
baseline      ~0.50     ~0.90   ← savunmasiz, saldiri kolay basarili
defense       ~0.35     ~0.05   ← savunmali ama win-rate duser
utility       ~0.45     ~0.05   ← win-rate geri kazanilir, asr korunur
```

---

## Tek Parca Colab Blogu (baseline + defense)

```python
%cd /content
!git clone --branch tez_salih https://github.com/ABerkeBilgin/prompt-injection-defense.git prompt-injection-defense
%cd /content/prompt-injection-defense
!pip install -r requirements.txt

%%writefile src/official_stacks/meta_secalign/data/openai_configs.yaml
default:
  - client_class: "openai.OpenAI"
    api_key: "YOUR_OPENAI_API_KEY"
    model: "gpt-4o-mini"
    min_interval_seconds: 1.5
    max_retries: 8
    backoff_seconds: 10.0

!python scripts/bootstrap_qwen_alpaca_data.py
!python src/model/setup.py
!python scripts/run_qwen_alpaca_eval.py --mode baseline --skip-gcg
!python scripts/run_qwen_alpaca_eval.py --mode defense  --skip-gcg
```

Utility icin adimlar 7-11'i ayri olarak izleyin (egitim notebook'u gerektirir).
