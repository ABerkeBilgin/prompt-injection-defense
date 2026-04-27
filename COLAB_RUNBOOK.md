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

## 7. Utility Token Egitimi

Asagidaki hucreleri sirayla calistirin. Her sey ayni Colab oturumunda calisiyor,
ayri bir notebook acmaniza gerek yok.

### 7a. Egitim kutuphanelerini yukle

```python
%cd /content/prompt-injection-defense
!pip install tqdm -q
```

### 7b. Utility token embedding'lerini egit

```python
%cd /content/prompt-injection-defense
import json, math, random, sys, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")

# ── Konfigürasyon ───────────────────────────────────────────────
TARGET_MODEL    = "Qwen/Qwen2.5-7B-Instruct"
N_DEFENSIVE     = 5
N_UTILITY       = 5
MAX_LENGTH      = 512
TRAIN_SAMPLES   = 1000
BATCH_SIZE      = 4
GRAD_ACCUM      = 4
EPOCHS          = 3
LR              = 5e-3
SEED            = 42
DEFENSIVE_JSON  = "src/official_stacks/defensivetoken/defensivetokens.json"
ALPACA_JSON     = "src/official_stacks/meta_secalign/data/davinci_003_outputs.json"
OUTPUT_JSON     = "src/official_stacks/utilitytokens/utilitytokens.json"

random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda")

# ── Dosyaları yükle ─────────────────────────────────────────────
with open(DEFENSIVE_JSON, "r", encoding="utf-8") as f:
    defensive_token_data = json.load(f)
with open(ALPACA_JSON, "r", encoding="utf-8") as f:
    alpaca_data = json.load(f)

# ── Tokenizer + model ───────────────────────────────────────────
defensive_token_names = [f"[DefensiveToken{i}]" for i in range(N_DEFENSIVE)]
utility_token_names   = [f"[UtilityToken{i}]"   for i in range(N_UTILITY)]

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
tokenizer.add_special_tokens(
    {"additional_special_tokens": defensive_token_names + utility_token_names}
)

FULL_PREFIX = "".join(defensive_token_names) + "".join(utility_token_names)
tokenizer.chat_template = (
    "{%- if add_defensive_tokens %}\n"
    "{{- '" + FULL_PREFIX + "' }}\n"
    "{%- endif %}\n"
    "{%- for message in messages %}\n"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] | trim + '\\n\\n<|im_end|>\\n' }}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "{{- '<|im_start|>assistant\\n' }}\n"
    "{%- endif %}\n"
)

defensive_ids = [tokenizer.convert_tokens_to_ids(t) for t in defensive_token_names]
utility_ids   = [tokenizer.convert_tokens_to_ids(t) for t in utility_token_names]

model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
)
model.resize_token_embeddings(len(tokenizer))

# DefensiveToken vektörlerini modele yaz
def_vecs = torch.tensor(defensive_token_data[TARGET_MODEL], dtype=torch.float32)
with torch.no_grad():
    for i, did in enumerate(defensive_ids):
        model.get_input_embeddings().weight[did] = def_vecs[i].to(model.dtype)

model.requires_grad_(False)
model.gradient_checkpointing_enable()
model = model.to(device).eval()

# ── UtilityToken embedding'lerini başlat ────────────────────────
embed_layer = model.get_input_embeddings()
seed_words  = ["helpful","certainly","sure","assist","gladly","absolutely","happy","pleasure"]
with torch.no_grad():
    seed_ids   = [tid for w in seed_words for tid in tokenizer.encode(w, add_special_tokens=False)]
    seed_embeds = embed_layer.weight[list(set(seed_ids))].float().mean(dim=0)

utility_embeddings = nn.Parameter(
    seed_embeds.unsqueeze(0).repeat(N_UTILITY, 1) + 0.01 * torch.randn(N_UTILITY, seed_embeds.shape[0])
)
optimizer = torch.optim.AdamW([utility_embeddings], lr=LR, weight_decay=0.0)

# ── Dataset ─────────────────────────────────────────────────────
class AlpacaDataset(Dataset):
    def __init__(self, rows):
        self.samples = []
        for row in rows:
            instruction = str(row.get("instruction","")).strip()
            user_input  = str(row.get("input","")).strip()
            ref_output  = str(row.get("output","")).strip()
            if not instruction or not ref_output:
                continue
            messages = [{"role":"system","content":instruction}]
            if user_input:
                messages.append({"role":"user","content":user_input})
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_defensive_tokens=True
            )
            full_str  = prompt_str + ref_output + tokenizer.eos_token
            full_ids  = tokenizer(full_str,   add_special_tokens=False, return_tensors="pt").input_ids[0]
            prompt_ids = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt").input_ids[0]
            if len(full_ids) > MAX_LENGTH:
                full_ids = full_ids[:MAX_LENGTH]
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = full_ids.clone()
            labels[:prompt_len] = -100
            if (labels != -100).sum() == 0:
                continue
            self.samples.append({"input_ids": full_ids, "labels": labels})
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    max_len = max(s["input_ids"].shape[0] for s in batch)
    B = len(batch)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    labels    = torch.full((B, max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros(B, max_len, dtype=torch.long)
    for i, s in enumerate(batch):
        L = s["input_ids"].shape[0]
        input_ids[i, :L] = s["input_ids"]
        labels[i, :L]    = s["labels"]
        attn_mask[i, :L] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

random.shuffle(alpaca_data)
dataset    = AlpacaDataset(alpaca_data[:TRAIN_SAMPLES])
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
total_steps = math.ceil(len(dataset) / BATCH_SIZE) * EPOCHS
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LR*0.1)
print(f"Dataset: {len(dataset)} ornek | Toplam adim: {total_steps}")

# ── Egitim dongusu ───────────────────────────────────────────────
def embed_with_utility(embed_layer, input_ids, utility_embeddings, utility_ids):
    with torch.no_grad():
        embeds = embed_layer(input_ids).to(utility_embeddings.dtype)
    for i, uid in enumerate(utility_ids):
        mask   = (input_ids == uid).float().unsqueeze(-1)
        embeds = embeds + mask * utility_embeddings[i] - mask * embeds.detach()
    return embeds

best_loss, best_embeddings = float("inf"), utility_embeddings.data.clone()

for epoch in range(EPOCHS):
    epoch_loss, epoch_steps = 0.0, 0
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch_idx, batch in enumerate(pbar):
        ids   = batch["input_ids"].to(device)
        lbls  = batch["labels"].to(device)
        mask  = batch["attention_mask"].to(device)
        embeds = embed_with_utility(embed_layer, ids, utility_embeddings.to(device), utility_ids)
        loss   = model(inputs_embeds=embeds.to(model.dtype), attention_mask=mask, labels=lbls).loss / GRAD_ACCUM
        loss.backward()
        epoch_loss  += loss.item() * GRAD_ACCUM
        epoch_steps += 1
        if (batch_idx + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_([utility_embeddings], max_norm=1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        pbar.set_postfix(loss=f"{epoch_loss/epoch_steps:.4f}")
    avg = epoch_loss / epoch_steps
    print(f"Epoch {epoch+1} — loss: {avg:.4f}")
    if avg < best_loss:
        best_loss = avg
        best_embeddings = utility_embeddings.data.clone()

print(f"\nEgitim tamamlandi. En iyi loss: {best_loss:.4f}")
```

### 7c. Vektörleri kaydet

```python
%cd /content/prompt-injection-defense
import json
from pathlib import Path

Path("src/official_stacks/utilitytokens").mkdir(parents=True, exist_ok=True)
output = {"Qwen/Qwen2.5-7B-Instruct": best_embeddings.cpu().float().tolist()}
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Kaydedildi: {OUTPUT_JSON}")
print(f"Vektör sayisi: {len(output['Qwen/Qwen2.5-7B-Instruct'])}")
```

### 7d. Modeli bellekten temizle

Sonraki adimlar vLLM ile modeli yeniden yukleyecegi icin GPU bellegini bosalt.

```python
import gc
del model, utility_embeddings, embed_layer, dataloader, dataset
gc.collect()
torch.cuda.empty_cache()
print("GPU bellegi temizlendi.")
```

Kontrol:

```python
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
