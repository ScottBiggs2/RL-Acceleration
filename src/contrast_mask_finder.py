import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path

from data.load_openr1 import load_openr1_subset
from data.load_contrast import load_contrast_subset



# ============ CONFIG ============
MODEL_NAME = "google/gemma-2-2b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("masks_contrast")
OUT_DIR.mkdir(exist_ok=True)
N_SAMPLES = 10
BATCH_SIZE = 2
# ================================

openr1_ds, tokenizer = load_openr1_subset(subset_size=10)
pile_ds, _ = load_contrast_subset("monology/pile-uncopyrighted", subset_size=10)
# fineweb_ds, _ = load_contrast_subset("HuggingFaceFW/fineweb-2M", subset_size=10)

contrast_texts = [r["text"] for r in pile_ds] #+ [r["text"] for r in fineweb_ds]
openr1_texts = [r["prompt"] for r in openr1_ds]


print(f"[DATA] Loaded {len(openr1_texts)} OpenR1 + {len(contrast_texts)} contrast samples")

# --- Model + tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval()

# --- Activation capture ---
activations_open = {}
activations_contrast = {}

def make_hook(name, store):
    def hook(module, inp, out):
        store[name].append(out.detach().to("cpu"))
    return hook

# --- Register hooks on all MLP layers ---
hooks = []
for name, module in model.named_modules():
    if "mlp" in name:
        activations_open[name] = []
        activations_contrast[name] = []
        hooks.append(module.register_forward_hook(make_hook(name, activations_open)))

# --- Tokenize + forward pass for OpenR1 ---
print("[RUN] Collecting OpenR1 activations...")
for txt in tqdm(openr1):
    inputs = tokenizer(txt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        _ = model(**inputs)

# --- Swap hooks to record contrast data ---
for h in hooks:
    h.remove()

hooks = []
for name, module in model.named_modules():
    if "mlp" in name:
        hooks.append(module.register_forward_hook(make_hook(name, activations_contrast)))

print("[RUN] Collecting contrast activations...")
for txt in tqdm(contrast_texts):
    inputs = tokenizer(txt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        _ = model(**inputs)

for h in hooks:
    h.remove()

# --- Compute masks based on activation contrasts ---
mask_stats = {}
for layer in activations_open.keys():
    open_cat = torch.cat(activations_open[layer], dim=0).float()
    contrast_cat = torch.cat(activations_contrast[layer], dim=0).float()

    # Align shapes (truncate to smallest batch)
    min_len = min(open_cat.shape[0], contrast_cat.shape[0])
    open_cat = open_cat[:min_len]
    contrast_cat = contrast_cat[:min_len]

    # Contrastive activation difference
    diff = (open_cat - contrast_cat).pow(2).mean(dim=0)

    # Normalize + threshold
    normed = diff / (diff.max() + 1e-6)
    mask = (normed > 0.8).float()  # keep top ~20% most differing features

    torch.save(mask, OUT_DIR / f"mask_{layer.replace('.', '_')}.pt")

    mask_stats[layer] = {
        "mean_diff": diff.mean().item(),
        "max_diff": diff.max().item(),
        "mask_nonzero": int(mask.sum().item()),
        "mask_shape": list(mask.shape),
    }

# --- Save summary ---
with open(OUT_DIR / "mask_stats.json", "w") as f:
    json.dump(mask_stats, f, indent=2)

print(f"[DONE] Saved {len(mask_stats)} masks to {OUT_DIR}")
