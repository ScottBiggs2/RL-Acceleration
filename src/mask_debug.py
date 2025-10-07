import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import math

def frac_finite(t):
    if not torch.is_tensor(t) or t.numel() == 0:
        return 1.0
    finite = torch.isfinite(t)
    return finite.sum().item() / finite.numel()

def summarize_tensor(name, t):
    try:
        mean = t.float().mean().item()
        std = t.float().std().item()
    except Exception:
        mean, std = float("nan"), float("nan")
    return {
        "shape": tuple(t.shape),
        "frac_finite": round(frac_finite(t), 4),
        "mean": round(mean, 4),
        "std": round(std, 4),
    }

def sanitize_output(output):
    """Return a tensor even if the module returns a tuple or list."""
    if torch.is_tensor(output):
        return output
    elif isinstance(output, (tuple, list)) and len(output) > 0:
        # choose the first tensor-like element
        for x in output:
            if torch.is_tensor(x):
                return x
    return None

print("[DEBUG] Loading google/gemma-3-270m -> cpu (float32)")
model_id = "google/gemma-3-270m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map="cpu")

inputs = tokenizer(["hello world", "mask test input"], return_tensors="pt", padding=True)

probe_data = {}
hooks = []

def make_hook(name):
    def fn(_, __, output):
        t = sanitize_output(output)
        if t is not None:
            probe_data[name] = summarize_tensor(name, t.detach().cpu())
    return fn

def register_hooks():
    for name, mod in model.named_modules():
        if any(key in name for key in [
            "embed_tokens",
            "self_attn",
            "mlp",
            "norm",
            "layernorm",
            "pre_feedforward",
            "post_attention"
        ]):
            hooks.append(mod.register_forward_hook(make_hook(name)))

def remove_hooks():
    for h in hooks:
        h.remove()
    hooks.clear()

register_hooks()

print("[DEBUG] Running forward pass ...")
with torch.no_grad():
    _ = model(**inputs)

remove_hooks()

# Sort by layer order
sorted_data = dict(sorted(probe_data.items(), key=lambda kv: kv[0]))

print("[DEBUG] Probe summary:")
bad_layers = []
for name, stats in sorted_data.items():
    line = f"  {name:50s} | frac_finite={stats['frac_finite']:.3f}, mean={stats['mean']:.2e}, std={stats['std']:.2e}"
    print(line)
    if stats["frac_finite"] < 0.999:
        bad_layers.append((name, stats))

if bad_layers:
    print("\n[RESULT] Non-finite values detected:")
    for name, stats in bad_layers:
        print(f"  ❌ {name}: {stats}")
else:
    print("\n[RESULT] ✅ All probed tensors are finite!")

with open("outputs/debug_instability.json", "w") as f:
    json.dump(sorted_data, f, indent=2)

print("[DEBUG] Wrote outputs/debug_instability.json")
