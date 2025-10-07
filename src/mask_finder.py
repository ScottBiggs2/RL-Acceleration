# mask_finder.py
import torch
import torch.nn as nn
import json
import os
from collections import defaultdict
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, AutoTokenizer


def collect_module_stats(model, device, input_texts, max_length=512, out_dir="outputs"):
    """
    Run a forward pass and record per-layer activation stats.
    Excludes attention projections; focuses on MLP and norm layers.
    """

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "mask_finder_summary.json")

    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    stats = defaultdict(list)

    def make_hook(name):
        def hook_fn(module, inp, out):
            def safe_stats(tensor):
                if not torch.is_floating_point(tensor):
                    return None
                tensor = tensor.detach()
                numel = tensor.numel()
                num_nan = torch.isnan(tensor).sum().item()
                num_inf = torch.isinf(tensor).sum().item()
                num_finite = numel - num_nan - num_inf
                frac_finite = num_finite / max(1, numel)
                t_finite = tensor[torch.isfinite(tensor)]
                if t_finite.numel() == 0:
                    mean = std = t_min = t_max = None
                else:
                    mean = t_finite.mean().item()
                    std = t_finite.std().item()
                    t_min = t_finite.min().item()
                    t_max = t_finite.max().item()

                return dict(
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    min=t_min,
                    max=t_max,
                    mean=mean,
                    std=std,
                    num_nan=num_nan,
                    num_inf=num_inf,
                    num_finite=num_finite,
                    frac_finite=frac_finite,
                )

            s_in = safe_stats(inp[0]) if isinstance(inp, tuple) and len(inp) > 0 else None
            s_out = safe_stats(out) if isinstance(out, torch.Tensor) else None
            stats[name].append({"phase": "input", "stats": s_in})
            stats[name].append({"phase": "output", "stats": s_out})

        return hook_fn

    # attach hooks only for relevant layers
    patterns = ["mlp", "feedforward", "ffn", "layernorm", "norm"]
    hooks = []
    for name, module in model.named_modules():
        lname = name.lower()
        if any(p in lname for p in patterns) and not any(attn in lname for attn in ["self_attn", "q_proj", "k_proj", "v_proj", "o_proj"]):
            hooks.append(module.register_forward_hook(make_hook(name)))

    print(f"[MASK_FINDER] Attached {len(hooks)} hooks to target modules")

    with torch.no_grad():
        _ = model(**inputs)

    for h in hooks:
        h.remove()

    # identify problem layers
    problem_layers = {}
    for name, recs in stats.items():
        for entry in recs:
            s = entry["stats"]
            if s is None:
                continue
            if s["num_nan"] > 0 or s["num_inf"] > 0 or (s["std"] is not None and s["std"] > 50):
                problem_layers[name] = entry
                break

    summary = {
        "model": model.config._name_or_path,
        "num_hooks": len(hooks),
        "problem_layers": problem_layers,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[MASK_FINDER] Wrote summary to {summary_path}")
    if not problem_layers:
        print("[MASK_FINDER] No instabilities found — all monitored layers stable.")
    else:
        print(f"[MASK_FINDER] {len(problem_layers)} problematic layers identified.")

    return summary


if __name__ == "__main__":
    device = torch.device("cpu")  # safer for debugging
    model_name = "google/gemma-3-270m"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)

    texts = ["This is a simple test sequence to evaluate layer stability."]
    collect_module_stats(model, device, texts)
