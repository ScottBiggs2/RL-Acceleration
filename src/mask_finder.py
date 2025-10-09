import torch
import torch.nn as nn
import json
import os
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer


def collect_activation_masks(
    model,
    device,
    input_texts,
    max_length=512,
    out_dir="outputs",
    activation_threshold=50.0,
    percentile_clip=0.99,
    top_k=50,
):
    """
    Runs a forward pass through target (MLP) layers, collects activations, and computes:
      - per-layer binary activation masks
      - per-layer top-K neuron indices
    Saves both as .pt (tensor) and .json for later use.
    """
    os.makedirs(out_dir, exist_ok=True)
    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "mask_summary.json")

    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    activations = {}
    stats = defaultdict(list)

    # =====================================================
    # --- Hook setup ---
    # =====================================================
    def make_hook(name):
        def hook_fn(module, inp, out):
            if not torch.is_tensor(out):
                return
            out = out.detach().to("cpu")
            finite_mask = torch.isfinite(out)
            out_finite = out[finite_mask]
            if out_finite.numel() == 0:
                return

            # Store full activation for mask computation
            activations[name] = out.view(-1, out.shape[-1]).float().cpu()

            # Basic stats
            stats[name].append(
                dict(
                    mean=out_finite.mean().item(),
                    std=out_finite.std().item(),
                    min=out_finite.min().item(),
                    max=out_finite.max().item(),
                    frac_finite=finite_mask.float().mean().item(),
                )
            )

        return hook_fn

    # Attach hooks only to MLP/feedforward layers
    patterns = ["mlp", "feedforward", "ffn"]
    hooks = []
    for name, module in model.named_modules():
        lname = name.lower()
        if any(p in lname for p in patterns) and not any(
            a in lname for a in ["self_attn", "q_proj", "k_proj", "v_proj", "o_proj"]
        ):
            hooks.append(module.register_forward_hook(make_hook(name)))

    print(f"[MASK_FINDER] Attached {len(hooks)} MLP hooks")

    # =====================================================
    # --- Forward pass ---
    # =====================================================
    with torch.no_grad():
        _ = model(**inputs)

    for h in hooks:
        h.remove()

    # =====================================================
    # --- Compute masks ---
    # =====================================================
    masks = {}
    mask_summary = {}

    for name, acts in activations.items():
        if acts.numel() == 0:
            continue

        threshold = min(
            activation_threshold,
            torch.quantile(acts.abs(), percentile_clip).item(),
        )

        # Binary activation mask (bounded)
        mask = (acts.abs() < threshold).float()
        masks[name] = mask

        # Compute per-neuron mean abs activation → ranking
        neuron_scores = acts.abs().mean(dim=0)
        ranked = torch.argsort(neuron_scores, descending=True)
        top_indices = ranked[:top_k].cpu().tolist()

        # Save both tensor + JSON
        torch.save(mask, os.path.join(mask_dir, f"{name.replace('.', '_')}.pt"))
        json_path = os.path.join(mask_dir, f"{name.replace('.', '_')}_mask.json")
        with open(json_path, "w") as f:
            json.dump({"top_k": top_indices}, f, indent=2)

        mask_summary[name] = {
            "threshold": threshold,
            "mean": stats[name][0]["mean"],
            "std": stats[name][0]["std"],
            "frac_finite": stats[name][0]["frac_finite"],
            "mask_keep_ratio": mask.mean().item(),
            "top_k": top_indices,
        }

    # =====================================================
    # --- Write summary ---
    # =====================================================
    with open(summary_path, "w") as f:
        json.dump(mask_summary, f, indent=2)

    print(f"[MASK_FINDER] Saved {len(masks)} masks to {mask_dir}")
    print(f"[MASK_FINDER] Summary written to {summary_path}")
    return mask_summary


if __name__ == "__main__":
    device = torch.device("cpu")
    model_name = "google/gemma-3-270m"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)

    texts = ["This is a simple test sequence to evaluate mask stability."]
    mask_summary = collect_activation_masks(model, device, texts)
