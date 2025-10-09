import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from difflib import get_close_matches


# ----------------------------
# Utility functions
# ----------------------------

def fuzzy_find_module(model, mask_name):
    """Fuzzy-match a mask name like 'model_layers_3_mlp_up_proj' to a model submodule."""
    target = mask_name.replace("_", ".")
    all_modules = list(dict(model.named_modules()).keys())

    if target in all_modules:
        return target

    # Candidates that share all meaningful tokens
    tokens = [t for t in target.split(".") if t]
    candidates = [m for m in all_modules if all(tok in m for tok in tokens)]

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        close = get_close_matches(target, candidates, n=1)
        if close:
            return close[0]

    # fallback: pick the one with most overlap
    best_match = None
    best_score = -1
    for m in all_modules:
        overlap = sum(tok in m for tok in tokens)
        if overlap > best_score:
            best_score = overlap
            best_match = m
    return best_match


def apply_mask_to_module(module, mask, noise_std=0.0):
    """Wrap module.forward so its output is multiplied by mask (+ optional Gaussian noise)."""
    orig_forward = module.forward

    def masked_forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        if torch.is_tensor(out) and out.shape[-1] == mask.shape[-1]:
            m = mask.to(out.device)
            if noise_std > 0:
                m = m + torch.randn_like(m) * noise_std
            return out * m
        return out

    module.forward = masked_forward
    return module


def load_all_masks(mask_dir, target_substring="mlp"):
    """Load all MLP-related mask tensors, excluding activation functions."""
    masks = []
    for fname in os.listdir(mask_dir):
        if (
            fname.endswith(".pt")
            and target_substring in fname
            and not any(skip in fname for skip in ["act_fn", "activation"])
        ):
            mask = torch.load(os.path.join(mask_dir, fname))
            masks.append((fname.replace(".pt", ""), mask))
    if not masks:
        raise FileNotFoundError(f"No usable MLP masks found in {mask_dir}")
    print(f"[TEST] Loaded {len(masks)} MLP masks.")
    return masks


def compare(model_name, logits_full, logits_masked, logits_masked_noise):
    """Compare logits between full, masked, and masked+noise runs."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Decode top token sequences
    decoded_full = tokenizer.decode(torch.argmax(logits_full[0], dim=-1))
    decoded_masked = tokenizer.decode(torch.argmax(logits_masked[0], dim=-1))
    decoded_masked_noise = tokenizer.decode(torch.argmax(logits_masked_noise[0], dim=-1))

    # Compute quantitative differences
    def diff_metrics(a, b):
        return (
            torch.mean(torch.abs(a - b)).item(),
            torch.max(torch.abs(a - b)).item(),
            torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
        )

    mean_diff_m, max_diff_m, cos_m = diff_metrics(logits_full, logits_masked)
    mean_diff_n, max_diff_n, cos_n = diff_metrics(logits_full, logits_masked_noise)

    print("\n=== Mask Effect Validation ===\n")
    print("[Masked vs Full]")
    print(f"mean_diff={mean_diff_m:.6f} max_diff={max_diff_m:.6f} cos_sim={cos_m:.6f}")
    print("\nSample decoded comparison:")
    print(f"FULL:   {decoded_full[:200]}")
    print(f"MASKED: {decoded_masked[:200]}\n")

    print("[Masked+Noise vs Full]")
    print(f"mean_diff={mean_diff_n:.6f} max_diff={max_diff_n:.6f} cos_sim={cos_n:.6f}")
    print("\nSample decoded comparison:")
    print(f"FULL:           {decoded_full[:200]}")
    print(f"MASKED+NOISE:   {decoded_masked_noise[:200]}")
    print("==============================\n")

    return {
        "masked": (mean_diff_m, max_diff_m, cos_m),
        "masked_noise": (mean_diff_n, max_diff_n, cos_n),
    }


# ----------------------------
# Main test procedure
# ----------------------------

def test_all_mlp_masks(model_name="google/gemma-3-270m", mask_dir="outputs/masks"):
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare input
    text = "The masked model should behave slightly differently than the original."
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # --- Full baseline ---
    with torch.no_grad():
        logits_full = model(**inputs).logits.detach().cpu()

    # --- Load and apply all MLP masks ---
    masks = load_all_masks(mask_dir)
    applied = []
    for mask_name, mask in masks:
        matched_name = fuzzy_find_module(model, mask_name)
        if matched_name is None:
            print(f"[WARN] Could not find module for {mask_name}")
            continue
        target_module = dict(model.named_modules())[matched_name]
        apply_mask_to_module(target_module, mask, noise_std=0.0)
        applied.append(matched_name)

    print(f"[TEST] Applied {len(applied)} masks across MLPs:\n  " + "\n  ".join(applied))

    # --- Run masked ---
    with torch.no_grad():
        logits_masked = model(**inputs).logits.detach().cpu()

    # --- Masked + noise run ---
    model_noise = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    for mask_name, mask in masks:
        matched_name = fuzzy_find_module(model_noise, mask_name)
        if matched_name is None:
            continue
        target_module = dict(model_noise.named_modules())[matched_name]
        apply_mask_to_module(target_module, mask, noise_std=0.001)

    with torch.no_grad():
        logits_masked_noise = model_noise(**inputs).logits.detach().cpu()

    # --- Compare ---
    results = compare(model_name, logits_full, logits_masked, logits_masked_noise)
    return results


if __name__ == "__main__":
    results = test_all_mlp_masks()
