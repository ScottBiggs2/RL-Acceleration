import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from difflib import get_close_matches


def fuzzy_find_module(model, mask_name):
    """
    Given a mask file name like 'model_layers_3_mlp_act_fn',
    try to find the best-matching module in the model by
    hierarchical token matching.
    """
    target = mask_name.replace("_", ".")
    all_modules = list(dict(model.named_modules()).keys())

    # Exact or near-exact match first
    if target in all_modules:
        return target

    # Otherwise fuzzy-match by longest substring overlap
    candidates = [m for m in all_modules if all(tok in m for tok in target.split(".") if tok)]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # choose closest by edit distance
        close = get_close_matches(target, candidates, n=1)
        if close:
            return close[0]
    else:
        # fallback: pick the one with the most shared subsequence tokens
        best_match = None
        best_score = -1
        for m in all_modules:
            overlap = sum(tok in m for tok in target.split("."))
            if overlap > best_score:
                best_score = overlap
                best_match = m
        return best_match

    return None


def apply_mask_to_module(module, mask, noise_std=0.0):
    """
    Wrap module.forward so its output is multiplied by mask (+ optional noise).
    """
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


def load_mask(mask_dir, target_substring="mlp"):
    """
    Load a representative mask tensor and its corresponding name.
    """
    for fname in os.listdir(mask_dir):
        if fname.endswith(".pt") and target_substring in fname:
            print(f"[TEST] Using mask {fname}")
            mask = torch.load(os.path.join(mask_dir, fname))
            return fname.replace(".pt", ""), mask
    raise FileNotFoundError(f"No mask found containing '{target_substring}' in {mask_dir}")


def compare(model_name, logits_full, logits_masked, logits_masked_noise):
    """
    Compare logits between full, masked, and masked+noise models and print human-readable diagnostics.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Decode top token sequences for human inspection
    decoded_full = tokenizer.decode(torch.argmax(logits_full[0], dim=-1))
    decoded_masked = tokenizer.decode(torch.argmax(logits_masked[0], dim=-1))
    decoded_masked_noise = tokenizer.decode(torch.argmax(logits_masked_noise[0], dim=-1))

    # Compute quantitative differences
    mean_diff_mask = torch.mean(torch.abs(logits_full - logits_masked)).item()
    max_diff_mask = torch.max(torch.abs(logits_full - logits_masked)).item()
    cos_sim_mask = torch.nn.functional.cosine_similarity(
        logits_full.flatten(), logits_masked.flatten(), dim=0
    ).item()

    mean_diff_noise = torch.mean(torch.abs(logits_full - logits_masked_noise)).item()
    max_diff_noise = torch.max(torch.abs(logits_full - logits_masked_noise)).item()
    cos_sim_noise = torch.nn.functional.cosine_similarity(
        logits_full.flatten(), logits_masked_noise.flatten(), dim=0
    ).item()

    print("\n=== Mask Effect Validation ===\n")
    print("[Masked vs Full]")
    print(f"mean_diff={mean_diff_mask:.6f} max_diff={max_diff_mask:.6f} cos_sim={cos_sim_mask:.6f}")
    print("\nSample decoded comparison:")
    print(f"FULL:   {decoded_full[:300]}")
    print(f"MASKED: {decoded_masked[:300]}\n")

    print("[Masked+Noise vs Full]")
    print(f"mean_diff={mean_diff_noise:.6f} max_diff={max_diff_noise:.6f} cos_sim={cos_sim_noise:.6f}")
    print("\nSample decoded comparison:")
    print(f"FULL:           {decoded_full[:300]}")
    print(f"MASKED+NOISE:   {decoded_masked_noise[:300]}")
    print("==============================\n")

    return {
        "masked": (mean_diff_mask, max_diff_mask, cos_sim_mask),
        "masked_noise": (mean_diff_noise, max_diff_noise, cos_sim_noise)
    }


def test_mask_effect(model_name="google/gemma-3-270m", mask_dir="outputs/masks"):
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare input
    text = "The masked model should behave slightly differently than the original."
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # --- Full baseline ---
    with torch.no_grad():
        logits_full = model(**inputs).logits.detach().cpu()

    # --- Load mask ---
    mask_name, mask = load_mask(mask_dir)
    matched_name = fuzzy_find_module(model, mask_name)
    if matched_name is None:
        raise RuntimeError(f"Could not find module matching {mask_name}")
    print(f"[TEST] Applying mask to module: {matched_name}")

    # --- Masked run ---
    target_module = dict(model.named_modules())[matched_name]
    apply_mask_to_module(target_module, mask, noise_std=0.0)
    with torch.no_grad():
        logits_masked = model(**inputs).logits.detach().cpu()

    # --- Masked + noise run ---
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    target_module = dict(model.named_modules())[matched_name]
    apply_mask_to_module(target_module, mask, noise_std=0.001)
    with torch.no_grad():
        logits_masked_noise = model(**inputs).logits.detach().cpu()

    # --- Compare all three ---
    results = compare(model_name, logits_full, logits_masked, logits_masked_noise)
    return results


if __name__ == "__main__":
    results = test_mask_effect()
