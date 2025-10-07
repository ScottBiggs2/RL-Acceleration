# src/mask/mask_stats.py
import torch
import numpy as np
import os
import json


# =====================================================
# --- Basic neuron ranking helper ---
# =====================================================
def rank_neurons(acts, cav=None):
    """
    acts: tensor [n_samples, hidden_dim]
    cav:  tensor [hidden_dim] or None
    Returns ranked indices and scores (mean |activation| or |projection| per neuron).
    """
    acts = torch.nan_to_num(acts, nan=0.0) # doesnt seem to do anything

    if cav is None:
        scores = acts.abs().mean(dim=0)
    else:
        projections = torch.matmul(acts, cav)
        scores = projections.abs().mean(dim=0)
    ranked = torch.argsort(scores, descending=True)
    return ranked, scores


# =====================================================
# --- Aggregated stats computation and mask saving ---
# =====================================================
def compute_neuron_stats(activations, cavs=None, save_dir="outputs/masks", top_k=50):
    """
    Compute neuron importance statistics across layers and save top-K masks.

    Args:
        activations: dict {layer_name: [tensor(batch, seq, hidden), ...]}
        cavs: dict {layer_name: tensor(hidden_dim)} or None
        save_dir: where to save results
        top_k: number of top neurons to include in mask per layer
    Returns:
        neuron_stats: dict {layer_name: { "ranked": [...], "scores": [...], "top_k": [...] }}
    """
    os.makedirs(save_dir, exist_ok=True)
    neuron_stats = {}

    for layer_name, layer_acts_list in activations.items():

        # if not layer_acts_list:
        #     continue

        # concatenate across batches → [N, hidden_dim]
        acts = torch.nan_to_num( 
                torch.clamp( 
                    torch.cat(
                        [a.flatten(0, -2) if a.ndim > 2 else a for a in layer_acts_list],
                        dim=0
                    ), -1e3, 1e3
                )
            , nan = 0.0)
        # ensure acts is [N, hidden]
        if acts.ndim > 2:
            print(f"DEBUG: acts.ndim = {acts.ndim} > 2, acts = {acts.mean(dim=-2)}, acts shape: {acts.shape}")
            acts = torch.clamp(acts, -1e3, 1e3) # clamp before mean 
            acts = acts.mean(dim=-2)
        


        # optional concept vector
        cav = None
        if cavs and layer_name in cavs:
            cav = cavs[layer_name]

        print(f"[DEBUG] {layer_name}: acts.shape={acts.shape}")
        print(f"[DEBUG] scores.shape={(acts.abs().mean(dim=0)).shape}")
        print(f"[DEBUG] top_10_scores={(acts.abs().mean(dim=0))[:10].tolist()}")

        ranked, scores = rank_neurons(acts, cav)
        top_indices = ranked[:top_k].cpu().tolist()
        neuron_stats[layer_name] = {
            "ranked": ranked.cpu().tolist(),
            "scores": scores.cpu().tolist(),
            "top_k": top_indices,
        }

        # Save mask as JSON for easy reloading later
        mask_path = os.path.join(save_dir, f"{layer_name.replace('.', '_')}_mask.json")
        with open(mask_path, "w") as f:
            json.dump({"top_k": top_indices}, f)

    print(f"[mask_stats] Saved masks for {len(neuron_stats)} layers → {save_dir}")
    return neuron_stats


# =====================================================
# --- Optional: Load previously saved masks ---
# =====================================================
def load_masks(mask_dir):
    masks = {}
    for fname in os.listdir(mask_dir):
        if fname.endswith("_mask.json"):
            with open(os.path.join(mask_dir, fname), "r") as f:
                data = json.load(f)
            layer_name = fname.replace("_mask.json", "").replace("_", ".")
            masks[layer_name] = data["top_k"]
    return masks
