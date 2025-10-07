# src/debug/diagnose_instability.py
"""
Diagnostic script to find where non-finite activations (NaN / Inf) appear
in a transformer model's forward pass. Robust to tuple/list inputs and
uses safe finite-only statistics (no torch.nanmean / torch.nanstd).
Outputs a JSON summary to outputs/debug_instability.json.
"""
import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OUTPATH = "outputs/debug_instability.json"
os.makedirs(os.path.dirname(OUTPATH), exist_ok=True)

# ----------------------------
# safe stats helpers
# ----------------------------
def tensor_finite_summary(t: torch.Tensor):
    """
    Return a JSON-serializable summary for tensor `t`, computed only over finite values.
    """
    if not torch.is_tensor(t):
        return {"type": str(type(t))}
    # move to CPU, float32 for numerics
    tt = t.detach().cpu().to(torch.float32)
    total = int(tt.numel())
    isfinite = torch.isfinite(tt)
    n_finite = int(isfinite.sum().item())
    n_nan = int(torch.isnan(tt).sum().item())
    n_inf = int(torch.isinf(tt).sum().item())
    frac_finite = float(n_finite / total) if total > 0 else 0.0

    # compute stats over finite entries only; guard if none
    if n_finite > 0:
        finite_vals = tt[isfinite]
        # compute safe scalars
        try:
            v_min = float(finite_vals.min().item())
            v_max = float(finite_vals.max().item())
            v_mean = float(finite_vals.mean().item())
            v_std = float(finite_vals.std().item())
        except Exception as e:
            v_min = v_max = v_mean = v_std = None
    else:
        v_min = v_max = v_mean = v_std = None

    return {
        "shape": tuple(tt.shape),
        "dtype": str(tt.dtype),
        "num_total": total,
        "num_finite": n_finite,
        "num_nan": n_nan,
        "num_inf": n_inf,
        "frac_finite": frac_finite,
        "min_finite": v_min,
        "max_finite": v_max,
        "mean_finite": v_mean,
        "std_finite": v_std,
    }

# ----------------------------
# hook factories
# ----------------------------
def make_probe_hook(mod_name, records):
    """
    Hook that records a summary for inputs and outputs (handling tensors, lists, tuples).
    We record only a compact summary (no tensors).
    """
    def hook(module, inputs, outputs):
        try:
            # inputs may be a single tensor, tuple/list or other types
            in_summaries = []
            if isinstance(inputs, (list, tuple)):
                for idx, inp in enumerate(inputs):
                    if torch.is_tensor(inp):
                        in_summaries.append({"idx": idx, "summary": tensor_finite_summary(inp)})
            elif torch.is_tensor(inputs):
                in_summaries.append({"idx": 0, "summary": tensor_finite_summary(inputs)})
            else:
                # not tensor (e.g., None), skip
                pass

            out_summaries = []
            if isinstance(outputs, (list, tuple)):
                for idx, out in enumerate(outputs):
                    if torch.is_tensor(out):
                        out_summaries.append({"idx": idx, "summary": tensor_finite_summary(out)})
            elif torch.is_tensor(outputs):
                out_summaries.append({"idx": 0, "summary": tensor_finite_summary(outputs)})
            else:
                # could be dict or other; try to extract tensors if it's a dict (rare)
                if isinstance(outputs, dict):
                    for k, v in outputs.items():
                        if torch.is_tensor(v):
                            out_summaries.append({"key": str(k), "summary": tensor_finite_summary(v)})

            records.setdefault(mod_name, []).append({
                "inputs": in_summaries,
                "outputs": out_summaries
            })
        except Exception as e:
            # keep execution going; record the exception
            records.setdefault(mod_name, []).append({"error": f"hook-exception: {repr(e)}"})
    return hook

# ----------------------------
# main diagnostic flow
# ----------------------------
def run_diagnostic(model_name, device, prompt, max_len, attach_patterns):
    device = torch.device(device)
    print(f"[DIAG] Loading {model_name} -> device={device} (float32)")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Prepare a simple tokenized batch (pad to max_len to mimic dataset shapes)
    tok = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    inputs = {k: v.to(device) for k, v in tok.items()}

    # attach hooks to modules whose names match any pattern in attach_patterns
    records = {}
    handles = []
    for name, module in model.named_modules():
        lname = name.lower()
        if any(pat in lname for pat in attach_patterns):
            h = module.register_forward_hook(make_probe_hook(name, records))
            handles.append(h)

    print(f"[DIAG] Attached {len(handles)} hooks (patterns: {attach_patterns})")

    # run forward (single batch)
    with torch.no_grad():
        try:
            outputs = model(**inputs)
            print("[DIAG] forward done; logits shape:", getattr(outputs, "logits", None).shape if outputs is not None else None)
        except Exception as e:
            print("[DIAG] Forward raised exception:", repr(e))

    # remove hooks
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    # assemble a compact summary
    summary = {"model": model_name, "device": str(device), "prompt_len": max_len, "records": {}}
    first_bad = None
    for mod_name, rec_list in records.items():
        # look for any rec in rec_list with non-finite fraction < 1.0
        keep_list = []
        mod_bad = False
        for rec in rec_list:
            # inspect inputs and outputs for finite fraction
            rec_summary = {"inputs": [], "outputs": []}
            for inp in rec.get("inputs", []):
                s = inp["summary"]
                rec_summary["inputs"].append(s)
                if isinstance(s, dict) and s.get("frac_finite", 1.0) < 1.0:
                    mod_bad = True
                    if first_bad is None:
                        first_bad = {"module": mod_name, "phase": "input", "summary": s}
            for out in rec.get("outputs", []):
                s = out["summary"]
                rec_summary["outputs"].append(s)
                if isinstance(s, dict) and s.get("frac_finite", 1.0) < 1.0:
                    mod_bad = True
                    if first_bad is None:
                        first_bad = {"module": mod_name, "phase": "output", "summary": s}
            keep_list.append(rec_summary)
        summary["records"][mod_name] = {
            "num_records": len(rec_list),
            "has_nonfinite": mod_bad,
            "records": keep_list[:3]  # keep at most 3 entries per module to limit JSON size
        }

    summary["first_nonfinite_module"] = first_bad
    # write JSON
    with open(OUTPATH, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[DIAG] Wrote summary to {OUTPATH}")
    if first_bad:
        print("[DIAG] First non-finite seen at module:", first_bad)
    else:
        print("[DIAG] No non-finite values detected in probed modules.")

    return summary

# ----------------------------
# CLI
# ----------------------------
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="google/gemma-3-270m")
    p.add_argument("--device", type=str, default="cpu", help="device for model (cpu or cuda)")
    p.add_argument("--prompt", type=str, default="Hello, world.")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--patterns", type=str, nargs="+", default=["self_attn", "o_proj", "k_proj", "v_proj", "q_proj", "mlp", "norm", "layernorm", "pre_feedforward", "post_attention"])
    args = p.parse_args()
    run_diagnostic(args.model, args.device, args.prompt, args.max_len, args.patterns)

if __name__ == "__main__":
    cli()
