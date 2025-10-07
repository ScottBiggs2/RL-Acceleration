# src/mask/activation_hooks.py
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
import re
import traceback

class ActivationCollector:
    """
    Collect forward activations from selected modules.

    Usage:
        collector = ActivationCollector()
        layers = collector.register_hooks(model, target="mlp", include_patterns=["mlp", "gate_proj"])
        # run forward passes...
        activations = collector.activations  # dict: name -> list[tensor(cpu)]
        counts = collector.call_counts       # dict: name -> int
        collector.clear_hooks()
    """

    def __init__(self, max_per_layer: Optional[int] = None):
        # activations: layer_name -> list of torch.Tensor (on CPU)
        self.activations: Dict[str, List[torch.Tensor]] = {}
        # call counts: layer_name -> number of times hook fired
        self.call_counts: Dict[str, int] = {}
        # handle list so we can remove hooks
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        # optionally limit how many activations we store per layer
        self.max_per_layer = max_per_layer

    def _pick_tensor(self, o: Any) -> Optional[torch.Tensor]:
        """Given a forward output (tensor/tuple/list/dict/other), return the first tensor found."""
        if torch.is_tensor(o):
            return o
        if isinstance(o, (list, tuple)):
            for el in o:
                t = self._pick_tensor(el)
                if t is not None:
                    return t
        if isinstance(o, dict):
            for v in o.values():
                t = self._pick_tensor(v)
                if t is not None:
                    return t
        return None

    def hook_fn(self, name: str, capture_inputs: bool = False):
        """
        Returns a function suitable as a forward hook.
        If capture_inputs is True, the hook will also capture inputs[0] if tensor-like.
        """
        def fn(module, inputs, output):
            try:
                # increment count
                self.call_counts[name] = self.call_counts.get(name, 0) + 1

                # prefer capturing outputs; fallback to inputs[0] if asked
                tensor = self._pick_tensor(output)
                if tensor is None and capture_inputs:
                    tensor = self._pick_tensor(inputs)

                if tensor is None:
                    # nothing tensor-like to store
                    return

                # detach and move to CPU (avoid blocking autograd in case)
                cpu_tensor = tensor.detach().cpu()

                # append to list for this layer
                lst = self.activations.setdefault(name, [])
                # optionally limit stored per-layer
                if self.max_per_layer is None or len(lst) < self.max_per_layer:
                    lst.append(cpu_tensor)
            except Exception as e:
                # never allow a hook exception to break forward
                print(f"[Hook ERROR] {name}: {e}\n{traceback.format_exc()}")
        return fn

    def register_hooks(
        self,
        model: nn.Module,
        target: str = "mlp",
        include_patterns: Optional[List[str]] = None,
        capture_inputs: bool = False,
    ) -> List[Tuple[str, nn.Module]]:
        """
        Attach forward hooks to selected modules.

        Args:
            model: the model to search
            target: "mlp" to attach to linear layers in MLP-related names,
                    "linear" to attach to all nn.Linear,
                    "all" to attempt hooking many modules (be careful!)
            include_patterns: optional list of substrings or regex to further restrict hook targets
            capture_inputs: whether to fallback to capturing the first input tensor if output is non-tensor

        Returns:
            list of (module_name, module) that were hooked
        """
        hooked = []
        # normalized patterns
        patterns = [p.lower() for p in (include_patterns or [])]

        for name, module in model.named_modules():
            lname = name.lower()

            # decide whether to attach based on `target` and patterns
            attach = False
            if target == "all":
                attach = True
            elif target == "linear":
                attach = isinstance(module, nn.Linear)
            elif target == "mlp":
                # heuristic: hook linear modules that are in MLP/gate/up/down names OR explicitly called 'mlp' in path
                if isinstance(module, nn.Linear) and (
                    ("mlp" in lname) or ("gate" in lname) or ("up_proj" in lname) or ("down_proj" in lname) or ("ffn" in lname)
                ):
                    attach = True

            # further filter by include_patterns if provided
            if attach and patterns:
                # require at least one pattern match
                attach = any((pat in lname) or re.search(pat, lname) for pat in patterns)

            if attach:
                handle = module.register_forward_hook(self.hook_fn(name, capture_inputs=capture_inputs))
                self.handles.append(handle)
                hooked.append((name, module))

        return hooked

    def clear_hooks(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        print("[Hooks] Cleared all registered hooks.")

    def summary(self, top_k: int = 10):
        """
        Return a small diagnostic summary about what was captured.
        """
        keys = list(self.activations.keys())
        total_layers = len(keys)
        print(f"[Collector] captured {total_layers} layers. Sample (first {min(top_k, total_layers)}):")
        for name in keys[:top_k]:
            lst = self.activations.get(name, [])
            cnt = self.call_counts.get(name, 0)
            shapes = [tuple(t.shape) for t in lst[:3]]
            # compute fraction finite for sample (first tensor)
            frac_fin = None
            if len(lst) > 0:
                t0 = lst[0]
                frac_fin = float(torch.isfinite(t0).sum().item()) / t0.numel()
            print(f"  {name}: calls={cnt}, captured={len(lst)}, shapes_sample={shapes}, frac_finite_first={frac_fin}")
        return {"num_layers": total_layers, "examples": keys[:top_k]}

# convenience wrapper
def attach_hooks(model, collector: Optional[ActivationCollector] = None, target: str = "mlp", include_patterns: Optional[List[str]] = None, capture_inputs: bool = False):
    if collector is None:
        collector = ActivationCollector()
    hooked = collector.register_hooks(model, target=target, include_patterns=include_patterns, capture_inputs=capture_inputs)
    print(f"[Hooks] Attached to {len(hooked)} modules (target={target}).")
    return hooked, collector

def clear_hooks(collector: ActivationCollector):
    if collector is not None:
        collector.clear_hooks()
