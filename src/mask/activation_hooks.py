import torch

class ActivationCollector:
    def __init__(self):
        self.activations = {}

    def hook_fn(self, name):
        def fn(module, input, output):
            self.activations[name] = output.detach().cpu()
        return fn

    def register_hooks(self, model, layers):
        hooks = []
        for name, module in layers:
            hook = module.register_forward_hook(self.hook_fn(name))
            hooks.append(hook)
        return hooks
