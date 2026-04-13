"""Forward hook utilities for capturing hidden states and attention weights.

Used for targeted analysis and during ablation experiments when we need
fine-grained control over which layers to capture from.
"""

import torch


class HiddenStateCapture:
    """Registers forward hooks on specific transformer layers to capture hidden states."""

    def __init__(self):
        self.captured = {}
        self.hooks = []

    def register(self, model_adapter, layer_type, layer_indices=None):
        """Register hooks on specified layers.

        Args:
            model_adapter: T5Adapter or SarvamAdapter instance.
            layer_type: 'encoder' or 'decoder'.
            layer_indices: List of layer indices to capture. If None, captures all layers.
        """
        num_layers = model_adapter.get_num_layers()[layer_type]
        if layer_indices is None:
            layer_indices = list(range(num_layers))

        for idx in layer_indices:
            module = model_adapter.get_layer_module(layer_type, idx)
            key = f"{layer_type}_layer_{idx}"
            hook = module.register_forward_hook(self._make_hook(key))
            self.hooks.append(hook)

    def _make_hook(self, key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.captured[key] = output[0].detach().cpu()
            else:
                self.captured[key] = output.detach().cpu()
        return hook_fn

    def get_captured(self):
        """Return captured states as dict of numpy arrays."""
        return {k: v.numpy() for k, v in self.captured.items()}

    def clear(self):
        self.captured.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def __del__(self):
        self.remove_hooks()


class AttentionCapture:
    """Registers forward hooks on attention modules to capture attention weights."""

    def __init__(self):
        self.captured = {}
        self.hooks = []

    def register(self, model_adapter, layer_type, layer_indices=None):
        """Register hooks on attention modules of specified layers."""
        num_layers = model_adapter.get_num_layers()[layer_type]
        if layer_indices is None:
            layer_indices = list(range(num_layers))

        for idx in layer_indices:
            attn_module = model_adapter.get_attention_module(layer_type, idx)
            key = f"{layer_type}_attn_{idx}"
            hook = attn_module.register_forward_hook(self._make_hook(key))
            self.hooks.append(hook)

    def _make_hook(self, key):
        def hook_fn(module, input, output):
            # Attention modules typically return (attn_output, attn_weights, ...)
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self.captured[key] = output[1].detach().cpu()
        return hook_fn

    def get_captured(self):
        return {k: v.numpy() for k, v in self.captured.items()}

    def clear(self):
        self.captured.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def __del__(self):
        self.remove_hooks()
