"""Ablation utilities for layer and attention head masking.

Supports:
- Layer bypass: Replace layer output with input (remove layer's contribution while preserving residual)
- Layer zeroing: Replace layer output with zeros
- Attention head masking: Zero out specific attention head outputs
"""

import torch


class LayerAblation:
    """Ablates transformer layers by registering forward hooks."""

    def __init__(self):
        self.hooks = []

    def ablate_layer(self, model_adapter, layer_type, layer_idx, method="bypass"):
        """Ablate a single layer.

        Args:
            model_adapter: T5Adapter or SarvamAdapter.
            layer_type: 'encoder' or 'decoder'.
            layer_idx: Index of the layer to ablate.
            method: 'bypass' (pass input through, preserving residual) or
                    'zero' (replace output with zeros).

        Returns:
            The registered hook handle.
        """
        module = model_adapter.get_layer_module(layer_type, layer_idx)

        if method == "bypass":
            def hook_fn(module, input, output):
                # Replace layer output with its input (skip the layer's transformation)
                if isinstance(output, tuple):
                    return (input[0],) + output[1:]
                return input[0]
        elif method == "zero":
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
        else:
            raise ValueError(f"Unknown ablation method: {method}")

        hook = module.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        return hook

    def remove_all(self):
        """Remove all registered ablation hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


class AttentionHeadAblation:
    """Ablates individual attention heads by zeroing their output slice."""

    def __init__(self, num_heads):
        self.num_heads = num_heads
        self.hooks = []

    def ablate_head(self, model_adapter, layer_type, layer_idx, head_idx):
        """Zero out a specific attention head's contribution.

        Args:
            model_adapter: T5Adapter or SarvamAdapter.
            layer_type: 'encoder' or 'decoder'.
            layer_idx: Layer index.
            head_idx: Attention head index to ablate.

        Returns:
            The registered hook handle.
        """
        attn_module = model_adapter.get_attention_module(layer_type, layer_idx)
        num_heads = self.num_heads

        def hook_fn(module, input, output):
            attn_output = output[0]  # [batch, seq_len, hidden_dim]
            hidden_dim = attn_output.shape[-1]
            head_dim = hidden_dim // num_heads
            start = head_idx * head_dim
            end = start + head_dim
            modified = attn_output.clone()
            modified[:, :, start:end] = 0
            return (modified,) + output[1:]

        hook = attn_module.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        return hook

    def remove_all(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def run_ablated_inference(model_adapter, prompt, gen_kwargs, extract_emotion_fn):
    """Run inference on a single prompt with current ablation hooks active.

    Args:
        model_adapter: Loaded model adapter with ablation hooks registered.
        prompt: Input prompt string.
        gen_kwargs: Generation keyword arguments.
        extract_emotion_fn: Function to extract emotion from model output.

    Returns:
        dict with 'emotion' and 'raw_output'.
    """
    response, _ = model_adapter.generate(prompt, **gen_kwargs)
    emotion = extract_emotion_fn(response)
    return {"emotion": emotion, "raw_output": response}
