"""Model adapters for T5 and Sarvam with unified interface for Phase 3 analysis.

Provides:
- Consistent model loading and inference
- Hidden state and attention weight extraction via forward pass
- Layer module access for hook registration and ablation
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_config import (
    DEVICE, T5_MODEL_NAME, SARVAM_MODEL_NAME,
    T5_NUM_ENCODER_LAYERS, T5_NUM_DECODER_LAYERS, T5_HIDDEN_DIM, T5_NUM_HEADS,
    SARVAM_NUM_LAYERS, SARVAM_HIDDEN_DIM, SARVAM_NUM_HEADS,
)


class ModelAdapter:
    """Base class for model-specific adapters."""

    def __init__(self, device=DEVICE):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_name = None

    def load(self):
        raise NotImplementedError

    def unload(self):
        """Free model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        import gc
        gc.collect()

    def generate(self, prompt, **gen_kwargs):
        """Generate text from a prompt. Returns (response_text, output_ids)."""
        raise NotImplementedError

    def forward_with_states(self, input_ids, **kwargs):
        """Run forward pass returning hidden states and attentions."""
        raise NotImplementedError

    def get_num_layers(self):
        """Return dict of layer counts, e.g. {'encoder': 24, 'decoder': 24}."""
        raise NotImplementedError

    def get_hidden_dim(self):
        raise NotImplementedError

    def get_num_heads(self):
        raise NotImplementedError

    def get_layer_module(self, layer_type, layer_idx):
        """Return the nn.Module for a specific layer."""
        raise NotImplementedError

    def get_attention_module(self, layer_type, layer_idx):
        """Return the attention nn.Module for a specific layer."""
        raise NotImplementedError

    def tokenize(self, text):
        """Tokenize text and return input_ids on device."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        return inputs


class T5Adapter(ModelAdapter):
    """Adapter for google/flan-t5-large (encoder-decoder)."""

    def __init__(self, device=DEVICE):
        super().__init__(device)
        self.model_name = T5_MODEL_NAME

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True
        self.model.eval()

    def generate(self, prompt, **gen_kwargs):
        inputs = self.tokenize(prompt)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, outputs[0]

    def forward_with_states(self, input_ids, decoder_input_ids=None, **kwargs):
        """Teacher-forced forward pass to extract all hidden states and attentions.

        Args:
            input_ids: Encoder input token ids [batch, enc_seq_len].
            decoder_input_ids: Decoder input token ids [batch, dec_seq_len].

        Returns:
            dict with keys:
                encoder_hidden_states: list of [enc_seq_len, hidden_dim] numpy arrays (25 layers)
                decoder_hidden_states: list of [dec_seq_len, hidden_dim] numpy arrays (25 layers)
                encoder_attentions: list of [num_heads, enc_seq_len, enc_seq_len] numpy arrays (24 layers)
                decoder_attentions: list of [num_heads, dec_seq_len, dec_seq_len] numpy arrays (24 layers)
                cross_attentions: list of [num_heads, dec_seq_len, enc_seq_len] numpy arrays (24 layers)
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                output_attentions=True,
            )

        result = {
            "encoder_hidden_states": [
                h[0].detach().cpu().numpy() for h in outputs.encoder_hidden_states
            ],
            "decoder_hidden_states": [
                h[0].detach().cpu().numpy() for h in outputs.decoder_hidden_states
            ],
            "encoder_attentions": [
                a[0].detach().cpu().numpy() for a in outputs.encoder_attentions
            ],
            "decoder_attentions": [
                a[0].detach().cpu().numpy() for a in outputs.decoder_attentions
            ],
            "cross_attentions": [
                a[0].detach().cpu().numpy() for a in outputs.cross_attentions
            ],
        }
        return result

    def get_num_layers(self):
        return {"encoder": T5_NUM_ENCODER_LAYERS, "decoder": T5_NUM_DECODER_LAYERS}

    def get_hidden_dim(self):
        return T5_HIDDEN_DIM

    def get_num_heads(self):
        return T5_NUM_HEADS

    def get_layer_module(self, layer_type, layer_idx):
        if layer_type == "encoder":
            return self.model.encoder.block[layer_idx]
        elif layer_type == "decoder":
            return self.model.decoder.block[layer_idx]
        raise ValueError(f"Unknown layer_type: {layer_type}")

    def get_attention_module(self, layer_type, layer_idx):
        if layer_type == "encoder":
            return self.model.encoder.block[layer_idx].layer[0].SelfAttention
        elif layer_type == "decoder":
            return self.model.decoder.block[layer_idx].layer[0].SelfAttention
        raise ValueError(f"Unknown layer_type: {layer_type}")


class SarvamAdapter(ModelAdapter):
    """Adapter for sarvamai/sarvam-2b (decoder-only, Llama-based)."""

    def __init__(self, device=DEVICE):
        super().__init__(device)
        self.model_name = SARVAM_MODEL_NAME

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        ).to(self.device)
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True
        self.model.eval()

    def generate(self, prompt, **gen_kwargs):
        inputs = self.tokenize(prompt)
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return response, outputs[0]

    def forward_with_states(self, input_ids, **kwargs):
        """Forward pass on full sequence (prompt + generated tokens).

        Args:
            input_ids: Full sequence token ids [batch, seq_len].

        Returns:
            dict with keys:
                decoder_hidden_states: list of [seq_len, hidden_dim] numpy arrays (29 layers)
                decoder_attentions: list of [num_heads, seq_len, seq_len] numpy arrays (28 layers)
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=True,
            )

        result = {
            "decoder_hidden_states": [
                h[0].detach().cpu().float().numpy() for h in outputs.hidden_states
            ],
            "decoder_attentions": [
                a[0].detach().cpu().float().numpy() for a in outputs.attentions
            ],
        }
        return result

    def get_num_layers(self):
        return {"decoder": SARVAM_NUM_LAYERS}

    def get_hidden_dim(self):
        return SARVAM_HIDDEN_DIM

    def get_num_heads(self):
        return SARVAM_NUM_HEADS

    def get_layer_module(self, layer_type, layer_idx):
        if layer_type == "decoder":
            return self.model.model.layers[layer_idx]
        raise ValueError(f"Unknown layer_type for Sarvam: {layer_type}")

    def get_attention_module(self, layer_type, layer_idx):
        if layer_type == "decoder":
            return self.model.model.layers[layer_idx].self_attn
        raise ValueError(f"Unknown layer_type for Sarvam: {layer_type}")
