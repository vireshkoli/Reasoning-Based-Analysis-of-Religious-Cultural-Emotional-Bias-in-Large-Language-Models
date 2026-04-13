"""Storage utilities for saving and loading hidden states, attention weights, and results.

Uses numpy .npz files for efficient storage of pooled hidden states.
Supports checkpointing for long-running extraction processes.
"""

import json
import os
import numpy as np
import warnings

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_config import (
    HIDDEN_STATES_DIR, ATTENTION_DIR, CHECKPOINTS_DIR,
)


def _make_key(scenario_id, religion, mode):
    """Create a standardized key for indexing states."""
    return f"s{scenario_id}_r{religion}_m{mode}"


def save_pooled_states(model_name, all_states, metadata):
    """Save pooled hidden states to .npz file.

    Args:
        model_name: 't5' or 'sarvam'.
        all_states: dict mapping sample keys to dicts with layer type keys,
            each containing a numpy array of shape [num_layers+1, hidden_dim].
        metadata: list of dicts with scenario_id, religion, mode, emotion, raw_output.
    """
    os.makedirs(HIDDEN_STATES_DIR, exist_ok=True)

    # Separate by layer type and stack into arrays
    layer_types = set()
    for sample in all_states.values():
        layer_types.update(sample.keys())

    arrays = {}
    for lt in sorted(layer_types):
        keys_ordered = sorted(all_states.keys())
        stacked = np.stack([all_states[k][lt] for k in keys_ordered])
        arrays[lt] = stacked  # [num_samples, num_layers+1, hidden_dim]

    np.savez_compressed(
        os.path.join(HIDDEN_STATES_DIR, f"{model_name}_states.npz"),
        **arrays,
    )

    # Save metadata alongside
    with open(os.path.join(HIDDEN_STATES_DIR, f"{model_name}_metadata.json"), "w") as f:
        json.dump({
            "keys_order": sorted(all_states.keys()),
            "samples": metadata,
        }, f, indent=2)


def load_pooled_states(model_name):
    """Load pooled hidden states and metadata.

    Returns:
        arrays: dict mapping layer type to numpy array [num_samples, num_layers+1, hidden_dim].
        metadata: dict with 'keys_order' and 'samples'.
    """
    data = np.load(
        os.path.join(HIDDEN_STATES_DIR, f"{model_name}_states.npz"),
        allow_pickle=False,
    )
    arrays = {k: data[k] for k in data.files}

    with open(os.path.join(HIDDEN_STATES_DIR, f"{model_name}_metadata.json")) as f:
        metadata = json.load(f)

    return arrays, metadata


def save_attention_summary(model_name, summary_stats):
    """Save attention summary statistics (entropy, RTAS, etc.).

    Args:
        model_name: 't5' or 'sarvam'.
        summary_stats: dict mapping sample keys to per-layer/head stats.
    """
    os.makedirs(ATTENTION_DIR, exist_ok=True)
    with open(os.path.join(ATTENTION_DIR, f"{model_name}_attn_summary.json"), "w") as f:
        json.dump(summary_stats, f, indent=2)


def load_attention_summary(model_name):
    """Load attention summary statistics."""
    path = os.path.join(ATTENTION_DIR, f"{model_name}_attn_summary.json")
    with open(path) as f:
        return json.load(f)


def save_attention_full(model_name, sample_key, attentions):
    """Save full attention matrices for a single sample.

    Args:
        model_name: 't5' or 'sarvam'.
        sample_key: Unique key for this sample.
        attentions: dict mapping layer type to list of numpy arrays.
    """
    subdir = os.path.join(ATTENTION_DIR, f"{model_name}_full")
    os.makedirs(subdir, exist_ok=True)

    arrays = {}
    for lt, attn_list in attentions.items():
        for i, attn in enumerate(attn_list):
            arrays[f"{lt}_layer_{i}"] = attn

    np.savez_compressed(os.path.join(subdir, f"{sample_key}.npz"), **arrays)


def load_attention_full(model_name, sample_key):
    """Load full attention matrices for a single sample."""
    path = os.path.join(ATTENTION_DIR, f"{model_name}_full", f"{sample_key}.npz")
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def save_checkpoint(model_name, stage, completed_ids, partial_data):
    """Save extraction checkpoint.

    Args:
        model_name: 't5' or 'sarvam'.
        stage: Checkpoint stage name (e.g. 'extract').
        completed_ids: List of completed scenario IDs.
        partial_data: Any partial results to preserve.
    """
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINTS_DIR, f"{model_name}_{stage}_checkpoint.json")
    with open(path, "w") as f:
        json.dump({
            "completed_ids": completed_ids,
            "num_completed": len(completed_ids),
        }, f, indent=2)

    # Save partial states separately if provided
    if partial_data:
        np.savez_compressed(
            os.path.join(CHECKPOINTS_DIR, f"{model_name}_{stage}_partial.npz"),
            **{k: v for k, v in partial_data.items() if isinstance(v, np.ndarray)},
        )


def load_checkpoint(model_name, stage):
    """Load extraction checkpoint if it exists.

    Returns:
        (completed_ids, partial_data) or ([], None) if no checkpoint.
    """
    path = os.path.join(CHECKPOINTS_DIR, f"{model_name}_{stage}_checkpoint.json")
    if not os.path.exists(path):
        return [], None

    with open(path) as f:
        info = json.load(f)

    partial_path = os.path.join(CHECKPOINTS_DIR, f"{model_name}_{stage}_partial.npz")
    partial_data = None
    if os.path.exists(partial_path):
        data = np.load(partial_path, allow_pickle=False)
        partial_data = {k: data[k] for k in data.files}

    return info.get("completed_ids", []), partial_data


def validate_states(states_dict, label=""):
    """Check for NaN/Inf in extracted states. Warns if found."""
    for key, arr in states_dict.items():
        if np.isnan(arr).any():
            warnings.warn(f"NaN detected in {label} {key}")
            return False
        if np.isinf(arr).any():
            warnings.warn(f"Inf detected in {label} {key}")
            return False
    return True
