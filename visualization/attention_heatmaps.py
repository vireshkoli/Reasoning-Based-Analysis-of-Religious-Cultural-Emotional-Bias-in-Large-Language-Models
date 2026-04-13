"""
Visualization: Attention pattern heatmaps.

Generates:
- Per-layer, per-head attention heatmaps for selected high-bias scenarios
- RTAS bar charts (which heads attend most to religion tokens)
- Attention entropy heatmaps across layers and heads
- Comparison of attention patterns across religions for the same scenario

"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_config import ATTENTION_DIR, TABLES_DIR, PLOTS_DIR

sns.set(style="white")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_rtas_bars(model_name):
    """Bar chart: Top religion-attending heads across layers."""
    summary_path = os.path.join(TABLES_DIR, f"attention_summary_{model_name}.json")
    if not os.path.exists(summary_path):
        print(f"Skipping RTAS bars for {model_name}: no attention summary")
        return

    summary = load_json(summary_path)
    top_rtas = summary.get("top_rtas_heads", [])
    if not top_rtas:
        print(f"Skipping RTAS bars for {model_name}: no RTAS data")
        return

    labels = [f"{h['layer'][-6:]}\nH{int(h['head'])}" for h in top_rtas]
    values = [h["rtas_mean"] for h in top_rtas]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, values, color=sns.color_palette("Reds_r", len(labels)))
    ax.set_title(f"{model_name.upper()} — Top Religion-Attending Heads (RTAS)", fontsize=12)
    ax.set_xlabel("Layer / Head")
    ax.set_ylabel("Mean RTAS")
    ax.tick_params(axis="x", labelsize=8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"rtas_bars_{model_name}.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: rtas_bars_{model_name}.png")


def plot_attention_entropy_heatmap(model_name):
    """Heatmap: mean attention entropy across layers × heads."""
    import pandas as pd

    csv_path = os.path.join(TABLES_DIR, f"attention_analysis_{model_name}.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping entropy heatmap for {model_name}: no CSV")
        return

    df = pd.read_csv(csv_path)

    # Pivot to layers × heads
    pivot = df.pivot_table(index="layer", columns="head", values="entropy_mean")

    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.35)))
    sns.heatmap(pivot, ax=ax, cmap="Blues",
                xticklabels=True,
                yticklabels=[str(l)[-10:] for l in pivot.index],
                cbar_kws={"label": "Mean Attention Entropy"})
    ax.set_title(f"{model_name.upper()} — Attention Entropy per Layer × Head", fontsize=12)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer")
    ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"attention_entropy_heatmap_{model_name}.png"),
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: attention_entropy_heatmap_{model_name}.png")


def plot_single_attention_matrix(attn_matrix, title, token_labels=None,
                                 religion_token_indices=None, ax=None, cmap="Blues"):
    """Plot a single attention matrix [seq_len, seq_len]."""
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(attn_matrix, ax=ax, cmap=cmap, vmin=0, vmax=attn_matrix.max(),
                cbar=True, square=True,
                xticklabels=token_labels if token_labels else False,
                yticklabels=token_labels if token_labels else False)

    # Highlight religion tokens
    if religion_token_indices:
        for idx in religion_token_indices:
            if idx < attn_matrix.shape[1]:
                ax.add_patch(patches.Rectangle(
                    (idx, 0), 1, attn_matrix.shape[0],
                    linewidth=2, edgecolor="red", facecolor="none", alpha=0.7
                ))

    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="both", labelsize=6)

    if created_fig:
        plt.tight_layout()
        return fig


def plot_sample_attention_grid(model_name, scenario_id, religions=None,
                               n_layers_to_show=4, n_heads_to_show=4):
    """Grid of attention maps for a scenario across religions and selected layers/heads."""
    if religions is None:
        religions = ["None", "Hindu", "Muslim", "Sikh", "Christian"]

    os.makedirs(PLOTS_DIR, exist_ok=True)
    attn_dir = os.path.join(ATTENTION_DIR, f"{model_name}_full")
    if not os.path.exists(attn_dir):
        print(f"No full attention matrices available for {model_name}")
        return

    # For each religion, load attention for the given scenario (direct mode)
    import numpy as np

    # Determine available layers to sample
    sample_file = None
    for f in os.listdir(attn_dir):
        if f"s{scenario_id}_rNone_mdirect" in f:
            sample_file = f
            break

    if sample_file is None:
        print(f"No attention data for scenario {scenario_id} in {model_name}")
        return

    data = np.load(os.path.join(attn_dir, sample_file.replace(".npz", ".npz")), allow_pickle=False)
    all_layers = sorted([k for k in data.files if "layer" in k])
    n_available = len(all_layers)
    layer_step = max(1, n_available // n_layers_to_show)
    selected_layers = [all_layers[i] for i in range(0, n_available, layer_step)][:n_layers_to_show]

    fig, axes = plt.subplots(len(religions), len(selected_layers),
                              figsize=(len(selected_layers) * 3.5, len(religions) * 3.5))

    for row_i, religion in enumerate(religions):
        key = f"s{scenario_id}_r{religion}_mdirect"
        file_path = os.path.join(attn_dir, f"{key}.npz")
        if not os.path.exists(file_path):
            continue

        rel_data = np.load(file_path, allow_pickle=False)

        for col_i, layer_key in enumerate(selected_layers):
            ax = axes[row_i][col_i] if len(religions) > 1 else axes[col_i]
            if layer_key not in rel_data:
                ax.set_visible(False)
                continue

            # Take head 0 attention: [num_heads, seq, seq] → pick head 0
            attn = rel_data[layer_key]  # [num_heads, seq, seq]
            head_attn = attn[0]  # [seq, seq]

            # Clip to first 30 tokens for readability
            clip = min(30, head_attn.shape[0])
            head_attn = head_attn[:clip, :clip]

            sns.heatmap(head_attn, ax=ax, cmap="Blues", cbar=False, square=True,
                        xticklabels=False, yticklabels=False)
            if row_i == 0:
                ax.set_title(layer_key.split("_")[-1].replace("layer", "L"), fontsize=8)
            if col_i == 0:
                ax.set_ylabel(religion, fontsize=8)

    fig.suptitle(f"{model_name.upper()} — Scenario {scenario_id}: Attention (Head 0) by Religion × Layer",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"attention_grid_{model_name}_s{scenario_id}.png"),
                bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Saved: attention_grid_{model_name}_s{scenario_id}.png")


def main():
    for model_name in ["t5", "sarvam"]:
        plot_rtas_bars(model_name)
        plot_attention_entropy_heatmap(model_name)

        # Plot attention grids for top 3 high-bias scenarios
        attn_dir = os.path.join(ATTENTION_DIR, f"{model_name}_full")
        if os.path.exists(attn_dir):
            sample_files = os.listdir(attn_dir)
            seen_scenarios = set()
            for f in sample_files:
                sid = f.split("_")[0].replace("s", "")
                if sid.isdigit() and sid not in seen_scenarios:
                    seen_scenarios.add(sid)
                    if len(seen_scenarios) <= 3:
                        plot_sample_attention_grid(model_name, int(sid))

    print("\nAttention heatmap visualizations complete!")


if __name__ == "__main__":
    main()
