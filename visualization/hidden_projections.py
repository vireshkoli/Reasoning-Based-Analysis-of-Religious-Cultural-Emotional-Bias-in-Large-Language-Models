"""
Visualization: t-SNE projections of hidden states.

Projects hidden representations at early / middle / late layers to 2D using t-SNE,
colored by religion, emotion, and domain. Shows how religion clusters emerge or
dissolve as information propagates through layers.

"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_config import PLOTS_DIR, EMOTION_LIST
from lib.storage import load_pooled_states

sns.set(style="white")

RELIGION_COLORS = {
    "None": "#888888",
    "Hindu": "#e6194b",
    "Muslim": "#3cb44b",
    "Sikh": "#4363d8",
    "Christian": "#f58231",
}

EMOTION_COLORS = {
    "Joy": "#f8d210",
    "Sadness": "#4363d8",
    "Anger": "#e6194b",
    "Fear": "#911eb4",
    "Neutral": "#888888",
    "Unknown": "#aaaaaa",
}

DOMAIN_COLORS = {
    "family": "#e6194b",
    "workspace": "#3cb44b",
    "legal": "#4363d8",
    "general": "#f58231",
}


def run_tsne(states, seed=42):
    """Run t-SNE on [n, dim] array. Returns [n, 2]."""
    scaler = StandardScaler()
    X = scaler.fit_transform(states)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=seed,
                init="pca", learning_rate="auto")
    return tsne.fit_transform(X)


def plot_tsne_grid(model_name, layer_type, states_all, metadata,
                   selected_layer_indices=None, color_by=("religion", "emotion", "domain")):
    """Plot t-SNE projections at selected layers, colored by different attributes.

    Args:
        model_name: 't5' or 'sarvam'.
        layer_type: 'encoder' or 'decoder'.
        states_all: numpy array [num_samples, num_layers, hidden_dim].
        metadata: metadata dict with 'samples'.
        selected_layer_indices: list of layer indices to visualize (default: early, mid, late).
        color_by: tuple of color attributes to plot per row.
    """
    samples = metadata["samples"]
    n_layers = states_all.shape[1]

    if selected_layer_indices is None:
        selected_layer_indices = [0, n_layers // 4, n_layers // 2,
                                  3 * n_layers // 4, n_layers - 1]

    n_rows = len(color_by)
    n_cols = len(selected_layer_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col_i, layer_idx in enumerate(selected_layer_indices):
        print(f"  t-SNE for layer {layer_idx}...")
        layer_states = states_all[:, layer_idx, :]
        coords = run_tsne(layer_states)

        for row_i, attr in enumerate(color_by):
            ax = axes[row_i][col_i]

            if attr == "religion":
                labels = [s.get("religion", "Unknown") for s in samples]
                color_map = RELIGION_COLORS
            elif attr == "emotion":
                labels = [s.get("emotion", "Unknown") for s in samples]
                color_map = EMOTION_COLORS
            elif attr == "domain":
                labels = [s.get("domain", "Unknown") for s in samples]
                color_map = DOMAIN_COLORS
            else:
                labels = ["Unknown"] * len(samples)
                color_map = {}

            unique_labels = sorted(set(labels))
            palette = sns.color_palette("tab10", n_colors=len(unique_labels))
            label_to_color = {
                lb: color_map.get(lb, palette[i])
                for i, lb in enumerate(unique_labels)
            }
            point_colors = [label_to_color[lb] for lb in labels]

            ax.scatter(coords[:, 0], coords[:, 1], c=point_colors,
                       s=12, alpha=0.6, linewidths=0)

            if col_i == 0:
                ax.set_ylabel(f"Color: {attr}", fontsize=9)
            if row_i == 0:
                ax.set_title(f"Layer {layer_idx}\n(rel={layer_idx/(n_layers-1):.2f})", fontsize=9)

            # Legend
            legend_patches = [
                mpatches.Patch(color=label_to_color[lb], label=lb)
                for lb in unique_labels
            ]
            ax.legend(handles=legend_patches, fontsize=6, loc="best",
                      markerscale=0.8, framealpha=0.7)
            ax.axis("off")

    fig.suptitle(f"{model_name.upper()} {layer_type} — t-SNE Hidden State Projections",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, f"tsne_{model_name}_{layer_type}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Saved: tsne_{model_name}_{layer_type}.png")


def main():
    for model_name in ["t5", "sarvam"]:
        try:
            arrays, metadata = load_pooled_states(model_name)
        except FileNotFoundError:
            print(f"No pooled states for {model_name}, skipping t-SNE")
            continue

        for layer_type, states in arrays.items():
            print(f"\nt-SNE: {model_name} {layer_type} ({states.shape})")
            plot_tsne_grid(model_name, layer_type, states, metadata)

    print("\nt-SNE projection visualizations complete!")


if __name__ == "__main__":
    main()
