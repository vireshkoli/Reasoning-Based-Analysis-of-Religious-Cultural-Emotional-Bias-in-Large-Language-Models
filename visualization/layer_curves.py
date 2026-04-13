"""
Visualization: Layer-wise sensitivity and probing accuracy curves.

Generates:
- LSS curves per religion, per model, with bootstrap CI
- LBS curves per model
- Probing accuracy curves (emotion vs religion probe) per layer type
- Side-by-side T5 encoder / T5 decoder / Sarvam subplots
- Paper-ready dashboard combining all key curves

"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_config import METRICS_DIR, PROBING_DIR, PLOTS_DIR

sns.set(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("tab10")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def bootstrap_ci(values, n_boot=200, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a 1D array."""
    rng = np.random.default_rng(seed)
    boots = [rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return lo, hi


def plot_lss_curves(ax, metrics, layer_type, title, normalize=True):
    """Plot LSS curves for each religion and the average on a given axis."""
    lss = np.array(metrics[f"{layer_type}_lss"])
    lss_per_rel = metrics.get(f"{layer_type}_lss_per_religion", {})
    n = len(lss)
    x = np.arange(n) / max(n - 1, 1) if normalize else np.arange(n)
    xlabel = "Relative Layer Depth" if normalize else "Layer Index"

    # Per-religion lines
    for i, (religion, vals) in enumerate(lss_per_rel.items()):
        ax.plot(x, vals, label=religion, color=COLORS[i], linewidth=1.4, alpha=0.8)

    # Average with shaded CI
    lo, hi = bootstrap_ci(lss)
    ax.plot(x, lss, label="Average", color="black", linewidth=2, linestyle="--")
    ax.fill_between(x, lo, hi, alpha=0.15, color="black", label="95% CI")

    # Mark peak
    peak_idx = int(np.argmax(lss))
    ax.axvline(x[peak_idx], color="red", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(x[peak_idx], ax.get_ylim()[1] * 0.95, f"L{peak_idx}",
            ha="center", va="top", fontsize=8, color="red")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("LSS")
    ax.legend(fontsize=8, loc="upper left")


def plot_lbs_curve(ax, metrics, layer_type, title, normalize=True):
    """Plot LBS curve."""
    lbs = np.array(metrics[f"{layer_type}_lbs"])
    n = len(lbs)
    x = np.arange(n) / max(n - 1, 1) if normalize else np.arange(n)

    ax.plot(x, lbs, color="darkgreen", linewidth=2)
    ax.fill_between(x, 0, lbs, alpha=0.2, color="darkgreen")

    peak_idx = int(np.argmax(lbs))
    ax.axvline(x[peak_idx], color="red", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(x[peak_idx], lbs.max() * 0.95, f"L{peak_idx}",
            ha="center", va="top", fontsize=8, color="red")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Relative Layer Depth" if normalize else "Layer Index")
    ax.set_ylabel("LBS")


def plot_probing_curves(ax, probing, layer_type, title, normalize=True):
    """Plot emotion vs religion probing accuracy curves."""
    emotion_results = probing["results"].get(f"{layer_type}_emotion", [])
    religion_results = probing["results"].get(f"{layer_type}_religion", [])

    if not emotion_results:
        ax.set_visible(False)
        return

    n = len(emotion_results)
    x = np.arange(n) / max(n - 1, 1) if normalize else np.arange(n)

    em_acc = [r["accuracy_mean"] for r in emotion_results]
    em_std = [r["accuracy_std"] for r in emotion_results]
    rel_acc = [r["accuracy_mean"] for r in religion_results] if religion_results else []
    rel_std = [r["accuracy_std"] for r in religion_results] if religion_results else []

    ax.plot(x, em_acc, label="Emotion Probe", color="steelblue", linewidth=2)
    ax.fill_between(x,
                    np.array(em_acc) - np.array(em_std),
                    np.array(em_acc) + np.array(em_std),
                    alpha=0.2, color="steelblue")

    if rel_acc:
        ax.plot(x, rel_acc, label="Religion Probe", color="tomato", linewidth=2)
        ax.fill_between(x,
                        np.array(rel_acc) - np.array(rel_std),
                        np.array(rel_acc) + np.array(rel_std),
                        alpha=0.2, color="tomato")

    # Mark emotion crystallization (first layer at ≥80% of max)
    max_em = max(em_acc)
    for i, a in enumerate(em_acc):
        if a >= 0.8 * max_em:
            ax.axvline(x[i], color="steelblue", linestyle=":", linewidth=1, alpha=0.7)
            ax.text(x[i], max_em * 0.5, f"E{i}", ha="center", fontsize=7, color="steelblue")
            break

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Relative Layer Depth" if normalize else "Layer Index")
    ax.set_ylabel("Probe Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)


def plot_all_lss():
    """Side-by-side LSS comparison: T5 encoder | T5 decoder | Sarvam decoder."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    try:
        t5_metrics = load_json(os.path.join(METRICS_DIR, "layerwise_metrics_t5.json"))
        sarvam_metrics = load_json(os.path.join(METRICS_DIR, "layerwise_metrics_sarvam.json"))
    except FileNotFoundError as e:
        print(f"Skipping LSS plot: {e}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_lss_curves(axes[0], t5_metrics, "encoder", "T5 Encoder — LSS by Religion")
    plot_lss_curves(axes[1], t5_metrics, "decoder", "T5 Decoder — LSS by Religion")
    plot_lss_curves(axes[2], sarvam_metrics, "decoder", "Sarvam Decoder — LSS by Religion")

    fig.suptitle("Layer-wise Sensitivity Score (LSS) — All Models", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "lss_curves_all.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: lss_curves_all.png")


def plot_all_lbs():
    """Side-by-side LBS comparison."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    try:
        t5_metrics = load_json(os.path.join(METRICS_DIR, "layerwise_metrics_t5.json"))
        sarvam_metrics = load_json(os.path.join(METRICS_DIR, "layerwise_metrics_sarvam.json"))
    except FileNotFoundError as e:
        print(f"Skipping LBS plot: {e}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_lbs_curve(axes[0], t5_metrics, "encoder", "T5 Encoder — LBS")
    plot_lbs_curve(axes[1], t5_metrics, "decoder", "T5 Decoder — LBS")
    plot_lbs_curve(axes[2], sarvam_metrics, "decoder", "Sarvam Decoder — LBS")

    fig.suptitle("Layer-wise Bias Score (LBS) — All Models", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "lbs_curves_all.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: lbs_curves_all.png")


def plot_all_probing():
    """Probing accuracy curves for all layer types and models."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    try:
        t5_probing = load_json(os.path.join(PROBING_DIR, "probing_results_t5.json"))
        sarvam_probing = load_json(os.path.join(PROBING_DIR, "probing_results_sarvam.json"))
    except FileNotFoundError as e:
        print(f"Skipping probing plot: {e}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_probing_curves(axes[0], t5_probing, "encoder", "T5 Encoder — Probing Accuracy")
    plot_probing_curves(axes[1], t5_probing, "decoder", "T5 Decoder — Probing Accuracy")
    plot_probing_curves(axes[2], sarvam_probing, "decoder", "Sarvam Decoder — Probing Accuracy")

    fig.suptitle("Probing Classifier Accuracy per Layer", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "probing_curves_all.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: probing_curves_all.png")


def plot_lbs_heatmap():
    """Heatmap: LBS per religion per layer for each model."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    try:
        t5_metrics = load_json(os.path.join(METRICS_DIR, "layerwise_metrics_t5.json"))
        sarvam_metrics = load_json(os.path.join(METRICS_DIR, "layerwise_metrics_sarvam.json"))
    except FileNotFoundError as e:
        print(f"Skipping LBS heatmap: {e}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, metrics, model_label, lt in [
        (axes[0], t5_metrics, "T5 Encoder", "encoder"),
        (axes[1], sarvam_metrics, "Sarvam Decoder", "decoder"),
    ]:
        lss_per_rel = metrics.get(f"{lt}_lss_per_religion", {})
        if not lss_per_rel:
            ax.set_visible(False)
            continue

        religions = sorted(lss_per_rel.keys())
        data = np.stack([lss_per_rel[r] for r in religions])  # [num_religions, num_layers]

        sns.heatmap(data, ax=ax, cmap="YlOrRd", yticklabels=religions,
                    xticklabels=[str(i) if i % 4 == 0 else "" for i in range(data.shape[1])],
                    cbar_kws={"label": "LSS"})
        ax.set_title(f"{model_label} — LSS per Religion per Layer", fontsize=11)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Religion")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "lss_religion_heatmap.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: lss_religion_heatmap.png")


def main():
    plot_all_lss()
    plot_all_lbs()
    plot_all_probing()
    plot_lbs_heatmap()
    print("\nLayer curve visualizations complete!")


if __name__ == "__main__":
    main()
