"""
Visualization: Ablation experiment results.

Generates:
- LCS bar charts per layer (sorted by magnitude)
- Ablation impact heatmap: layers × metrics, colored by change from control
- Per-religion LCS comparison
- Head ablation impact: which heads drive the effect in critical layers

"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_config import ABLATION_DIR, PLOTS_DIR, RELIGIONS

sns.set(style="whitegrid")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def compute_lcs(control_metrics, ablation_data):
    """Compute LCS values from ablation data."""
    control_bias = control_metrics["scenario_bias_rate"]
    lcs_vals = {}
    for key, data in ablation_data.items():
        ablated_bias = data["metrics"]["scenario_bias_rate"]
        lcs_vals[key] = control_bias - ablated_bias
    return lcs_vals


def plot_lcs_bars(model_name, ablation_data, control_metrics):
    """Bar chart of LCS values per layer sorted by absolute magnitude."""
    lcs_vals = compute_lcs(control_metrics, ablation_data)

    # Separate by layer type
    by_type = {}
    for key, val in lcs_vals.items():
        parts = key.split("_layer_")
        lt = parts[0]
        layer_idx = int(parts[1])
        by_type.setdefault(lt, []).append((layer_idx, val))

    n_types = len(by_type)
    fig, axes = plt.subplots(1, n_types, figsize=(10 * n_types, 6))
    if n_types == 1:
        axes = [axes]

    for ax, (lt, layer_vals) in zip(axes, sorted(by_type.items())):
        layer_vals = sorted(layer_vals, key=lambda x: x[0])
        indices, values = zip(*layer_vals)
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]
        bars = ax.bar(indices, values, color=colors, width=0.7)
        ax.axhline(0, color="black", linewidth=0.8)

        # Annotate top bars
        sorted_by_abs = sorted(enumerate(values), key=lambda x: abs(x[1]), reverse=True)
        for rank, (i, v) in enumerate(sorted_by_abs[:5]):
            ax.text(indices[i], v + (0.2 if v >= 0 else -0.2),
                    f"L{indices[i]}", ha="center", fontsize=7, color="black")

        ax.set_title(f"{model_name.upper()} — {lt.capitalize()}\nLayer Contribution Score (LCS)",
                     fontsize=11)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("LCS (positive = contributes to bias)")
        ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"lcs_bars_{model_name}.png"),
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: lcs_bars_{model_name}.png")


def plot_ablation_heatmap(model_name, ablation_data, control_metrics):
    """Heatmap: rows=ablated layers, cols=metrics, color=change from control."""
    control_bias = control_metrics["scenario_bias_rate"]
    control_direct = control_metrics["mode_bias"].get("direct", 0)
    control_reasoning = control_metrics["mode_bias"].get("reasoning", 0)
    control_religion = control_metrics["religion_trigger_bias"]

    rows = []
    for key in sorted(ablation_data.keys()):
        data = ablation_data[key]
        m = data["metrics"]
        row = {
            "key": key,
            "layer_type": data["layer_type"],
            "layer_idx": data["layer_idx"],
            "Δ Bias Rate": m["scenario_bias_rate"] - control_bias,
            "Δ Direct Bias": m["mode_bias"].get("direct", 0) - control_direct,
            "Δ Reasoning Bias": m["mode_bias"].get("reasoning", 0) - control_reasoning,
        }
        for religion in RELIGIONS:
            ctrl_r = control_religion.get(religion, 0)
            ablated_r = m["religion_trigger_bias"].get(religion, 0)
            row[f"Δ {religion}"] = ablated_r - ctrl_r
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows).set_index("key")
    metric_cols = [c for c in df.columns if c.startswith("Δ")]
    heatmap_data = df[metric_cols].astype(float)

    fig_h = max(8, len(heatmap_data) * 0.3)
    fig, ax = plt.subplots(figsize=(len(metric_cols) * 1.5, fig_h))

    sns.heatmap(heatmap_data, ax=ax, cmap="RdBu_r", center=0,
                linewidths=0.3, linecolor="lightgray",
                yticklabels=[f"{r['layer_type'][0].upper()}{r['layer_idx']}"
                             for r in rows],
                cbar_kws={"label": "Change from control"},
                robust=True)

    ax.set_title(f"{model_name.upper()} — Ablation Impact Matrix\n"
                 f"(red=increased bias, blue=decreased bias)", fontsize=11)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Ablated Layer")
    ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"ablation_heatmap_{model_name}.png"),
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: ablation_heatmap_{model_name}.png")


def plot_head_ablation(model_name):
    """Heatmap of head ablation impact for top-K layers."""
    path = os.path.join(ABLATION_DIR, f"head_ablation_{model_name}.json")
    if not os.path.exists(path):
        print(f"No head ablation data for {model_name}, skipping")
        return

    data = load_json(path)

    # Group by layer
    layers = {}
    for key, res in data.items():
        layer_key = f"{res['layer_type']}_layer_{res['layer_idx']}"
        layers.setdefault(layer_key, {})[res["head_idx"]] = res["metrics"]["scenario_bias_rate"]

    n_layers = len(layers)
    if n_layers == 0:
        return

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    for ax, (layer_key, head_data) in zip(axes, sorted(layers.items())):
        head_indices = sorted(head_data.keys())
        bias_rates = [head_data[h] for h in head_indices]

        colors = ["#d62728" if b > np.mean(bias_rates) else "#1f77b4" for b in bias_rates]
        ax.bar(head_indices, bias_rates, color=colors, width=0.7)
        ax.axhline(np.mean(bias_rates), color="black", linestyle="--",
                   linewidth=0.8, label=f"Mean ({np.mean(bias_rates):.1f}%)")

        ax.set_title(f"{layer_key}\nHead Ablation Bias Rate", fontsize=10)
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Bias Rate %")
        ax.legend(fontsize=8)

    fig.suptitle(f"{model_name.upper()} — Attention Head Ablation", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"head_ablation_{model_name}.png"),
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: head_ablation_{model_name}.png")


def main():
    for model_name in ["t5", "sarvam"]:
        ablation_path = os.path.join(ABLATION_DIR, f"layer_ablation_{model_name}.json")
        if not os.path.exists(ablation_path):
            print(f"No layer ablation data for {model_name}, skipping")
            continue

        ablation_data = load_json(ablation_path)

        # Load control (Phase 2) metrics
        from config.experiment_config import RESULTS_T5_FILE, RESULTS_SARVAM_FILE
        control_file = RESULTS_T5_FILE if model_name == "t5" else RESULTS_SARVAM_FILE
        control_raw = load_json(control_file)
        from experiments.phase3_ablation import compute_bias_metrics
        control_metrics = compute_bias_metrics(control_raw)

        plot_lcs_bars(model_name, ablation_data, control_metrics)
        plot_ablation_heatmap(model_name, ablation_data, control_metrics)
        plot_head_ablation(model_name)

    print("\nAblation visualizations complete!")


if __name__ == "__main__":
    main()
