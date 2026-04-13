"""
Visualization: Cross-model comparison plots.

Generates:
- CKA similarity heatmap (T5 layers × Sarvam layers)
- Overlay LSS curves (normalized layer position)
- Overlay probing accuracy curves
- Per-religion LSS comparison bars
- Paper-ready 4-panel dashboard figure

"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_config import METRICS_DIR, PROBING_DIR, ABLATION_DIR, PLOTS_DIR, RELIGIONS

sns.set(style="whitegrid", font_scale=1.1)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_cka_heatmap():
    """CKA similarity matrix: T5 encoder/decoder rows vs Sarvam decoder columns."""
    path = os.path.join(METRICS_DIR, "cross_model_cka.json")
    if not os.path.exists(path):
        print("No cross-model CKA data, skipping heatmap")
        return

    cka_data = load_json(path)
    n_keys = len(cka_data)
    if n_keys == 0:
        return

    fig, axes = plt.subplots(1, n_keys, figsize=(10 * n_keys, 8))
    if n_keys == 1:
        axes = [axes]

    for ax, (key, cka_mat) in zip(axes, cka_data.items()):
        mat = np.array(cka_mat)
        parts = key.split("_vs_")
        row_label = parts[0].replace("_", " ").title() if parts else "T5"
        col_label = parts[1].replace("_", " ").title() if len(parts) > 1 else "Sarvam"

        sns.heatmap(mat, ax=ax, cmap="viridis", vmin=0, vmax=1,
                    xticklabels=[str(i) if i % 4 == 0 else "" for i in range(mat.shape[1])],
                    yticklabels=[str(i) if i % 4 == 0 else "" for i in range(mat.shape[0])],
                    cbar_kws={"label": "CKA Similarity"})
        ax.set_title(f"CKA: {row_label} (rows) vs {col_label} (cols)", fontsize=11)
        ax.set_xlabel(f"{col_label} Layer Index")
        ax.set_ylabel(f"{row_label} Layer Index")

        # Mark best-match diagonal tendency
        best_matches = np.argmax(mat, axis=1)
        for i, j in enumerate(best_matches):
            ax.plot(j + 0.5, i + 0.5, "r.", markersize=3, alpha=0.5)

    fig.suptitle("Cross-Model CKA Layer Similarity (T5 vs Sarvam)", fontsize=13, y=1.02)
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "cka_cross_model.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: cka_cross_model.png")


def plot_overlay_lss():
    """Overlay LSS curves using normalized layer position (0→1)."""
    try:
        t5_m = load_json(os.path.join(METRICS_DIR, "layerwise_metrics_t5.json"))
        sarvam_m = load_json(os.path.join(METRICS_DIR, "layerwise_metrics_sarvam.json"))
    except FileNotFoundError as e:
        print(f"Skipping overlay LSS: {e}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    plot_specs = [
        (t5_m, "encoder", "T5 Encoder", "steelblue", "-"),
        (t5_m, "decoder", "T5 Decoder", "steelblue", "--"),
        (sarvam_m, "decoder", "Sarvam Decoder", "tomato", "-"),
    ]

    for metrics, lt, label, color, ls in plot_specs:
        key = f"{lt}_lss"
        if key not in metrics:
            continue
        lss = np.array(metrics[key])
        n = len(lss)
        x = np.arange(n) / max(n - 1, 1)
        ax.plot(x, lss, label=label, color=color, linestyle=ls, linewidth=2)

        peak_i = int(np.argmax(lss))
        ax.plot(x[peak_i], lss[peak_i], "o", color=color, markersize=8)

    ax.set_title("Layer-wise Sensitivity Score (LSS) — Normalized Comparison", fontsize=12)
    ax.set_xlabel("Relative Layer Depth (0 = first, 1 = last)")
    ax.set_ylabel("LSS")
    ax.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "lss_overlay_comparison.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: lss_overlay_comparison.png")


def plot_overlay_probing():
    """Overlay probing accuracy curves with normalized layer positions."""
    try:
        t5_p = load_json(os.path.join(PROBING_DIR, "probing_results_t5.json"))
        sarvam_p = load_json(os.path.join(PROBING_DIR, "probing_results_sarvam.json"))
    except FileNotFoundError as e:
        print(f"Skipping overlay probing: {e}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, task in zip(axes, ["emotion", "religion"]):
        plot_specs = [
            (t5_p, "encoder", f"T5 Encoder", "steelblue", "-"),
            (t5_p, "decoder", f"T5 Decoder", "steelblue", "--"),
            (sarvam_p, "decoder", f"Sarvam Decoder", "tomato", "-"),
        ]

        for probing, lt, label, color, ls in plot_specs:
            results = probing["results"].get(f"{lt}_{task}", [])
            if not results:
                continue
            n = len(results)
            x = np.arange(n) / max(n - 1, 1)
            acc = [r["accuracy_mean"] for r in results]
            ax.plot(x, acc, label=label, color=color, linestyle=ls, linewidth=2)

        ax.set_title(f"{task.capitalize()} Probing Accuracy — Comparison", fontsize=11)
        ax.set_xlabel("Relative Layer Depth")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "probing_overlay_comparison.png"),
                bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: probing_overlay_comparison.png")


def plot_paper_dashboard():
    """4-panel paper-ready dashboard figure.

    Panel A: LSS overlay (both models, normalized)
    Panel B: Probing accuracy (emotion + religion, both models)
    Panel C: LCS bar chart (top-20 layers by abs LCS, stacked T5/Sarvam)
    Panel D: CKA cross-model similarity (T5 encoder vs Sarvam decoder)
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # --- Panel A: LSS Overlay ---
    try:
        t5_m = load_json(os.path.join(METRICS_DIR, "layerwise_metrics_t5.json"))
        sarvam_m = load_json(os.path.join(METRICS_DIR, "layerwise_metrics_sarvam.json"))

        for metrics, lt, label, color, ls in [
            (t5_m, "encoder", "T5 Encoder", "steelblue", "-"),
            (t5_m, "decoder", "T5 Decoder", "steelblue", "--"),
            (sarvam_m, "decoder", "Sarvam", "tomato", "-"),
        ]:
            if f"{lt}_lss" not in metrics:
                continue
            lss = np.array(metrics[f"{lt}_lss"])
            n = len(lss)
            x = np.arange(n) / max(n - 1, 1)
            ax_a.plot(x, lss, label=label, color=color, linestyle=ls, linewidth=2)
            peak = int(np.argmax(lss))
            ax_a.plot(x[peak], lss[peak], "o", color=color, markersize=7)

        ax_a.set_title("(A) Layer Sensitivity Score (LSS)", fontsize=12, fontweight="bold")
        ax_a.set_xlabel("Relative Layer Depth")
        ax_a.set_ylabel("LSS")
        ax_a.legend(fontsize=9)
    except FileNotFoundError:
        ax_a.text(0.5, 0.5, "LSS data not found", ha="center", va="center",
                  transform=ax_a.transAxes)

    # --- Panel B: Probing Curves ---
    try:
        t5_p = load_json(os.path.join(PROBING_DIR, "probing_results_t5.json"))
        sarvam_p = load_json(os.path.join(PROBING_DIR, "probing_results_sarvam.json"))

        for probing, lt, task, label, color, ls in [
            (t5_p, "encoder", "emotion", "T5 Enc (emotion)", "#1f77b4", "-"),
            (t5_p, "encoder", "religion", "T5 Enc (religion)", "#ff7f0e", "-"),
            (sarvam_p, "decoder", "emotion", "Sarvam (emotion)", "#1f77b4", "--"),
            (sarvam_p, "decoder", "religion", "Sarvam (religion)", "#ff7f0e", "--"),
        ]:
            res = probing["results"].get(f"{lt}_{task}", [])
            if not res:
                continue
            n = len(res)
            x = np.arange(n) / max(n - 1, 1)
            acc = [r["accuracy_mean"] for r in res]
            ax_b.plot(x, acc, label=label, color=color, linestyle=ls, linewidth=2)

        ax_b.set_title("(B) Probing Accuracy per Layer", fontsize=12, fontweight="bold")
        ax_b.set_xlabel("Relative Layer Depth")
        ax_b.set_ylabel("Accuracy")
        ax_b.set_ylim(0, 1.05)
        ax_b.legend(fontsize=8)
    except FileNotFoundError:
        ax_b.text(0.5, 0.5, "Probing data not found", ha="center", va="center",
                  transform=ax_b.transAxes)

    # --- Panel C: LCS Top Layers ---
    from config.experiment_config import RESULTS_T5_FILE, RESULTS_SARVAM_FILE
    from experiments.phase3_ablation import compute_bias_metrics

    lcs_combined = {}
    for model_name, results_file in [("t5", RESULTS_T5_FILE), ("sarvam", RESULTS_SARVAM_FILE)]:
        abl_path = os.path.join(ABLATION_DIR, f"layer_ablation_{model_name}.json")
        if not os.path.exists(abl_path):
            continue
        abl = load_json(abl_path)
        ctrl = compute_bias_metrics(load_json(results_file))
        ctrl_bias = ctrl["scenario_bias_rate"]
        for key, data in abl.items():
            lcs = ctrl_bias - data["metrics"]["scenario_bias_rate"]
            lcs_combined[f"{model_name}/{key}"] = lcs

    if lcs_combined:
        sorted_lcs = sorted(lcs_combined.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        labels = [k.replace("_layer_", " L").replace("t5/", "T5:").replace("sarvam/", "SV:") for k, _ in sorted_lcs]
        values = [v for _, v in sorted_lcs]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]
        ax_c.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.7)
        ax_c.axvline(0, color="black", linewidth=0.8)
        ax_c.set_title("(C) Top Layer Contribution Scores (LCS)", fontsize=12, fontweight="bold")
        ax_c.set_xlabel("LCS (positive = amplifies bias)")
        ax_c.tick_params(axis="y", labelsize=7)
    else:
        ax_c.text(0.5, 0.5, "Ablation data not found", ha="center", va="center",
                  transform=ax_c.transAxes)

    # --- Panel D: CKA Heatmap ---
    cka_path = os.path.join(METRICS_DIR, "cross_model_cka.json")
    if os.path.exists(cka_path):
        cka_data = load_json(cka_path)
        first_key = next(iter(cka_data))
        cka_mat = np.array(cka_data[first_key])
        im = ax_d.imshow(cka_mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax_d, label="CKA")
        ax_d.set_title("(D) CKA Cross-Model Similarity", fontsize=12, fontweight="bold")
        ax_d.set_xlabel("Sarvam Layer")
        ax_d.set_ylabel("T5 Encoder Layer")
        ax_d.tick_params(axis="both", labelsize=8)
    else:
        ax_d.text(0.5, 0.5, "CKA data not found", ha="center", va="center",
                  transform=ax_d.transAxes)

    fig.suptitle("Phase 3: Layer-wise Interpretability — Summary Dashboard",
                 fontsize=15, fontweight="bold", y=0.98)

    out_path = os.path.join(PLOTS_DIR, "dashboard_phase3.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: dashboard_phase3.png")


def main():
    plot_cka_heatmap()
    plot_overlay_lss()
    plot_overlay_probing()
    plot_paper_dashboard()
    print("\nCross-model visualizations complete!")


if __name__ == "__main__":
    main()
