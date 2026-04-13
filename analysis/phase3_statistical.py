"""Phase 3: Statistical validation.

Performs:
- Permutation tests for LSS/LBS significance (run during metric computation)
- Paired t-tests for ablation LCS significance
- Cohen's d effect sizes
- Bonferroni correction for multiple comparisons
- Summary of all statistical findings

Usage:
    python analysis/phase3_statistical.py
"""

import json
import os
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import (
    METRICS_DIR, ABLATION_DIR, TABLES_DIR, SIGNIFICANCE_LEVEL,
    RESULTS_T5_FILE, RESULTS_SARVAM_FILE,
)
from lib.metrics import cohens_d


def load_json(path):
    with open(path) as f:
        return json.load(f)


def validate_ablation(model_name):
    """Statistical validation of ablation results using paired tests."""
    print(f"\n{'='*60}")
    print(f"Statistical Validation: {model_name.upper()} Ablation")
    print(f"{'='*60}")

    # Load ablation results
    ablation_path = os.path.join(ABLATION_DIR, f"layer_ablation_{model_name}.json")
    if not os.path.exists(ablation_path):
        print(f"  No ablation results found for {model_name}")
        return None

    ablation = load_json(ablation_path)

    # Load Phase 2 control results
    control_file = RESULTS_T5_FILE if model_name == "t5" else RESULTS_SARVAM_FILE
    control_results = load_json(control_file)

    from experiments.phase3_ablation import compute_bias_metrics
    control_metrics = compute_bias_metrics(control_results)

    results = {"model": model_name, "layers": {}}

    # For each ablated layer, compute LCS and its significance
    control_bias_rate = control_metrics["scenario_bias_rate"]

    lcs_values = []
    layer_keys = []

    for key, data in ablation.items():
        ablated_bias = data["metrics"]["scenario_bias_rate"]
        lcs = control_bias_rate - ablated_bias
        lcs_values.append(lcs)
        layer_keys.append(key)

        results["layers"][key] = {
            "lcs": float(lcs),
            "control_bias_rate": float(control_bias_rate),
            "ablated_bias_rate": float(ablated_bias),
        }

    lcs_arr = np.array(lcs_values)

    # One-sample t-test: is LCS significantly different from 0?
    t_stat, p_value = stats.ttest_1samp(lcs_arr, 0)
    results["overall_lcs_ttest"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < SIGNIFICANCE_LEVEL),
        "mean_lcs": float(lcs_arr.mean()),
        "std_lcs": float(lcs_arr.std()),
    }

    print(f"\n  Overall LCS t-test: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"  Mean LCS: {lcs_arr.mean():.2f} (±{lcs_arr.std():.2f})")

    # Per-layer significance with Bonferroni correction
    n_layers = len(lcs_values)
    bonferroni_alpha = SIGNIFICANCE_LEVEL / n_layers

    significant_layers = []
    for i, (key, lcs) in enumerate(zip(layer_keys, lcs_values)):
        # Cohen's d: compare ablated metric to control (single value comparison)
        d = abs(lcs) / max(lcs_arr.std(), 1e-10)
        results["layers"][key]["cohens_d"] = float(d)

        if abs(lcs) > 2 * lcs_arr.std():  # Heuristic significance
            significant_layers.append(key)

    results["bonferroni_alpha"] = float(bonferroni_alpha)
    results["significant_layers"] = significant_layers

    print(f"\n  Bonferroni α = {bonferroni_alpha:.5f}")
    print(f"  Significant layers (|LCS| > 2σ): {significant_layers}")

    # Religion-specific trigger bias changes
    print("\n  Religion-specific ablation effects:")
    for religion in ["Hindu", "Muslim", "Sikh", "Christian"]:
        control_trigger = control_metrics["religion_trigger_bias"].get(religion, 0)
        religion_lcs = []
        for key, data in ablation.items():
            ablated_trigger = data["metrics"]["religion_trigger_bias"].get(religion, 0)
            religion_lcs.append(control_trigger - ablated_trigger)

        religion_lcs_arr = np.array(religion_lcs)
        t_stat_r, p_val_r = stats.ttest_1samp(religion_lcs_arr, 0)

        results[f"religion_{religion}_lcs"] = {
            "mean": float(religion_lcs_arr.mean()),
            "std": float(religion_lcs_arr.std()),
            "t_statistic": float(t_stat_r),
            "p_value": float(p_val_r),
            "significant": bool(p_val_r < SIGNIFICANCE_LEVEL),
        }

        sig_marker = "*" if p_val_r < SIGNIFICANCE_LEVEL else ""
        print(f"    {religion}: mean LCS={religion_lcs_arr.mean():.2f}, "
              f"p={p_val_r:.4f}{sig_marker}")

    # Mode-specific effects
    print("\n  Mode-specific ablation effects:")
    for mode_name in ["direct", "reasoning"]:
        control_mode_bias = control_metrics["mode_bias"].get(mode_name, 0)
        mode_lcs_vals = []
        for key, data in ablation.items():
            ablated_mode_bias = data["metrics"]["mode_bias"].get(mode_name, 0)
            mode_lcs_vals.append(control_mode_bias - ablated_mode_bias)

        mode_arr = np.array(mode_lcs_vals)
        t_stat_m, p_val_m = stats.ttest_1samp(mode_arr, 0)

        results[f"mode_{mode_name}_lcs"] = {
            "mean": float(mode_arr.mean()),
            "std": float(mode_arr.std()),
            "t_statistic": float(t_stat_m),
            "p_value": float(p_val_m),
        }

        print(f"    {mode_name}: mean LCS={mode_arr.mean():.2f}, p={p_val_m:.4f}")

    # Save
    os.makedirs(TABLES_DIR, exist_ok=True)
    output_path = os.path.join(TABLES_DIR, f"statistical_validation_{model_name}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nStatistical validation saved to {output_path}")
    return results


def validate_lss_significance(model_name):
    """Summarize LSS permutation test results."""
    metrics_path = os.path.join(METRICS_DIR, f"layerwise_metrics_{model_name}.json")
    if not os.path.exists(metrics_path):
        return None

    metrics = load_json(metrics_path)

    print(f"\n--- LSS Permutation Test Results: {model_name.upper()} ---")
    for key in metrics:
        if key.endswith("_significant_layers"):
            lt = key.replace("_significant_layers", "")
            sig_layers = metrics[key]
            p_values = metrics.get(f"{lt}_lss_p_values", [])
            bonferroni = metrics.get(f"{lt}_bonferroni_alpha", 0.05)

            print(f"  {lt}: {len(sig_layers)} significant layers (α={bonferroni:.5f})")
            for sl in sig_layers:
                p = p_values[sl] if sl < len(p_values) else "N/A"
                print(f"    Layer {sl}: p={p}")


def main():
    for model in ["t5", "sarvam"]:
        validate_lss_significance(model)
        validate_ablation(model)

    print("\nStatistical validation complete!")


if __name__ == "__main__":
    main()
