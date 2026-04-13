"""Phase 3: Layer-wise sensitivity analysis.

Aggregates LSS, LBS, probing results, and ablation metrics into a comprehensive
per-layer analysis. Identifies critical layers for religion sensitivity and
emotion processing.

"""

import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import METRICS_DIR, PROBING_DIR, ABLATION_DIR, TABLES_DIR


def load_json(path):
    with open(path) as f:
        return json.load(f)


def analyze_model(model_name):
    """Compile layer-wise analysis for a single model."""
    print(f"\n{'='*60}")
    print(f"Layer Analysis: {model_name.upper()}")
    print(f"{'='*60}")

    # Load metrics
    metrics_path = os.path.join(METRICS_DIR, f"layerwise_metrics_{model_name}.json")
    metrics = load_json(metrics_path)

    # Load probing results
    probing_path = os.path.join(PROBING_DIR, f"probing_results_{model_name}.json")
    probing = load_json(probing_path)

    # Load ablation results (if available)
    ablation_path = os.path.join(ABLATION_DIR, f"layer_ablation_{model_name}.json")
    ablation = load_json(ablation_path) if os.path.exists(ablation_path) else None

    # Also load Phase 2 control metrics for LCS computation
    from config.experiment_config import RESULTS_T5_FILE, RESULTS_SARVAM_FILE
    control_file = RESULTS_T5_FILE if model_name == "t5" else RESULTS_SARVAM_FILE
    control_results = load_json(control_file)

    from experiments.phase3_ablation import compute_bias_metrics
    control_metrics = compute_bias_metrics(control_results)

    analysis = {"model": model_name, "layer_types": {}}

    for layer_type_key in [k.replace("_lss", "") for k in metrics.keys() if k.endswith("_lss")]:
        lt = layer_type_key
        lss = np.array(metrics[f"{lt}_lss"])
        lbs = np.array(metrics[f"{lt}_lbs"])
        p_values = np.array(metrics.get(f"{lt}_lss_p_values", []))
        significant = metrics.get(f"{lt}_significant_layers", [])
        num_layers = len(lss)

        # Probing accuracy curves
        emotion_probing = probing["results"].get(f"{lt}_emotion", [])
        religion_probing = probing["results"].get(f"{lt}_religion", [])

        emotion_acc = [r["accuracy_mean"] for r in emotion_probing]
        religion_acc = [r["accuracy_mean"] for r in religion_probing]

        # LCS from ablation
        lcs = np.zeros(num_layers)
        if ablation:
            for i in range(num_layers):
                key = f"{lt}_layer_{i}"
                if key in ablation:
                    ablated_bias = ablation[key]["metrics"]["scenario_bias_rate"]
                    lcs[i] = control_metrics["scenario_bias_rate"] - ablated_bias

        # Build per-layer table
        rows = []
        for i in range(num_layers):
            row = {
                "layer_idx": i,
                "layer_type": lt,
                "relative_position": i / max(num_layers - 1, 1),
                "lss": float(lss[i]),
                "lbs": float(lbs[i]),
                "lcs": float(lcs[i]),
                "emotion_probe_acc": emotion_acc[i] if i < len(emotion_acc) else None,
                "religion_probe_acc": religion_acc[i] if i < len(religion_acc) else None,
                "lss_p_value": float(p_values[i]) if i < len(p_values) else None,
                "lss_significant": i in significant,
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Identify critical layers
        lss_peak = int(np.argmax(lss))
        lbs_peak = int(np.argmax(lbs))
        lcs_peak = int(np.argmax(np.abs(lcs))) if ablation else None

        # Emotion crystallization: layer where emotion probe accuracy exceeds 80%
        # of its max
        emotion_threshold = None
        if emotion_acc:
            max_acc = max(emotion_acc)
            for i, a in enumerate(emotion_acc):
                if a >= 0.8 * max_acc:
                    emotion_threshold = i
                    break

        # Religion encoding peak
        religion_peak = int(np.argmax(religion_acc)) if religion_acc else None

        layer_analysis = {
            "num_layers": num_layers,
            "lss_peak_layer": lss_peak,
            "lss_peak_value": float(lss[lss_peak]),
            "lbs_peak_layer": lbs_peak,
            "lbs_peak_value": float(lbs[lbs_peak]),
            "lcs_peak_layer": lcs_peak,
            "lcs_peak_value": float(lcs[lcs_peak]) if lcs_peak is not None else None,
            "emotion_crystallization_layer": emotion_threshold,
            "religion_encoding_peak_layer": religion_peak,
            "significant_lss_layers": significant,
            "per_religion_lss": metrics.get(f"{lt}_lss_per_religion", {}),
        }

        analysis["layer_types"][lt] = layer_analysis

        # Save table
        os.makedirs(TABLES_DIR, exist_ok=True)
        df.to_csv(os.path.join(TABLES_DIR, f"layer_analysis_{model_name}_{lt}.csv"), index=False)

        # Print summary
        print(f"\n  {lt} ({num_layers} layers):")
        print(f"    LSS peak: layer {lss_peak} (value={lss[lss_peak]:.4f})")
        print(f"    LBS peak: layer {lbs_peak} (value={lbs[lbs_peak]:.6f})")
        if lcs_peak is not None:
            print(f"    LCS peak: layer {lcs_peak} (value={lcs[lcs_peak]:.2f})")
        if emotion_threshold is not None:
            print(f"    Emotion crystallization: layer {emotion_threshold}")
        if religion_peak is not None:
            print(f"    Religion encoding peak: layer {religion_peak}")
        print(f"    Significant LSS layers: {significant}")

    # Save full analysis
    output_path = os.path.join(TABLES_DIR, f"layer_analysis_{model_name}.json")
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nAnalysis saved to {output_path}")
    return analysis


def main():
    for model in ["t5", "sarvam"]:
        try:
            analyze_model(model)
        except FileNotFoundError as e:
            print(f"Skipping {model}: {e}")

    print("\nLayer analysis complete!")


if __name__ == "__main__":
    main()
