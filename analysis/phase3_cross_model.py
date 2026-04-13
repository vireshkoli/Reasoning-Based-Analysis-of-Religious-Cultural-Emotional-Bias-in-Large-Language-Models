"""Phase 3: Cross-model comparison using CKA and metric alignment.

Compares T5 (encoder-decoder) and Sarvam (decoder-only) at the layer level to
identify functional correspondence and differences in how they process religious context.

"""

import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import METRICS_DIR, TABLES_DIR


def load_json(path):
    with open(path) as f:
        return json.load(f)


def normalize_layer_positions(metrics, layer_type):
    """Normalize layer indices to [0, 1] for cross-architecture comparison."""
    lss = np.array(metrics[f"{layer_type}_lss"])
    n = len(lss)
    positions = np.arange(n) / max(n - 1, 1)
    return positions, lss


def compare_models():
    """Run comprehensive cross-model comparison."""
    print("=" * 60)
    print("Cross-Model Comparison: T5 vs Sarvam")
    print("=" * 60)

    # Load metrics for both models
    t5_metrics = load_json(os.path.join(METRICS_DIR, "layerwise_metrics_t5.json"))
    sarvam_metrics = load_json(os.path.join(METRICS_DIR, "layerwise_metrics_sarvam.json"))

    # Load cross-model CKA if available
    cka_path = os.path.join(METRICS_DIR, "cross_model_cka.json")
    cross_cka = load_json(cka_path) if os.path.exists(cka_path) else None

    comparison = {"models": ["t5", "sarvam"]}

    # --- LSS Comparison ---
    print("\n--- LSS Comparison ---")
    for lt in ["encoder", "decoder"]:
        key = f"{lt}_lss"
        if key in t5_metrics:
            t5_lss = np.array(t5_metrics[key])
            print(f"  T5 {lt}: max LSS = {t5_lss.max():.4f} at layer {np.argmax(t5_lss)}")
            comparison[f"t5_{lt}_lss_peak"] = {
                "layer": int(np.argmax(t5_lss)),
                "value": float(t5_lss.max()),
                "relative_position": float(np.argmax(t5_lss) / max(len(t5_lss) - 1, 1)),
            }

    sarvam_lss = np.array(sarvam_metrics["decoder_lss"])
    print(f"  Sarvam decoder: max LSS = {sarvam_lss.max():.4f} at layer {np.argmax(sarvam_lss)}")
    comparison["sarvam_decoder_lss_peak"] = {
        "layer": int(np.argmax(sarvam_lss)),
        "value": float(sarvam_lss.max()),
        "relative_position": float(np.argmax(sarvam_lss) / max(len(sarvam_lss) - 1, 1)),
    }

    # --- LBS Comparison ---
    print("\n--- LBS Comparison ---")
    for lt in ["encoder", "decoder"]:
        key = f"{lt}_lbs"
        if key in t5_metrics:
            t5_lbs = np.array(t5_metrics[key])
            print(f"  T5 {lt}: max LBS = {t5_lbs.max():.6f} at layer {np.argmax(t5_lbs)}")

    sarvam_lbs = np.array(sarvam_metrics["decoder_lbs"])
    print(f"  Sarvam decoder: max LBS = {sarvam_lbs.max():.6f} at layer {np.argmax(sarvam_lbs)}")

    # --- CKA-based Layer Mapping ---
    if cross_cka:
        print("\n--- CKA Cross-Model Layer Mapping ---")

        for cka_key, cka_mat in cross_cka.items():
            cka_arr = np.array(cka_mat)
            print(f"\n  {cka_key}: shape {cka_arr.shape}")

            # Find best Sarvam match for each T5 layer
            best_matches = np.argmax(cka_arr, axis=1)
            best_values = np.max(cka_arr, axis=1)

            mapping = []
            for t5_layer, (sarvam_layer, cka_val) in enumerate(zip(best_matches, best_values)):
                mapping.append({
                    "t5_layer": int(t5_layer),
                    "sarvam_layer": int(sarvam_layer),
                    "cka_similarity": float(cka_val),
                })
                if t5_layer % 6 == 0:
                    print(f"    T5 layer {t5_layer} → Sarvam layer {sarvam_layer} (CKA={cka_val:.3f})")

            comparison[f"cka_mapping_{cka_key}"] = mapping

    # --- Per-Religion LSS Comparison ---
    print("\n--- Per-Religion LSS Comparison ---")
    t5_religions = t5_metrics.get("encoder_lss_per_religion", {})
    sarvam_religions = sarvam_metrics.get("decoder_lss_per_religion", {})

    religion_comparison = {}
    for religion in set(list(t5_religions.keys()) + list(sarvam_religions.keys())):
        rc = {}
        if religion in t5_religions:
            t5_r_lss = np.array(t5_religions[religion])
            rc["t5_encoder_peak"] = int(np.argmax(t5_r_lss))
            rc["t5_encoder_max"] = float(t5_r_lss.max())
        if religion in sarvam_religions:
            s_r_lss = np.array(sarvam_religions[religion])
            rc["sarvam_decoder_peak"] = int(np.argmax(s_r_lss))
            rc["sarvam_decoder_max"] = float(s_r_lss.max())
        religion_comparison[religion] = rc
        print(f"  {religion}: T5 enc peak={rc.get('t5_encoder_peak', 'N/A')}, "
              f"Sarvam dec peak={rc.get('sarvam_decoder_peak', 'N/A')}")

    comparison["religion_comparison"] = religion_comparison

    # --- Key Findings ---
    print("\n--- Key Findings ---")

    # Does sensitivity peak at same relative depth?
    t5_enc_peak_pos = comparison.get("t5_encoder_lss_peak", {}).get("relative_position")
    sarvam_peak_pos = comparison.get("sarvam_decoder_lss_peak", {}).get("relative_position")

    if t5_enc_peak_pos is not None and sarvam_peak_pos is not None:
        depth_diff = abs(t5_enc_peak_pos - sarvam_peak_pos)
        same_depth = depth_diff < 0.15
        print(f"  Sensitivity peak depth: T5 enc={t5_enc_peak_pos:.2f}, "
              f"Sarvam={sarvam_peak_pos:.2f}, diff={depth_diff:.2f} "
              f"({'similar' if same_depth else 'different'})")
        comparison["depth_similarity"] = {
            "similar": bool(same_depth),
            "difference": float(depth_diff),
        }

    # Save
    os.makedirs(TABLES_DIR, exist_ok=True)
    output_path = os.path.join(TABLES_DIR, "cross_model_comparison.json")
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to {output_path}")


def main():
    compare_models()


if __name__ == "__main__":
    main()
