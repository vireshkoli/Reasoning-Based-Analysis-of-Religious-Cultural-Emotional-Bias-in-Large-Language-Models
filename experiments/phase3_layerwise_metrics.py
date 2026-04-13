"""
Phase 3: Compute layer-wise metrics from stored hidden states.

Computes LSS, LBS, and CKA from extracted pooled states. Fast operation
(pure numpy, no model needed in memory).

"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import METRICS_DIR, TABLES_DIR, PERMUTATION_N_ITER, SIGNIFICANCE_LEVEL
from lib.storage import load_pooled_states
from lib.metrics import (
    layer_sensitivity_score, layer_bias_score, linear_cka, cka_matrix,
    permutation_test, cohens_d,
)


def compute_metrics_for_model(model_name):
    """Compute LSS, LBS for a single model."""
    print(f"\n{'='*60}")
    print(f"Computing layer-wise metrics for {model_name.upper()}")
    print(f"{'='*60}")

    arrays, metadata = load_pooled_states(model_name)
    results = {}

    for layer_type, states in arrays.items():
        print(f"\n--- {layer_type} ({states.shape}) ---")

        # LSS
        print("  Computing LSS...")
        lss, lss_per_religion = layer_sensitivity_score(states, metadata)
        results[f"{layer_type}_lss"] = lss.tolist()
        results[f"{layer_type}_lss_per_religion"] = {
            r: v.tolist() for r, v in lss_per_religion.items()
        }

        # LBS
        print("  Computing LBS...")
        lbs, _ = layer_bias_score(states, metadata)
        results[f"{layer_type}_lbs"] = lbs.tolist()

        # Intra-model CKA
        print("  Computing intra-model CKA...")
        n_layers = states.shape[1]
        cka_mat = np.zeros((n_layers, n_layers))
        for i in range(n_layers):
            for j in range(i, n_layers):
                cka_val = linear_cka(states[:, i, :], states[:, j, :])
                cka_mat[i, j] = cka_val
                cka_mat[j, i] = cka_val
        results[f"{layer_type}_cka_intra"] = cka_mat.tolist()

        # Permutation test for LSS significance
        print(f"  Running permutation test ({PERMUTATION_N_ITER} iterations)...")

        def lss_fn(states, meta):
            lss_val, _ = layer_sensitivity_score(states, meta)
            return lss_val

        p_values, _ = permutation_test(lss_fn, states, metadata,
                                       n_permutations=PERMUTATION_N_ITER)

        # Bonferroni correction
        bonferroni_alpha = SIGNIFICANCE_LEVEL / n_layers
        significant_layers = [
            int(i) for i in range(n_layers) if p_values[i] < bonferroni_alpha
        ]

        results[f"{layer_type}_lss_p_values"] = p_values.tolist()
        results[f"{layer_type}_significant_layers"] = significant_layers
        results[f"{layer_type}_bonferroni_alpha"] = bonferroni_alpha

        print(f"  Significant layers (Bonferroni α={bonferroni_alpha:.4f}): {significant_layers}")

    # Save
    os.makedirs(METRICS_DIR, exist_ok=True)
    output_path = os.path.join(METRICS_DIR, f"layerwise_metrics_{model_name}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nMetrics saved to {output_path}")
    return results, arrays, metadata


def compute_cross_model_cka():
    """Compute CKA between T5 and Sarvam layer representations."""
    print(f"\n{'='*60}")
    print("Computing cross-model CKA (T5 vs Sarvam)")
    print(f"{'='*60}")

    t5_arrays, t5_meta = load_pooled_states("t5")
    sarvam_arrays, sarvam_meta = load_pooled_states("sarvam")

    # Align samples by matching keys
    t5_keys = set(t5_meta["keys_order"])
    sarvam_keys = set(sarvam_meta["keys_order"])
    common_keys = sorted(t5_keys & sarvam_keys)
    print(f"Common samples: {len(common_keys)}")

    if len(common_keys) < 10:
        print("WARNING: Too few common samples for meaningful CKA")
        return

    # Build aligned index
    t5_key_to_idx = {k: i for i, k in enumerate(t5_meta["keys_order"])}
    sarvam_key_to_idx = {k: i for i, k in enumerate(sarvam_meta["keys_order"])}
    t5_indices = [t5_key_to_idx[k] for k in common_keys]
    sarvam_indices = [sarvam_key_to_idx[k] for k in common_keys]

    results = {}

    # Compare T5 encoder vs Sarvam decoder
    if "encoder" in t5_arrays and "decoder" in sarvam_arrays:
        t5_enc = t5_arrays["encoder"][t5_indices]
        sarvam_dec = sarvam_arrays["decoder"][sarvam_indices]

        print(f"  T5 encoder ({t5_enc.shape}) vs Sarvam decoder ({sarvam_dec.shape})")
        cross_cka = cka_matrix(t5_enc, sarvam_dec)
        results["t5_encoder_vs_sarvam_decoder"] = cross_cka.tolist()

    # Compare T5 decoder vs Sarvam decoder
    if "decoder" in t5_arrays and "decoder" in sarvam_arrays:
        t5_dec = t5_arrays["decoder"][t5_indices]
        sarvam_dec = sarvam_arrays["decoder"][sarvam_indices]

        print(f"  T5 decoder ({t5_dec.shape}) vs Sarvam decoder ({sarvam_dec.shape})")
        cross_cka = cka_matrix(t5_dec, sarvam_dec)
        results["t5_decoder_vs_sarvam_decoder"] = cross_cka.tolist()

    # Save
    os.makedirs(METRICS_DIR, exist_ok=True)
    output_path = os.path.join(METRICS_DIR, "cross_model_cka.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCross-model CKA saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Layer-wise metrics")
    parser.add_argument("--model", choices=["t5", "sarvam", "both"], default="both")
    parser.add_argument("--cross-model", action="store_true",
                        help="Also compute cross-model CKA")
    args = parser.parse_args()

    if args.model in ("t5", "both"):
        compute_metrics_for_model("t5")

    if args.model in ("sarvam", "both"):
        compute_metrics_for_model("sarvam")

    if args.cross_model or args.model == "both":
        compute_cross_model_cka()

    print("\nLayer-wise metric computation complete!")


if __name__ == "__main__":
    main()
