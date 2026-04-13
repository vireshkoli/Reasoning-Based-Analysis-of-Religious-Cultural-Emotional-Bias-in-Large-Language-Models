"""
Phase 3: Probing classifier training per layer.

Trains logistic regression probes at each layer to predict:
1. Emotion labels — reveals where emotion info crystallizes
2. Religion labels — reveals where religious context is encoded

The intersection of these curves identifies layers where religion entangles with emotion.

"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import PROBING_DIR
from lib.storage import load_pooled_states
from lib.probing import probe_all_layers, encode_emotion_labels, encode_religion_labels


def run_probing(model_name):
    """Run probing classifiers for a single model."""
    print(f"\n{'='*60}")
    print(f"Probing classifiers for {model_name.upper()}")
    print(f"{'='*60}")

    arrays, metadata = load_pooled_states(model_name)
    print(f"Loaded {arrays[list(arrays.keys())[0]].shape[0]} samples")

    emotion_labels = encode_emotion_labels(metadata)
    religion_labels, religion_map = encode_religion_labels(metadata)

    all_results = {}

    for layer_type, states in arrays.items():
        print(f"\n--- Layer type: {layer_type} ---")
        print(f"  Shape: {states.shape}")

        # Emotion probing
        print(f"\n  Emotion probing ({layer_type}):")
        emotion_results = probe_all_layers(states, emotion_labels, task_name="emotion")
        all_results[f"{layer_type}_emotion"] = emotion_results

        # Religion probing
        print(f"\n  Religion probing ({layer_type}):")
        religion_results = probe_all_layers(states, religion_labels, task_name="religion")
        all_results[f"{layer_type}_religion"] = religion_results

    # Save results
    os.makedirs(PROBING_DIR, exist_ok=True)
    output_path = os.path.join(PROBING_DIR, f"probing_results_{model_name}.json")
    with open(output_path, "w") as f:
        json.dump({
            "model": model_name,
            "religion_label_map": religion_map,
            "results": all_results,
        }, f, indent=2)

    print(f"\nProbing results saved to {output_path}")

    # Print summary
    for key, results in all_results.items():
        print(f"\n  {key}:")
        for r in results:
            print(f"    Layer {r['layer_idx']}: acc={r['accuracy_mean']:.3f} "
                  f"(±{r['accuracy_std']:.3f}), F1={r['f1_macro_mean']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Probing classifiers")
    parser.add_argument("--model", choices=["t5", "sarvam", "both"], default="both")
    args = parser.parse_args()

    if args.model in ("t5", "both"):
        run_probing("t5")

    if args.model in ("sarvam", "both"):
        run_probing("sarvam")

    print("\nProbing complete!")


if __name__ == "__main__":
    main()
