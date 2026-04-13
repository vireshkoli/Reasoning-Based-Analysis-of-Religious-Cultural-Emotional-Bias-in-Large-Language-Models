"""
Phase 3: Attention pattern analysis.

Analyzes attention weight summaries (entropy, RTAS) to identify:
- Which layers/heads attend most to religion tokens
- How attention patterns differ across religious contexts
- Heads that are candidates for targeted ablation

"""

import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import ATTENTION_DIR, TABLES_DIR, PLOTS_DIR


def analyze_attention(model_name):
    """Analyze attention patterns for a model."""
    print(f"\n{'='*60}")
    print(f"Attention Analysis: {model_name.upper()}")
    print(f"{'='*60}")

    summary_path = os.path.join(ATTENTION_DIR, f"{model_name}_attn_summary.json")
    with open(summary_path) as f:
        summaries = json.load(f)

    print(f"Loaded attention summaries for {len(summaries)} samples")

    # Collect entropy and RTAS statistics across all samples
    entropy_by_layer = {}
    rtas_by_layer = {}

    for sample_key, stats in summaries.items():
        for stat_key, values in stats.items():
            if stat_key.endswith("_entropy"):
                layer_key = stat_key.replace("_entropy", "")
                if layer_key not in entropy_by_layer:
                    entropy_by_layer[layer_key] = []
                entropy_by_layer[layer_key].append(values)

            elif stat_key.endswith("_rtas"):
                layer_key = stat_key.replace("_rtas", "")
                if layer_key not in rtas_by_layer:
                    rtas_by_layer[layer_key] = []
                rtas_by_layer[layer_key].append(values)

    # Compute aggregate statistics
    rows = []
    for layer_key in sorted(entropy_by_layer.keys()):
        ent_arr = np.array(entropy_by_layer[layer_key])  # [num_samples, num_heads]
        mean_ent = ent_arr.mean(axis=0)  # [num_heads]
        std_ent = ent_arr.std(axis=0)

        rtas_data = rtas_by_layer.get(layer_key)
        if rtas_data:
            rtas_arr = np.array(rtas_data)
            mean_rtas = rtas_arr.mean(axis=0)
            std_rtas = rtas_arr.std(axis=0)
        else:
            num_heads = len(mean_ent)
            mean_rtas = np.zeros(num_heads)
            std_rtas = np.zeros(num_heads)

        for h in range(len(mean_ent)):
            rows.append({
                "layer": layer_key,
                "head": h,
                "entropy_mean": float(mean_ent[h]),
                "entropy_std": float(std_ent[h]),
                "rtas_mean": float(mean_rtas[h]),
                "rtas_std": float(std_rtas[h]),
            })

    df = pd.DataFrame(rows)

    # Identify top religion-attending heads
    if "rtas_mean" in df.columns and df["rtas_mean"].sum() > 0:
        top_rtas = df.nlargest(10, "rtas_mean")
        print("\nTop 10 religion-attending heads (by RTAS):")
        for _, row in top_rtas.iterrows():
            print(f"  {row['layer']} head {int(row['head'])}: "
                  f"RTAS={row['rtas_mean']:.4f}, entropy={row['entropy_mean']:.3f}")

    # Identify most focused heads (lowest entropy)
    top_focused = df.nsmallest(10, "entropy_mean")
    print("\nTop 10 most focused heads (lowest entropy):")
    for _, row in top_focused.iterrows():
        print(f"  {row['layer']} head {int(row['head'])}: "
              f"entropy={row['entropy_mean']:.3f}, RTAS={row['rtas_mean']:.4f}")

    # Save
    os.makedirs(TABLES_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLES_DIR, f"attention_analysis_{model_name}.csv"), index=False)

    summary = {
        "model": model_name,
        "num_samples": len(summaries),
        "top_rtas_heads": top_rtas[["layer", "head", "rtas_mean"]].to_dict("records") if "rtas_mean" in df.columns and df["rtas_mean"].sum() > 0 else [],
        "top_focused_heads": top_focused[["layer", "head", "entropy_mean"]].to_dict("records"),
    }

    with open(os.path.join(TABLES_DIR, f"attention_summary_{model_name}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else x)

    print(f"\nAttention analysis saved to {TABLES_DIR}")
    return df


def main():
    for model in ["t5", "sarvam"]:
        try:
            analyze_attention(model)
        except FileNotFoundError as e:
            print(f"Skipping {model}: {e}")

    print("\nAttention analysis complete!")


if __name__ == "__main__":
    main()
