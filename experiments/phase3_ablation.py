"""
Phase 3: Ablation experiments.

Experiment A: Single-layer bypass ablation
  - For each layer, bypass it and re-run all 1000 predictions
  - Measure impact on emotion accuracy, bias rate, religion trigger bias

Experiment B: Attention head ablation (targeted)
  - For top-K layers with highest LCS from Experiment A
  - Ablate each head individually on a scenario subset

"""

import argparse
import json
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import (
    SCENARIOS_FILE, RELIGIONS, ABLATION_DIR,
    T5_GEN_PARAMS, SARVAM_GEN_PARAMS_EMOTION, SARVAM_GEN_PARAMS_REASONING,
    ABLATION_TOP_K_LAYERS, ABLATION_SUBSET_SCENARIOS,
)
from lib.models import T5Adapter, SarvamAdapter
from lib.prompts import inject_religion, direct_prompt, reasoning_prompt, emotion_from_reasoning_prompt, clean_reasoning
from lib.emotions import extract_emotion
from lib.ablation import LayerAblation, AttentionHeadAblation


def load_scenarios():
    with open(SCENARIOS_FILE) as f:
        return json.load(f)


def compute_bias_metrics(results):
    """Compute bias metrics from a list of prediction results.

    Returns dict with scenario_bias_rate, religion_trigger_bias, mode_bias.
    """
    from collections import defaultdict

    # Scenario bias rate
    scenario_emotions = defaultdict(set)
    for r in results:
        scenario_emotions[r["scenario_id"]].add(r["emotion"])

    bias_flags = [len(v) > 1 for v in scenario_emotions.values()]
    scenario_bias_rate = float(np.mean(bias_flags) * 100) if bias_flags else 0.0

    # Religion trigger bias
    baseline = {}
    for r in results:
        if r["religion"] == "None":
            baseline[(r["scenario_id"], r["mode"])] = r["emotion"]

    religion_trigger = {}
    for rel in RELIGIONS:
        changes, total = 0, 0
        for r in results:
            if r["religion"] == rel:
                key = (r["scenario_id"], r["mode"])
                if key in baseline:
                    total += 1
                    if baseline[key] != r["emotion"]:
                        changes += 1
        if total > 0:
            religion_trigger[rel] = float(changes / total * 100)

    # Mode bias
    mode_bias = {}
    for mode in ["direct", "reasoning"]:
        mode_results = [r for r in results if r["mode"] == mode]
        mode_scenario_emotions = defaultdict(set)
        for r in mode_results:
            mode_scenario_emotions[r["scenario_id"]].add(r["emotion"])
        flags = [len(v) > 1 for v in mode_scenario_emotions.values()]
        mode_bias[mode] = float(np.mean(flags) * 100) if flags else 0.0

    return {
        "scenario_bias_rate": scenario_bias_rate,
        "religion_trigger_bias": religion_trigger,
        "mode_bias": mode_bias,
    }


def run_inference_batch(adapter, scenarios, model_type, gen_kwargs_direct,
                        gen_kwargs_reasoning=None, extract_fn=None):
    """Run full inference on all scenarios with current model state (including any hooks)."""
    results = []

    for scenario in scenarios:
        base_text = scenario["scenario"]
        scenario_id = scenario["id"]
        domain = scenario["domain"]

        contexts = [("None", base_text)]
        for r in RELIGIONS:
            contexts.append((r, inject_religion(base_text, domain, r)))

        for religion, text in contexts:
            # Direct mode
            prompt_text = direct_prompt(text, model_type=model_type)
            response, _ = adapter.generate(prompt_text, **gen_kwargs_direct)
            emotion = extract_fn(response) if extract_fn else extract_emotion(response)

            results.append({
                "scenario_id": scenario_id,
                "religion": religion,
                "mode": "direct",
                "emotion": emotion,
            })

            # Reasoning mode
            r_prompt = reasoning_prompt(text, model_type=model_type)
            r_kwargs = gen_kwargs_reasoning if gen_kwargs_reasoning else gen_kwargs_direct
            reasoning_output, _ = adapter.generate(r_prompt, **r_kwargs)
            if model_type == "sarvam":
                reasoning_output = clean_reasoning(reasoning_output)

            e_prompt = emotion_from_reasoning_prompt(reasoning_output, model_type=model_type)
            e_response, _ = adapter.generate(e_prompt, **gen_kwargs_direct)
            emotion2 = extract_fn(e_response) if extract_fn else extract_emotion(e_response)

            results.append({
                "scenario_id": scenario_id,
                "religion": religion,
                "mode": "reasoning",
                "emotion": emotion2,
            })

    return results


def layer_ablation_experiment(model_name, scenarios):
    """Experiment A: Single-layer bypass ablation."""
    print(f"\n{'='*60}")
    print(f"Layer Ablation Experiment: {model_name.upper()}")
    print(f"{'='*60}")

    if model_name == "t5":
        adapter = T5Adapter()
        model_type = "t5"
        gen_kwargs_direct = T5_GEN_PARAMS.copy()
        gen_kwargs_reasoning = T5_GEN_PARAMS.copy()
        extract_fn = None
    else:
        adapter = SarvamAdapter()
        model_type = "sarvam"
        gen_kwargs_direct = SARVAM_GEN_PARAMS_EMOTION.copy()
        gen_kwargs_reasoning = SARVAM_GEN_PARAMS_REASONING.copy()

        # Load embedding fallback
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        emotion_embeddings = embed_model.encode(["Joy", "Sadness", "Anger", "Fear", "Neutral"])
        extract_fn = lambda text: extract_emotion(
            text, use_embedding_fallback=True,
            embed_model=embed_model, emotion_embeddings=emotion_embeddings,
        )

    adapter.load()
    print(f"Model loaded on {adapter.device}")

    layer_info = adapter.get_num_layers()
    ablation = LayerAblation()
    all_ablation_results = {}

    for layer_type, num_layers in layer_info.items():
        print(f"\n--- Ablating {layer_type} layers ({num_layers} total) ---")

        for layer_idx in tqdm(range(num_layers), desc=f"{layer_type} ablation"):
            # Register ablation hook
            ablation.ablate_layer(adapter, layer_type, layer_idx, method="bypass")

            # Run inference
            if model_name == "sarvam":
                gen_kwargs_direct["pad_token_id"] = adapter.tokenizer.eos_token_id
                gen_kwargs_reasoning["pad_token_id"] = adapter.tokenizer.eos_token_id

            results = run_inference_batch(
                adapter, scenarios, model_type,
                gen_kwargs_direct, gen_kwargs_reasoning, extract_fn,
            )

            # Compute metrics
            metrics = compute_bias_metrics(results)
            key = f"{layer_type}_layer_{layer_idx}"
            all_ablation_results[key] = {
                "layer_type": layer_type,
                "layer_idx": layer_idx,
                "metrics": metrics,
                "num_predictions": len(results),
                "predictions": results,
            }

            # Remove hook for next iteration
            ablation.remove_all()

    # Save results
    os.makedirs(ABLATION_DIR, exist_ok=True)
    output_path = os.path.join(ABLATION_DIR, f"layer_ablation_{model_name}.json")
    with open(output_path, "w") as f:
        # Save metrics only (predictions are large)
        save_data = {}
        for key, data in all_ablation_results.items():
            save_data[key] = {
                "layer_type": data["layer_type"],
                "layer_idx": data["layer_idx"],
                "metrics": data["metrics"],
                "num_predictions": data["num_predictions"],
            }
        json.dump(save_data, f, indent=2)

    print(f"\nLayer ablation results saved to {output_path}")
    adapter.unload()
    return all_ablation_results


def head_ablation_experiment(model_name, scenarios, top_k_layers=ABLATION_TOP_K_LAYERS):
    """Experiment B: Attention head ablation on top-K layers."""
    print(f"\n{'='*60}")
    print(f"Head Ablation Experiment: {model_name.upper()}")
    print(f"{'='*60}")

    # Load layer ablation results to identify top-K layers
    layer_results_path = os.path.join(ABLATION_DIR, f"layer_ablation_{model_name}.json")
    if not os.path.exists(layer_results_path):
        print("ERROR: Run layer ablation first (--experiment layer)")
        return

    with open(layer_results_path) as f:
        layer_results = json.load(f)

    # Also need control (Phase 2) metrics for LCS
    from config.experiment_config import RESULTS_T5_FILE, RESULTS_SARVAM_FILE
    control_file = RESULTS_T5_FILE if model_name == "t5" else RESULTS_SARVAM_FILE
    with open(control_file) as f:
        control_results = json.load(f)
    control_metrics = compute_bias_metrics(control_results)

    # Compute LCS and find top-K layers
    lcs_scores = {}
    for key, data in layer_results.items():
        lcs = control_metrics["scenario_bias_rate"] - data["metrics"]["scenario_bias_rate"]
        lcs_scores[key] = abs(lcs)

    top_layers = sorted(lcs_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_layers]
    print(f"Top-{top_k_layers} layers by |LCS|:")
    for key, score in top_layers:
        print(f"  {key}: |LCS| = {score:.2f}")

    # Use scenario subset
    subset_scenarios = scenarios[:ABLATION_SUBSET_SCENARIOS]

    if model_name == "t5":
        adapter = T5Adapter()
        model_type = "t5"
        gen_kwargs_direct = T5_GEN_PARAMS.copy()
        gen_kwargs_reasoning = T5_GEN_PARAMS.copy()
        extract_fn = None
    else:
        adapter = SarvamAdapter()
        model_type = "sarvam"
        gen_kwargs_direct = SARVAM_GEN_PARAMS_EMOTION.copy()
        gen_kwargs_reasoning = SARVAM_GEN_PARAMS_REASONING.copy()
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        emotion_embeddings = embed_model.encode(["Joy", "Sadness", "Anger", "Fear", "Neutral"])
        extract_fn = lambda text: extract_emotion(
            text, use_embedding_fallback=True,
            embed_model=embed_model, emotion_embeddings=emotion_embeddings,
        )

    adapter.load()
    num_heads = adapter.get_num_heads()
    head_ablation = AttentionHeadAblation(num_heads)
    all_head_results = {}

    for layer_key, _ in top_layers:
        layer_data = layer_results[layer_key]
        layer_type = layer_data["layer_type"]
        layer_idx = layer_data["layer_idx"]

        print(f"\n--- Head ablation: {layer_key} ({num_heads} heads) ---")

        for head_idx in tqdm(range(num_heads), desc=f"{layer_key} heads"):
            head_ablation.ablate_head(adapter, layer_type, layer_idx, head_idx)

            if model_name == "sarvam":
                gen_kwargs_direct["pad_token_id"] = adapter.tokenizer.eos_token_id
                gen_kwargs_reasoning["pad_token_id"] = adapter.tokenizer.eos_token_id

            results = run_inference_batch(
                adapter, subset_scenarios, model_type,
                gen_kwargs_direct, gen_kwargs_reasoning, extract_fn,
            )

            metrics = compute_bias_metrics(results)
            result_key = f"{layer_key}_head_{head_idx}"
            all_head_results[result_key] = {
                "layer_type": layer_type,
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "metrics": metrics,
            }

            head_ablation.remove_all()

    # Save
    output_path = os.path.join(ABLATION_DIR, f"head_ablation_{model_name}.json")
    with open(output_path, "w") as f:
        json.dump(all_head_results, f, indent=2)

    print(f"\nHead ablation results saved to {output_path}")
    adapter.unload()


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Ablation experiments")
    parser.add_argument("--model", choices=["t5", "sarvam"], required=True)
    parser.add_argument("--experiment", choices=["layer", "head", "both"], default="both")
    args = parser.parse_args()

    scenarios = load_scenarios()

    if args.experiment in ("layer", "both"):
        layer_ablation_experiment(args.model, scenarios)

    if args.experiment in ("head", "both"):
        head_ablation_experiment(args.model, scenarios)

    print("\nAblation experiments complete!")


if __name__ == "__main__":
    main()
