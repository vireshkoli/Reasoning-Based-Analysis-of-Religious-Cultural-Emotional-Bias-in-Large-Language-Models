"""
Phase 3: Hidden state and attention weight extraction.

For each model, runs inference on all 1000 samples (100 scenarios x 5 religions x 2 modes),
captures hidden states from all layers via teacher-forced forward pass, mean-pools across
sequence dimension, and saves compact representations to disk.

For a subset of high-bias scenarios, also stores full attention matrices.

"""

import argparse
import json
import os
import sys
import warnings
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import (
    SCENARIOS_FILE, RESULTS_T5_FILE, RESULTS_SARVAM_FILE,
    RELIGIONS, RANDOM_SEED, CHECKPOINT_INTERVAL,
    T5_GEN_PARAMS, SARVAM_GEN_PARAMS_EMOTION, SARVAM_GEN_PARAMS_REASONING,
    ATTENTION_SUBSET_SCENARIOS, DEVICE,
)
from lib.models import T5Adapter, SarvamAdapter
from lib.prompts import inject_religion, direct_prompt, reasoning_prompt, emotion_from_reasoning_prompt, clean_reasoning
from lib.emotions import extract_emotion
from lib.storage import (
    _make_key, save_pooled_states, save_attention_summary, save_attention_full,
    save_checkpoint, load_checkpoint, validate_states,
)
from lib.metrics import attention_entropy, religion_token_attention_score


def load_scenarios():
    with open(SCENARIOS_FILE) as f:
        return json.load(f)


def identify_high_bias_scenarios(results_file, n=ATTENTION_SUBSET_SCENARIOS):
    """Identify scenarios with highest counterfactual sensitivity from Phase 2 results."""
    with open(results_file) as f:
        results = json.load(f)

    # Count unique emotions per scenario (across all religions and modes)
    from collections import defaultdict
    scenario_emotions = defaultdict(set)
    for r in results:
        scenario_emotions[r["scenario_id"]].add(r["emotion"])

    # Sort by number of unique emotions (most varied first)
    ranked = sorted(scenario_emotions.items(), key=lambda x: len(x[1]), reverse=True)
    return [sid for sid, _ in ranked[:n]]


def mean_pool(hidden_states_list):
    """Mean-pool a list of per-layer hidden state arrays across sequence dimension.

    Args:
        hidden_states_list: list of numpy arrays, each [seq_len, hidden_dim].

    Returns:
        numpy array [num_layers, hidden_dim].
    """
    return np.stack([h.mean(axis=0) for h in hidden_states_list])


def find_religion_token_positions(tokenizer, prompt_with_religion, prompt_without_religion):
    """Find token positions that differ between religion-injected and baseline prompts.

    Returns list of token indices corresponding to religion-specific tokens.
    """
    tokens_with = tokenizer.encode(prompt_with_religion)
    tokens_without = tokenizer.encode(prompt_without_religion)

    # Find differing positions (simple heuristic: tokens in 'with' not aligned with 'without')
    religion_positions = []
    min_len = min(len(tokens_with), len(tokens_without))

    for i in range(min_len):
        if tokens_with[i] != tokens_without[i]:
            religion_positions.append(i)

    # Extra tokens at the end of the longer sequence
    if len(tokens_with) > len(tokens_without):
        religion_positions.extend(range(min_len, len(tokens_with)))

    return religion_positions


def extract_t5(scenarios, high_bias_ids):
    """Extract hidden states and attention for T5 model."""
    print("=" * 60)
    print("Extracting hidden states from T5 (FLAN-T5-Large)")
    print("=" * 60)

    adapter = T5Adapter()
    adapter.load()
    print(f"Model loaded on {adapter.device}")

    torch.manual_seed(RANDOM_SEED)

    completed_ids, _ = load_checkpoint("t5", "extract")
    all_states = {}
    all_metadata = []
    attn_summaries = {}

    for scenario in tqdm(scenarios, desc="T5 extraction"):
        scenario_id = scenario["id"]
        if scenario_id in completed_ids:
            continue

        base_text = scenario["scenario"]
        domain = scenario["domain"]

        contexts = [("None", base_text)]
        for r in RELIGIONS:
            contexts.append((r, inject_religion(base_text, domain, r)))

        for religion, text in contexts:
            for mode in ["direct", "reasoning"]:
                key = _make_key(scenario_id, religion, mode)

                # Build prompt based on mode
                if mode == "direct":
                    prompt_text = direct_prompt(text, model_type="t5")
                    gen_kwargs = T5_GEN_PARAMS.copy()
                else:
                    # Step 1: Generate reasoning
                    r_prompt = reasoning_prompt(text, model_type="t5")
                    reasoning_output, _ = adapter.generate(r_prompt, **T5_GEN_PARAMS)
                    # Step 2: Classify from reasoning
                    prompt_text = emotion_from_reasoning_prompt(reasoning_output, model_type="t5")
                    gen_kwargs = T5_GEN_PARAMS.copy()

                # Generate output tokens
                response, output_ids = adapter.generate(prompt_text, **gen_kwargs)
                emotion = extract_emotion(response)

                # Teacher-forced forward pass for hidden states
                inputs = adapter.tokenize(prompt_text)
                states = adapter.forward_with_states(
                    input_ids=inputs["input_ids"],
                    decoder_input_ids=output_ids.unsqueeze(0) if output_ids.dim() == 1 else output_ids,
                )

                # Validate
                if not validate_states(
                    {"enc": np.stack(states["encoder_hidden_states"]),
                     "dec": np.stack(states["decoder_hidden_states"])},
                    label=f"T5 {key}"
                ):
                    warnings.warn(f"Invalid states for {key}, skipping")
                    continue

                # Mean-pool
                pooled = {
                    "encoder": mean_pool(states["encoder_hidden_states"]),
                    "decoder": mean_pool(states["decoder_hidden_states"]),
                }

                all_states[key] = pooled
                all_metadata.append({
                    "scenario_id": scenario_id,
                    "religion": religion,
                    "mode": mode,
                    "domain": domain,
                    "emotion": emotion,
                    "raw_output": response,
                })

                # Attention summary stats
                attn_summary_entry = {}
                for lt in ["encoder_attentions", "decoder_attentions", "cross_attentions"]:
                    for layer_i, attn in enumerate(states[lt]):
                        ent = attention_entropy(attn).tolist()
                        attn_summary_entry[f"{lt}_{layer_i}_entropy"] = ent

                        # RTAS for religion contexts
                        if religion != "None" and lt == "encoder_attentions":
                            base_prompt = direct_prompt(base_text) if mode == "direct" else prompt_text
                            rel_positions = find_religion_token_positions(
                                adapter.tokenizer, prompt_text, direct_prompt(base_text)
                            )
                            rtas = religion_token_attention_score(attn, rel_positions).tolist()
                            attn_summary_entry[f"{lt}_{layer_i}_rtas"] = rtas

                attn_summaries[key] = attn_summary_entry

                # Store full attention for high-bias subset (direct mode only)
                if scenario_id in high_bias_ids and mode == "direct":
                    save_attention_full("t5", key, {
                        "encoder_attentions": states["encoder_attentions"],
                        "decoder_attentions": states["decoder_attentions"],
                        "cross_attentions": states["cross_attentions"],
                    })

        completed_ids.append(scenario_id)

        # Checkpoint
        if len(completed_ids) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint("t5", "extract", completed_ids, {})
            print(f"  Checkpoint saved: {len(completed_ids)} scenarios complete")

    # Save all states
    save_pooled_states("t5", all_states, all_metadata)
    save_attention_summary("t5", attn_summaries)
    print(f"T5 extraction complete: {len(all_states)} samples saved")

    adapter.unload()


def extract_sarvam(scenarios, high_bias_ids):
    """Extract hidden states and attention for Sarvam model."""
    print("=" * 60)
    print("Extracting hidden states from Sarvam (sarvam-2b)")
    print("=" * 60)

    adapter = SarvamAdapter()
    adapter.load()
    print(f"Model loaded on {adapter.device}")

    # Load embedding model for emotion extraction fallback
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    emotion_embeddings = embed_model.encode(["Joy", "Sadness", "Anger", "Fear", "Neutral"])

    completed_ids, _ = load_checkpoint("sarvam", "extract")
    all_states = {}
    all_metadata = []
    attn_summaries = {}

    for scenario in tqdm(scenarios, desc="Sarvam extraction"):
        scenario_id = scenario["id"]
        if scenario_id in completed_ids:
            continue

        base_text = scenario["scenario"]
        domain = scenario["domain"]

        contexts = [("None", base_text)]
        for r in RELIGIONS:
            contexts.append((r, inject_religion(base_text, domain, r)))

        for religion, text in contexts:
            for mode in ["direct", "reasoning"]:
                key = _make_key(scenario_id, religion, mode)

                if mode == "direct":
                    prompt_text = direct_prompt(text, model_type="sarvam")
                    gen_kwargs = SARVAM_GEN_PARAMS_EMOTION.copy()
                    gen_kwargs["pad_token_id"] = adapter.tokenizer.eos_token_id
                else:
                    # Step 1: Generate reasoning
                    r_prompt = reasoning_prompt(text, model_type="sarvam")
                    r_kwargs = SARVAM_GEN_PARAMS_REASONING.copy()
                    r_kwargs["pad_token_id"] = adapter.tokenizer.eos_token_id
                    reasoning_output, _ = adapter.generate(r_prompt, **r_kwargs)
                    reasoning_output = clean_reasoning(reasoning_output)

                    # Step 2: Classify from reasoning
                    prompt_text = emotion_from_reasoning_prompt(reasoning_output, model_type="sarvam")
                    gen_kwargs = SARVAM_GEN_PARAMS_EMOTION.copy()
                    gen_kwargs["pad_token_id"] = adapter.tokenizer.eos_token_id

                # Generate output tokens
                response, full_ids = adapter.generate(prompt_text, **gen_kwargs)
                emotion = extract_emotion(
                    response, use_embedding_fallback=True,
                    embed_model=embed_model, emotion_embeddings=emotion_embeddings,
                )

                # Forward pass on full sequence for hidden states
                states = adapter.forward_with_states(
                    input_ids=full_ids.unsqueeze(0) if full_ids.dim() == 1 else full_ids,
                )

                # Validate
                if not validate_states(
                    {"dec": np.stack(states["decoder_hidden_states"])},
                    label=f"Sarvam {key}"
                ):
                    warnings.warn(f"Invalid states for {key}, skipping")
                    continue

                # Mean-pool
                pooled = {
                    "decoder": mean_pool(states["decoder_hidden_states"]),
                }

                all_states[key] = pooled
                all_metadata.append({
                    "scenario_id": scenario_id,
                    "religion": religion,
                    "mode": mode,
                    "domain": domain,
                    "emotion": emotion,
                    "raw_output": response,
                })

                # Attention summary stats
                attn_summary_entry = {}
                for layer_i, attn in enumerate(states["decoder_attentions"]):
                    ent = attention_entropy(attn).tolist()
                    attn_summary_entry[f"decoder_attn_{layer_i}_entropy"] = ent

                    if religion != "None":
                        base_prompt = direct_prompt(base_text) if mode == "direct" else prompt_text
                        rel_positions = find_religion_token_positions(
                            adapter.tokenizer, prompt_text, direct_prompt(base_text)
                        )
                        rtas = religion_token_attention_score(attn, rel_positions).tolist()
                        attn_summary_entry[f"decoder_attn_{layer_i}_rtas"] = rtas

                attn_summaries[key] = attn_summary_entry

                # Full attention for high-bias subset
                if scenario_id in high_bias_ids and mode == "direct":
                    save_attention_full("sarvam", key, {
                        "decoder_attentions": states["decoder_attentions"],
                    })

        completed_ids.append(scenario_id)

        if len(completed_ids) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint("sarvam", "extract", completed_ids, {})
            print(f"  Checkpoint saved: {len(completed_ids)} scenarios complete")

    save_pooled_states("sarvam", all_states, all_metadata)
    save_attention_summary("sarvam", attn_summaries)
    print(f"Sarvam extraction complete: {len(all_states)} samples saved")

    adapter.unload()


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Hidden state extraction")
    parser.add_argument("--model", choices=["t5", "sarvam", "both"], default="both")
    args = parser.parse_args()

    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios")

    if args.model in ("t5", "both"):
        high_bias_t5 = identify_high_bias_scenarios(RESULTS_T5_FILE)
        print(f"High-bias scenarios for T5 attention subset: {high_bias_t5[:5]}...")
        extract_t5(scenarios, high_bias_t5)

    if args.model in ("sarvam", "both"):
        high_bias_sarvam = identify_high_bias_scenarios(RESULTS_SARVAM_FILE)
        print(f"High-bias scenarios for Sarvam attention subset: {high_bias_sarvam[:5]}...")
        extract_sarvam(scenarios, high_bias_sarvam)

    print("\nExtraction complete!")


if __name__ == "__main__":
    main()
