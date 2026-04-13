"""Centralized configuration for all experiments and analyses."""

import os
import torch

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PHASE3_DIR = os.path.join(RESULTS_DIR, "phase3")

SCENARIOS_FILE = os.path.join(DATASET_DIR, "scenarios.json")
RESULTS_T5_FILE = os.path.join(RESULTS_DIR, "results_T5.json")
RESULTS_SARVAM_FILE = os.path.join(RESULTS_DIR, "results_sarvam.json")

HIDDEN_STATES_DIR = os.path.join(PHASE3_DIR, "hidden_states")
ATTENTION_DIR = os.path.join(PHASE3_DIR, "attention_weights")
PROBING_DIR = os.path.join(PHASE3_DIR, "probing")
ABLATION_DIR = os.path.join(PHASE3_DIR, "ablation")
METRICS_DIR = os.path.join(PHASE3_DIR, "metrics")
PLOTS_DIR = os.path.join(PHASE3_DIR, "plots")
TABLES_DIR = os.path.join(PHASE3_DIR, "tables")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")

# --- Device ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# --- Models ---
T5_MODEL_NAME = "google/flan-t5-large"
SARVAM_MODEL_NAME = "sarvamai/sarvam-2b"

# T5 architecture: 24 encoder + 24 decoder layers, hidden_dim=1024, 16 heads
T5_NUM_ENCODER_LAYERS = 24
T5_NUM_DECODER_LAYERS = 24
T5_HIDDEN_DIM = 1024
T5_NUM_HEADS = 16

# Sarvam architecture: 28 decoder layers, hidden_dim=2048, 16 heads (8 KV heads, GQA)
SARVAM_NUM_LAYERS = 28
SARVAM_HIDDEN_DIM = 2048
SARVAM_NUM_HEADS = 16
SARVAM_NUM_KV_HEADS = 8

# --- Experiment Parameters ---
RELIGIONS = ["Hindu", "Muslim", "Sikh", "Christian"]
EMOTION_LIST = ["Joy", "Sadness", "Anger", "Fear", "Neutral"]
EMOTION_MAP = {e: i for i, e in enumerate(EMOTION_LIST)}
DOMAINS = ["family", "workspace", "legal", "general"]

# --- Generation Parameters ---
T5_GEN_PARAMS = {
    "max_new_tokens": 200,
    "temperature": 0.8,
    "do_sample": True,
    "top_p": 0.9,
}

SARVAM_GEN_PARAMS_EMOTION = {
    "max_new_tokens": 5,
    "do_sample": False,
}

SARVAM_GEN_PARAMS_REASONING = {
    "max_new_tokens": 50,
    "do_sample": False,
}

# --- Phase 3 Parameters ---
RANDOM_SEED = 42
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N scenarios

# Probing classifier
PROBING_CV_FOLDS = 5
PROBING_MAX_ITER = 1000

# Ablation
ABLATION_TOP_K_LAYERS = 5  # Number of top layers for attention head ablation
ABLATION_SUBSET_SCENARIOS = 20  # Number of scenarios for head ablation subset

# Attention storage
ATTENTION_SUBSET_SCENARIOS = 20  # Number of high-bias scenarios for full attention storage

# Statistical tests
PERMUTATION_N_ITER = 1000
SIGNIFICANCE_LEVEL = 0.05
