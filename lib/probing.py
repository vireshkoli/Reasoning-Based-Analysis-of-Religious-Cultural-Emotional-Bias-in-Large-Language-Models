"""Probing classifiers for layer-wise analysis.

Trains logistic regression probes at each layer to predict:
1. Emotion labels — where does emotion information crystallize?
2. Religion labels — where is religious context encoded?
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_config import PROBING_CV_FOLDS, PROBING_MAX_ITER, EMOTION_MAP


def train_layer_probe(layer_states, labels, n_folds=PROBING_CV_FOLDS):
    """Train a logistic regression probe on hidden states from a single layer.

    Args:
        layer_states: numpy array [num_samples, hidden_dim].
        labels: numpy array [num_samples] of integer-encoded labels.
        n_folds: Number of cross-validation folds.

    Returns:
        dict with accuracy, f1_macro, and per-fold scores.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(layer_states)

    clf = LogisticRegression(
        max_iter=PROBING_MAX_ITER,
        multi_class="multinomial",
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
    )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    acc_scores = cross_val_score(clf, X, labels, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(clf, X, labels, cv=cv, scoring="f1_macro")

    return {
        "accuracy_mean": float(acc_scores.mean()),
        "accuracy_std": float(acc_scores.std()),
        "f1_macro_mean": float(f1_scores.mean()),
        "f1_macro_std": float(f1_scores.std()),
        "fold_accuracies": acc_scores.tolist(),
        "fold_f1s": f1_scores.tolist(),
    }


def probe_all_layers(pooled_states, labels, task_name="emotion"):
    """Train probing classifiers at every layer.

    Args:
        pooled_states: numpy array [num_samples, num_layers, hidden_dim].
        labels: numpy array [num_samples] of integer labels.
        task_name: Name for logging ('emotion' or 'religion').

    Returns:
        results: list of dicts, one per layer, each containing probe metrics.
    """
    num_layers = pooled_states.shape[1]
    results = []

    for l in range(num_layers):
        print(f"  Probing layer {l}/{num_layers-1} for {task_name}...")
        layer_data = pooled_states[:, l, :]
        probe_result = train_layer_probe(layer_data, labels)
        probe_result["layer_idx"] = l
        probe_result["task"] = task_name
        results.append(probe_result)

    return results


def encode_emotion_labels(metadata):
    """Convert emotion strings to integer labels.

    Args:
        metadata: dict with 'samples' list, each having 'emotion' key.

    Returns:
        labels: numpy array [num_samples] of integers.
    """
    return np.array([EMOTION_MAP.get(s["emotion"], -1) for s in metadata["samples"]])


def encode_religion_labels(metadata):
    """Convert religion strings to integer labels.

    Args:
        metadata: dict with 'samples' list, each having 'religion' key.

    Returns:
        labels: numpy array [num_samples] of integers.
        label_map: dict mapping religion string to int.
    """
    religions = sorted({s["religion"] for s in metadata["samples"]})
    label_map = {r: i for i, r in enumerate(religions)}
    labels = np.array([label_map[s["religion"]] for s in metadata["samples"]])
    return labels, label_map
