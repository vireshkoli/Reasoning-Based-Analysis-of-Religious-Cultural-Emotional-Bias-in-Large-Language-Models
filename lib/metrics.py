"""Layer-wise interpretability metrics.

Implements:
- LSS (Layer-wise Sensitivity Score): Representation displacement from religion injection
- LBS (Layer-wise Bias Score): Differential treatment across religions
- LCS (Layer Contribution Score): Causal contribution from ablation
- Attention Entropy: Focus/distribution of attention
- RTAS (Religion Token Attention Score): Attention directed at religion tokens
- CKA (Centered Kernel Alignment): Representational similarity between layers/models
"""

import numpy as np
from scipy import stats


def layer_sensitivity_score(pooled_states, metadata, religion_key="religion",
                            baseline_religion="None"):
    """Compute LSS for each layer.

    LSS(l) = (1/|S|) * (1/|R|) * sum_s sum_r ||h_l^r(s) - h_l^none(s)||_2 / ||h_l^none(s)||_2

    Args:
        pooled_states: numpy array [num_samples, num_layers, hidden_dim].
        metadata: dict with 'samples' list containing scenario_id, religion, mode.

    Returns:
        lss: numpy array [num_layers] with LSS values.
        lss_per_religion: dict mapping religion -> numpy array [num_layers].
    """
    samples = metadata["samples"]
    num_layers = pooled_states.shape[1]

    # Index samples by (scenario_id, mode)
    baseline_idx = {}
    religion_indices = {}

    for i, s in enumerate(samples):
        key = (s["scenario_id"], s["mode"])
        if s[religion_key] == baseline_religion:
            baseline_idx[key] = i
        else:
            if key not in religion_indices:
                religion_indices[key] = {}
            religion_indices[key][s[religion_key]] = i

    religions = sorted({s[religion_key] for s in samples if s[religion_key] != baseline_religion})
    lss_per_religion = {r: np.zeros(num_layers) for r in religions}
    counts_per_religion = {r: 0 for r in religions}

    for key, base_i in baseline_idx.items():
        h_base = pooled_states[base_i]  # [num_layers, hidden_dim]
        base_norm = np.linalg.norm(h_base, axis=1, keepdims=False)  # [num_layers]
        base_norm = np.maximum(base_norm, 1e-10)  # avoid division by zero

        if key not in religion_indices:
            continue

        for r, rel_i in religion_indices[key].items():
            h_rel = pooled_states[rel_i]
            displacement = np.linalg.norm(h_rel - h_base, axis=1) / base_norm
            lss_per_religion[r] += displacement
            counts_per_religion[r] += 1

    for r in religions:
        if counts_per_religion[r] > 0:
            lss_per_religion[r] /= counts_per_religion[r]

    lss = np.mean([lss_per_religion[r] for r in religions], axis=0)
    return lss, lss_per_religion


def layer_bias_score(pooled_states, metadata, religion_key="religion",
                     baseline_religion="None"):
    """Compute LBS for each layer.

    LBS(l) = (1/|S|) * sum_s Var_r[ ||h_l^r(s) - h_l^none(s)||_2 ]

    High LBS = layer treats religions differently.

    Args:
        pooled_states: numpy array [num_samples, num_layers, hidden_dim].
        metadata: dict with 'samples' list.

    Returns:
        lbs: numpy array [num_layers].
        lbs_per_scenario: dict mapping (scenario_id, mode) -> numpy array [num_layers].
    """
    samples = metadata["samples"]
    num_layers = pooled_states.shape[1]

    baseline_idx = {}
    religion_indices = {}

    for i, s in enumerate(samples):
        key = (s["scenario_id"], s["mode"])
        if s[religion_key] == baseline_religion:
            baseline_idx[key] = i
        else:
            if key not in religion_indices:
                religion_indices[key] = {}
            religion_indices[key][s[religion_key]] = i

    lbs_per_scenario = {}
    for key, base_i in baseline_idx.items():
        h_base = pooled_states[base_i]

        if key not in religion_indices:
            continue

        displacements = []
        for r, rel_i in religion_indices[key].items():
            h_rel = pooled_states[rel_i]
            disp = np.linalg.norm(h_rel - h_base, axis=1)  # [num_layers]
            displacements.append(disp)

        if len(displacements) >= 2:
            displacements = np.stack(displacements)  # [num_religions, num_layers]
            lbs_per_scenario[key] = np.var(displacements, axis=0)  # [num_layers]

    if not lbs_per_scenario:
        return np.zeros(num_layers), {}

    lbs = np.mean(list(lbs_per_scenario.values()), axis=0)
    return lbs, lbs_per_scenario


def layer_contribution_score(control_results, ablated_results, metric_fn):
    """Compute LCS from ablation results.

    LCS(l) = metric(full_model) - metric(model_ablated_at_l)

    Args:
        control_results: Phase 2 results (no ablation).
        ablated_results: dict mapping layer_idx -> ablated results.
        metric_fn: Function that computes a scalar metric from results list.

    Returns:
        lcs: dict mapping layer_idx -> float.
    """
    control_metric = metric_fn(control_results)
    lcs = {}
    for layer_idx, results in ablated_results.items():
        ablated_metric = metric_fn(results)
        lcs[layer_idx] = control_metric - ablated_metric
    return lcs


def attention_entropy(attn_weights, epsilon=1e-10):
    """Compute attention entropy per head per layer.

    H(A, i) = -sum_j A[i,j] * log(A[i,j] + eps)
    Entropy(l, h) = mean_i H(A, i)

    Args:
        attn_weights: numpy array [num_heads, seq_len, seq_len] (softmax probabilities).

    Returns:
        entropy: numpy array [num_heads] with mean entropy per head.
    """
    log_attn = np.log(attn_weights + epsilon)
    pointwise_entropy = -attn_weights * log_attn  # [num_heads, seq_len, seq_len]
    per_query_entropy = pointwise_entropy.sum(axis=-1)  # [num_heads, seq_len]
    return per_query_entropy.mean(axis=-1)  # [num_heads]


def religion_token_attention_score(attn_weights, religion_token_indices):
    """Compute RTAS — average attention directed at religion tokens.

    RTAS(l, h) = (1/T_q) * sum_i sum_{j in religion_tokens} A[i,j]

    Args:
        attn_weights: numpy array [num_heads, seq_len, seq_len].
        religion_token_indices: list of token position indices for religion tokens.

    Returns:
        rtas: numpy array [num_heads].
    """
    if not religion_token_indices:
        return np.zeros(attn_weights.shape[0])

    # Sum attention to religion token positions across all query positions
    rtas = attn_weights[:, :, religion_token_indices].sum(axis=-1).mean(axis=-1)
    return rtas  # [num_heads]


def linear_cka(X, Y):
    """Compute Linear Centered Kernel Alignment between two representation matrices.

    Args:
        X: numpy array [n, p] — representations from source.
        Y: numpy array [n, q] — representations from target.

    Returns:
        cka: float in [0, 1]. Higher = more similar representations.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    hsic_xy = np.linalg.norm(YtX, "fro") ** 2
    hsic_xx = np.linalg.norm(XtX, "fro") ** 2
    hsic_yy = np.linalg.norm(YtY, "fro") ** 2

    denom = np.sqrt(hsic_xx * hsic_yy) + 1e-10
    return float(hsic_xy / denom)


def cka_matrix(states_a, states_b):
    """Compute CKA between all layer pairs of two models.

    Args:
        states_a: numpy array [num_samples, num_layers_a, hidden_dim_a].
        states_b: numpy array [num_samples, num_layers_b, hidden_dim_b].

    Returns:
        cka_mat: numpy array [num_layers_a, num_layers_b].
    """
    n_a = states_a.shape[1]
    n_b = states_b.shape[1]
    mat = np.zeros((n_a, n_b))

    for i in range(n_a):
        for j in range(n_b):
            mat[i, j] = linear_cka(states_a[:, i, :], states_b[:, j, :])

    return mat


def permutation_test(metric_fn, pooled_states, metadata, n_permutations=1000,
                     seed=42):
    """Permutation test for LSS significance.

    Permutes religion labels and recomputes metric to build null distribution.

    Args:
        metric_fn: Function(pooled_states, metadata) -> numpy array [num_layers].
        pooled_states: numpy array.
        metadata: dict with 'samples'.
        n_permutations: Number of permutations.

    Returns:
        p_values: numpy array [num_layers].
        observed: numpy array [num_layers] (the real metric values).
    """
    rng = np.random.default_rng(seed)
    observed = metric_fn(pooled_states, metadata)
    num_layers = len(observed)

    null_dist = np.zeros((n_permutations, num_layers))

    for p in range(n_permutations):
        perm_metadata = {
            "keys_order": metadata["keys_order"],
            "samples": [],
        }
        religions = [s["religion"] for s in metadata["samples"]]
        rng.shuffle(religions)
        for i, s in enumerate(metadata["samples"]):
            perm_s = dict(s)
            perm_s["religion"] = religions[i]
            perm_metadata["samples"].append(perm_s)

        null_dist[p] = metric_fn(pooled_states, perm_metadata)

    p_values = np.mean(null_dist >= observed[np.newaxis, :], axis=0)
    return p_values, observed


def cohens_d(group1, group2):
    """Compute Cohen's d effect size.

    Args:
        group1, group2: 1D numpy arrays.

    Returns:
        d: float effect size.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)
