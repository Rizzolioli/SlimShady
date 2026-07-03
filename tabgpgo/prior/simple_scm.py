"""
Clean-room structural-causal-model (SCM) synthetic dataset generator.

Fallback / ablation backend for the TabPFN-style prior: samples a random DAG,
propagates noise through per-node random linear maps + nonlinearities, then
picks a random subset of nodes as features and one downstream node as target.
Follows the description of the causal prior in the TabPFN paper (Hollmann et
al., 2023) without reusing its code.
"""
import math
import random

import torch

_ACTIVATIONS = [
    lambda t: t,            # identity
    torch.tanh,
    torch.relu,
    torch.sin,
    torch.abs,
]


def generate_dataset(n_rows, n_features):
    """Generate one synthetic regression dataset (X: (n_rows, k), y: (n_rows,)).

    Uses the global `random`/`torch` RNG state — callers seed per dataset.
    """
    n_hidden = random.randint(4, 32)
    n_nodes = n_features + 1 + n_hidden          # features + target + slack
    n_roots = random.randint(2, max(3, n_nodes // 4))
    noise_std = math.exp(random.uniform(math.log(1e-3), math.log(0.3)))
    init_std = math.exp(random.uniform(math.log(0.1), math.log(3.0)))

    values = []                                  # node values in topological order
    for i in range(n_nodes):
        if i < n_roots:
            mean = random.gauss(0, 1)
            std = abs(random.gauss(0, 1)) + 0.1
            values.append(torch.normal(mean, std, (n_rows,)))
            continue
        n_parents = random.randint(1, min(i, 5))
        parents = random.sample(range(i), n_parents)
        weights = torch.normal(0.0, init_std, (n_parents,))
        combined = sum(w * values[p] for w, p in zip(weights, parents))
        combined = combined + random.gauss(0, 1)                    # bias
        combined = combined + torch.normal(0.0, noise_std, (n_rows,))
        act = random.choice(_ACTIVATIONS)
        values.append(act(combined))

    # target: a downstream (late topological order) non-root node
    y_idx = random.randint(max(n_roots, n_nodes - 1 - n_hidden), n_nodes - 1)
    candidates = [i for i in range(n_nodes) if i != y_idx]
    feature_idx = random.sample(candidates, n_features)

    X = torch.stack([values[i] for i in feature_idx], dim=1).float()
    y = values[y_idx].float()
    return X, y
