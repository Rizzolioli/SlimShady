"""
Synthetic dataset generation backends for the TabGPGO prior.

`generate_datasets` is the single entry point; each backend module exposes
`generate_dataset(n_rows, n_features) -> (X, y)` operating on the global RNG
state (seeded here per dataset for reproducibility).
"""
import random

import numpy as np
import torch

from . import simple_scm, tabpfn_v1

_BACKENDS = {
    "tabpfn_v1": tabpfn_v1.generate_dataset,
    "simple_scm": simple_scm.generate_dataset,
}


def generate_datasets(n_datasets, n_rows, max_features, min_features, seed,
                      backend="tabpfn_v1"):
    """Yield `n_datasets` synthetic regression datasets as CPU float32 tensors.

    Each dataset gets a random feature count k in [min_features, max_features]
    and is generated under a deterministic per-dataset seed derived from
    `seed`, so the stream is reproducible regardless of consumption order.
    """
    generate = _BACKENDS[backend]
    for i in range(n_datasets):
        ds_seed = seed + i
        random.seed(ds_seed)
        np.random.seed(ds_seed % (2 ** 32))
        torch.manual_seed(ds_seed)
        k = random.randint(min_features, max_features)
        for _ in range(10):
            X, y = generate(n_rows, k)
            if (torch.isfinite(X).all() and torch.isfinite(y).all()
                    and y.std() > 1e-8):
                break
        else:
            raise RuntimeError(f"{backend}: only degenerate datasets for seed {ds_seed}")
        yield X, y
