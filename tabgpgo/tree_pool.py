"""
Phase 3: the precomputed base-tree pool.

A fixed pool of random GP trees over the 512 latent tokens is generated once
(ramped half-and-half, like rhh), each tree's raw semantics are evaluated once
on the static training/validation tokens, and evolution afterwards only ever
indexes into the resulting (pool_size, n_rows) tensors.

`evaluate_structure` is the single tree interpreter used both for pool
construction and for inference on unseen data, so the two paths agree by
construction.
"""
import random

import torch

from algorithms.GP.representations.tree_utils import (
    create_full_random_tree, create_grow_random_tree, flatten, tree_depth)
from utils.utils import protected_div

FUNCTIONS = {
    'add':      {'function': lambda x, y: torch.add(x, y), 'arity': 2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
    'divide':   {'function': lambda x, y: protected_div(x, y), 'arity': 2},
}

# Python floats (not CPU tensors) so binary ops broadcast on any device.
CONSTANTS = {
    'constant_2':  lambda _: 2.0,
    'constant_3':  lambda _: 3.0,
    'constant_4':  lambda _: 4.0,
    'constant_5':  lambda _: 5.0,
    'constant__1': lambda _: -1.0,
}

BOUND = 1e12


def make_terminals(latent_dim):
    return {f"z{i}": i for i in range(latent_dim)}


def evaluate_structure(structure, T, TERMINALS):
    """Evaluate a nested-tuple tree on latent tokens T (n_rows, latent_dim)."""
    if isinstance(structure, tuple):
        fname = structure[0]
        left = evaluate_structure(structure[1], T, TERMINALS)
        right = evaluate_structure(structure[2], T, TERMINALS)
        out = FUNCTIONS[fname]['function'](left, right)
        if isinstance(out, torch.Tensor):
            out = torch.clamp(out, -BOUND, BOUND)
        return out
    if structure in TERMINALS:
        return T[:, TERMINALS[structure]]
    return CONSTANTS[structure](None)


def build_pool(cfg, TERMINALS):
    """Generate cfg.pool_size random tree structures (ramped, half grow/full).

    Returns the registry: list of {"structure", "nodes", "depth"} dicts.
    Uses the global `random` RNG — caller seeds beforehand.
    """
    depth_fn = tree_depth(FUNCTIONS)
    registry = []
    depths = list(range(2, cfg.init_depth + 1))
    for i in range(cfg.pool_size):
        depth = depths[i % len(depths)]
        if i % 2 == 0:
            structure = create_grow_random_tree(depth, FUNCTIONS, TERMINALS,
                                                CONSTANTS, p_c=cfg.p_c)
        else:
            structure = create_full_random_tree(depth, FUNCTIONS, TERMINALS,
                                                CONSTANTS, p_c=cfg.p_c)
        nodes = len(list(flatten(structure))) if isinstance(structure, tuple) else 1
        registry.append({"structure": structure, "nodes": nodes,
                         "depth": depth_fn(structure)})
    return registry


@torch.no_grad()
def evaluate_pool(registry, T, TERMINALS, dtype=torch.float32):
    """Raw semantics of every pool tree on tokens T -> (pool_size, n_rows)."""
    n_rows = T.shape[0]
    pool = torch.empty((len(registry), n_rows), dtype=dtype, device=T.device)
    for i, entry in enumerate(registry):
        sem = evaluate_structure(entry["structure"], T, TERMINALS)
        if not isinstance(sem, torch.Tensor) or sem.dim() == 0:
            sem = torch.full((n_rows,), float(sem), device=T.device)
        pool[i] = sem.to(dtype)
    return pool
