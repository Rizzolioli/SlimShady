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
from utils.utils import protected_div, protected_log, protected_sqrt

FUNCTIONS = {
    # Binary arithmetic
    'add':      {'function': lambda x, y: torch.add(x, y), 'arity': 2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
    'divide':   {'function': lambda x, y: protected_div(x, y), 'arity': 2},
}


EXTRA_FUNCTIONS = {

    # ----------------------------------------------------
    # Classical symbolic regression nonlinearities
    # ----------------------------------------------------

    'sin': {
        'function': lambda x: torch.sin(x),
        'arity': 1
    },

    'cos': {
        'function': lambda x: torch.cos(x),
        'arity': 1
    },

    'tan': {
        'function': lambda x: torch.tan(torch.clamp(x, -10, 10)),
        'arity': 1
    },


    # ----------------------------------------------------
    # Domain-protected mathematical functions
    # ----------------------------------------------------

    'log': {
        'function': lambda x: protected_log(x),
        'arity': 1
    },

    'sqrt': {
        'function': lambda x: protected_sqrt(x),
        'arity': 1
    },

    'exp': {
        'function': lambda x: torch.exp(torch.clamp(x, -10, 10)),
        'arity': 1
    },

    'neg_exp': {
        'function': lambda x: torch.exp(torch.clamp(-x, -10, 10)),
        'arity': 1
    },

    'logabs': {
        'function': lambda x: torch.log(torch.abs(x) + 1e-8),
        'arity': 1
    },

    'reciprocal': {
        'function': lambda x: protected_div(torch.ones_like(x), x),
        'arity': 1
    },


    # ----------------------------------------------------
    # ML activation functions
    # ----------------------------------------------------

    'tanh': {
        'function': lambda x: torch.tanh(x),
        'arity': 1
    },

    'sigmoid': {
        'function': lambda x: torch.sigmoid(x),
        'arity': 1
    },

    'relu': {
        'function': lambda x: torch.relu(x),
        'arity': 1
    },

    'softplus': {
        'function': lambda x: torch.nn.functional.softplus(x),
        'arity': 1
    },

    'softsign': {
        'function': lambda x: x / (1 + torch.abs(x)),
        'arity': 1
    },


    # ----------------------------------------------------
    # Hyperbolic functions
    # ----------------------------------------------------

    'sinh': {
        'function': lambda x: torch.sinh(torch.clamp(x, -10, 10)),
        'arity': 1
    },

    'cosh': {
        'function': lambda x: torch.cosh(torch.clamp(x, -10, 10)),
        'arity': 1
    },


    # ----------------------------------------------------
    # Polynomial feature operators
    # ----------------------------------------------------

    'square': {
        'function': lambda x: torch.square(x),
        'arity': 1
    },

    'cube': {
        'function': lambda x: x * x * x,
        'arity': 1
    },

    'quartic': {
        'function': lambda x: torch.pow(x, 4),
        'arity': 1
    },

    'pow2': {
        'function': lambda x: torch.pow(x, 2),
        'arity': 1
    },

    'pow3': {
        'function': lambda x: torch.pow(x, 3),
        'arity': 1
    },


    # ----------------------------------------------------
    # Magnitude / symmetry / threshold operators
    # ----------------------------------------------------

    'abs': {
        'function': lambda x: torch.abs(x),
        'arity': 1
    },

    'neg': {
        'function': lambda x: -x,
        'arity': 1
    },

    'sign': {
        'function': lambda x: torch.sign(x),
        'arity': 1
    },


    # ----------------------------------------------------
    # Binary geometry / nonlinear operators
    # ----------------------------------------------------

    'hypot': {
        'function': lambda x, y: torch.sqrt(
            x*x + y*y + 1e-8
        ),
        'arity': 2
    },


    # ----------------------------------------------------
    # Piecewise ML operators
    # ----------------------------------------------------

    'maximum': {
        'function': lambda x, y: torch.maximum(x, y),
        'arity': 2
    },

    'minimum': {
        'function': lambda x, y: torch.minimum(x, y),
        'arity': 2
    },
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
    """Evaluate a nested-tuple tree on latent tokens T (n_rows, latent_dim).

    Arity-2 nodes are 3-tuples (fname, left, right); arity-1 nodes are
    2-tuples (fname, child) -- same convention as
    algorithms/GP/representations/tree.py's Tree.apply_tree."""
    if isinstance(structure, tuple):
        fname = structure[0]
        if FUNCTIONS[fname]['arity'] == 2:
            left = evaluate_structure(structure[1], T, TERMINALS)
            right = evaluate_structure(structure[2], T, TERMINALS)
            out = FUNCTIONS[fname]['function'](left, right)
        else:
            arg = evaluate_structure(structure[1], T, TERMINALS)
            out = FUNCTIONS[fname]['function'](arg)
        if isinstance(out, torch.Tensor):
            out = torch.clamp(out, -BOUND, BOUND)
        return out
    if structure in TERMINALS:
        return T[:, TERMINALS[structure]]
    # Broadcast to a full (n_rows,) tensor immediately, not a bare float --
    # several torch ops (protected_div's torch.abs, any unary function like
    # sin/log) require a real Tensor even for a constant-only subtree, which
    # p_c=0.0 (the default everywhere until now) never exercised.
    value = CONSTANTS[structure](None)
    return torch.full((T.shape[0],), float(value), dtype=torch.float32, device=T.device)


def generate_ramped_structures(n, init_depth, p_c, TERMINALS):
    """Generate n random tree structures (ramped, half grow/full).

    Returns a registry: list of {"structure", "nodes", "depth"} dicts.
    Uses the global `random`/`np.random` RNGs — caller seeds beforehand.
    """
    depth_fn = tree_depth(FUNCTIONS)
    registry = []
    depths = list(range(2, init_depth + 1))
    for i in range(n):
        depth = depths[i % len(depths)]
        if i % 2 == 0:
            structure = create_grow_random_tree(depth, FUNCTIONS, TERMINALS,
                                                CONSTANTS, p_c=p_c)
        else:
            structure = create_full_random_tree(depth, FUNCTIONS, TERMINALS,
                                                CONSTANTS, p_c=p_c)
        nodes = len(list(flatten(structure))) if isinstance(structure, tuple) else 1
        registry.append({"structure": structure, "nodes": nodes,
                         "depth": depth_fn(structure)})
    return registry


def build_pool(cfg, TERMINALS):
    """Generate cfg.pool_size random tree structures (ramped, half grow/full).

    Returns the registry: list of {"structure", "nodes", "depth"} dicts.
    Uses the global `random` RNG — caller seeds beforehand.
    """
    return generate_ramped_structures(cfg.pool_size, cfg.init_depth, cfg.p_c, TERMINALS)


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
