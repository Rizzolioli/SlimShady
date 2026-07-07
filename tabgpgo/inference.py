"""
Phase 5: symbolic reconstruction and fast inference on unseen data.

The elite's genotype (pool indices + wrapper metadata) is resolved against the
tree registry to rebuild the full symbolic equation. Inference re-evaluates
those structures with the same `evaluate_structure` interpreter used to build
the pool, so evolution-time and inference-time semantics agree by
construction (asserted by `verify_inference` after every run).
"""
import json
import os
import pickle

import torch

from .autoencoder import MLPAutoencoder, encode
from .config import TabGPGOConfig
from .evolution import (Block, PoolIndividual, individual_semantics,
                        individual_semantics_sequential)
from .preprocessing import standardize_scale_pad, zscore
from .tree_pool import BOUND, evaluate_structure, make_terminals

_INFIX = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}


def _to_jsonable(structure):
    """Nested tuples -> nested lists (JSON-safe)."""
    if isinstance(structure, tuple):
        return [_to_jsonable(s) for s in structure]
    return structure


def _from_jsonable(structure):
    """Nested lists (from JSON) -> nested tuples."""
    if isinstance(structure, list):
        return tuple(_from_jsonable(s) for s in structure)
    return structure


def structure_to_str(structure):
    if isinstance(structure, tuple):
        fname, left, right = structure
        return (f"({structure_to_str(left)} {_INFIX[fname]} "
                f"{structure_to_str(right)})")
    if structure.startswith("constant_"):
        return structure.replace("constant__", "-").replace("constant_", "")
    return structure


def _block_to_str(block, registry):
    t1 = structure_to_str(registry[block.idx1]["structure"])
    ms = f"{block.ms:.6g}"
    if block.wrapper == "abs":
        delta = f"{ms}*(1 - 2/(1 + abs({t1})))"
    elif block.wrapper == "sig1":
        delta = f"{ms}*(2*sigmoid({t1}) - 1)"
    else:  # sig2
        t2 = structure_to_str(registry[block.idx2]["structure"])
        delta = f"{ms}*(sigmoid({t1}) - sigmoid({t2}))"
    return f"(1 + {delta})" if block.operator == "mul" else delta


def reconstruct_expression(ind, registry):
    """Full symbolic equation of an individual over latent tokens z0..z511.

    Blocks that all share one operator (the six fixed variants, and *MIX
    variants with a fixed aggregation) get the flat "a + b + c" / "a * b * c"
    form. A genuinely mixed sum/mul sequence (SLIM~MIX) folds left-to-right
    with explicit parens, since sum and mul mutations don't commute.
    """
    head_str = structure_to_str(registry[ind.head_idx]["structure"])
    if not ind.blocks:
        return head_str
    operators = {b.operator for b in ind.blocks}
    if len(operators) == 1:
        operator = next(iter(operators))
        parts = [head_str] + [_block_to_str(b, registry) for b in ind.blocks]
        return (" + " if operator == "sum" else " * ").join(parts)
    expr = head_str
    for b in ind.blocks:
        term = _block_to_str(b, registry)
        expr = f"({expr}) + {term}" if b.operator == "sum" else f"({expr}) * {term}"
    return expr


def semantics_from_tokens(ind, T, registry, TERMINALS):
    """Evaluate an individual directly on latent tokens (no pool lookup)."""
    def sem(idx):
        out = evaluate_structure(registry[idx]["structure"], T, TERMINALS)
        if not isinstance(out, torch.Tensor) or out.dim() == 0:
            out = torch.full((T.shape[0],), float(out), device=T.device)
        return out.float()

    agg = sem(ind.head_idx)
    for b in ind.blocks:
        tr1 = sem(b.idx1)
        if b.wrapper == "sig2":
            delta = b.ms * (torch.sigmoid(tr1) - torch.sigmoid(sem(b.idx2)))
        elif b.wrapper == "sig1":
            delta = b.ms * (2 * torch.sigmoid(tr1) - 1)
        else:
            delta = b.ms * (1 - 2 / (1 + torch.abs(tr1)))
        if b.operator == "mul":
            agg = agg * (1 + delta)
        else:
            agg = agg + delta
    return torch.clamp(agg, -BOUND, BOUND)


def verify_inference(ind, pool, T, registry, TERMINALS, atol=None):
    """Assert pool-path and token-path semantics agree on the same rows.

    Uses the batched path when every block shares one wrapper/operator (the
    fitness-eval path homogeneous variants actually ran on), and the
    sequential fold otherwise (SLIM*MIX/SLIM+MIX/SLIM~MIX, or a head-only
    individual with no blocks).
    """
    if ind.blocks and len({(b.wrapper, b.operator) for b in ind.blocks}) == 1:
        wrapper, operator = ind.blocks[0].wrapper, ind.blocks[0].operator
        from_pool = individual_semantics(ind, wrapper, operator, pool)
    else:
        from_pool = individual_semantics_sequential(ind, pool)
    from_tokens = semantics_from_tokens(ind, T, registry, TERMINALS)
    atol = atol if atol is not None else (1e-1 if pool.dtype == torch.float16 else 1e-3)
    if not torch.allclose(from_pool, from_tokens, atol=atol, rtol=1e-3):
        diff = (from_pool - from_tokens).abs().max().item()
        raise AssertionError(f"inference/evolution semantics diverge: "
                             f"max abs diff {diff:.3e}")
    return True


def make_predictor(ind, registry, model, cfg, reducer_meta=None):
    """Standalone predictor for raw unseen data.

    predict(X_raw): z-scores/scales/pads X_raw using its own statistics (or a
    fitted reducer for >max_features data), encodes it with the frozen
    encoder, and evaluates the symbolic equation. Predictions live in the
    z-scored target space the model was evolved in.
    """
    TERMINALS = make_terminals(cfg.latent_dim)
    device = cfg.get_device()

    @torch.no_grad()
    def predict(X_raw):
        X = X_raw.float()
        if X.shape[1] > cfg.max_features:
            if reducer_meta is None:
                raise ValueError(f"{X.shape[1]} features > {cfg.max_features}: "
                                 "a fitted reducer_meta is required")
            if reducer_meta["method"] == "rf":
                X = X[:, reducer_meta["columns"]]
            else:
                Xs, _, _ = zscore(X, *reducer_meta["x_stats"])
                X = torch.from_numpy(reducer_meta["pca"].transform(Xs.numpy())).float()
        X100, _, _ = standardize_scale_pad(X, torch.zeros(X.shape[0]),
                                           cfg.max_features)
        T = encode(model, X100.to(device))
        return semantics_from_tokens(ind, T, registry, TERMINALS)

    return predict


# ---------------------------------------------------------------------------
# Persistence: everything standalone inference needs, in one run directory.
# ---------------------------------------------------------------------------

def save_run(run_dir, cfg, model, registry, ind, wrapper, operator,
             algo, seed, expression):
    """Persist everything standalone inference needs.

    elite.json is self-contained: it stores the RESOLVED tree structures of
    the head and every block (not just pool indices), so inference only needs
    elite.json + encoder.pt + config.json. registry.pkl (the full pool) is
    kept for reproducibility. `wrapper`/`operator` here are the variant-level
    selectors (possibly "mix"), kept only as a human-readable label -- each
    block already carries the wrapper/operator it actually drew.
    """
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.encoder.state_dict(), os.path.join(run_dir, "encoder.pt"))
    with open(os.path.join(run_dir, "registry.pkl"), "wb") as fh:
        pickle.dump(registry, fh)
    elite = {"algo": algo, "wrapper": wrapper, "operator": operator,
             "seed": seed, "head_idx": ind.head_idx,
             "blocks": [[b.idx1, b.idx2, b.ms, b.wrapper, b.operator]
                       for b in ind.blocks],
             "head_structure": _to_jsonable(registry[ind.head_idx]["structure"]),
             "block_structures": [
                 [_to_jsonable(registry[b.idx1]["structure"]),
                  _to_jsonable(registry[b.idx2]["structure"]) if b.idx2 is not None else None]
                 for b in ind.blocks],
             "fitness": ind.fitness, "nodes_count": ind.nodes_count,
             "expression": expression}
    with open(os.path.join(run_dir, "elite.json"), "w") as fh:
        json.dump(elite, fh, indent=2)
    cfg_dict = {k: v for k, v in vars(cfg).items()
                if isinstance(v, (int, float, str, bool, tuple, list))}
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(cfg_dict, fh, indent=2, default=str)


def load_run(run_dir, device="cpu"):
    """Rebuild (cfg, model, registry, elite, wrapper, operator) from disk.

    The registry is reconstructed from the resolved structures inside
    elite.json (registry.pkl is NOT required), so a run directory with just
    elite.json + encoder.pt + config.json is enough for inference.
    """
    with open(os.path.join(run_dir, "config.json")) as fh:
        cfg_dict = json.load(fh)
    cfg_dict["variants"] = tuple(tuple(v) for v in cfg_dict.get("variants", ()))
    cfg_dict["val_datasets"] = tuple(cfg_dict.get("val_datasets", ()))
    cfg = TabGPGOConfig(**cfg_dict)
    cfg.device = device
    model = MLPAutoencoder(cfg.max_features, cfg.ae_hidden, cfg.latent_dim)
    model.encoder.load_state_dict(
        torch.load(os.path.join(run_dir, "encoder.pt"), map_location=device))
    model.to(cfg.get_device()).eval().requires_grad_(False)
    with open(os.path.join(run_dir, "elite.json")) as fh:
        e = json.load(fh)
    # minimal registry: only the structures the elite references
    registry = {e["head_idx"]: {"structure": _from_jsonable(e["head_structure"])}}
    for b, (s1, s2) in zip(e["blocks"], e["block_structures"]):
        registry[b[0]] = {"structure": _from_jsonable(s1)}
        if b[1] is not None:
            registry[b[1]] = {"structure": _from_jsonable(s2)}
    ind = PoolIndividual(e["head_idx"],
                         [Block(b[0], b[1], b[2], b[3], b[4]) for b in e["blocks"]],
                         fitness=e["fitness"], nodes_count=e["nodes_count"])
    return cfg, model, registry, ind, e["wrapper"], e["operator"]
