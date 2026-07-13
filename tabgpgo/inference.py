"""
Phase 5: symbolic reconstruction and fast inference on unseen data.

The elite's genotype (pool indices + wrapper metadata) is resolved against the
tree registry to rebuild the full symbolic equation. Inference re-evaluates
those structures with the same `evaluate_structure` interpreter used to build
the pool, so evolution-time and inference-time semantics agree by
construction (asserted by `verify_inference` after every run).
"""
import json
import math
import os
import pickle

import torch

from .autoencoder import MLPAutoencoder, encode
from .config import TabGPGOConfig
from .evolution import (Block, PoolIndividual, individual_semantics,
                        individual_semantics_sequential, wrapper_output)
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


def block_raw_terms(ind, T, registry, TERMINALS):
    """Head semantics, and each block's UNSCALED wrapper_output term (sR) on
    latent tokens T -- the same quantity `ms` scales in semantics_from_tokens
    (see evolution.py::wrapper_output), computed once and reused for both
    zero-shot evaluation and weight fine-tuning. Each term only depends on T
    and its own subtree, independent of sibling blocks.

    Returns (head_sem: (n_rows,), raw_terms: list[(n_rows,)] in block order).
    """
    def sem(idx):
        out = evaluate_structure(registry[idx]["structure"], T, TERMINALS)
        if not isinstance(out, torch.Tensor) or out.dim() == 0:
            out = torch.full((T.shape[0],), float(out), device=T.device)
        return out.float()

    head_sem = sem(ind.head_idx)
    raw_terms = []
    for b in ind.blocks:
        tr1 = sem(b.idx1)
        tr2 = sem(b.idx2) if b.idx2 is not None else None
        raw_terms.append(wrapper_output(b.wrapper, tr1, tr2))
    return head_sem, raw_terms


def weighted_semantics(ind, head_sem, raw_terms, weights, bound=BOUND):
    """Aggregate semantics from block_raw_terms's output, using an externally
    supplied per-block scalar `weights` in place of each block's frozen `ms`.

    Same order-preserving sum/mul fold as semantics_from_tokens (sum and mul
    blocks don't commute, so insertion order is respected regardless of
    homogeneity) -- differentiable w.r.t. weights, so it doubles as the
    forward pass for fine_tune_weights.

    `bound` clamps the running aggregate after EVERY block, not just at the
    end (default is the same permissive BOUND semantics_from_tokens uses, so
    zero-shot/predict-without-fit callers are unaffected): a long chain of
    "mul" blocks compounds multiplicatively, so an unclamped intermediate can
    overflow long before the final value would. fine_tune_weights passes a
    much tighter bound, since a freshly-optimized weight has no such
    guarantee the way an evolved `ms` implicitly does.
    """
    agg = head_sem
    for b, raw, w in zip(ind.blocks, raw_terms, weights):
        delta = w * raw
        agg = agg * (1 + delta) if b.operator == "mul" else agg + delta
        agg = torch.clamp(agg, -bound, bound)
    return agg


def fine_tune_weights(ind, head_sem, raw_terms, y, steps=200, lr=0.05,
                      weight_decay=0.0, inner_val_frac=0.2, bound=10.0, seed=0):
    """Re-fit each block's scalar weight against a labeled sample from the
    deployment dataset, via Adam over MSE -- structures/wrappers/operators/
    head all stay frozen; only the per-block scalars move. This is the
    inference-time analog of OMS (TensorSLIM._optimal_ms), which solves the
    same per-block-scalar role against the *synthetic* prior target; here it
    is refit post-hoc against real labels.

    head_sem/raw_terms (from block_raw_terms) are computed once by the
    caller and reused across every step -- no tree re-evaluation during
    optimization, only cheap re-weighting of a fixed (n_blocks, n_rows)
    quantity.

    A chain of "mul" blocks compounds multiplicatively, so even a tiny
    per-block weight excursion can blow up the aggregate on out-of-
    distribution (real, not synthetic) inputs -- unlike `ms`, which is
    implicitly bounded by construction (random draw or OMS's own clip), a
    freshly-optimized weight has no such guarantee, and empirically this
    isn't just a training-time instability: a configuration can genuinely
    improve training MSE while still extrapolating catastrophically on
    held-out rows. Two safeguards, not one, are needed:
      - `bound` (see weighted_semantics) caps every intermediate aggregate
        during both optimization and scoring, so a failed attempt is merely
        worse than zero-shot instead of numerically catastrophic.
      - An inner-validation slice, carved out of (head_sem, raw_terms, y)
        with a fixed seed independent of any outer split the caller is
        already doing, is scored at every step; the best-inner-val-loss
        weights are returned, including step 0 (the original `ms` values).
        Fine-tuning can therefore only match or improve inner-val
        generalization, never silently accept an in-sample-only improvement.
    weight_decay, if set, additionally regularizes toward each block's
    original `ms` (not toward zero).

    Returns the best weights tensor found (detached). Caller must skip this
    entirely when ind.blocks is empty (nothing to fit).
    """
    init = torch.tensor([b.ms for b in ind.blocks], dtype=torch.float32)
    n = head_sem.shape[0]
    if n >= 5:
        val_n = max(1, int(round(inner_val_frac * n)))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        val_idx, fit_idx = perm[:val_n], perm[val_n:]
    else:
        val_idx = fit_idx = torch.arange(n)   # too few rows for a meaningful split
    head_fit, head_val = head_sem[fit_idx], head_sem[val_idx]
    raw_fit = [r[fit_idx] for r in raw_terms]
    raw_val = [r[val_idx] for r in raw_terms]
    y_fit, y_val = y[fit_idx], y[val_idx]

    weights = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([weights], lr=lr)

    def _val_mse():
        with torch.no_grad():
            pred = weighted_semantics(ind, head_val, raw_val, weights, bound=bound)
            return torch.mean((pred - y_val) ** 2).item()

    best_loss, best_weights = _val_mse(), weights.detach().clone()
    for _ in range(steps):
        opt.zero_grad()
        pred = weighted_semantics(ind, head_fit, raw_fit, weights, bound=bound)
        fit_loss = torch.mean((pred - y_fit) ** 2)
        loss = fit_loss + weight_decay * torch.mean((weights - init) ** 2) if weight_decay else fit_loss
        loss.backward()
        opt.step()
        cur_loss = _val_mse()
        if not math.isfinite(cur_loss):
            break   # unrecoverable step -- keep whatever was best so far
        if cur_loss < best_loss:
            best_loss, best_weights = cur_loss, weights.detach().clone()
    return best_weights


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


class TabGPGOPredictor:
    """Sklearn-style fit/predict wrapper around a frozen elite, with an
    optional few-shot fine-tuning step.

    Structures/wrappers/operators/head and the frozen autoencoder are never
    touched -- the only thing `fit(..., fine_tune=True)` can change is each
    block's scalar weight (initialized from its evolved `ms`), re-optimized
    against a labeled sample from the deployment dataset (see
    fine_tune_weights). Without calling fit(), or with fine_tune=False
    (default), predictions are identical to the old zero-shot make_predictor
    path.
    """

    def __init__(self, ind, registry, model, cfg, reducer_meta=None):
        self.ind = ind
        self.registry = registry
        self.model = model
        self.cfg = cfg
        self.reducer_meta = reducer_meta
        self.TERMINALS = make_terminals(cfg.latent_dim)
        self.device = cfg.get_device()
        self.weights = torch.tensor([b.ms for b in ind.blocks], dtype=torch.float32)
        self.x_stats = None   # set by fit(); reused by predict() for consistent scaling
        self.y_stats = None   # set by fit(); predict() inverts to raw units if set
        self._ft_bound = None  # set by fit(fine_tune=True); predict() reuses it if set

    @torch.no_grad()
    def _encode(self, X_raw, fit_stats=False):
        X = X_raw.float()
        if X.shape[1] > self.cfg.max_features:
            if self.reducer_meta is None:
                raise ValueError(f"{X.shape[1]} features > {self.cfg.max_features}: "
                                 "a fitted reducer_meta is required")
            if self.reducer_meta["method"] == "rf":
                X = X[:, self.reducer_meta["columns"]]
            else:
                Xs, _, _ = zscore(X, *self.reducer_meta["x_stats"])
                X = torch.from_numpy(self.reducer_meta["pca"].transform(Xs.numpy())).float()
        x_stats = None if fit_stats else self.x_stats
        X100, _, meta = standardize_scale_pad(X, torch.zeros(X.shape[0]),
                                              self.cfg.max_features, x_stats=x_stats)
        if fit_stats:
            self.x_stats = meta["x_stats"]
        return encode(self.model, X100.to(self.device))

    def fit(self, X, y=None, fine_tune=False, steps=200, lr=0.05, weight_decay=0.0,
           inner_val_frac=0.2, bound=10.0):
        """Fix feature-scaling stats from X; if y is given, derive y_stats
        for raw-unit predict() inversion; if fine_tune=True (and y is given,
        and the elite has blocks), re-optimize per-block weights against
        (X, y) via fine_tune_weights (see there for `inner_val_frac`/`bound`,
        the two safeguards against multiplicative-chain instability). A
        no-op on the weights whenever fine_tune=False or the elite is
        head-only. `bound` is remembered and reused by predict() so
        fit-time validation and deployment-time evaluation stay consistent;
        zero-shot predict() (fit() never called, or fine_tune=False) is
        unaffected."""
        T = self._encode(X, fit_stats=True)
        if y is not None:
            y_z, y_mean, y_std = zscore(y.float().reshape(-1, 1))
            y_z = y_z.reshape(-1).to(T.device)
            self.y_stats = (y_mean, y_std)
            if fine_tune and self.ind.blocks:
                head_sem, raw_terms = block_raw_terms(self.ind, T, self.registry,
                                                       self.TERMINALS)
                self.weights = fine_tune_weights(self.ind, head_sem, raw_terms, y_z,
                                                 steps=steps, lr=lr,
                                                 weight_decay=weight_decay,
                                                 inner_val_frac=inner_val_frac, bound=bound)
                self._ft_bound = bound
        return self

    @torch.no_grad()
    def predict(self, X):
        T = self._encode(X)
        head_sem, raw_terms = block_raw_terms(self.ind, T, self.registry, self.TERMINALS)
        bound = self._ft_bound if self._ft_bound is not None else BOUND
        sem = weighted_semantics(self.ind, head_sem, raw_terms, self.weights, bound=bound)
        if self.y_stats is not None:
            mean, std = self.y_stats
            mean, std = mean.to(sem.device).squeeze(), std.to(sem.device).squeeze()
            sem = sem * std + mean
        return sem


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
