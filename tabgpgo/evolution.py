"""
Phase 4: tensor-pool SLIM-GSGP evolution.

Mirrors the control flow of algorithms/SLIM_GSGP/slim_gsgp.py::solve, but an
individual is only a genotype of pool indices: a head tree (raw semantics)
plus mutation blocks (wrapper, tree index/indices, mutation step ms). All
semantics are gathered from the precomputed pool and combined with native
torch ops, so a generation is pure tensor arithmetic.

Wrapper formulas are exactly the deltas of
algorithms/SLIM_GSGP/operators/mutators.py (with sigmoid applied here, since
the pool stores RAW tree semantics):
    abs :  ms * (1 - 2 / (1 + |TR|))
    sig1:  ms * (2*sigmoid(TR) - 1)
    sig2:  ms * (sigmoid(TR1) - sigmoid(TR2))
mul-operator variants use 1 + delta; blocks aggregate with sum or prod and the
result is clamped to +-1e12 like Individual.evaluate.
"""
import random
import threading
import time
from dataclasses import dataclass, field

import torch

from evaluators.fitness_functions import r2, rmse
from utils.logger import logger

from .config import ALGO_NAMES, OPERATORS, WRAPPERS
from .tree_pool import BOUND

# CSV writes happen from every concurrently-running TensorSLIM (see
# main_tabgpgo.py's thread-pooled evolve()); guard the shared log file.
_LOG_LOCK = threading.Lock()

# Per-block node/depth overhead of each wrapper, matching the intent of
# nested_nodes_calculator / nested_depth_calculator in
# algorithms/GSGP/representations/tree_utils.py.
WRAPPER_NODES = {("sig2", "sum"): 4, ("sig2", "mul"): 6,
                 ("sig1", "sum"): 7, ("sig1", "mul"): 9,
                 ("abs", "sum"): 9, ("abs", "mul"): 11}
WRAPPER_DEPTH = {("sig2", "sum"): 2, ("sig2", "mul"): 3,
                 ("sig1", "sum"): 3, ("sig1", "mul"): 4,
                 ("abs", "sum"): 4, ("abs", "mul"): 5}


@dataclass
class Block:
    idx1: int
    idx2: int | None   # only used by sig2
    ms: float
    wrapper: str       # "abs" | "sig1" | "sig2" -- this block's own mutation function
    operator: str      # "sum" | "mul" -- this block's own aggregation operator


@dataclass
class PoolIndividual:
    head_idx: int
    blocks: list = field(default_factory=list)
    fitness: float = float("inf")
    nodes_count: int = 0

    @property
    def size(self):
        return 1 + len(self.blocks)


def block_deltas(blocks, wrapper, operator, pool):
    """Delta semantics of all blocks at once -> (n_blocks, n_rows) float32."""
    device = pool.device
    tr1 = pool[torch.tensor([b.idx1 for b in blocks], device=device)].float()
    ms = torch.tensor([b.ms for b in blocks], device=device).unsqueeze(1)
    if wrapper == "sig2":
        tr2 = pool[torch.tensor([b.idx2 for b in blocks], device=device)].float()
        delta = ms * (torch.sigmoid(tr1) - torch.sigmoid(tr2))
    elif wrapper == "sig1":
        delta = ms * (2 * torch.sigmoid(tr1) - 1)
    elif wrapper == "abs":
        delta = ms * (1 - 2 / (1 + torch.abs(tr1)))
    else:
        raise ValueError(f"unknown wrapper: {wrapper}")
    if operator == "mul":
        delta = 1 + delta
    return delta


def individual_semantics(ind, wrapper, operator, pool):
    """Aggregate semantics of an individual on the rows of `pool`.

    Fast batched path: valid only when every block shares the same wrapper
    and operator (true by construction for the six fixed-variant runs).
    """
    agg = pool[ind.head_idx].float()
    if ind.blocks:
        deltas = block_deltas(ind.blocks, wrapper, operator, pool)
        if operator == "sum":
            agg = agg + deltas.sum(dim=0)
        else:
            agg = agg * deltas.prod(dim=0)
    return torch.clamp(agg, -BOUND, BOUND)


def individual_semantics_sequential(ind, pool):
    """Aggregate semantics folding blocks in order, each using its own
    wrapper/operator (Block.wrapper / Block.operator).

    Needed whenever blocks can be heterogeneous (the *MIX variants): sum and
    mul mutations don't commute with each other, so the fold must respect
    insertion order, unlike the homogeneous batched path above. Also usable
    as a reference implementation for any variant (homogeneous chains give
    the same result either way, since a single operator's blocks do commute).
    """
    agg = pool[ind.head_idx].float()
    for b in ind.blocks:
        tr1 = pool[b.idx1].float()
        if b.wrapper == "sig2":
            delta = b.ms * (torch.sigmoid(tr1) - torch.sigmoid(pool[b.idx2].float()))
        elif b.wrapper == "sig1":
            delta = b.ms * (2 * torch.sigmoid(tr1) - 1)
        else:  # abs
            delta = b.ms * (1 - 2 / (1 + torch.abs(tr1)))
        agg = agg * (1 + delta) if b.operator == "mul" else agg + delta
    return torch.clamp(agg, -BOUND, BOUND)


class TensorSLIM:
    """Tensor-pool SLIM-GSGP optimizer for one variant and one seed.

    Uses an instance-local `random.Random` (not the global `random` module)
    so multiple TensorSLIM runs can execute concurrently (see
    main_tabgpgo.py's thread-pooled evolve()) without clobbering each other's
    RNG state -- seeding the global module would make concurrent runs
    non-reproducible and mutually corrupting.
    """

    def __init__(self, cfg, variant, registry, pool_train, y_target,
                 val_pools, val_targets, val_y_stats, seed, ms_hi=None):
        wrapper, operator = variant
        self.cfg = cfg
        self.variant = variant
        self.ms_hi = cfg.ms_hi if ms_hi is None else ms_hi
        self.algo = f"{ALGO_NAMES[variant]}_ms{self.ms_hi:g}"
        self.wrapper = wrapper          # "abs" | "sig1" | "sig2" | "mix"
        self.operator = operator        # "sum" | "mul" | "mix"
        self.mixed = wrapper == "mix" or operator == "mix"
        self.registry = registry
        self.pool_train = pool_train        # (pool_size, n_train)
        self.y_target = y_target            # (n_train,)
        self.val_pools = val_pools          # {name: (pool_size, n_val)}
        self.val_targets = val_targets      # {name: (n_val,)} -- z-scored
        self.val_y_stats = val_y_stats      # {name: (y_mean, y_std)} -- to invert to raw units
        self.seed = seed
        self.rng = random.Random(seed)
        self.ms_fn = lambda: self.rng.uniform(cfg.ms_lo, self.ms_hi)
        self.p_deflate = 1 - cfg.p_inflate
        self.elite = None

    # -- helpers -----------------------------------------------------------

    def _semantics(self, ind, pool):
        if self.mixed:
            return individual_semantics_sequential(ind, pool)
        return individual_semantics(ind, self.wrapper, self.operator, pool)

    def _evaluate(self, ind):
        sem = self._semantics(ind, self.pool_train)
        ind.fitness = float(rmse(self.y_target, sem))
        ind.nodes_count = self._nodes_count(ind)

    def _nodes_count(self, ind):
        nodes = self.registry[ind.head_idx]["nodes"]
        for b in ind.blocks:
            nodes += self.registry[b.idx1]["nodes"] + WRAPPER_NODES[(b.wrapper, b.operator)]
            if b.idx2 is not None:
                nodes += self.registry[b.idx2]["nodes"]
        return nodes + len(ind.blocks)   # size-1 linkage operators

    def _tournament(self, population):
        contestants = self.rng.sample(population, self.cfg.tournament_size)
        return min(contestants, key=lambda i: i.fitness)

    def _best(self, population):
        return min(population, key=lambda i: (i.fitness, i.nodes_count))

    def _inflate(self, parent):
        wrapper = self.rng.choice(WRAPPERS) if self.wrapper == "mix" else self.wrapper
        operator = self.rng.choice(OPERATORS) if self.operator == "mix" else self.operator
        idx2 = self.rng.randrange(self.cfg.pool_size) if wrapper == "sig2" else None
        block = Block(idx1=self.rng.randrange(self.cfg.pool_size), idx2=idx2,
                      ms=self.ms_fn(), wrapper=wrapper, operator=operator)
        return PoolIndividual(parent.head_idx, [*parent.blocks, block])

    def _deflate(self, parent):
        if not parent.blocks:   # cannot deflate: copy parent (copy_parent=True)
            return PoolIndividual(parent.head_idx, list(parent.blocks))
        point = self.rng.randrange(len(parent.blocks))
        return PoolIndividual(parent.head_idx,
                              [b for i, b in enumerate(parent.blocks) if i != point])

    def elite_val_metrics(self):
        """Elite RMSE (scaled + raw units) and R^2 on each validation dataset.

        "Scaled" is RMSE in the z-scored target space the model was evolved
        in. "Raw" inverts both the prediction and the target with that
        dataset's own y_stats (mean, std), so it's directly comparable to
        other methods reporting error in the dataset's native units. R^2 is
        reported separately because it's affine-invariant (identical whether
        computed on scaled or raw values), so it's the one metric directly
        comparable *across* datasets regardless of their scale.
        Returns (scaled, raw, r2), each a list in cfg.val_datasets order.
        """
        scaled, raw, r2s = [], [], []
        for name in self.cfg.val_datasets:
            sem = self._semantics(self.elite, self.val_pools[name])
            y = self.val_targets[name]
            scaled.append(float(rmse(y, sem)))
            r2s.append(float(r2(y, sem)))
            mean, std = self.val_y_stats[name]
            mean, std = mean.to(sem.device).squeeze(), std.to(sem.device).squeeze()
            raw.append(float(rmse(y * std + mean, sem * std + mean)))
        return scaled, raw, r2s

    # -- main loop ----------------------------------------------------------

    def solve(self, run_info, log_path, verbose=1):
        cfg = self.cfg

        start = time.time()
        population = [PoolIndividual(self.rng.randrange(cfg.pool_size))
                      for _ in range(cfg.pop_size)]
        for ind in population:
            self._evaluate(ind)
        self.elite = self._best(population)
        self._log(0, time.time() - start, population, run_info, log_path, verbose)

        for gen in range(1, cfg.n_gens + 1):
            start = time.time()
            offspring = sorted(population,
                               key=lambda i: (i.fitness, i.nodes_count))[:cfg.n_elites]
            while len(offspring) < cfg.pop_size:
                parent = self._tournament(population)
                if self.rng.random() < self.p_deflate:
                    child = self._deflate(parent)
                else:
                    child = self._inflate(parent)
                self._evaluate(child)
                offspring.append(child)
            population = offspring
            self.elite = self._best(population)
            self._log(gen, time.time() - start, population, run_info, log_path, verbose)
        return self.elite

    def _log(self, gen, elapsed, population, run_info, log_path, verbose):
        scaled, raw, r2s = self.elite_val_metrics()
        train_r2 = float(r2(self.y_target, self._semantics(self.elite, self.pool_train)))
        total_nodes = sum(i.nodes_count for i in population)
        # CSV columns: [algo, run_id, dataset, seed, generation,
        #               elite_train_rmse, time, population_nodes,
        #               val_{d}_rmse_scaled, val_{d}_rmse_raw, val_{d}_r2
        #               (cfg.val_datasets order), elite_size, elite_nodes,
        #               elite_train_r2]
        val_cols = [v for triple in zip(scaled, raw, r2s) for v in triple]
        with _LOG_LOCK:
            logger(log_path, gen, self.elite.fitness, elapsed, float(total_nodes),
                   additional_infos=[*val_cols, self.elite.size,
                                     self.elite.nodes_count, train_r2],
                   run_info=run_info, seed=self.seed)
        if verbose:
            val_str = " ".join(f"{name}={scaled[i]:.3f}/{raw[i]:.3f}/R2={r2s[i]:.3f}"
                               for i, name in enumerate(self.cfg.val_datasets))
            print(f"  [{self.algo} seed {self.seed}] gen {gen}/{self.cfg.n_gens} "
                  f"train_rmse={self.elite.fitness:.4f} train_R2={train_r2:.3f} "
                  f"size={self.elite.size} | val(scaled/raw/R2): {val_str} | {elapsed:.2f}s")
