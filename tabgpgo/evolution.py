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
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from evaluators.fitness_functions import rmse
from utils.logger import logger

from .config import ALGO_NAMES, wrapper_name
from .tree_pool import BOUND

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
    """Aggregate semantics of an individual on the rows of `pool`."""
    agg = pool[ind.head_idx].float()
    if ind.blocks:
        deltas = block_deltas(ind.blocks, wrapper, operator, pool)
        if operator == "sum":
            agg = agg + deltas.sum(dim=0)
        else:
            agg = agg * deltas.prod(dim=0)
    return torch.clamp(agg, -BOUND, BOUND)


class TensorSLIM:
    """Tensor-pool SLIM-GSGP optimizer for one variant and one seed."""

    def __init__(self, cfg, variant, registry, pool_train, y_target,
                 val_pools, val_targets, seed):
        sig, two_trees, operator = variant
        self.cfg = cfg
        self.variant = variant
        self.algo = ALGO_NAMES[variant]
        self.wrapper = wrapper_name(sig, two_trees)
        self.operator = operator
        self.registry = registry
        self.pool_train = pool_train        # (pool_size, n_train)
        self.y_target = y_target            # (n_train,)
        self.val_pools = val_pools          # {name: (pool_size, n_val)}
        self.val_targets = val_targets      # {name: (n_val,)}
        self.seed = seed
        self.ms_fn = lambda: random.uniform(cfg.ms_lo, cfg.ms_hi)
        self.p_deflate = 1 - cfg.p_inflate
        self.elite = None

    # -- helpers -----------------------------------------------------------

    def _evaluate(self, ind):
        sem = individual_semantics(ind, self.wrapper, self.operator, self.pool_train)
        ind.fitness = float(rmse(self.y_target, sem))
        ind.nodes_count = self._nodes_count(ind)

    def _nodes_count(self, ind):
        nodes = self.registry[ind.head_idx]["nodes"]
        overhead = WRAPPER_NODES[(self.wrapper, self.operator)]
        for b in ind.blocks:
            nodes += self.registry[b.idx1]["nodes"] + overhead
            if b.idx2 is not None:
                nodes += self.registry[b.idx2]["nodes"]
        return nodes + len(ind.blocks)   # size-1 linkage operators

    def _tournament(self, population):
        contestants = random.sample(population, self.cfg.tournament_size)
        return min(contestants, key=lambda i: i.fitness)

    def _best(self, population):
        return min(population, key=lambda i: (i.fitness, i.nodes_count))

    def _inflate(self, parent):
        idx2 = (random.randrange(self.cfg.pool_size)
                if self.wrapper == "sig2" else None)
        block = Block(idx1=random.randrange(self.cfg.pool_size),
                      idx2=idx2, ms=self.ms_fn())
        return PoolIndividual(parent.head_idx, [*parent.blocks, block])

    def _deflate(self, parent):
        if not parent.blocks:   # cannot deflate: copy parent (copy_parent=True)
            return PoolIndividual(parent.head_idx, list(parent.blocks))
        point = random.randrange(len(parent.blocks))
        return PoolIndividual(parent.head_idx,
                              [b for i, b in enumerate(parent.blocks) if i != point])

    def elite_val_rmse(self):
        """Elite RMSE on each validation dataset -> list in cfg order."""
        out = []
        for name in self.cfg.val_datasets:
            sem = individual_semantics(self.elite, self.wrapper,
                                       self.operator, self.val_pools[name])
            out.append(float(rmse(self.val_targets[name], sem)))
        return out

    # -- main loop ----------------------------------------------------------

    def solve(self, run_info, log_path, verbose=1):
        cfg = self.cfg
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        start = time.time()
        population = [PoolIndividual(random.randrange(cfg.pool_size))
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
                if random.random() < self.p_deflate:
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
        val_rmses = self.elite_val_rmse()
        total_nodes = sum(i.nodes_count for i in population)
        # CSV columns: [algo, run_id, dataset, seed, generation,
        #               elite_train_rmse, time, population_nodes,
        #               val_{d}_rmse (cfg.val_datasets order),
        #               elite_size, elite_nodes]
        logger(log_path, gen, self.elite.fitness, elapsed, float(total_nodes),
               additional_infos=[*val_rmses, self.elite.size, self.elite.nodes_count],
               run_info=run_info, seed=self.seed)
        if verbose:
            val_str = " ".join(f"{name}={val_rmses[i]:.3f}"
                               for i, name in enumerate(self.cfg.val_datasets))
            print(f"  [{self.algo} seed {self.seed}] gen {gen}/{self.cfg.n_gens} "
                  f"train_rmse={self.elite.fitness:.4f} size={self.elite.size} "
                  f"| val: {val_str} | {elapsed:.2f}s")
