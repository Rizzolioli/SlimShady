"""
Phase 4 (fresh-pool variant): SLIM-GSGP evolution with no static upfront pool.

`tabgpgo/evolution.py`'s TensorSLIM builds one large pool of random trees
once, evaluates all of their semantics upfront, and samples from it WITH
replacement for the whole run. This module is the ablation: every generation,
trees are drawn from a per-run RESERVOIR of freshly generated random
structures that is topped up (never discarded/reset) as needed, and each
structure is consumed by at most one mutation event, ever -- `reservoir.pop()`
enforces single-use directly.

Storage strategy (worked out against real data from an HPT sweep on this
codebase -- see the design doc): individuals cache ONE incremental aggregate
tensor each (updated O(1) on inflate), not one tensor per block. A block only
stores its own tree structure(s) (cheap). This bounds memory by
`pop_size x one (n_train,) tensor`, independent of how large any individual's
block count grows or how many generations run -- unlike caching a tensor per
block, which real HPT data showed can reach ~10GB for a single run's
population (and isn't shared across concurrent runs the way the static pool
is).

Deflate (removing an arbitrary historical block) stays O(1) for the 8 of 9
variants with a single fixed aggregation operator per individual (sum/mul are
commutative there, so the removed block's own delta can just be subtracted/
divided back out). Only SLIM~MIX (operator itself drawn per block, so sum and
mul mutations are interleaved and don't commute) needs a full from-scratch
refold on deflate. Every individual's elite gets a periodic full refold
regardless, to correct the small floating-point drift incremental updates
accumulate over many generations.

Persistence/inference reuse `tabgpgo/inference.py` completely unmodified via
`to_registry`/`build_adapter`, converting a FreshIndividual into the
(PoolIndividual, registry[, pool]) vocabulary those functions already expect.
"""
import random
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from evaluators.fitness_functions import r2, rmse
from utils.logger import logger
from utils.utils import protected_div

from .config import ALGO_NAMES, OPERATORS, WRAPPERS
from .evolution import Block, PoolIndividual, WRAPPER_NODES, wrapper_output
from .inference import semantics_from_tokens
from .tree_pool import BOUND, evaluate_structure, generate_ramped_structures

# Structure generation (create_grow_random_tree/create_full_random_tree) draws
# from the global random/np.random modules with no injectable RNG -- today's
# TensorSLIM sidesteps this by building its pool once, single-threaded,
# before evolve()'s ThreadPoolExecutor starts. This module generates trees
# repeatedly *inside* the per-generation loop, which DOES run concurrently
# across cfg.max_workers jobs, so structure generation must be serialized.
# Reseeding on every acquisition (not just locking) keeps each run's batches
# reproducible regardless of thread-scheduling order.
_TREE_GEN_LOCK = threading.Lock()

# CSV writes happen from every concurrently-running FreshPoolSLIM; guard the
# shared log file (separate lock from evolution.py's _LOG_LOCK -- this module
# is self-contained and writes to its own log_path anyway).
_LOG_LOCK = threading.Lock()


@dataclass
class FreshBlock:
    structure1: tuple
    structure2: tuple | None   # only used by sig2
    nodes1: int
    nodes2: int | None
    ms: float
    wrapper: str       # "abs" | "sig1" | "sig2" -- this block's own mutation function
    operator: str       # "sum" | "mul" -- this block's own aggregation operator


@dataclass
class FreshIndividual:
    head_structure: tuple
    head_nodes: int
    aggregate: torch.Tensor          # (n_train,) -- incremental cache, O(1) to update
    blocks: list = field(default_factory=list)
    fitness: float = float("inf")
    nodes_count: int = 0

    @property
    def size(self):
        return 1 + len(self.blocks)


def _raw_semantics(structure, T, TERMINALS):
    """Evaluate one tree structure on tokens T, broadcasting a scalar result
    (a constant-only tree) into a full (n_rows,) tensor -- same handling
    evaluate_pool's loop and inference.py's semantics_from_tokens both use."""
    out = evaluate_structure(structure, T, TERMINALS)
    if not isinstance(out, torch.Tensor) or out.dim() == 0:
        out = torch.full((T.shape[0],), float(out), device=T.device)
    return out.float()


def to_registry(ind):
    """FreshIndividual -> (PoolIndividual, registry), no semantics tensors --
    a tiny local index space over only the structures `ind` references.
    Cheap (no evaluate_structure calls), safe to call every generation."""
    registry = {}

    def register(structure, nodes):
        idx = len(registry)
        registry[idx] = {"structure": structure, "nodes": nodes}
        return idx

    head_idx = register(ind.head_structure, ind.head_nodes)
    blocks = []
    for b in ind.blocks:
        idx1 = register(b.structure1, b.nodes1)
        idx2 = register(b.structure2, b.nodes2) if b.structure2 is not None else None
        blocks.append(Block(idx1, idx2, b.ms, b.wrapper, b.operator))
    pi = PoolIndividual(head_idx, blocks, fitness=ind.fitness, nodes_count=ind.nodes_count)
    return pi, registry


def build_adapter(ind, T_train, TERMINALS):
    """FreshIndividual -> (PoolIndividual, registry, pool), also stacking a
    small pool tensor of just this individual's own referenced structures'
    train semantics -- for verify_inference's pool-gather comparison path.
    Called once per run (not per generation): evaluates every referenced
    structure via evaluate_structure, unlike the cheap to_registry above."""
    pi, registry = to_registry(ind)
    pool_rows = [None] * len(registry)
    for idx, entry in registry.items():
        pool_rows[idx] = _raw_semantics(entry["structure"], T_train, TERMINALS)
    pool = torch.stack(pool_rows)
    return pi, registry, pool


class FreshPoolSLIM:
    """Fresh-pool SLIM-GSGP optimizer for one variant and one seed.

    Uses an instance-local `random.Random` (not the global `random` module)
    for selection/mutation decisions, same as TensorSLIM -- the global
    `random`/`np.random` modules are only ever touched, briefly and under
    _TREE_GEN_LOCK, for tree-structure generation.
    """

    def __init__(self, cfg, variant, T_train, T_val, y_target,
                 val_targets, val_y_stats, TERMINALS, seed, ms_spec=None):
        wrapper, operator = variant
        self.cfg = cfg
        self.variant = variant
        ms_spec = cfg.ms_hi if ms_spec is None else ms_spec
        self.use_oms = ms_spec == "oms"
        self.ms_hi = cfg.oms_bound if self.use_oms else ms_spec
        patience = cfg.stagnation_patience
        patience_label = "patnone" if not patience or patience <= 0 else f"pat{patience}"
        self.algo = (f"{ALGO_NAMES[variant]}_{'oms' if self.use_oms else f'ms{self.ms_hi:g}'}"
                    f"_{patience_label}")
        self.wrapper = wrapper          # "abs" | "sig1" | "sig2" | "mix"
        self.operator = operator        # "sum" | "mul" | "mix"
        self.T_train = T_train
        self.T_val = T_val              # {name: (n_val, latent_dim)}
        self.y_target = y_target        # (n_train,)
        self.val_targets = val_targets      # {name: (n_val,)} -- z-scored
        self.val_y_stats = val_y_stats      # {name: (y_mean, y_std)} -- to invert to raw units
        self.TERMINALS = TERMINALS
        self.seed = seed
        self.rng = random.Random(seed)
        self.ms_fn = lambda: self.rng.uniform(cfg.ms_lo, self.ms_hi)
        self.p_deflate = 1 - cfg.p_inflate
        self.reservoir = []   # list of {"structure", "nodes", "depth"} dicts, single-use
        self.elite = None
        self.best_fitness_ever = float("inf")   # anti-stagnation tracking
        self.stall_count = 0

    # -- reservoir -----------------------------------------------------------

    def _worst_case_demand(self, gen):
        """Upper bound on trees this generation could need: gen 0 draws one
        head per individual; later generations' offspring each inflate with
        at most the two-tree sig2 wrapper."""
        if gen == 0:
            return self.cfg.pop_size
        return 2 * (self.cfg.pop_size - self.cfg.n_elites)

    def _refill_reservoir(self, n, gen):
        if n <= 0:
            return
        seed = (self.seed * 1_000_003 + gen) % (2**31 - 1)
        with _TREE_GEN_LOCK:
            random.seed(seed)
            np.random.seed(seed)
            new_structures = generate_ramped_structures(
                n, self.cfg.init_depth, self.cfg.p_c, self.TERMINALS)
        self.reservoir.extend(new_structures)

    def _ensure_reservoir_at_least(self, n_needed, gen):
        shortfall = n_needed - len(self.reservoir)
        if shortfall > 0:
            self._refill_reservoir(shortfall, gen)

    def _ensure_reservoir(self, gen):
        self._ensure_reservoir_at_least(self._worst_case_demand(gen), gen)

    def _pop_tree(self):
        try:
            return self.reservoir.pop()
        except IndexError:
            raise RuntimeError(
                "tree reservoir exhausted mid-generation -- "
                "_ensure_reservoir's worst-case estimate was wrong") from None

    # -- helpers ---------------------------------------------------------------

    def _refold(self, head_structure, blocks):
        """Full from-scratch aggregate, folding blocks in order from stored
        structures. Used for SLIM~MIX deflate (order-dependent, can't be
        undone with O(1) arithmetic) and the periodic elite drift-resync."""
        agg = _raw_semantics(head_structure, self.T_train, self.TERMINALS)
        for b in blocks:
            tr1 = _raw_semantics(b.structure1, self.T_train, self.TERMINALS)
            tr2 = (_raw_semantics(b.structure2, self.T_train, self.TERMINALS)
                  if b.structure2 is not None else None)
            delta = b.ms * wrapper_output(b.wrapper, tr1, tr2)
            agg = agg * (1 + delta) if b.operator == "mul" else agg + delta
        return torch.clamp(agg, -BOUND, BOUND)

    def _evaluate(self, ind):
        ind.fitness = float(rmse(self.y_target, ind.aggregate))
        ind.nodes_count = self._nodes_count(ind)

    def _nodes_count(self, ind):
        nodes = ind.head_nodes
        for b in ind.blocks:
            nodes += b.nodes1 + WRAPPER_NODES[(b.wrapper, b.operator)]
            if b.nodes2 is not None:
                nodes += b.nodes2
        return nodes + len(ind.blocks)   # size-1 linkage operators

    def _tournament(self, population):
        contestants = self.rng.sample(population, self.cfg.tournament_size)
        return min(contestants, key=lambda i: i.fitness)

    def _best(self, population):
        return min(population, key=lambda i: (i.fitness, i.nodes_count))

    def _optimal_ms(self, parent, wrapper, operator, tr1, tr2):
        """Regularized Optimal Mutation Step -- same math as TensorSLIM's, but
        `s` comes from the parent's already-cached aggregate (no refold) and
        tr1/tr2 come from the freshly popped candidate trees' own semantics
        (no pool gather)."""
        s = parent.aggregate
        sR = wrapper_output(wrapper, tr1, tr2)
        residual = (self.y_target - s if operator == "sum"
                   else protected_div(self.y_target, s) - 1)
        denom = float(torch.dot(sR, sR))
        if denom < 1e-12:
            return 0.0
        ms = float(torch.dot(sR, residual)) / denom
        if abs(ms) < self.cfg.oms_eps:
            return 0.0
        return max(-self.cfg.oms_bound, min(self.cfg.oms_bound, ms))

    def _inflate(self, parent):
        wrapper = self.rng.choice(WRAPPERS) if self.wrapper == "mix" else self.wrapper
        operator = self.rng.choice(OPERATORS) if self.operator == "mix" else self.operator
        tree1 = self._pop_tree()
        tree2 = self._pop_tree() if wrapper == "sig2" else None
        tr1 = _raw_semantics(tree1["structure"], self.T_train, self.TERMINALS)
        tr2 = (_raw_semantics(tree2["structure"], self.T_train, self.TERMINALS)
              if tree2 is not None else None)
        ms = (self._optimal_ms(parent, wrapper, operator, tr1, tr2)
             if self.use_oms else self.ms_fn())
        delta = ms * wrapper_output(wrapper, tr1, tr2)
        new_agg = parent.aggregate * (1 + delta) if operator == "mul" else parent.aggregate + delta
        new_agg = torch.clamp(new_agg, -BOUND, BOUND)
        block = FreshBlock(tree1["structure"], tree2["structure"] if tree2 else None,
                           tree1["nodes"], tree2["nodes"] if tree2 else None,
                           ms, wrapper, operator)
        return FreshIndividual(parent.head_structure, parent.head_nodes, new_agg,
                               [*parent.blocks, block])

    def _deflate(self, parent):
        if not parent.blocks:   # cannot deflate: copy parent (copy_parent=True)
            return FreshIndividual(parent.head_structure, parent.head_nodes,
                                   parent.aggregate, list(parent.blocks))
        point = self.rng.randrange(len(parent.blocks))
        removed = parent.blocks[point]
        new_blocks = [b for i, b in enumerate(parent.blocks) if i != point]
        if self.operator == "mix":
            # sum/mul interleaved per-block -- order-dependent, can't undo one
            # block with O(1) arithmetic; full refold from stored structures.
            new_agg = self._refold(parent.head_structure, new_blocks)
        else:
            # single fixed operator for the whole individual -- commutative,
            # so the removed block's own delta can be subtracted/divided out
            # directly without touching any other block.
            tr1 = _raw_semantics(removed.structure1, self.T_train, self.TERMINALS)
            tr2 = (_raw_semantics(removed.structure2, self.T_train, self.TERMINALS)
                  if removed.structure2 is not None else None)
            delta = removed.ms * wrapper_output(removed.wrapper, tr1, tr2)
            if removed.operator == "mul":
                new_agg = protected_div(parent.aggregate, 1 + delta)
            else:
                new_agg = parent.aggregate - delta
            new_agg = torch.clamp(new_agg, -BOUND, BOUND)
        return FreshIndividual(parent.head_structure, parent.head_nodes, new_agg, new_blocks)

    # -- anti-stagnation ------------------------------------------------------

    def _check_stagnation(self, population, gen):
        """Track generations since the elite last improved; once
        cfg.stagnation_patience is reached, sweep the worst-fitness half
        (cfg.stagnation_replace_frac) of the population and reset the
        counter. cfg.stagnation_patience of None or <=0 disables this
        entirely."""
        if self.elite.fitness < self.best_fitness_ever - 1e-9:
            self.best_fitness_ever = self.elite.fitness
            self.stall_count = 0
        else:
            self.stall_count += 1
        patience = self.cfg.stagnation_patience
        if patience and patience > 0 and self.stall_count >= patience:
            self._sweep_stagnant(population, gen)
            self.stall_count = 0

    def _sweep_stagnant(self, population, gen):
        """Replace the worst-ranked individuals (by the same (fitness,
        nodes_count) key elitism uses) with brand-new head-only individuals
        drawn from the reservoir -- the elite (always rank 0) is never
        touched. Tops up the reservoir for its own demand first, since the
        generation's own offspring-creation loop may already have drawn it
        down."""
        n_replace = int(len(population) * self.cfg.stagnation_replace_frac)
        if n_replace <= 0:
            return
        order = sorted(range(len(population)),
                       key=lambda i: (population[i].fitness, population[i].nodes_count))
        worst = order[-n_replace:]
        self._ensure_reservoir_at_least(n_replace, gen)
        for i in worst:
            tree = self._pop_tree()
            agg = torch.clamp(_raw_semantics(tree["structure"], self.T_train, self.TERMINALS),
                              -BOUND, BOUND)
            fresh = FreshIndividual(tree["structure"], tree["nodes"], agg)
            self._evaluate(fresh)
            population[i] = fresh

    def elite_val_metrics(self):
        """Elite RMSE (scaled + raw units) and R^2 on each validation dataset
        -- lazy, elite-only: re-evaluates the elite's own structures against
        each validation set's tokens on demand, rather than materializing
        every candidate's val semantics up front (most never become elite)."""
        pi, registry = to_registry(self.elite)
        scaled, raw, r2s = [], [], []
        for name in self.cfg.val_datasets:
            sem = semantics_from_tokens(pi, self.T_val[name], registry, self.TERMINALS)
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
        self._ensure_reservoir(0)
        population = []
        for _ in range(cfg.pop_size):
            tree = self._pop_tree()
            agg = torch.clamp(_raw_semantics(tree["structure"], self.T_train, self.TERMINALS),
                              -BOUND, BOUND)
            population.append(FreshIndividual(tree["structure"], tree["nodes"], agg))
        for ind in population:
            self._evaluate(ind)
        self.elite = self._best(population)
        self.best_fitness_ever = self.elite.fitness   # nothing to compare against yet at gen 0
        self._log(0, time.time() - start, population, run_info, log_path, verbose)

        for gen in range(1, cfg.n_gens + 1):
            start = time.time()
            self._ensure_reservoir(gen)
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
            self._resync_elite()   # bound incremental-update drift
            self._check_stagnation(population, gen)
            self._log(gen, time.time() - start, population, run_info, log_path, verbose)
        return self.elite

    def _resync_elite(self):
        """Correct floating-point drift accumulated by O(1) incremental
        inflate/deflate updates: recompute the elite's aggregate from scratch
        via its own stored structures, once per generation (cheap -- one
        individual). Mutates self.elite in place, so if it's carried forward
        as an elite next generation its cached state stays accurate."""
        self.elite.aggregate = self._refold(self.elite.head_structure, self.elite.blocks)
        self.elite.fitness = float(rmse(self.y_target, self.elite.aggregate))

    def _log(self, gen, elapsed, population, run_info, log_path, verbose):
        scaled, raw, r2s = self.elite_val_metrics()
        train_r2 = float(r2(self.y_target, self.elite.aggregate))
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
                  f"size={self.elite.size} reservoir={len(self.reservoir)} | "
                  f"val(scaled/raw/R2): {val_str} | {elapsed:.2f}s")

