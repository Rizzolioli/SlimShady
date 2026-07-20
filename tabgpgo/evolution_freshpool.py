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
import dataclasses
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from evaluators.fitness_functions import linear_scaling, r2, rmse
from utils.logger import logger
from utils.utils import protected_div

from . import tree_pool
from .config import ALGO_NAMES, OPERATORS, WRAPPERS
from .evolution import (Block, FreshBlock, FreshIndividual, PoolIndividual,
                        WRAPPER_NODES, individual_semantics,
                        individual_semantics_sequential, wrapper_output)
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


def _raw_semantics(structure, T, TERMINALS):
    """Evaluate one tree structure on tokens T, broadcasting a scalar result
    (a constant-only tree) into a full (n_rows,) tensor -- same handling
    evaluate_pool's loop and inference.py's semantics_from_tokens both use.

    `structure` may also be a nested FreshIndividual (an "omt" block's TO,
    itself a compact head+blocks recipe -- see FreshPoolSLIM._omt_search) --
    dispatches to _nested_semantics instead of the tree interpreter so
    persistence/re-evaluation on unseen data stay exact without ever
    materializing TO as one giant literal tree."""
    if isinstance(structure, FreshIndividual):
        return _nested_semantics(structure, T, TERMINALS)
    out = evaluate_structure(structure, T, TERMINALS)
    if not isinstance(out, torch.Tensor) or out.dim() == 0:
        out = torch.full((T.shape[0],), float(out), device=T.device)
    return out.float()


def _nested_semantics(ind, T, TERMINALS):
    """Recursively evaluate a nested FreshIndividual (an OMT block's TO
    recipe) on arbitrary tokens T -- same fold logic as FreshPoolSLIM._refold,
    but a free function parametrized over T so it works equally for T_train
    (evolution time) and T_val/unseen data (inference.py's own copy of this
    dispatch, in semantics_from_tokens/block_raw_terms, reuses the same
    _raw_semantics recursion via evaluate_structure's sibling handling)."""
    agg = _raw_semantics(ind.head_structure, T, TERMINALS)
    for b in ind.blocks:
        tr1 = _raw_semantics(b.structure1, T, TERMINALS)
        tr2 = _raw_semantics(b.structure2, T, TERMINALS) if b.structure2 is not None else None
        delta = b.ms * wrapper_output(b.wrapper, tr1, tr2)
        agg = agg * (1 + delta) if b.operator == "mul" else agg + delta
    return torch.clamp(agg, -BOUND, BOUND)


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
                 val_targets, val_y_stats, TERMINALS, seed, ms_spec=None, use_ls=False,
                 reservoir_owner=None, dataset_chunks=None, dataset_switch_every=0):
        """dataset_chunks (optional): a list of (T_i, y_i) tuples, one per
        synthetic dataset, used to rotate the ACTIVE training data every
        `dataset_switch_every` generations instead of training against the
        fixed (T_train, y_target) pool for the whole run -- see
        main_tabgpgo_tabpfn_rotate.py. T_train/y_target as passed in are kept
        untouched as self.T_train_full/self.y_target_full (the whole pool),
        used only to compute a "global" elite-fitness logging column that
        stays comparable across dataset switches; self.T_train/self.y_target
        themselves get overwritten to the currently-active chunk when
        rotation is enabled. dataset_chunks=None or dataset_switch_every<=0
        (the defaults) disable rotation entirely, reproducing every existing
        script's behavior exactly."""
        wrapper, operator = variant
        self.cfg = cfg
        self.use_ls = use_ls
        self.omt_frac = cfg.omt_frac
        self.omt_pop_size = cfg.omt_pop_size
        self.omt_gens = cfg.omt_gens
        self.omt_max_workers = cfg.omt_max_workers
        self._omt_calls = 0   # own seed counter, independent of the main reservoir's generation counter
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
        self.T_train_full = T_train     # the whole synthetic pool, never overwritten
        self.y_target_full = y_target   # by dataset rotation -- see _switch_dataset
        self.val_targets = val_targets      # {name: (n_val,)} -- z-scored
        self.val_y_stats = val_y_stats      # {name: (y_mean, y_std)} -- to invert to raw units
        self.TERMINALS = TERMINALS
        self.seed = seed
        self._dataset_chunks = dataset_chunks
        self._dataset_switch_every = dataset_switch_every or 0
        self._rotating = bool(dataset_chunks) and self._dataset_switch_every > 0
        self._chunk_order = []
        self._chunk_cursor = 0
        self._gens_since_switch = 0
        self.active_dataset_idx = -1
        self.rng = random.Random(seed)
        self.ms_fn = lambda: self.rng.uniform(cfg.ms_lo, self.ms_hi)
        self.p_deflate = 1 - cfg.p_inflate
        self.reservoir = []   # list of {"structure", "nodes", "depth"} dicts, single-use
        self._reservoir_lock = threading.Lock()
        # None: this instance owns/consumes its own reservoir (the normal
        # case). Set (to another FreshPoolSLIM): delegate every reservoir
        # read/write to THAT instance's reservoir/lock instead of keeping a
        # private one -- used by OMT's inner searches (see
        # _omt_search_seeded) so they draw from -- and single-use-consume,
        # exactly like the outer algorithm's own mutations do -- the SAME
        # reservoir the outer run uses, rather than generating and
        # evaluating an entirely separate set of trees per search. This
        # instance's own self.reservoir then simply stays empty/unused.
        self._reservoir_owner = reservoir_owner
        self._refill_counter = 0   # monotonic, unique per actual refill -- see _refill_reservoir
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
        """`gen` is accepted for signature compatibility but no longer used
        for seeding -- see self._refill_counter. Seeding off a caller-
        supplied `gen` was only ever safe because, before OMT's reservoir-
        sharing existed, at most one refill happened per generation per
        instance. Now that many concurrent OMT inner searches (and this
        instance's own mid-generation top-ups -- see _make_offspring's
        pre-warm) can all trigger refills passing small, easily-colliding
        gen values, two DIFFERENT refill events reusing the same gen would
        reseed to the exact same value and generate IDENTICAL "random"
        trees. self._refill_counter instead guarantees every actual refill
        this instance ever performs gets a distinct seed, regardless of who
        triggered it or what gen they passed."""
        if n <= 0:
            return
        seed = (self.seed * 1_000_003 + self._refill_counter) % (2**31 - 1)
        self._refill_counter += 1
        with _TREE_GEN_LOCK:
            random.seed(seed)
            np.random.seed(seed)
            new_structures = generate_ramped_structures(
                n, self.cfg.init_depth, self.cfg.p_c, self.TERMINALS)
        self.reservoir.extend(new_structures)

    def _ensure_reservoir_at_least(self, n_needed, gen):
        """Delegates to self._reservoir_owner's reservoir/lock when this
        instance doesn't own its own (an OMT inner search sharing the outer
        algorithm's reservoir -- see _omt_search_seeded), so every consumer,
        outer or inner, tops up and draws from the exact same underlying
        supply. Thread-safe: guarded by the owner's _reservoir_lock, since
        _make_offspring's OMT batch calls into this concurrently across
        threads."""
        owner = self._reservoir_owner or self
        with owner._reservoir_lock:
            shortfall = n_needed - len(owner.reservoir)
            if shortfall > 0:
                owner._refill_reservoir(shortfall, gen)

    def _ensure_reservoir(self, gen):
        self._ensure_reservoir_at_least(self._worst_case_demand(gen), gen)

    def _pop_tree(self):
        owner = self._reservoir_owner or self
        with owner._reservoir_lock:
            try:
                return owner.reservoir.pop()
            except IndexError:
                raise RuntimeError(
                    "tree reservoir exhausted mid-generation -- "
                    "_ensure_reservoir's worst-case estimate was wrong") from None

    # -- helpers ---------------------------------------------------------------

    def _refold_on(self, head_structure, blocks, T):
        """Full from-scratch aggregate on arbitrary tokens T, folding blocks
        in order from stored structures -- for a SINGLE individual (no
        cross-individual structure-sharing to exploit here; see
        _switch_dataset's registry-based batch path for the population-wide
        case). Parametrized over T (rather than hardcoding self.T_train) so
        the same logic serves _refold (the active dataset) and
        _global_pool_metrics (the whole synthetic pool, for a
        rotation-invariant logging column) alike."""
        agg = _raw_semantics(head_structure, T, self.TERMINALS)
        for b in blocks:
            tr1 = _raw_semantics(b.structure1, T, self.TERMINALS)
            tr2 = (_raw_semantics(b.structure2, T, self.TERMINALS)
                  if b.structure2 is not None else None)
            delta = b.ms * wrapper_output(b.wrapper, tr1, tr2)
            agg = agg * (1 + delta) if b.operator == "mul" else agg + delta
        return torch.clamp(agg, -BOUND, BOUND)

    def _refold(self, head_structure, blocks):
        """Full from-scratch aggregate on the currently-active training
        data. Used for SLIM~MIX deflate (order-dependent, can't be undone
        with O(1) arithmetic) and the periodic elite drift-resync."""
        return self._refold_on(head_structure, blocks, self.T_train)

    def _evaluate(self, ind):
        if self.use_ls:
            a, b = linear_scaling(self.y_target, ind.aggregate)
            ind.ls_a, ind.ls_b = float(a), float(b)
            ind.fitness = float(rmse(self.y_target, a + b * ind.aggregate))
        else:
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

    def _optimal_ms(self, parent, wrapper, operator, tr1, tr2, sR=None):
        """Regularized Optimal Mutation Step -- same math as TensorSLIM's, but
        `s` comes from the parent's already-cached aggregate (no refold) and
        tr1/tr2 come from the freshly popped candidate trees' own semantics
        (no pool gather).

        `sR` lets a caller supply the un-scaled term directly instead of
        deriving it from (wrapper, tr1, tr2) -- used by the OMT path, whose
        candidate tree already IS the un-scaled term (no squashing wrapper),
        so wrapper/tr2 are otherwise unused/ignored in that case."""
        s = parent.aggregate
        if sR is None:
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

    def _next_omt_seed(self):
        """Own seed counter, independent of the main reservoir's generation
        counter -- consumed strictly sequentially (see _make_offspring's
        sequential pass) so a batched/parallel OMT run gets the exact same
        per-search seeds a fully-sequential run would, regardless of
        max_workers."""
        seed = (self.seed * 1_000_003 + 999_983 * self._omt_calls) % (2**31 - 1)
        self._omt_calls += 1
        return seed

    def _omt_search_seeded(self, residual, inner_seed):
        """Optimal Mutation Tree via genuine GSGP: nests an actual
        FreshPoolSLIM run -- SLIM*MIX, the same established variant used
        throughout this project -- targeting `residual` (on T_train) instead
        of the real y_target, for self.omt_pop_size individuals x
        self.omt_gens generations.

        Its elite is a compact FreshIndividual (head + blocks, semantics
        cached incrementally exactly like the outer algorithm -- no tree
        bloat from repeatedly re-expressing a growing structure), returned
        AS-IS to become the outer mutation's TO: stored directly as an
        "omt" block's structure1 (see FreshBlock/_finish_omt_inflate below),
        evaluated via _raw_semantics' FreshIndividual dispatch to
        _nested_semantics wherever it's later needed (elite drift-resync,
        validation scoring, persistence/reconstruction) -- so nothing
        downstream needs to know TO came from a nested search rather than a
        single tree.

        T_val/val_targets/val_y_stats are empty -- "validation" isn't a
        meaningful concept for an inner search whose only job is matching
        `residual` on T_train, so solve(track_val=False) skips that (and the
        CSV logging that would otherwise go with it) entirely.

        Takes an explicit seed (rather than drawing its own) so a whole
        generation's worth of these can be fired off concurrently via
        _make_offspring's thread pool -- each call here only touches its own
        freshly-constructed `inner` instance and locals, no shared mutable
        state besides the module-level _TREE_GEN_LOCK and the shared
        reservoir itself (both already thread-safe -- see the module
        docstring and _ensure_reservoir_at_least/_pop_tree).

        reservoir_owner=self: the inner search draws its random candidate
        trees from -- and single-use-consumes them out of -- the OUTER
        run's own reservoir, rather than generating and evaluating an
        entirely separate batch per search. Its own self.reservoir stays
        empty/unused."""
        # omt_frac=0.0 is critical here, not cosmetic: without it the nested
        # run would inherit the outer cfg's own omt_frac and try to launch
        # ITS OWN nested OMT search on every one of its mutations -- infinite
        # recursion. The inner search is always plain SLIM*MIX.
        inner_cfg = dataclasses.replace(self.cfg, pop_size=self.omt_pop_size,
                                        n_gens=self.omt_gens, omt_frac=0.0)
        inner = FreshPoolSLIM(inner_cfg, ("mix", "mul"), self.T_train, {}, residual,
                              {}, {}, self.TERMINALS, seed=inner_seed, ms_spec="oms",
                              reservoir_owner=self)
        return inner.solve(run_info=None, log_path=None, verbose=0, track_val=False)

    def _omt_search(self, residual):
        """Convenience wrapper: draws the next sequential seed and runs one
        OMT search immediately (unbatched). See _make_offspring for the
        batched path _inflate's OMT branch actually uses during solve()."""
        return self._omt_search_seeded(residual, self._next_omt_seed())

    def _apply_block(self, parent, operator, delta, block):
        new_agg = parent.aggregate * (1 + delta) if operator == "mul" else parent.aggregate + delta
        new_agg = torch.clamp(new_agg, -BOUND, BOUND)
        return FreshIndividual(parent.head_structure, parent.head_nodes, new_agg,
                               [*parent.blocks, block])

    def _inflate_random(self, parent, operator):
        """The non-OMT inflate path: draw one random reservoir tree, wrap it
        (abs/sig1/sig2), and add -- split out from _inflate so
        _make_offspring's sequential pass can resolve it immediately without
        ever touching OMT."""
        wrapper = self.rng.choice(WRAPPERS) if self.wrapper == "mix" else self.wrapper
        tree1 = self._pop_tree()
        tree2 = self._pop_tree() if wrapper == "sig2" else None
        tr1 = _raw_semantics(tree1["structure"], self.T_train, self.TERMINALS)
        tr2 = (_raw_semantics(tree2["structure"], self.T_train, self.TERMINALS)
              if tree2 is not None else None)
        ms = (self._optimal_ms(parent, wrapper, operator, tr1, tr2)
             if self.use_oms else self.ms_fn())
        delta = ms * wrapper_output(wrapper, tr1, tr2)
        block = FreshBlock(tree1["structure"], tree2["structure"] if tree2 else None,
                           tree1["nodes"], tree2["nodes"] if tree2 else None,
                           ms, wrapper, operator)
        return self._apply_block(parent, operator, delta, block)

    def _finish_omt_inflate(self, parent, operator, to_individual):
        """Completes an inflate mutation whose TO was already found by a
        (possibly batched/concurrent) OMT search -- see _make_offspring."""
        wrapper = "omt"
        tr1 = to_individual.aggregate
        ms = (self._optimal_ms(parent, wrapper, operator, tr1, None, sR=tr1)
             if self.use_oms else self.ms_fn())
        delta = ms * tr1
        block = FreshBlock(to_individual, None, to_individual.nodes_count, None, ms, wrapper, operator)
        return self._apply_block(parent, operator, delta, block)

    def _inflate(self, parent):
        """Unbatched single-mutation inflate (used by _make_offspring only
        for the non-OMT/already-resolved slots; the OMT-using slots go
        through _finish_omt_inflate instead, after a batched
        _omt_search_seeded call)."""
        operator = self.rng.choice(OPERATORS) if self.operator == "mix" else self.operator
        if self.omt_frac > 0 and self.rng.random() < self.omt_frac:
            residual = (self.y_target - parent.aggregate if operator == "sum"
                       else protected_div(self.y_target, parent.aggregate) - 1)
            return self._finish_omt_inflate(parent, operator, self._omt_search(residual))
        return self._inflate_random(parent, operator)

    def _make_offspring(self, population, n_needed, gen, verbose):
        """Two-pass offspring creation for one generation's remaining
        n_needed slots (after elitism).

        Pass 1 (sequential, fast): for each slot, run the exact same RNG-
        driven decisions the old fully-sequential loop made -- tournament-
        select a parent, decide deflate vs. inflate, and for inflate, decide
        mix-resolved operator and OMT-vs-random -- in the same order, so
        which trees end up as parents and which mutations end up using OMT
        is identical regardless of max_workers. Deflate and non-OMT inflate
        are cheap and resolved immediately; OMT-using slots are deferred
        (their residual is cheap to compute -- one subtract/divide -- so
        that happens now too, only the expensive nested search waits).

        Pass 2 (parallel): every deferred slot's OMT search is independent
        (own parent/residual/seed, no shared mutable state besides the
        already-thread-safe _TREE_GEN_LOCK), so they run concurrently via a
        thread pool sized by cfg.omt_max_workers. torch's own intra-op
        thread pool is temporarily pinned to 1 while this runs, so it
        doesn't oversubscribe CPU cores alongside the outer thread-level
        concurrency (restored afterward regardless of how the block exits).

        Pass 3 (sequential, fast): each completed search's TO is turned into
        a finished offspring and dropped into its reserved slot.
        """
        slots = [None] * n_needed
        pending = []   # (slot_index, parent, operator, residual, seed)
        for i in range(n_needed):
            parent = self._tournament(population)
            if self.rng.random() < self.p_deflate:
                slots[i] = self._deflate(parent)
                continue
            operator = self.rng.choice(OPERATORS) if self.operator == "mix" else self.operator
            if self.omt_frac > 0 and self.rng.random() < self.omt_frac:
                residual = (self.y_target - parent.aggregate if operator == "sum"
                           else protected_div(self.y_target, parent.aggregate) - 1)
                pending.append((i, parent, operator, residual, self._next_omt_seed()))
            else:
                slots[i] = self._inflate_random(parent, operator)

        if not pending:
            return slots

        # Pre-warm the shared reservoir for this batch's expected demand
        # BEFORE launching the parallel searches: each inner run draws from
        # -- and single-use-consumes out of -- this same reservoir (see
        # _omt_search_seeded), and can need up to omt_pop_size (its own
        # initial population) + omt_gens * 2*(omt_pop_size - n_elites)
        # trees in the worst case (up to 2 per inflate -- the sig2 wrapper
        # -- across every inner generation). len(pending) concurrent
        # searches sharing one reservoir need it sized well beyond the
        # outer generation's own (much smaller) per-generation demand.
        # _ensure_reservoir_at_least still tops up on demand if this
        # estimate falls short, so this is a size hint, not a hard cap.
        per_search_worst_case = self.omt_pop_size + self.omt_gens * 2 * max(0, self.omt_pop_size - self.cfg.n_elites)
        self._ensure_reservoir_at_least(len(pending) * per_search_worst_case, gen)

        max_workers = max(1, self.omt_max_workers)
        if verbose:
            print(f"  [{self.algo} seed {self.seed}] gen {gen}: running {len(pending)} "
                  f"OMT search{'es' if len(pending) != 1 else ''} "
                  f"(max_workers={max_workers}, pop={self.omt_pop_size}, gens={self.omt_gens})...")
        omt_start = time.time()
        if max_workers == 1:
            results = [self._omt_search_seeded(residual, seed)
                      for _, _, _, residual, seed in pending]
        else:
            prev_threads = torch.get_num_threads()
            torch.set_num_threads(1)   # avoid intra-op x inter-op oversubscription
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    results = list(pool.map(
                        lambda job: self._omt_search_seeded(job[3], job[4]), pending))
            finally:
                torch.set_num_threads(prev_threads)
        if verbose:
            print(f"  [{self.algo} seed {self.seed}] gen {gen}: {len(pending)} OMT "
                  f"searches done in {time.time() - omt_start:.2f}s")

        for (i, parent, operator, _, _), to_individual in zip(pending, results):
            slots[i] = self._finish_omt_inflate(parent, operator, to_individual)
        return slots

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

    # -- dataset rotation ------------------------------------------------------

    def _next_chunk_index(self):
        """Shuffled-cycle draw: consume a random permutation of every chunk
        index before any index repeats, reshuffling (via self.rng, so this
        stays reproducible given the run's seed) once exhausted. Guarantees
        even coverage of the whole synthetic pool before any dataset is
        revisited, unlike plain uniform-random-with-replacement."""
        if self._chunk_cursor >= len(self._chunk_order):
            self._chunk_order = list(range(len(self._dataset_chunks)))
            self.rng.shuffle(self._chunk_order)
            self._chunk_cursor = 0
        idx = self._chunk_order[self._chunk_cursor]
        self._chunk_cursor += 1
        return idx

    def _switch_dataset(self, population, gen, verbose):
        """Draws the next synthetic dataset and makes it the active
        (T_train, y_target), then refolds EVERY individual in `population`
        from scratch against it -- their cached .aggregate/.fitness were
        computed on the OLD dataset and are meaningless for selection under
        the new one. Also resets the anti-stagnation tracker: "no
        improvement" isn't a meaningful signal across a dataset change, so
        patience is measured relative to the new dataset only.

        Same "evaluate once into an indexed pool" design as TensorSLIM's own
        engine (tabgpgo/evolution.py), reused rather than re-derived: builds
        ONE registry shared across the WHOLE population (deduped by
        structure object IDENTITY -- population members routinely share
        block/head-structure objects via common ancestry: mutation only
        ever appends a block, see _apply_block, and head_structure is passed
        through unchanged forever), stacks each distinct structure's
        semantics on the new dataset into a single (n_distinct, n_rows)
        `pool` tensor (exactly one evaluate_structure call per distinct
        structure -- this is the actual floor; nothing can need fewer since
        none of them have ever been evaluated against this dataset before),
        then folds every individual via pool[idx] tensor indexing instead of
        a fresh tree-interpreter walk. individual_semantics_sequential is
        used unconditionally rather than individual_semantics' faster
        homogeneous-wrapper path, mirroring TensorSLIM's own
        `self.mixed = wrapper == "mix" or operator == "mix"` check: SLIM*MIX
        (wrapper="mix") needs per-block wrapper dispatch regardless of
        precomputation, so the fold itself stays a per-block Python loop --
        precomputation only removes the LEAF (tree) evaluation cost, not
        that dispatch, exactly like the reference engine."""
        idx = self._next_chunk_index()
        self.active_dataset_idx = idx
        self.T_train, self.y_target = self._dataset_chunks[idx]

        registry = []
        row_of = {}

        def _row(structure):
            key = id(structure)
            row = row_of.get(key)
            if row is None:
                row = len(registry)
                row_of[key] = row
                registry.append(structure)
            return row

        pool_inds = []
        for ind in population:
            head_idx = _row(ind.head_structure)
            blocks = [Block(_row(b.structure1),
                            _row(b.structure2) if b.structure2 is not None else None,
                            b.ms, b.wrapper, b.operator)
                     for b in ind.blocks]
            pool_inds.append(PoolIndividual(head_idx, blocks))

        pool = torch.stack([_raw_semantics(s, self.T_train, self.TERMINALS) for s in registry])

        # omt_frac>0 can inject wrapper="omt" blocks even when self.wrapper
        # is otherwise fixed (see _finish_omt_inflate) -- unlike TensorSLIM
        # (which never has OMT blocks, see config.py), so the homogeneity
        # check here also requires OMT disabled, not just a non-mix wrapper.
        homogeneous = self.wrapper != "mix" and self.operator != "mix" and self.omt_frac == 0
        for ind, pi in zip(population, pool_inds):
            ind.aggregate = (individual_semantics(pi, self.wrapper, self.operator, pool)
                             if homogeneous else individual_semantics_sequential(pi, pool))
            self._evaluate(ind)

        self.elite = self._best(population)
        self.best_fitness_ever = self.elite.fitness
        self.stall_count = 0
        if verbose:
            n_structs = sum(1 + sum(1 + (b.structure2 is not None) for b in ind.blocks)
                           for ind in population)
            print(f"  [{self.algo} seed {self.seed}] gen {gen}: switched to synthetic "
                  f"dataset #{idx} (refolded {len(population)} individuals via an indexed "
                  f"pool of {len(registry)} distinct structures, {n_structs} total block "
                  f"references, elite_rmse={self.elite.fitness:.4f})")

    def _maybe_switch_dataset(self, population, gen, verbose):
        if not self._rotating:
            return
        self._gens_since_switch += 1
        if self._gens_since_switch >= self._dataset_switch_every:
            self._switch_dataset(population, gen, verbose)
            self._gens_since_switch = 0

    def _global_pool_metrics(self):
        """Elite RMSE/R^2 against the WHOLE synthetic pool (self.T_train_full
        /self.y_target_full, fixed regardless of rotation) -- a logging-only
        metric, comparable across generations/dataset switches, alongside
        the "local" self.elite.fitness (which is only ever comparable within
        one active-dataset block)."""
        agg = self._refold_on(self.elite.head_structure, self.elite.blocks, self.T_train_full)
        pred = self.elite.ls_a + self.elite.ls_b * agg if self.use_ls else agg
        return float(rmse(self.y_target_full, pred)), float(r2(self.y_target_full, pred))

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
        every candidate's val semantics up front (most never become elite).

        When use_ls is on, applies the elite's (ls_a, ls_b) -- fit on the
        training set only, in _evaluate/_resync_elite -- to the validation
        semantics too. This is the standard LS-GP protocol: the affine
        transform is frozen from training data and never refit against
        validation targets, so this stays a genuine held-out evaluation."""
        pi, registry = to_registry(self.elite)
        scaled, raw, r2s = [], [], []
        for name in self.cfg.val_datasets:
            sem = semantics_from_tokens(pi, self.T_val[name], registry, self.TERMINALS)
            if self.use_ls:
                sem = self.elite.ls_a + self.elite.ls_b * sem
            y = self.val_targets[name]
            scaled.append(float(rmse(y, sem)))
            r2s.append(float(r2(y, sem)))
            mean, std = self.val_y_stats[name]
            mean, std = mean.to(sem.device).squeeze(), std.to(sem.device).squeeze()
            raw.append(float(rmse(y * std + mean, sem * std + mean)))
        return scaled, raw, r2s

    # -- main loop ----------------------------------------------------------

    def solve(self, run_info, log_path, verbose=1, track_val=True):
        """`track_val=False` skips elite_val_metrics()/CSV logging entirely
        (run_info/log_path are then unused and may be None) -- for a nested
        OMT search, whose only job is matching a residual on T_train, where
        "validation" isn't a meaningful concept and per-generation CSV rows
        would otherwise flood the outer algorithm's own results file."""
        cfg = self.cfg

        if self._rotating:
            idx = self._next_chunk_index()
            self.active_dataset_idx = idx
            self.T_train, self.y_target = self._dataset_chunks[idx]
            if verbose:
                print(f"  [{self.algo} seed {self.seed}] gen 0: starting on synthetic "
                      f"dataset #{idx}")

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
        self._log(0, time.time() - start, population, run_info, log_path, verbose, track_val)

        for gen in range(1, cfg.n_gens + 1):
            start = time.time()
            self._maybe_switch_dataset(population, gen, verbose)
            self._ensure_reservoir(gen)
            offspring = sorted(population,
                               key=lambda i: (i.fitness, i.nodes_count))[:cfg.n_elites]
            new_offspring = self._make_offspring(population, cfg.pop_size - len(offspring), gen, verbose)
            for child in new_offspring:
                self._evaluate(child)
            offspring.extend(new_offspring)
            population = offspring
            self.elite = self._best(population)
            self._resync_elite()   # bound incremental-update drift
            self._check_stagnation(population, gen)
            self._log(gen, time.time() - start, population, run_info, log_path, verbose, track_val)
        return self.elite

    def _resync_elite(self):
        """Correct floating-point drift accumulated by O(1) incremental
        inflate/deflate updates: recompute the elite's aggregate from scratch
        via its own stored structures, once per generation (cheap -- one
        individual). Mutates self.elite in place, so if it's carried forward
        as an elite next generation its cached state stays accurate."""
        self.elite.aggregate = self._refold(self.elite.head_structure, self.elite.blocks)
        if self.use_ls:
            a, b = linear_scaling(self.y_target, self.elite.aggregate)
            self.elite.ls_a, self.elite.ls_b = float(a), float(b)
            self.elite.fitness = float(rmse(self.y_target, a + b * self.elite.aggregate))
        else:
            self.elite.fitness = float(rmse(self.y_target, self.elite.aggregate))

    def _log(self, gen, elapsed, population, run_info, log_path, verbose, track_val=True):
        train_pred = (self.elite.ls_a + self.elite.ls_b * self.elite.aggregate
                     if self.use_ls else self.elite.aggregate)
        train_r2 = float(r2(self.y_target, train_pred))
        if not track_val:
            if verbose:
                print(f"  [{self.algo} seed {self.seed}] gen {gen}/{self.cfg.n_gens} "
                      f"train_rmse={self.elite.fitness:.4f} train_R2={train_r2:.3f} "
                      f"size={self.elite.size} reservoir={len(self.reservoir)} | {elapsed:.2f}s")
            return
        scaled, raw, r2s = self.elite_val_metrics()
        total_nodes = sum(i.nodes_count for i in population)
        # CSV columns: [algo, run_id, dataset, seed, generation,
        #               elite_train_rmse, time, population_nodes,
        #               val_{d}_rmse_scaled, val_{d}_rmse_raw, val_{d}_r2
        #               (cfg.val_datasets order), elite_size, elite_nodes,
        #               elite_train_r2] + (only when dataset rotation is
        #               enabled) [global_pool_rmse, global_pool_r2,
        #               active_dataset_idx] -- "elite_train_rmse"/train_r2
        #               above are the LOCAL fitness on whichever synthetic
        #               dataset is currently active (only comparable within
        #               one active-dataset block); global_pool_* is the same
        #               elite scored against the WHOLE synthetic pool, and
        #               stays comparable across every generation/switch.
        val_cols = [v for triple in zip(scaled, raw, r2s) for v in triple]
        extra_cols = []
        if self._rotating:
            g_rmse, g_r2 = self._global_pool_metrics()
            extra_cols = [g_rmse, g_r2, self.active_dataset_idx]
        with _LOG_LOCK:
            logger(log_path, gen, self.elite.fitness, elapsed, float(total_nodes),
                   additional_infos=[*val_cols, self.elite.size,
                                     self.elite.nodes_count, train_r2, *extra_cols],
                   run_info=run_info, seed=self.seed)
        if verbose:
            val_str = " ".join(f"{name}={scaled[i]:.3f}/{raw[i]:.3f}/R2={r2s[i]:.3f}"
                               for i, name in enumerate(self.cfg.val_datasets))
            global_str = (f" global_pool_rmse={extra_cols[0]:.4f}/R2={extra_cols[1]:.3f} "
                         f"dataset#{extra_cols[2]}" if self._rotating else "")
            print(f"  [{self.algo} seed {self.seed}] gen {gen}/{self.cfg.n_gens} "
                  f"train_rmse={self.elite.fitness:.4f} train_R2={train_r2:.3f} "
                  f"size={self.elite.size} reservoir={len(self.reservoir)} |{global_str} "
                  f"val(scaled/raw/R2): {val_str} | {elapsed:.2f}s")

