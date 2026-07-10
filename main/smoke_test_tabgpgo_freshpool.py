"""
End-to-end CPU smoke test for the fresh-pool TabGPGO variant.

Runs the whole pipeline (synthetic data, AE, latent tokens, fresh-pool
evolution -- NO static tree pool) at tiny sizes with hard assertions specific
to this variant: prepare() actually skips stage 3, the reservoir never goes
negative, the incrementally-cached elite aggregate matches a from-scratch
structural refold every generation (bounds floating-point drift), elite
fitness is non-increasing (elitism), and verify_inference/
reconstruct_expression pass via the build_adapter() shim into
tabgpgo/inference.py (unmodified). Must finish in minutes on a CPU-only
machine.

Usage:  python main/smoke_test_tabgpgo_freshpool.py
"""
import csv
import dataclasses
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from main_tabgpgo import prepare
from main_tabgpgo_freshpool import evolve_freshpool
from tabgpgo.config import TabGPGOConfig
from tabgpgo.evolution_freshpool import (BOUND, FreshIndividual, FreshPoolSLIM,
                                         _raw_semantics, build_adapter)
from tabgpgo.inference import reconstruct_expression, verify_inference

tmp_dir = tempfile.mkdtemp(prefix="tabgpgo_freshpool_smoke_")
cfg = TabGPGOConfig(
    device="cpu",
    prior_backend="simple_scm",
    n_synth_datasets=20, n_rows=100, min_features=2, max_features=100,
    val_datasets=("instanbul", "energy", "concrete", "ppb"),  # incl. one >100-feat
    reducer="pca",
    ae_epochs=5, ae_batch=512,
    pop_size=20, n_gens=8, n_runs=1,
    tournament_size=2, p_inflate=0.5, n_elites=1,
    stagnation_patience_values=(5,),   # pinned to 1 value here; the patience sweep axis itself
                                        # is tested separately below with its own tiny cfg
    log_path=os.path.join(tmp_dir, "results.csv"),
    run_dir_base=os.path.join(tmp_dir, "runs"),
    artifacts_dir=os.path.join(tmp_dir, "artifacts"),
)

# --- prepare(build_static_pool=False) must skip stage 3 entirely -------------
ctx = prepare(cfg, verbose=1, build_static_pool=False)
assert ctx["registry"] is None and ctx["pool_train"] is None and ctx["val_pools"] is None
assert ctx["T_val"] is not None and set(ctx["T_val"]) == set(cfg.val_datasets)
print("prepare(build_static_pool=False) skipped stage 3 correctly")

# --- single tracked run: reservoir/drift/elitism invariants ------------------
# SLIM~MIX (per-block sum/mul) exercises the order-dependent deflate refold
# path, and OMS exercises _optimal_ms -- the two riskiest code paths.
variant = ("mix", "mix")
optimizer = FreshPoolSLIM(cfg, variant, ctx["T_train"], ctx["T_val"], ctx["y_target"],
                          ctx["val_targets"], ctx["val_y_stats"], ctx["TERMINALS"],
                          seed=0, ms_spec="oms")

fitness_history = []
orig_log = optimizer._log
def tracking_log(gen, elapsed, population, run_info, log_path, verbose):
    assert len(optimizer.reservoir) >= 0, "reservoir went negative"
    fitness_history.append(optimizer.elite.fitness)
    refolded = optimizer._refold(optimizer.elite.head_structure, optimizer.elite.blocks)
    assert torch.allclose(optimizer.elite.aggregate, refolded, atol=1e-3, rtol=1e-3), (
        f"gen {gen}: incrementally-cached elite aggregate diverged from a "
        f"from-scratch structural refold (max abs diff "
        f"{(optimizer.elite.aggregate - refolded).abs().max().item():.3e})")
    orig_log(gen, elapsed, population, run_info, log_path, verbose)
optimizer._log = tracking_log

elite = optimizer.solve(run_info=[optimizer.algo, "smoke-tracked-run", "synthetic_prior"],
                        log_path=cfg.log_path, verbose=0)
assert len(fitness_history) == cfg.n_gens + 1
assert all(fitness_history[i] >= fitness_history[i + 1] - 1e-3
          for i in range(len(fitness_history) - 1)), fitness_history
print(f"elite fitness non-increasing across {cfg.n_gens} generations OK: "
      f"{fitness_history[0]:.4f} -> {fitness_history[-1]:.4f} "
      f"(final elite size={elite.size})")

# --- adapter shim: reuses tabgpgo/inference.py completely unmodified ---------
pi, registry, pool = build_adapter(elite, ctx["T_train"], ctx["TERMINALS"])
verify_inference(pi, pool, ctx["T_train"], registry, ctx["TERMINALS"])
expression = reconstruct_expression(pi, registry)
assert isinstance(expression, str) and len(expression) > 0
print(f"verify_inference + reconstruct_expression via build_adapter() OK "
      f"(expression length={len(expression)})")

# --- anti-stagnation sweep: forced, deterministic (doesn't rely on natural
# stagnation occurring within a short random run) -----------------------------
stag_cfg = dataclasses.replace(cfg, stagnation_patience=2, stagnation_replace_frac=0.5)
stag_opt = FreshPoolSLIM(stag_cfg, ("abs", "sum"), ctx["T_train"], ctx["T_val"],
                         ctx["y_target"], ctx["val_targets"], ctx["val_y_stats"],
                         ctx["TERMINALS"], seed=1)
stag_opt._ensure_reservoir(0)
stag_population = []
for _ in range(stag_cfg.pop_size):
    tree = stag_opt._pop_tree()
    agg = torch.clamp(_raw_semantics(tree["structure"], stag_opt.T_train, stag_opt.TERMINALS),
                      -BOUND, BOUND)
    stag_population.append(FreshIndividual(tree["structure"], tree["nodes"], agg))
for ind in stag_population:
    stag_opt._evaluate(ind)
stag_opt.elite = stag_opt._best(stag_population)
stag_opt.best_fitness_ever = stag_opt.elite.fitness

elite_before = stag_opt.elite
elite_fitness_before = stag_opt.elite.fitness
# Keep strong references to the original objects (not just their ids): once an
# old individual is overwritten in-place, CPython can immediately reclaim and
# reuse its memory address for a newly-created one, making an id()-only
# before/after set comparison unreliable -- compare identity per-index instead.
old_individuals = list(stag_population)
n_replace = int(stag_cfg.pop_size * stag_cfg.stagnation_replace_frac)

assert stag_opt.stall_count == 0
stag_opt._check_stagnation(stag_population, gen=1)   # no improvement -> stall_count=1, no sweep yet
assert stag_opt.stall_count == 1, stag_opt.stall_count
assert all(stag_population[i] is old_individuals[i] for i in range(len(stag_population))), \
    "no sweep should have happened yet"

stag_opt._check_stagnation(stag_population, gen=2)   # 2nd non-improving gen (patience=2) -> sweep
assert stag_opt.stall_count == 0, "stall_count must reset immediately after a sweep"

replaced = [stag_population[i] for i in range(len(stag_population))
           if stag_population[i] is not old_individuals[i]]
assert len(replaced) == n_replace, f"expected {n_replace} replaced, got {len(replaced)}"
assert all(len(ind.blocks) == 0 for ind in replaced), "swept-in individuals must be head-only"
assert all(torch.isfinite(torch.tensor(ind.fitness)) and ind.nodes_count > 0 for ind in replaced)
assert stag_opt.elite is elite_before and stag_opt.elite.fitness == elite_fitness_before, \
    "the elite must never be touched by a stagnation sweep"
# `in` would fall back to FreshIndividual's dataclass-generated __eq__, which
# compares the `aggregate` tensor field element-wise -- use identity directly.
assert any(ind is elite_before for ind in stag_population), \
    "the elite individual must still be present"
print(f"anti-stagnation sweep OK: {len(replaced)}/{stag_cfg.pop_size} replaced, "
      f"elite untouched, stall_count reset")

# --- stagnation_patience as a sweep axis in evolve_freshpool -----------------
# Tiny cross (1 variant x 1 ms x 2 patience values) confirms the plumbing:
# each combo gets its own cfg/algo label, "None" is accepted without raising,
# and distinct patience values produce distinct, separately-logged runs.
patience_cfg = dataclasses.replace(
    cfg, variants=(("mix", "mix"),), ms_hi_values=("oms",),
    stagnation_patience_values=(None, 5))
patience_elites, patience_run_id = evolve_freshpool(patience_cfg, ctx, verbose=0)
assert len(patience_elites) == 2, patience_elites.keys()
algos = {algo for algo, seed in patience_elites}
assert any(a.endswith("_patnone") for a in algos), algos
assert any(a.endswith("_pat5") for a in algos), algos
for (algo, seed), e in patience_elites.items():
    assert torch.isfinite(torch.tensor(e.fitness)), f"{algo}: bad fitness"
print(f"stagnation_patience sweep axis OK: {sorted(algos)}")

# --- full concurrent sweep: all variants x ms_hi_values x n_runs -------------
elites, run_id = evolve_freshpool(cfg, ctx, verbose=0)
assert len(elites) == len(cfg.variants) * len(cfg.ms_hi_values) * cfg.n_runs
for (algo, seed), e in elites.items():
    assert torch.isfinite(torch.tensor(e.fitness)), f"{algo}: bad fitness"
print("concurrent sweep OK:", len(elites), "runs")

# --- CSV: one row per generation (incl. gen 0) per sweep run ------------------
with open(cfg.log_path) as fh:
    rows = list(csv.reader(fh))
sweep_rows = [r for r in rows if r[1] == str(run_id)]
expected = len(cfg.variants) * len(cfg.ms_hi_values) * cfg.n_runs * (cfg.n_gens + 1)
assert len(sweep_rows) == expected, f"CSV rows {len(sweep_rows)} != {expected}"
print("CSV logging OK")

# --- persisted run artifacts, one dir per sweep run ---------------------------
run_dirs = [d for d in os.listdir(cfg.run_dir_base) if d.startswith(str(run_id))]
assert len(run_dirs) == len(cfg.variants) * len(cfg.ms_hi_values) * cfg.n_runs
print("persisted run artifacts OK:", len(run_dirs), "dirs")

shutil.rmtree(tmp_dir, ignore_errors=True)
print("\nFRESH-POOL SMOKE TEST PASSED")
