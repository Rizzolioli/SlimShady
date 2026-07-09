"""
Small hyperparameter grid for TabGPGO, restricted to the 3 mix variants
(SLIM+MIX, SLIM*MIX, SLIM~MIX) with the mutation-step strategy fixed to OMS.

Sweeps pop_size x tournament_size x p_inflate x n_elites -- pure
evolution-loop knobs (TensorSLIM reads them straight off cfg/self.p_deflate
at solve() time), so they never touch the tree pool, registry, or
autoencoder. prepare() therefore runs exactly once and every combo in the
grid reuses the same pool_train/registry/val_pools.

    python main/hpt_tabgpgo.py

Writes:
  main/log/tabgpgo_hpt_results.csv   -- same per-generation columns as
                                         tabgpgo_results.csv (see
                                         main_tabgpgo.py's docstring)
  main/log/tabgpgo_hpt_manifest.csv  -- one row per run, mapping its run_id
                                         back to the grid combo that produced
                                         it (pop_size, tournament_size,
                                         p_inflate, n_elites, variant, seed)
                                         plus its final elite_train_rmse/
                                         size/nodes, since the algo label
                                         alone ("SLIM+MIX_oms") doesn't
                                         encode the grid axes.
  main/log/tabgpgo_hpt_runs/          -- persisted run artifacts, one dir per
                                         run (elite.json/config.json/etc.),
                                         same as main_tabgpgo.py's evolve()
"""
import csv
import dataclasses
import itertools
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_tabgpgo import _run_one, prepare
from tabgpgo.config import ALGO_NAMES, TabGPGOConfig

MIX_VARIANTS = [("mix", "sum"), ("mix", "mul"), ("mix", "mix")]  # SLIM+MIX, SLIM*MIX, SLIM~MIX

GRID = {
    "pop_size": [50, 100, 200],
    "tournament_size": [2, 5, 10],
    "p_inflate": [0.3, 0.5, 0.7],
    "n_elites": [1, 3, 5, 10],
}

LOG_PATH = os.path.join("main", "log", "tabgpgo_hpt_results.csv")
RUN_DIR_BASE = os.path.join("main", "log", "tabgpgo_hpt_runs")
MANIFEST_PATH = os.path.join("main", "log", "tabgpgo_hpt_manifest.csv")


def build_jobs(base_cfg):
    """One (cfg, variant, overrides) tuple per (grid combo x MIX variant)."""
    keys = list(GRID)
    jobs = []
    for values in itertools.product(*(GRID[k] for k in keys)):
        overrides = dict(zip(keys, values))
        cfg = dataclasses.replace(base_cfg, log_path=LOG_PATH,
                                  run_dir_base=RUN_DIR_BASE, **overrides)
        for variant in MIX_VARIANTS:
            jobs.append((cfg, variant, overrides))
    return jobs


def _submit(cfg, variant, overrides, ctx, seed, verbose):
    run_id = uuid.uuid1()
    (algo, run_seed), elite = _run_one(cfg, variant, "oms", seed, ctx, run_id, verbose)
    return {
        "run_id": run_id, "algo": algo, "variant": ALGO_NAMES[variant], "seed": run_seed,
        **overrides,
        "elite_train_rmse": elite.fitness, "elite_size": elite.size,
        "elite_nodes": elite.nodes_count,
    }


def run_hpt(max_workers=4, seed=0, verbose=1):
    base_cfg = TabGPGOConfig()
    ctx = prepare(base_cfg, verbose=True)

    jobs = build_jobs(base_cfg)
    print(f"HPT grid: {len(jobs)} runs "
          f"({len(GRID['pop_size'])}x{len(GRID['tournament_size'])}x"
          f"{len(GRID['p_inflate'])}x{len(GRID['n_elites'])} combos x "
          f"{len(MIX_VARIANTS)} variants)")

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    manifest_rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_submit, cfg, variant, overrides, ctx, seed, verbose)
                  for cfg, variant, overrides in jobs]
        for future in futures:
            manifest_rows.append(future.result())

    with open(MANIFEST_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"wrote {len(manifest_rows)} rows -> {MANIFEST_PATH}")
    return manifest_rows


if __name__ == "__main__":
    run_hpt()
