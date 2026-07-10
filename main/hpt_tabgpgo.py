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
Resumable: before submitting jobs, scans main/log/tabgpgo_hpt_runs/ for run
directories that already have both config.json and elite.json (save_run()
only writes those after solve() + verify_inference() succeed, so a directory
missing either file was interrupted mid-run and gets redone) and skips any
(pop_size, tournament_size, p_inflate, n_elites, variant, seed) combo that's
already there. The manifest is rebuilt from that same directory scan every
time run_hpt() finishes, so it always reflects everything on disk -- previous
invocations' runs included -- regardless of how many times the process gets
interrupted.
"""
import csv
import dataclasses
import itertools
import json
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


def _scan_completed(run_dir_base):
    """(pop_size, tournament_size, p_inflate, n_elites, algo, seed) keys for
    every run directory that finished (has both config.json and elite.json).
    A directory missing either file was interrupted mid-run, so it's NOT
    counted as completed and its combo will be resubmitted."""
    completed = set()
    if not os.path.isdir(run_dir_base):
        return completed
    for d in os.listdir(run_dir_base):
        full = os.path.join(run_dir_base, d)
        cfg_path = os.path.join(full, "config.json")
        elite_path = os.path.join(full, "elite.json")
        if not (os.path.isfile(cfg_path) and os.path.isfile(elite_path)):
            continue
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        with open(elite_path) as fh:
            elite = json.load(fh)
        completed.add((cfg["pop_size"], cfg["tournament_size"], cfg["p_inflate"],
                       cfg["n_elites"], elite["algo"], elite["seed"]))
    return completed


def build_jobs(base_cfg, seed, completed):
    """One (cfg, variant, overrides) tuple per (grid combo x MIX variant),
    skipping combos already present in `completed`."""
    keys = list(GRID)
    jobs = []
    for values in itertools.product(*(GRID[k] for k in keys)):
        overrides = dict(zip(keys, values))
        cfg = dataclasses.replace(base_cfg, log_path=LOG_PATH,
                                  run_dir_base=RUN_DIR_BASE, **overrides)
        for variant in MIX_VARIANTS:
            algo = f"{ALGO_NAMES[variant]}_oms"
            key = (overrides["pop_size"], overrides["tournament_size"],
                  overrides["p_inflate"], overrides["n_elites"], algo, seed)
            if key in completed:
                continue
            jobs.append((cfg, variant, overrides))
    return jobs


def _rebuild_manifest(run_dir_base, manifest_path):
    """Regenerate the manifest CSV from every completed run directory on disk
    (this invocation's runs plus any from previous, interrupted invocations),
    so it's always consistent with what's actually there."""
    rows = []
    for d in sorted(os.listdir(run_dir_base)) if os.path.isdir(run_dir_base) else []:
        full = os.path.join(run_dir_base, d)
        cfg_path = os.path.join(full, "config.json")
        elite_path = os.path.join(full, "elite.json")
        if not (os.path.isfile(cfg_path) and os.path.isfile(elite_path)):
            continue
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        with open(elite_path) as fh:
            elite = json.load(fh)
        rows.append({
            "run_id": d.split("_", 1)[0], "algo": elite["algo"],
            "variant": elite["algo"].rsplit("_", 1)[0], "seed": elite["seed"],
            "pop_size": cfg["pop_size"], "tournament_size": cfg["tournament_size"],
            "p_inflate": cfg["p_inflate"], "n_elites": cfg["n_elites"],
            "elite_train_rmse": elite["fitness"], "elite_size": len(elite["blocks"]) + 1,
            "elite_nodes": elite["nodes_count"],
        })
    if rows:
        with open(manifest_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def run_hpt(max_workers=4, seed=0, verbose=1):
    base_cfg = TabGPGOConfig()
    ctx = prepare(base_cfg, verbose=True)

    completed = _scan_completed(RUN_DIR_BASE)
    jobs = build_jobs(base_cfg, seed, completed)
    total_planned = (len(GRID["pop_size"]) * len(GRID["tournament_size"]) *
                     len(GRID["p_inflate"]) * len(GRID["n_elites"]) * len(MIX_VARIANTS))
    print(f"HPT grid: {total_planned} total combos, {len(completed)} already done, "
          f"{len(jobs)} left to run")

    if jobs:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_run_one, cfg, variant, "oms", seed, ctx,
                                   uuid.uuid1(), verbose)
                      for cfg, variant, overrides in jobs]
            for future in futures:
                future.result()

    manifest_rows = _rebuild_manifest(RUN_DIR_BASE, MANIFEST_PATH)
    print(f"wrote {len(manifest_rows)} rows -> {MANIFEST_PATH}")
    return manifest_rows


if __name__ == "__main__":
    run_hpt()
