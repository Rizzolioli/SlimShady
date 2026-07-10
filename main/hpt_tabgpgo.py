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

The scan (_scan_run_dirs) tolerates a messy run_dir_base: directories with
truncated/malformed config.json or elite.json (e.g. from a partial zip/rsync
transfer) are skipped with a printed warning rather than crashing the scan or
silently miscounting, and if more than one directory maps to the same grid
combo (e.g. two machines/attempts both completed the same combo and their run
directories got merged into one folder), the first one found is kept and the
rest are reported as duplicates instead of being silently dropped. Run with
verbose=1 (the default) to see this diagnostic output.

LOG_PATH/RUN_DIR_BASE/MANIFEST_PATH are anchored to REPO_ROOT, not bare
relative paths, so they resolve correctly regardless of the cwd this script
is invoked from.
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
from tabgpgo.config import ALGO_NAMES, REPO_ROOT, TabGPGOConfig

MIX_VARIANTS = [("mix", "sum"), ("mix", "mul"), ("mix", "mix")]  # SLIM+MIX, SLIM*MIX, SLIM~MIX

GRID = {
    "pop_size": [50, 100, 200],
    "tournament_size": [2, 5, 10],
    "p_inflate": [0.3, 0.5, 0.7],
    "n_elites": [1, 3, 5, 10],
}

# Anchored to REPO_ROOT (not a bare relative path) so these resolve correctly
# regardless of the cwd this script is invoked from -- a relative path here
# previously caused a `cd main && python hpt_tabgpgo.py` invocation to nest
# everything under main/main/log/, silently failing to find/resume the real
# run directory and duplicating an already-completed grid slice from scratch.
LOG_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_hpt_results.csv")
RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_hpt_runs")
MANIFEST_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_hpt_manifest.csv")


def _read_run_dir(full, d, verbose=True):
    """Load (cfg, elite) from a run directory, or None if it isn't a valid,
    complete run. Tolerates directories that merely *look* done (both files
    present) but aren't -- e.g. truncated/partial copies from a zip or rsync
    transfer that cut a JSON file off mid-write, or an elite.json missing
    fields because save_run() was interrupted mid-write. Prints a warning and
    treats the directory as incomplete (never as a crash) so one bad transfer
    can't take down the whole scan or silently miscount what's really done."""
    cfg_path = os.path.join(full, "config.json")
    elite_path = os.path.join(full, "elite.json")
    if not (os.path.isfile(cfg_path) and os.path.isfile(elite_path)):
        return None
    try:
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        with open(elite_path) as fh:
            elite = json.load(fh)
        for key in ("pop_size", "tournament_size", "p_inflate", "n_elites"):
            _ = cfg[key]
        for key in ("algo", "seed", "fitness", "nodes_count", "blocks"):
            _ = elite[key]
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        if verbose:
            print(f"  [skip] {d}: malformed/incomplete run directory ({exc})")
        return None
    return cfg, elite


def _scan_run_dirs(run_dir_base, verbose=True):
    """Scan run_dir_base for valid completed runs, keyed by
    (pop_size, tournament_size, p_inflate, n_elites, algo, seed) ->
    (dir_name, cfg, elite). Skips malformed directories (see _read_run_dir)
    and, when several directories map to the same combo key (duplicate/
    resumed-from-scratch runs of the same grid point -- e.g. from a transfer
    that landed run artifacts in more than one place), keeps the first one
    found and reports the rest as duplicates rather than silently dropping
    them, so mismatches between what a scan finds and what's actually unique
    on disk are visible instead of guessed at."""
    by_key = {}
    duplicates = []
    if not os.path.isdir(run_dir_base):
        if verbose:
            print(f"  [scan] {run_dir_base} does not exist")
        return by_key, duplicates
    entries = sorted(os.listdir(run_dir_base))
    n_valid = 0
    for d in entries:
        full = os.path.join(run_dir_base, d)
        if not os.path.isdir(full):
            continue
        parsed = _read_run_dir(full, d, verbose=verbose)
        if parsed is None:
            continue
        cfg, elite = parsed
        n_valid += 1
        key = (cfg["pop_size"], cfg["tournament_size"], cfg["p_inflate"],
              cfg["n_elites"], elite["algo"], elite["seed"])
        if key in by_key:
            duplicates.append((key, d, by_key[key][0]))
            continue
        by_key[key] = (d, cfg, elite)
    if verbose:
        print(f"  [scan] {run_dir_base}: {len(entries)} entries, {n_valid} valid runs, "
              f"{len(by_key)} unique combos, {len(duplicates)} duplicate(s)")
        for key, dropped_dir, kept_dir in duplicates:
            print(f"    duplicate combo {key}: kept {kept_dir}, ignored {dropped_dir}")
    return by_key, duplicates


def _scan_completed(run_dir_base, verbose=True):
    """(pop_size, tournament_size, p_inflate, n_elites, algo, seed) keys for
    every run directory that finished (has both config.json and elite.json,
    and both parse with the fields build_jobs()/the manifest need). A
    directory missing either file, or with malformed content, was interrupted
    mid-run or corrupted in transit, so it's NOT counted as completed and its
    combo will be resubmitted."""
    by_key, _ = _scan_run_dirs(run_dir_base, verbose=verbose)
    return set(by_key)


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


def _rebuild_manifest(run_dir_base, manifest_path, verbose=True):
    """Regenerate the manifest CSV from every valid, unique completed run
    directory on disk (this invocation's runs plus any from previous,
    interrupted invocations or merged-in transfers), so it always reflects
    what's actually there -- one row per unique grid combo, malformed or
    duplicate directories excluded (see _scan_run_dirs)."""
    by_key, _ = _scan_run_dirs(run_dir_base, verbose=verbose)
    rows = []
    for (pop_size, tournament_size, p_inflate, n_elites, algo, seed), (d, cfg, elite) in by_key.items():
        rows.append({
            "run_id": d.split("_", 1)[0], "algo": algo,
            "variant": algo.rsplit("_", 1)[0], "seed": seed,
            "pop_size": pop_size, "tournament_size": tournament_size,
            "p_inflate": p_inflate, "n_elites": n_elites,
            "elite_train_rmse": elite["fitness"], "elite_size": len(elite["blocks"]) + 1,
            "elite_nodes": elite["nodes_count"],
        })
    rows.sort(key=lambda r: r["run_id"])
    if rows:
        with open(manifest_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def run_hpt(max_workers=4, seed=0, verbose=1):
    base_cfg = TabGPGOConfig()
    ctx = prepare(base_cfg, verbose=True)

    completed = _scan_completed(RUN_DIR_BASE, verbose=bool(verbose))
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

    manifest_rows = _rebuild_manifest(RUN_DIR_BASE, MANIFEST_PATH, verbose=bool(verbose))
    print(f"wrote {len(manifest_rows)} rows -> {MANIFEST_PATH}")
    return manifest_rows


if __name__ == "__main__":
    run_hpt()
