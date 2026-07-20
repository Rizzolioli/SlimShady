"""
TabGPGO experiment (TabPFN encoder + rotating synthetic dataset): tests
whether forcing the elite to keep performing well as its TRAINING data
itself keeps changing -- rather than fitting one fixed pooled synthetic
tensor for the whole run, the way every other experiment this session does
-- improves zero-shot generalization to the real validation datasets.

Encoder: the same frozen TabPFN embedding pipeline as
main_tabgpgo_tabpfn_encoder.py (tabgpgo/tabpfn_encoder.py, n_estimators=1).
Deliberately reuses that script's own cache directory
(main/log/tabgpgo_tabpfn_artifacts/tabpfn_pool_bundle.pkl): the TabPFN pool
only depends on cfg's synthetic-prior fields (n_synth_datasets, n_rows,
max_features, min_features, data_seed, prior_backend), none of which this
script's HPT_OVERRIDES touch, so it's the exact same expensive (~1000
TabPFNRegressor fits) artifact either script would produce -- if
main_tabgpgo_tabpfn_encoder.py has already been run on a machine, this
script reuses that cache directly with zero extra TabPFN fitting.

Dataset rotation: every synthetic dataset contributes exactly cfg.n_rows
rows to the pooled (T_train, y_target) tensor build_tabpfn_pool returns (see
tabgpgo.prior.generate_datasets), so torch.split by cfg.n_rows recovers
each dataset's own (T_i, y_i) slice with zero extra encoder computation --
see _split_chunks below. tabgpgo.evolution_freshpool.FreshPoolSLIM's new
dataset_chunks/dataset_switch_every constructor args (see that module) then
draw a shuffled-cycle sequence of these per-dataset slices, switching the
ACTIVE (T_train, y_target) every `switch_every` generations, refolding the
whole population against the new one, and resetting the anti-stagnation
tracker each time (a "no improvement" streak isn't meaningful across a
dataset change). The elite's fitness against the WHOLE pool (fixed,
independent of rotation) is also logged every generation as
global_pool_rmse/global_pool_r2 -- see FreshPoolSLIM._global_pool_metrics --
so the "local" per-block curve and the dataset-independent "global"
generalization curve can both be plotted.

Sweeps ms (explicit, non-OMS values, per the experiment request) x
switch_every, 3x3 = 9 combos:
  ms in {0.01, 0.1, 1.0} (uniform-random ms ~ U(0, that bound) per mutation)
  switch_every in {1, 5, 10} (generations per active dataset before rotating)
switch_every=1 is the most expensive setting: EVERY generation triggers a
full population refold (pop_size individuals, each re-evaluated from
scratch against the new dataset), forfeiting the usual O(1) incremental
semantics update for that generation.

Fixed for every combo: SLIM*MIX (wrapper="mix", operator="mul"), fresh pool
(FreshPoolSLIM), patience=5, pop_size=200, n_gens=200, HPT-tuned
hyperparameters, full_extended/small_ints function/constant set (the
established best-known winner from main_tabgpgo_funcset.py's sweep), OMT
disabled (omt_frac=0.0, plain SLIM*MIX mutations only).

Resumable the same way every other tabgpgo sweep script is: evolve_rotate()
skips any (ms, switch_every) combo that already has a completed run dir
(elite.json present).

Usage:
  python main/main_tabgpgo_tabpfn_rotate.py prepare   # phase 1-2 only (build/cache the TabPFN pool)
  python main/main_tabgpgo_tabpfn_rotate.py evolve    # phase 4-5 (reusing the cache, resumable)
  python main/main_tabgpgo_tabpfn_rotate.py           # everything

Writes to main/log/tabgpgo_rotate_results.csv / tabgpgo_rotate_runs/ and
main/log/tabgpgo_rotate_manifest.csv (run_id, algo, ms, switch_every,
function_set, constant_set).

CSV columns of tabgpgo_rotate_results.csv (extends the schema every other
tabgpgo sweep script uses -- see main_tabgpgo.py's own docstring -- with
three rotation-only columns appended at the end):
  [algo, run_id, dataset, seed, generation, elite_train_rmse (LOCAL: on
   whichever synthetic dataset is currently active), time_s,
   population_nodes,
   val_<d>_rmse_scaled, val_<d>_rmse_raw, val_<d>_r2 (per d in
   cfg.val_datasets order),
   elite_size, elite_nodes, elite_train_r2 (LOCAL),
   global_pool_rmse, global_pool_r2 (elite scored against the WHOLE
   synthetic pool -- comparable across every generation regardless of
   rotation), active_dataset_idx (which pool chunk was active this gen)]
"""
import csv
import dataclasses
import json
import os
import pickle
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from main_tabgpgo_funcset import CONSTANT_SETS, FUNCTION_SETS, function_constant_set
from tabgpgo.config import ALGO_NAMES, REPO_ROOT, TabGPGOConfig
from tabgpgo.evolution_freshpool import FreshPoolSLIM, build_adapter
from tabgpgo.inference import reconstruct_expression, save_run, verify_inference
from tabgpgo.preprocessing import load_validation_sets
from tabgpgo.tabpfn_encoder import build_tabpfn_pool, encode_val_tabpfn
from tabgpgo.tree_pool import make_terminals

HPT_OVERRIDES = dict(pop_size=200, tournament_size=2, p_inflate=0.5, n_elites=5)

WRAPPER = "mix"
VARIANT = (WRAPPER, "mul")   # SLIM*MIX
PATIENCE = 5
N_GENS = 200

TABPFN_N_ESTIMATORS = 1   # see tabgpgo/tabpfn_encoder.py

FUNCTION_SET_NAME = "full_extended"
CONSTANT_SET_NAME = "small_ints"

MS_VALUES = [0.01, 0.1, 1.0]
SWITCH_EVERY_VALUES = [1, 5, 10]
COMBOS = [{"ms": ms, "switch_every": se}
         for ms in MS_VALUES for se in SWITCH_EVERY_VALUES]

# Deliberately the SAME directory main_tabgpgo_tabpfn_encoder.py caches its
# TabPFN pool bundle in -- see module docstring.
TABPFN_ARTIFACTS_DIR = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_tabpfn_artifacts")
ROTATE_LOG_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_rotate_results.csv")
ROTATE_RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_rotate_runs")
ROTATE_MANIFEST_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_rotate_manifest.csv")


class _EncoderStub:
    """Stands in for the real MLPAutoencoder at save_run's `model` parameter
    -- see main_tabgpgo_tabpfn_encoder.py's identical class for why."""
    encoder = torch.nn.Identity()


def build_config(**extra):
    return dataclasses.replace(
        TabGPGOConfig(), log_path=ROTATE_LOG_PATH, variants=(VARIANT,),
        run_dir_base=ROTATE_RUN_DIR_BASE, stagnation_patience=PATIENCE, n_gens=N_GENS,
        artifacts_dir=TABPFN_ARTIFACTS_DIR,
        **HPT_OVERRIDES, **extra)


def _cached_pool(cfg, verbose):
    path = os.path.join(cfg.artifacts_dir, "tabpfn_pool_bundle.pkl")
    if cfg.reuse_artifacts and os.path.exists(path):
        if verbose:
            print(f"[cached] TabPFN pool <- {os.path.basename(path)}")
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    result = build_tabpfn_pool(cfg, n_estimators=TABPFN_N_ESTIMATORS, verbose=verbose)
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(result, fh)
    if verbose:
        print(f"[computed] TabPFN pool ({time.time() - t0:.1f}s) -> {os.path.basename(path)}")
    return result


def _split_chunks(T_train, y_target, cfg):
    """Recovers each synthetic dataset's own (T_i, y_i) slice from the
    already-concatenated pool -- every synthetic dataset contributes exactly
    cfg.n_rows rows, in generation order (tabgpgo.prior.generate_datasets),
    so torch.split by cfg.n_rows exactly reconstructs the per-dataset
    boundaries with zero extra encoder computation."""
    T_chunks = torch.split(T_train, cfg.n_rows)
    y_chunks = torch.split(y_target, cfg.n_rows)
    assert len(T_chunks) == cfg.n_synth_datasets, (
        f"expected {cfg.n_synth_datasets} chunks of {cfg.n_rows} rows, got {len(T_chunks)}")
    return list(zip(T_chunks, y_chunks))


def prepare_rotate(cfg, verbose=True):
    """Phase 1-2: builds/caches the TabPFN pool (see _cached_pool), splits
    it into per-synthetic-dataset chunks for rotation, and embeds the real
    validation sets via the frozen reference model -- same shape as
    main_tabgpgo_tabpfn_encoder.py's prepare_tabpfn, plus dataset_chunks."""
    device = cfg.get_device()
    T_train, y_target, reference_model, embed_dim = _cached_pool(cfg, verbose)
    T_train, y_target = T_train.to(device), y_target.to(device)
    dataset_chunks = _split_chunks(T_train, y_target, cfg)

    val_sets = load_validation_sets(cfg)
    T_val = {name: encode_val_tabpfn(reference_model, d["X"]).to(device)
             for name, d in val_sets.items()}
    val_targets = {name: d["y"].to(device) for name, d in val_sets.items()}
    val_y_stats = {name: d["meta"]["y_stats"] for name, d in val_sets.items()}
    TERMINALS = make_terminals(embed_dim)

    if verbose:
        print(f"TabPFN embed_dim={embed_dim}, {len(dataset_chunks)} rotation "
              f"chunks x {cfg.n_rows} rows")

    return {"T_train": T_train, "y_target": y_target, "T_val": T_val,
            "val_targets": val_targets, "val_y_stats": val_y_stats,
            "TERMINALS": TERMINALS, "reference_model": reference_model,
            "dataset_chunks": dataset_chunks}


def _base_algo_label(combo):
    return f"{ALGO_NAMES[VARIANT]}_ms{combo['ms']:g}_pat{PATIENCE}"


def _combo_tag(combo):
    algo = (f"{_base_algo_label(combo)}_fn-{FUNCTION_SET_NAME}_const-{CONSTANT_SET_NAME}"
            f"_ogens{N_GENS}_switch{combo['switch_every']}")
    return algo, algo.replace("*", "x").replace("~", "t")


def find_completed_run_dir(tag):
    if not os.path.isdir(ROTATE_RUN_DIR_BASE):
        return None
    suffix = f"_{tag}_0"
    for name in sorted(os.listdir(ROTATE_RUN_DIR_BASE)):
        if name.endswith(suffix) and os.path.isfile(os.path.join(ROTATE_RUN_DIR_BASE, name, "elite.json")):
            return os.path.join(ROTATE_RUN_DIR_BASE, name)
    return None


def _elite_fitness_only(run_dir):
    with open(os.path.join(run_dir, "elite.json")) as fh:
        return json.load(fh)["fitness"]


def _run_one(cfg, combo, ctx, unique_run_id, verbose):
    algo, tag = _combo_tag(combo)
    optimizer = FreshPoolSLIM(cfg, VARIANT, ctx["T_train"], ctx["T_val"],
                              ctx["y_target"], ctx["val_targets"], ctx["val_y_stats"],
                              ctx["TERMINALS"], seed=0, ms_spec=combo["ms"], use_ls=False,
                              dataset_chunks=ctx["dataset_chunks"],
                              dataset_switch_every=combo["switch_every"])
    optimizer.algo = algo
    elite = optimizer.solve(
        run_info=[optimizer.algo, unique_run_id, "synthetic_prior"],
        log_path=cfg.log_path, verbose=verbose)

    pi, registry, pool = build_adapter(elite, ctx["T_train"], ctx["TERMINALS"])
    verify_inference(pi, pool, ctx["T_train"], registry, ctx["TERMINALS"])
    expression = reconstruct_expression(pi, registry)
    run_dir = os.path.join(cfg.run_dir_base, f"{unique_run_id}_{tag}_0")
    save_run(run_dir, cfg, _EncoderStub(), registry, pi, WRAPPER, "mul",
             optimizer.algo, seed=0, expression=expression)
    if verbose:
        print(f"[{optimizer.algo}] done: train_rmse={elite.fitness:.4f} "
              f"size={elite.size} -> {run_dir}")
    return optimizer.algo, elite


def evolve_rotate(verbose=1):
    cfg = build_config()
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]
    cfg = dataclasses.replace(cfg, p_c=p_c)

    os.makedirs(os.path.dirname(ROTATE_LOG_PATH), exist_ok=True)
    unique_run_id = uuid.uuid1()
    manifest_rows = []
    elites = {}
    with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
        ctx = prepare_rotate(cfg, verbose=bool(verbose))
        for combo in COMBOS:
            algo, tag = _combo_tag(combo)
            existing_dir = find_completed_run_dir(tag)
            if existing_dir is not None:
                if verbose:
                    fitness = _elite_fitness_only(existing_dir)
                    print(f"[{algo}] already completed -> skipping "
                          f"({os.path.basename(existing_dir)}, fitness={fitness:.4f})")
                run_id = os.path.basename(existing_dir).split("_", 1)[0]
            else:
                algo, elite = _run_one(cfg, combo, ctx, unique_run_id, verbose)
                elites[algo] = elite
                run_id = unique_run_id
            manifest_rows.append({
                "run_id": run_id, "algo": algo, "ms": combo["ms"],
                "switch_every": combo["switch_every"],
                "function_set": FUNCTION_SET_NAME, "constant_set": CONSTANT_SET_NAME,
            })
    with open(ROTATE_MANIFEST_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nwrote manifest -> {ROTATE_MANIFEST_CSV}")
    return elites, unique_run_id


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "prepare":
        cfg = build_config()
        p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]
        cfg = dataclasses.replace(cfg, p_c=p_c)
        with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
            prepare_rotate(cfg)
    elif stage == "evolve":
        evolve_rotate()
    elif stage == "all":
        evolve_rotate()
    else:
        sys.exit(f"unknown stage '{stage}' (use: prepare | evolve | all)")
