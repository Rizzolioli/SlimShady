"""
TabGPGO experiment (TabPFN-encoder variant): same fresh-pool pipeline as
main_tabgpgo_omt.py, but Phase 2 (the a-priori encoder) is TabPFN itself
(tabgpgo/tabpfn_encoder.py) instead of the custom MLPAutoencoder -- see that
module's docstring for why TabPFN needs a fundamentally different
pool-building strategy (per-synthetic-dataset fitting, no single global
X -> latent function) and why real validation data is embedded via a frozen
"reference" TabPFN context rather than its own fit.

Motivation: every previous sweep (function sets, ms, linear scaling, latent
sizes, OMT) converged to the same near-zero zero-shot ceiling using our own
autoencoder. TabPFN itself gets R^2=0.94-0.99 on several of these datasets
from the same PCA-100 input -- so this tests whether swapping in TabPFN's
own (frozen, pretrained, never fine-tuned) representation as the GP's
terminal set changes that story at all.

Rather than committing everything to the single best-by-average combo,
this sweeps the 4 DISTINCT (function_set, constant_set) combos that won on
at least one real dataset in main_tabgpgo_funcset.py's sweep (see that
dashboard's comparison table) -- all with OMT disabled (omt_frac=0) -- plus
ONE additional combo (the overall-best full_extended/small_ints) with a
SMALL OMT setting (omt_frac=0.2, omt_pop_size=20), matching the scale that
did not catastrophically overfit in the OMT experiment (unlike the
pop=100/omt_frac=1.0 run, which blew up to ~11k nodes and went negative on
every dataset).

All 5 combos: SLIM*MIX, ms=oms, patience=5, HPT-tuned hyperparameters, 200
generations -- full budget, same as every other 200-generation baseline
this session, since the TabPFN pool is built ONCE and cached (unlike OMT,
nothing here is per-generation-expensive beyond ordinary evolution).

IMPORTANT persistence note: TabPFN is not a torch.nn.Module the way the
custom MLPAutoencoder is, so tabgpgo.inference.save_run's `model.encoder.
state_dict()` call needs a stand-in -- _EncoderStub below supplies a real
(empty) nn.Module so save_run works completely unmodified, matching every
other experiment script's persistence exactly. Resumability's "already
completed" skip path reads elite.json's fitness directly (plain JSON, no
model reconstruction) instead of calling tabgpgo.inference.load_run, which
would otherwise try to rebuild an MLPAutoencoder that was never used here.

Resumable the same way every other tabgpgo sweep script is: evolve_tabpfn()
skips any combo that already has a completed run dir (elite.json present).

Usage:
  python main/main_tabgpgo_tabpfn_encoder.py prepare   # phase 1-2 only (build/cache the TabPFN pool)
  python main/main_tabgpgo_tabpfn_encoder.py evolve    # phase 4-5 (reusing the cache, resumable)
  python main/main_tabgpgo_tabpfn_encoder.py           # everything

Writes to main/log/tabgpgo_tabpfn_results.csv / tabgpgo_tabpfn_runs/ (same
CSV column schema as main_tabgpgo_funcset.py) and
main/log/tabgpgo_tabpfn_manifest.csv (run_id, algo, function_set,
constant_set, omt_frac, omt_pop_size, omt_gens).
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

WRAPPER = "mix"    # SLIM*MIX for every combo -- the OMT combo's omt_frac<1.0
                   # means most of its mutations still use plain mix wrapping
VARIANT = (WRAPPER, "mul")
MS_SPEC = "oms"
PATIENCE = 5
N_GENS = 200       # full budget -- the TabPFN pool is built ONCE and cached,
                   # nothing here is per-generation-expensive the way OMT is

TABPFN_N_ESTIMATORS = 1   # see tabgpgo/tabpfn_encoder.py: ~22x cheaper than the
                          # TabPFNRegressor default of 8, negligible accuracy
                          # cost for our purposes (an encoder, not a predictor)

# The 4 distinct (function_set, constant_set) combos that won on at least one
# real dataset in main_tabgpgo_funcset.py's sweep, plus a small-OMT variant of
# the overall-best one -- see module docstring.
COMBOS = [
    {"function_set": "full_extended", "constant_set": "small_ints", "omt_frac": 0.0, "omt_pop_size": 0, "omt_gens": 0},
    {"function_set": "scientific_extended", "constant_set": "small_ints", "omt_frac": 0.0, "omt_pop_size": 0, "omt_gens": 0},
    {"function_set": "full_extended_piecewise", "constant_set": "none", "omt_frac": 0.0, "omt_pop_size": 0, "omt_gens": 0},
    {"function_set": "ml_smooth", "constant_set": "small_ints", "omt_frac": 0.0, "omt_pop_size": 0, "omt_gens": 0},
    {"function_set": "full_extended", "constant_set": "small_ints", "omt_frac": 0.2, "omt_pop_size": 20, "omt_gens": 10},
]

TABPFN_ARTIFACTS_DIR = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_tabpfn_artifacts")
TABPFN_LOG_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_tabpfn_results.csv")
TABPFN_RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_tabpfn_runs")
TABPFN_MANIFEST_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_tabpfn_manifest.csv")


class _EncoderStub:
    """Stands in for the real MLPAutoencoder at save_run's `model` parameter
    -- save_run only ever calls `model.encoder.state_dict()`, and an empty
    nn.Identity() satisfies that without special-casing the shared
    persistence function for the one experiment that has no real encoder to
    persist (TabPFN's weights are a fixed pretrained asset, not something
    this experiment trains -- see tabgpgo/tabpfn_encoder.py)."""
    encoder = torch.nn.Identity()


def build_config(**extra):
    return dataclasses.replace(
        TabGPGOConfig(), log_path=TABPFN_LOG_PATH, variants=(VARIANT,),
        run_dir_base=TABPFN_RUN_DIR_BASE, stagnation_patience=PATIENCE, n_gens=N_GENS,
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


def prepare_tabpfn(cfg, verbose=True):
    """Phase 1-2 (TabPFN-encoder variant): builds/caches the TabPFN pool,
    loads the real validation sets (unchanged), and embeds them via the
    frozen reference model -- never refit on real data (see
    tabgpgo/tabpfn_encoder.py). Returns a ctx dict with exactly the fields
    FreshPoolSLIM/build_adapter/verify_inference already expect, so nothing
    downstream needs to know the encoder was TabPFN instead of the usual
    MLPAutoencoder.
    """
    device = cfg.get_device()
    T_train, y_target, reference_model, embed_dim = _cached_pool(cfg, verbose)
    T_train, y_target = T_train.to(device), y_target.to(device)

    val_sets = load_validation_sets(cfg)
    T_val = {name: encode_val_tabpfn(reference_model, d["X"]).to(device)
             for name, d in val_sets.items()}
    val_targets = {name: d["y"].to(device) for name, d in val_sets.items()}
    val_y_stats = {name: d["meta"]["y_stats"] for name, d in val_sets.items()}
    TERMINALS = make_terminals(embed_dim)

    if verbose:
        print(f"TabPFN embed_dim={embed_dim}")

    return {"T_train": T_train, "y_target": y_target, "T_val": T_val,
            "val_targets": val_targets, "val_y_stats": val_y_stats,
            "TERMINALS": TERMINALS, "reference_model": reference_model}


def _base_algo_label(combo):
    ms_label = "oms" if MS_SPEC == "oms" else f"ms{MS_SPEC:g}"
    return f"{ALGO_NAMES[VARIANT]}_{ms_label}_pat{PATIENCE}"


def _combo_tag(combo):
    omt_label = f"_omtfrac{combo['omt_frac']:g}_omtpop{combo['omt_pop_size']}_omtgens{combo['omt_gens']}"
    algo = (f"{_base_algo_label(combo)}_fn-{combo['function_set']}_const-{combo['constant_set']}"
            f"_ogens{N_GENS}{omt_label}")
    return algo, algo.replace("*", "x").replace("~", "t")


def find_completed_run_dir(tag):
    if not os.path.isdir(TABPFN_RUN_DIR_BASE):
        return None
    suffix = f"_{tag}_0"
    for name in sorted(os.listdir(TABPFN_RUN_DIR_BASE)):
        if name.endswith(suffix) and os.path.isfile(os.path.join(TABPFN_RUN_DIR_BASE, name, "elite.json")):
            return os.path.join(TABPFN_RUN_DIR_BASE, name)
    return None


def _elite_fitness_only(run_dir):
    """Reads elite.json's fitness directly, without tabgpgo.inference.
    load_run's full model/registry reconstruction (which assumes an
    MLPAutoencoder was saved -- see _EncoderStub's docstring). Only used for
    the skip-path's informational print."""
    with open(os.path.join(run_dir, "elite.json")) as fh:
        return json.load(fh)["fitness"]


def _run_one(cfg, combo, ctx, unique_run_id, verbose):
    p_c, constants = CONSTANT_SETS[combo["constant_set"]]
    cfg = dataclasses.replace(cfg, p_c=p_c,
                              omt_frac=combo["omt_frac"], omt_pop_size=combo["omt_pop_size"],
                              omt_gens=combo["omt_gens"])
    with function_constant_set(FUNCTION_SETS[combo["function_set"]], constants):
        optimizer = FreshPoolSLIM(cfg, VARIANT, ctx["T_train"], ctx["T_val"],
                                  ctx["y_target"], ctx["val_targets"], ctx["val_y_stats"],
                                  ctx["TERMINALS"], seed=0, ms_spec=MS_SPEC, use_ls=False)
        optimizer.algo = f"{optimizer.algo}_fn-{combo['function_set']}_const-{combo['constant_set']}_tabpfn"
        elite = optimizer.solve(
            run_info=[optimizer.algo, unique_run_id, "synthetic_prior"],
            log_path=cfg.log_path, verbose=verbose)

        pi, registry, pool = build_adapter(elite, ctx["T_train"], ctx["TERMINALS"])
        verify_inference(pi, pool, ctx["T_train"], registry, ctx["TERMINALS"])
        expression = reconstruct_expression(pi, registry)
        tag = optimizer.algo.replace("*", "x").replace("~", "t")
        run_dir = os.path.join(cfg.run_dir_base, f"{unique_run_id}_{tag}_0")
        save_run(run_dir, cfg, _EncoderStub(), registry, pi, WRAPPER, "mul",
                 optimizer.algo, seed=0, expression=expression)
    if verbose:
        print(f"[{optimizer.algo}] done: train_rmse={elite.fitness:.4f} "
              f"size={elite.size} -> {run_dir}")
    return optimizer.algo, elite


def evolve_tabpfn(verbose=1):
    cfg = build_config()
    ctx = prepare_tabpfn(cfg, verbose=bool(verbose))

    os.makedirs(os.path.dirname(TABPFN_LOG_PATH), exist_ok=True)
    unique_run_id = uuid.uuid1()
    manifest_rows = []
    elites = {}
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
            "run_id": run_id, "algo": algo,
            "function_set": combo["function_set"], "constant_set": combo["constant_set"],
            "omt_frac": combo["omt_frac"], "omt_pop_size": combo["omt_pop_size"], "omt_gens": combo["omt_gens"],
        })
    with open(TABPFN_MANIFEST_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nwrote manifest -> {TABPFN_MANIFEST_CSV}")
    return elites, unique_run_id


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "prepare":
        prepare_tabpfn(build_config())
    elif stage == "evolve":
        evolve_tabpfn()
    elif stage == "all":
        evolve_tabpfn()
    else:
        sys.exit(f"unknown stage '{stage}' (use: prepare | evolve | all)")
