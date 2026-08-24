"""
TabGPGO experiment: re-runs the pop_size sweep's winning fixed combo (see
main_tabgpgo_tabpfn_norotate_popsweep.py) -- pop_size=200, ms=0.01,
no rotation -- but with the TabPFN embedding model explicitly
pinned to V3, instead of the ambiguous "auto" default every other
tabgpgo_tabpfn_* script's build_tabpfn_pool call has been using (which
resolves to whichever checkpoint the installed tabpfn package currently
treats as default -- confirmed NOT v1; see this session's conversation on
the synthetic-prior/embedding-model mismatch: our OWN synthetic training
data is drawn from the vendored TabPFN v1 prior, while the embedding model
and every TabPFN baseline comparison use v2/v3 checkpoints trained on a much
richer prior).

This is a single combo, not a sweep -- it exists to isolate one question:
does pinning the embedding to the LATEST TabPFN checkpoint (V3) change
evolution/zero-shot-transfer results for the exact same (pop_size, ms)
setting already evaluated with the ambiguous default elsewhere.

n_gens=50000 (10x the earlier 5000-gen version this session validated) --
feasible without the earlier pop_size=2000/n_gens=20000 attempt's RAM-
ceiling failure mode because that was driven by pop_size (aggregate
tensors scale as pop_size * n_train; at this script's pop_size=200 that
quantity is a small, CONSTANT ~400MB regardless of generation count, not
the risk here -- see main_tabgpgo_tabpfn_norotate_20k.py's own docstring).
The real risk at 10x the generations is TIME, not memory: FreshPoolSLIM's
periodic elite drift-resync (`_resync_elite`, a full from-scratch refold
whose cost grows with the elite's own block count) used to run every
single generation; TabGPGOConfig.resync_elite_every (default 25) now
throttles that to every 25 generations instead, validated (see this
session's smoke test) to introduce only ~1e-4-scale fitness drift at a
~200-block elite with no NaN/Inf, while cutting that cost ~1.8x already at
that modest scale -- expected to matter far more at the thousands of
blocks a real run reaches. FreshPoolSLIM.solve() also gained
checkpoint/resume (checkpoint_every/checkpoint_path/resume_population/
resume_start_gen) and SIGINT/SIGTERM-triggered checkpointing, so this
script's own CHECKPOINT_EVERY=2000 splits the run into segments each well
within the already-validated <=5000-gen stable range -- re-running this
script after an interrupt (manual or a crash) resumes from the last
checkpoint in main/log/tabgpgo_v3_pop200ms001_checkpoints/ instead of
restarting from generation 0.

Its own artifacts_dir (main/log/tabgpgo_tabpfn_v3_artifacts/) is
DELIBERATELY separate from every other tabgpgo_tabpfn_* script's shared
main/log/tabgpgo_tabpfn_artifacts/tabpfn_pool_bundle.pkl -- that shared
cache was built with the ambiguous "auto" version and must not be
overwritten or read by this V3-pinned run.

Usage:
  python main/main_tabgpgo_tabpfn_v3_pop200_ms001.py prepare   # phase 1-2 only (build/cache the V3 TabPFN pool)
  python main/main_tabgpgo_tabpfn_v3_pop200_ms001.py evolve    # phase 4-5 (reusing the cache, resumable)
  python main/main_tabgpgo_tabpfn_v3_pop200_ms001.py           # everything
Re-running any of these after an interrupt resumes from the last
checkpoint automatically -- no separate "resume" stage needed.

Writes to main/log/tabgpgo_v3_pop200ms001_results.csv /
tabgpgo_v3_pop200ms001_runs/ and
main/log/tabgpgo_v3_pop200ms001_manifest.csv. CSV schema: same 29-column
schema as every other non-rotating tabgpgo sweep script.
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
from tabpfn.constants import ModelVersion

from main_tabgpgo_funcset import CONSTANT_SETS, FUNCTION_SETS, function_constant_set
from tabgpgo.config import ALGO_NAMES, REPO_ROOT, TabGPGOConfig
from tabgpgo.evolution_freshpool import FreshPoolSLIM, RunInterrupted, build_adapter, to_registry
from tabgpgo.inference import reconstruct_expression, save_run, verify_inference
from tabgpgo.preprocessing import load_validation_sets
from tabgpgo.tabpfn_encoder import build_tabpfn_pool, encode_val_tabpfn
from tabgpgo.tree_pool import make_terminals

HPT_OVERRIDES = dict(pop_size=200, tournament_size=2, p_inflate=0.5, n_elites=5)

WRAPPER = "mix"
VARIANT = (WRAPPER, "mul")   # SLIM*MIX
PATIENCE = 5
N_GENS = 50000

TABPFN_N_ESTIMATORS = 1   # see tabgpgo/tabpfn_encoder.py
TABPFN_MODEL_VERSION = ModelVersion.V3

FUNCTION_SET_NAME = "full_extended"
CONSTANT_SET_NAME = "small_ints"

MS_VALUES = [0.01]   # single fixed combo, not a sweep -- see module docstring
COMBOS = [{"ms": ms} for ms in MS_VALUES]

# Deliberately SEPARATE from the shared tabgpgo_tabpfn_artifacts/ cache every
# other tabgpgo_tabpfn_* script uses -- see module docstring.
TABPFN_V3_ARTIFACTS_DIR = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_tabpfn_v3_artifacts")
V3_LOG_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_v3_pop200ms001_results.csv")
V3_RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_v3_pop200ms001_runs")
V3_MANIFEST_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_v3_pop200ms001_manifest.csv")

# Periodic checkpoint/resume (FreshPoolSLIM.solve()'s checkpoint_every/
# checkpoint_path/resume_population/resume_start_gen params) -- lets a run
# be split across process restarts, e.g. for a much-longer N_GENS than the
# 5000 this script has been validated at. Demonstrates usage; every other
# main_tabgpgo_tabpfn_*.py script is unaffected (they never pass these).
CHECKPOINT_EVERY = 2000
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_v3_pop200ms001_checkpoints")

# build_adapter's post-run verification pool is (registry_size, n_train) --
# at n_gens=50000 the elite can bloat far past anything a <=5000-gen run
# ever reached (observed: 14000+ nodes), so stacking that pool at the FULL
# training-set row count can need tens of GB, blowing past GPU memory even
# though the incremental evolution itself never materializes anything like
# it. See _run_one: caps the verification pool to this many bytes by
# subsampling T_train rows for that one post-run check only.
POOL_VERIFY_BUDGET_BYTES = 500_000_000


def _checkpoint_path(tag):
    return os.path.join(CHECKPOINT_DIR, f"{tag}.ckpt")


class _EncoderStub:
    """Stands in for the real MLPAutoencoder at save_run's `model` parameter
    -- see main_tabgpgo_tabpfn_encoder.py's identical class for why."""
    encoder = torch.nn.Identity()


def build_config(**extra):
    return dataclasses.replace(
        TabGPGOConfig(), log_path=V3_LOG_PATH, variants=(VARIANT,),
        run_dir_base=V3_RUN_DIR_BASE, stagnation_patience=PATIENCE, n_gens=N_GENS,
        artifacts_dir=TABPFN_V3_ARTIFACTS_DIR,
        **HPT_OVERRIDES, **extra)


def _cached_v3_pool(cfg, verbose):
    path = os.path.join(cfg.artifacts_dir, "tabpfn_v3_pool_bundle.pkl")
    if cfg.reuse_artifacts and os.path.exists(path):
        if verbose:
            print(f"[cached] TabPFN V3 pool <- {os.path.basename(path)}")
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    result = build_tabpfn_pool(cfg, n_estimators=TABPFN_N_ESTIMATORS, verbose=verbose,
                               model_version=TABPFN_MODEL_VERSION)
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(result, fh)
    if verbose:
        print(f"[computed] TabPFN V3 pool ({time.time() - t0:.1f}s) -> {os.path.basename(path)}")
    return result


def prepare_v3(cfg, verbose=True):
    """Phase 1-2: builds/caches the V3-pinned TabPFN pool, no rotation (the
    whole pool is one fixed training set for the entire run, matching every
    other norotate tabgpgo_tabpfn_* script)."""
    device = cfg.get_device()
    T_train, y_target, reference_model, embed_dim = _cached_v3_pool(cfg, verbose)
    T_train, y_target = T_train.to(device), y_target.to(device)

    val_sets = load_validation_sets(cfg)
    T_val = {name: encode_val_tabpfn(reference_model, d["X"]).to(device)
             for name, d in val_sets.items()}
    val_targets = {name: d["y"].to(device) for name, d in val_sets.items()}
    val_y_stats = {name: d["meta"]["y_stats"] for name, d in val_sets.items()}
    TERMINALS = make_terminals(embed_dim)

    if verbose:
        print(f"TabPFN V3 embed_dim={embed_dim}, pool={T_train.shape[0]} rows")

    return {"T_train": T_train, "T_val": T_val, "y_target": y_target,
            "val_targets": val_targets, "val_y_stats": val_y_stats, "TERMINALS": TERMINALS}


def _base_algo_label(combo):
    ms_label = "oms" if combo["ms"] == "oms" else f"ms{combo['ms']:g}"
    return f"{ALGO_NAMES[VARIANT]}_{ms_label}_pat{PATIENCE}"


def _combo_tag(combo):
    algo = (f"{_base_algo_label(combo)}_fn-{FUNCTION_SET_NAME}_const-{CONSTANT_SET_NAME}"
            f"_ogens{N_GENS}_pop{HPT_OVERRIDES['pop_size']}_tabpfnv3_norotate")
    return algo, algo.replace("*", "x").replace("~", "t")


def find_completed_run_dir(tag):
    if not os.path.isdir(V3_RUN_DIR_BASE):
        return None
    suffix = f"_{tag}_0"
    for name in sorted(os.listdir(V3_RUN_DIR_BASE)):
        if name.endswith(suffix) and os.path.isfile(os.path.join(V3_RUN_DIR_BASE, name, "elite.json")):
            return os.path.join(V3_RUN_DIR_BASE, name)
    return None


def _elite_fitness_only(run_dir):
    with open(os.path.join(run_dir, "elite.json")) as fh:
        return json.load(fh)["fitness"]


def _run_one(cfg, combo, ctx, unique_run_id, verbose):
    algo, tag = _combo_tag(combo)
    ckpt_path = _checkpoint_path(tag)

    if os.path.exists(ckpt_path):
        optimizer, population, completed_gen, run_info = FreshPoolSLIM.resume_from_checkpoint(
            ckpt_path, cfg, VARIANT, ctx["T_train"], ctx["T_val"], ctx["y_target"],
            ctx["val_targets"], ctx["val_y_stats"], ctx["TERMINALS"], seed=0,
            ms_spec=combo["ms"], use_ls=False)
        optimizer.algo = algo
        if verbose:
            print(f"[{optimizer.algo}] resuming from checkpoint at generation {completed_gen} "
                 f"-> {ckpt_path}")
        try:
            elite = optimizer.solve(
                run_info=run_info, log_path=V3_LOG_PATH, verbose=verbose,
                checkpoint_every=CHECKPOINT_EVERY, checkpoint_path=ckpt_path,
                resume_population=population, resume_start_gen=completed_gen + 1)
        except RunInterrupted as exc:
            print(f"[{optimizer.algo}] interrupted, checkpointed at generation "
                 f"{exc.completed_gen} -> {ckpt_path}. Re-run this script to resume.")
            raise SystemExit(0)
        unique_run_id = run_info[1]
    else:
        optimizer = FreshPoolSLIM(cfg, VARIANT, ctx["T_train"], ctx["T_val"],
                                  ctx["y_target"], ctx["val_targets"], ctx["val_y_stats"],
                                  ctx["TERMINALS"], seed=0, ms_spec=combo["ms"], use_ls=False)
        optimizer.algo = algo
        run_info = [optimizer.algo, unique_run_id, "synthetic_prior"]
        try:
            elite = optimizer.solve(
                run_info=run_info, log_path=V3_LOG_PATH, verbose=verbose,
                checkpoint_every=CHECKPOINT_EVERY, checkpoint_path=ckpt_path)
        except RunInterrupted as exc:
            print(f"[{optimizer.algo}] interrupted, checkpointed at generation "
                 f"{exc.completed_gen} -> {ckpt_path}. Re-run this script to resume.")
            raise SystemExit(0)

    # Verification pool is a one-off, post-run sanity check (save_run below
    # never touches `pool`) -- bound its memory to POOL_VERIFY_BUDGET_BYTES
    # by subsampling rows instead of always using the full training set, so
    # an unrestrained-bloat elite (see module docstring) can't OOM here
    # regardless of how large it grew during evolution.
    _, registry_probe = to_registry(elite)
    full_rows = ctx["T_train"].shape[0]
    max_rows = min(full_rows,
                   max(1000, POOL_VERIFY_BUDGET_BYTES // (max(len(registry_probe), 1) * 4)))
    T_verify = ctx["T_train"]
    if max_rows < full_rows:
        perm = torch.randperm(full_rows, device=T_verify.device)[:max_rows]
        T_verify = T_verify[perm]
        if verbose:
            print(f"[{optimizer.algo}] elite registry has {len(registry_probe)} entries -- "
                  f"verifying pool-path/token-path semantics on a {max_rows}/{full_rows}-row "
                  f"subsample to bound verification memory")
    pi, registry, pool = build_adapter(elite, T_verify, ctx["TERMINALS"])
    verify_inference(pi, pool, T_verify, registry, ctx["TERMINALS"])
    expression = reconstruct_expression(pi, registry)
    run_dir = os.path.join(cfg.run_dir_base, f"{unique_run_id}_{tag}_0")
    save_run(run_dir, cfg, _EncoderStub(), registry, pi, WRAPPER, "mul",
             optimizer.algo, seed=0, expression=expression)
    if verbose:
        print(f"[{optimizer.algo}] done: train_rmse={elite.fitness:.4f} "
              f"size={elite.size} -> {run_dir}")
    return optimizer.algo, elite, unique_run_id


def evolve_v3(verbose=1):
    cfg = build_config()
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]
    cfg = dataclasses.replace(cfg, p_c=p_c)

    os.makedirs(os.path.dirname(V3_LOG_PATH), exist_ok=True)
    unique_run_id = uuid.uuid1()
    manifest_rows = []
    elites = {}
    with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
        ctx = prepare_v3(cfg, verbose=bool(verbose))
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
                algo, elite, run_id = _run_one(cfg, combo, ctx, unique_run_id, verbose)
                elites[algo] = elite
            manifest_rows.append({
                "run_id": run_id, "algo": algo, "ms": combo["ms"],
                "function_set": FUNCTION_SET_NAME, "constant_set": CONSTANT_SET_NAME,
            })
    with open(V3_MANIFEST_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nwrote manifest -> {V3_MANIFEST_CSV}")
    return elites, unique_run_id


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "prepare":
        cfg = build_config()
        p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]
        cfg = dataclasses.replace(cfg, p_c=p_c)
        with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
            prepare_v3(cfg)
    elif stage == "evolve":
        evolve_v3()
    elif stage == "all":
        evolve_v3()
    else:
        sys.exit(f"unknown stage '{stage}' (use: prepare | evolve | all)")
