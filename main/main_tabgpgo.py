"""
TabGPGO experiment: TabPFN-style synthetic prior + tensor-pool SLIM-GSGP.

Pipeline (all knobs in tabgpgo/config.py):
  1. generate + standardize synthetic datasets, pool into one training tensor;
     load the real validation datasets (merged files, unsplit, reduced/padded
     to 100 features)
  2. train the MLP autoencoder on pooled X, freeze the encoder, produce
     static 512-dim latent tokens for training and validation data
  3. generate the base-tree pool and precompute its raw semantics on all tokens
  4. evolve every (SLIM variant, ms_hi) combination over the pool, running
     cfg.max_workers of them concurrently (validation RMSE/R^2 tracked per gen)
  5. verify inference equivalence, save artifacts per run

The expensive phase 1-3 products (synthetic data, encoder weights, latent
tokens, tree registry) are cached in cfg.artifacts_dir and reused on the next
invocation, so evolution runs never recompute the latent space. Stages can
also be run separately:

  python main/main_tabgpgo.py prepare   # phases 1-3 only (fill the cache)
  python main/main_tabgpgo.py eval-ae   # a-priori AE reconstruction check (no evolution)
  python main/main_tabgpgo.py evolve    # phases 4-5 (reusing the cache)
  python main/main_tabgpgo.py           # everything

Phase 4 sweeps cfg.variants x cfg.ms_hi_values x range(cfg.n_runs); each
combination gets its own algo label ("SLIM+ABS_ms10") and run directory.

CSV columns of main/log/tabgpgo_results.csv:
  [algo, run_id, dataset, seed, generation, elite_train_rmse, time_s,
   population_nodes,
   val_<d>_rmse_scaled, val_<d>_rmse_raw, val_<d>_r2 (per d in
   cfg.val_datasets order -- "scaled" is in the z-scored target space the
   model was evolved in; "raw" inverts the elite's prediction and the target
   back with that dataset's own y_stats, comparable to other methods; R^2
   is affine-invariant, so it's identical whether computed scaled or raw --
   the one metric directly comparable across differently-scaled datasets),
   elite_size, elite_nodes, elite_train_r2]
"""
import os
import pickle
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import torch

from tabgpgo.autoencoder import (MLPAutoencoder, encode, reconstruction_stats,
                                 train_autoencoder)
from tabgpgo.config import TabGPGOConfig
from tabgpgo.evolution import TensorSLIM
from tabgpgo.inference import (reconstruct_expression, save_run,
                               verify_inference)
from tabgpgo.preprocessing import build_synthetic_pool, load_validation_sets
from tabgpgo.tree_pool import build_pool, evaluate_pool, make_terminals


def _cached(cfg, fname, compute, verbose, desc):
    """Load `fname` from the artifact cache, or compute and cache it."""
    path = os.path.join(cfg.artifacts_dir, fname)
    if cfg.reuse_artifacts and os.path.exists(path):
        if verbose:
            print(f"[cached] {desc} <- {fname}")
        with open(path, "rb") as fh:
            return pickle.load(fh) if fname.endswith(".pkl") else torch.load(
                fh, map_location=cfg.get_device(), weights_only=False)
    t0 = time.time()
    result = compute()
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(result, fh) if fname.endswith(".pkl") else torch.save(result, fh)
    if verbose:
        print(f"[computed] {desc} ({time.time() - t0:.1f}s) -> {fname}")
    return result


def prepare(cfg, verbose=True):
    """Phases 1-3 with per-stage artifact caching.

    Pool semantics are always recomputed from registry + tokens (too large to
    cache); everything else is loaded from cfg.artifacts_dir when present.
    """
    device = cfg.get_device()
    if verbose:
        print(f"device: {device} | artifacts: {cfg.artifacts_dir}")

    # -- stage 1: data ---------------------------------------------------------
    X_train, y_target = _cached(
        cfg, "synthetic_pool.pt",
        lambda: build_synthetic_pool(cfg), verbose,
        f"synthetic pool ({cfg.n_synth_datasets} x {cfg.n_rows} rows)")
    X_train, y_target = X_train.to(device), y_target.to(device)
    val_sets = _cached(cfg, "val_sets.pkl",
                       lambda: load_validation_sets(cfg), verbose,
                       "validation datasets")

    # -- stage 2: autoencoder + latent tokens -----------------------------------
    def _train_ae():
        model = train_autoencoder(X_train, cfg, verbose=verbose)
        return model.state_dict()

    ae_state = _cached(cfg, "autoencoder.pt", _train_ae, verbose, "autoencoder")
    ae = MLPAutoencoder(cfg.max_features, cfg.ae_hidden, cfg.latent_dim).to(device)
    ae.load_state_dict(ae_state)
    ae.eval().requires_grad_(False)

    T_train = _cached(cfg, "T_train.pt", lambda: encode(ae, X_train),
                      verbose, "training latent tokens").to(device)
    T_val = _cached(
        cfg, "T_val.pt",
        lambda: {name: encode(ae, d["X"].to(device)) for name, d in val_sets.items()},
        verbose, "validation latent tokens")
    T_val = {name: T.to(device) for name, T in T_val.items()}
    val_targets = {name: d["y"].to(device) for name, d in val_sets.items()}
    val_y_stats = {name: d["meta"]["y_stats"] for name, d in val_sets.items()}

    # -- stage 3: tree pool -------------------------------------------------------
    TERMINALS = make_terminals(cfg.latent_dim)

    def _build_registry():
        random.seed(cfg.data_seed)
        return build_pool(cfg, TERMINALS)

    registry = _cached(cfg, "registry.pkl", _build_registry, verbose,
                       f"tree registry ({cfg.pool_size} trees)")

    t0 = time.time()
    dtype = cfg.get_pool_dtype()
    pool_train = evaluate_pool(registry, T_train, TERMINALS, dtype)
    val_pools = {name: evaluate_pool(registry, T, TERMINALS, dtype)
                 for name, T in T_val.items()}
    if verbose:
        gb = pool_train.numel() * pool_train.element_size() / 1e9
        print(f"pool semantics: {tuple(pool_train.shape)} "
              f"({gb:.2f} GB, {time.time() - t0:.1f}s, recomputed each start)")

    return {"ae": ae, "X_train": X_train, "T_train": T_train, "y_target": y_target,
            "registry": registry, "pool_train": pool_train,
            "val_pools": val_pools, "val_targets": val_targets,
            "val_y_stats": val_y_stats, "val_sets": val_sets, "TERMINALS": TERMINALS}


def evaluate_autoencoder(cfg, ctx, verbose=True):
    """A-priori sanity check of the trained encoder/decoder, run once before
    evolution (not on every prepare()/evolve() call): full-model (encoder+
    decoder) reconstruction MSE and R^2 on the synthetic training data and on
    every real validation set, so a synthetic-trained AE that fails to
    generalize to real data is caught before it's used to build latent
    tokens. R^2 (fraction of variance explained) is reported alongside raw
    MSE because datasets preprocessed via zero-padding (low feature count,
    100/k-scaled) and via PCA/RF reduction (>100 features) sit on very
    different raw scales and aren't comparable on MSE alone.
    """
    ae = ctx["ae"]
    device = ae.encoder[0].weight.device
    out = {}
    mse, r2 = reconstruction_stats(ae, ctx["X_train"][:20000])
    out["synthetic(train)"] = {"mse": mse, "r2": r2}
    for name, d in ctx["val_sets"].items():
        mse, r2 = reconstruction_stats(ae, d["X"].to(device))
        out[name] = {"mse": mse, "r2": r2}
    if verbose:
        for name, stats in out.items():
            print(f"AE reconstruction  {name}: "
                  f"MSE={stats['mse']:.5f}  R2={stats['r2']:.4f}")
    return out


def _run_one(cfg, variant, ms_hi, seed, ctx, unique_run_id, verbose):
    """One (variant, ms_hi, seed) run: evolve, verify, persist. Runs safely
    from a worker thread (TensorSLIM uses its own RNG instance, and CSV
    writes are lock-protected in evolution.py)."""
    wrapper, operator = variant
    optimizer = TensorSLIM(cfg, variant, ctx["registry"], ctx["pool_train"],
                           ctx["y_target"], ctx["val_pools"], ctx["val_targets"],
                           ctx["val_y_stats"], seed, ms_hi=ms_hi)
    elite = optimizer.solve(
        run_info=[optimizer.algo, unique_run_id, "synthetic_prior"],
        log_path=cfg.log_path, verbose=verbose)

    verify_inference(elite, ctx["pool_train"], ctx["T_train"],
                     ctx["registry"], ctx["TERMINALS"])
    expression = reconstruct_expression(elite, ctx["registry"])
    tag = optimizer.algo.replace("*", "x").replace("~", "t")
    run_dir = os.path.join(cfg.run_dir_base, f"{unique_run_id}_{tag}_{seed}")
    save_run(run_dir, cfg, ctx["ae"], ctx["registry"], elite,
             wrapper, operator, optimizer.algo, seed, expression)
    if verbose:
        print(f"[{optimizer.algo} seed {seed}] done: train_rmse="
              f"{elite.fitness:.4f} size={elite.size} -> {run_dir}")
    return (optimizer.algo, seed), elite


def evolve(cfg, ctx, verbose=1):
    """Phases 4-5: (variant, ms_hi, seed) sweep over prepared artifacts,
    run concurrently across cfg.max_workers threads. Safe because all shared
    state (pool_train, registry, val_pools, ...) is read-only after
    prepare(), and each TensorSLIM/its logging use their own RNG/a shared
    lock respectively (see tabgpgo/evolution.py)."""
    os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
    unique_run_id = uuid.uuid1()
    jobs = [(variant, ms_hi, seed)
           for variant in cfg.variants
           for ms_hi in cfg.ms_hi_values
           for seed in range(cfg.n_runs)]
    elites = {}
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = [pool.submit(_run_one, cfg, variant, ms_hi, seed, ctx,
                               unique_run_id, verbose)
                  for variant, ms_hi, seed in jobs]
        for future in futures:
            key, elite = future.result()
            elites[key] = elite
    return elites, unique_run_id


def run_experiment(cfg, verbose=1):
    ctx = prepare(cfg, verbose=bool(verbose))
    elites, unique_run_id = evolve(cfg, ctx, verbose=verbose)
    return ctx, elites, unique_run_id


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    config = TabGPGOConfig()
    if stage == "prepare":
        prepare(config)
    elif stage == "eval-ae":
        evaluate_autoencoder(config, prepare(config))
    elif stage == "evolve":
        evolve(config, prepare(config))
    elif stage == "all":
        run_experiment(config)
    else:
        sys.exit(f"unknown stage '{stage}' (use: prepare | eval-ae | evolve | all)")
