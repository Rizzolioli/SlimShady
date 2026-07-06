"""
TabGPGO experiment: TabPFN-style synthetic prior + tensor-pool SLIM-GSGP.

Pipeline (all knobs in tabgpgo/config.py):
  1. generate + standardize synthetic datasets, pool into one training tensor;
     load the real validation datasets (merged files, unsplit, reduced/padded
     to 100 features)
  2. train the MLP autoencoder on pooled X, freeze the encoder, produce
     static 512-dim latent tokens for training and validation data
  3. generate the base-tree pool and precompute its raw semantics on all tokens
  4. evolve every SLIM variant over the pool (validation RMSE tracked per gen)
  5. verify inference equivalence, save artifacts per run

The expensive phase 1-3 products (synthetic data, encoder weights, latent
tokens, tree registry) are cached in cfg.artifacts_dir and reused on the next
invocation, so evolution runs never recompute the latent space. Stages can
also be run separately:

  python main/main_tabgpgo.py prepare   # phases 1-3 only (fill the cache)
  python main/main_tabgpgo.py eval-ae   # a-priori AE reconstruction check (no evolution)
  python main/main_tabgpgo.py evolve    # phases 4-5 (reusing the cache)
  python main/main_tabgpgo.py           # everything

CSV columns of main/log/tabgpgo_results.csv:
  [algo, run_id, dataset, seed, generation, elite_train_rmse, time_s,
   population_nodes, val_<d>_rmse (for each d in cfg.val_datasets order),
   elite_size, elite_nodes]
"""
import os
import pickle
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import torch

from tabgpgo.autoencoder import (MLPAutoencoder, encode, reconstruction_mse,
                                 train_autoencoder)
from tabgpgo.config import ALGO_NAMES, TabGPGOConfig, wrapper_name
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
            "val_sets": val_sets, "TERMINALS": TERMINALS}


def evaluate_autoencoder(cfg, ctx, verbose=True):
    """A-priori sanity check of the trained encoder/decoder, run once before
    evolution (not on every prepare()/evolve() call): full-model (encoder+
    decoder) reconstruction MSE on the synthetic training data and on every
    real validation set, so a synthetic-trained AE that fails to generalize
    to real data is caught before it's used to build latent tokens.
    """
    ae = ctx["ae"]
    out = {"synthetic(train)": reconstruction_mse(ae, ctx["X_train"][:20000])}
    for name, d in ctx["val_sets"].items():
        out[name] = reconstruction_mse(ae, d["X"].to(ae.encoder[0].weight.device))
    if verbose:
        for name, mse in out.items():
            print(f"AE reconstruction MSE  {name}: {mse:.5f}")
    return out


def evolve(cfg, ctx, verbose=1):
    """Phases 4-5: variant sweep over prepared artifacts."""
    os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
    unique_run_id = uuid.uuid1()
    elites = {}
    for variant in cfg.variants:
        sig, two_trees, operator = variant
        algo = ALGO_NAMES[variant]
        wrapper = wrapper_name(sig, two_trees)
        for seed in range(cfg.n_runs):
            optimizer = TensorSLIM(cfg, variant, ctx["registry"],
                                   ctx["pool_train"], ctx["y_target"],
                                   ctx["val_pools"], ctx["val_targets"], seed)
            elite = optimizer.solve(
                run_info=[algo, unique_run_id, "synthetic_prior"],
                log_path=cfg.log_path, verbose=verbose)

            verify_inference(elite, wrapper, operator, ctx["pool_train"],
                             ctx["T_train"], ctx["registry"], ctx["TERMINALS"])
            expression = reconstruct_expression(elite, ctx["registry"],
                                                wrapper, operator)
            run_dir = os.path.join(cfg.run_dir_base,
                                   f"{unique_run_id}_{algo.replace('*', 'x')}_{seed}")
            save_run(run_dir, cfg, ctx["ae"], ctx["registry"], elite,
                     wrapper, operator, algo, seed, expression)
            elites[(algo, seed)] = elite
            if verbose:
                print(f"[{algo} seed {seed}] done: train_rmse="
                      f"{elite.fitness:.4f} size={elite.size} -> {run_dir}")
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
