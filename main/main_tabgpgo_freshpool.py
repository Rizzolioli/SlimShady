"""
TabGPGO experiment (fresh-pool variant): same pipeline as main_tabgpgo.py,
except phase 3 (the static upfront tree pool) is skipped entirely -- see
tabgpgo/evolution_freshpool.py's docstring for why. Every generation, each
run draws random trees from its own per-run reservoir, topped up as needed,
never reset.

Pipeline:
  1-2. identical to main_tabgpgo.py (synthetic data + AE + latent tokens),
       reused directly via `prepare(cfg, build_static_pool=False)`.
  3. (skipped -- ctx["registry"]/["pool_train"]/["val_pools"] are None)
  4. evolve every (ms_hi, stagnation_patience, seed) combination for the
     single SLIM*MIX variant (see VARIANTS_OVERRIDE below) with FreshPoolSLIM,
     running cfg.max_workers of them concurrently.
  5. verify inference equivalence, save artifacts per run -- via
     tabgpgo/evolution_freshpool.py's build_adapter(), reusing
     tabgpgo/inference.py's reconstruct_expression/save_run/verify_inference
     completely unmodified.

Usage:
  python main/main_tabgpgo_freshpool.py prepare   # phases 1-2 only (fill the cache)
  python main/main_tabgpgo_freshpool.py eval-ae   # a-priori AE reconstruction check
  python main/main_tabgpgo_freshpool.py evolve    # phase 4-5 (reusing the cache)
  python main/main_tabgpgo_freshpool.py           # everything

HPT_OVERRIDES below (pop_size/tournament_size/p_inflate/n_elites) is left
unfilled on purpose -- main/hpt_tabgpgo.py sweeps exactly these four knobs for
the 3 MIX variants; fill them in from its winning combo (see
main/log/tabgpgo_hpt_manifest.csv) before running this script. It raises
loudly if you try to run with placeholders still in place.

Writes to main/log/tabgpgo_freshpool_results.csv / tabgpgo_freshpool_runs/ --
separate from main_tabgpgo.py's outputs, same CSV column schema (see that
module's docstring). Shares the same artifacts_dir default, so a
previously-trained AE/latent tokens get reused via _cached() without
retraining.
"""
import dataclasses
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_tabgpgo import evaluate_autoencoder, prepare
from tabgpgo.config import REPO_ROOT, TabGPGOConfig
from tabgpgo.evolution_freshpool import FreshPoolSLIM, build_adapter
from tabgpgo.inference import reconstruct_expression, save_run, verify_inference

# Winning combo from main/hpt_tabgpgo.py's full 108-combo x 3-variant sweep
# (see main/log/tabgpgo_hpt_manifest.csv / tabgpgo_hpt_results.csv), picked by
# aggregating zero-shot val_r2 across all 3 MIX variants and all 6 real
# datasets (average R^2, mean rank, and median rank all agree on this combo
# among the top handful).
HPT_OVERRIDES = dict(pop_size=200, tournament_size=2, p_inflate=0.5, n_elites=5)

# SLIM*MIX only: of the 3 MIX variants, it's the one whose zero-shot val_r2
# stayed near 0 (never strongly negative) across all 6 real datasets in the
# HPT sweep, unlike SLIM+MIX/SLIM~MIX which swing sharply negative on some
# datasets -- see the winning-combo per-variant breakdown in the HPT
# dashboard. This script no longer sweeps variants at all.
VARIANTS_OVERRIDE = (("mix", "mul"),)  # SLIM*MIX

FRESHPOOL_LOG_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_freshpool_results.csv")
FRESHPOOL_RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_freshpool_runs")


def build_config(**extra):
    missing = [k for k, v in HPT_OVERRIDES.items() if v is None]
    if missing:
        raise ValueError(
            f"HPT_OVERRIDES has unfilled placeholders: {missing} -- fill from "
            "main/hpt_tabgpgo.py's sweep results (see "
            "main/log/tabgpgo_hpt_manifest.csv) before running this script.")
    return dataclasses.replace(TabGPGOConfig(), log_path=FRESHPOOL_LOG_PATH,
                               variants=VARIANTS_OVERRIDE,
                               run_dir_base=FRESHPOOL_RUN_DIR_BASE,
                               **HPT_OVERRIDES, **extra)


def _run_one_freshpool(cfg, variant, ms_spec, seed, ctx, unique_run_id, verbose):
    """One (variant, ms_spec, seed) run: evolve, verify, persist. Mirrors
    main_tabgpgo.py's _run_one, using FreshPoolSLIM + build_adapter() instead
    of TensorSLIM + the static registry/pool_train."""
    wrapper, operator = variant
    optimizer = FreshPoolSLIM(cfg, variant, ctx["T_train"], ctx["T_val"],
                              ctx["y_target"], ctx["val_targets"], ctx["val_y_stats"],
                              ctx["TERMINALS"], seed, ms_spec=ms_spec)
    elite = optimizer.solve(
        run_info=[optimizer.algo, unique_run_id, "synthetic_prior"],
        log_path=cfg.log_path, verbose=verbose)

    pi, registry, pool = build_adapter(elite, ctx["T_train"], ctx["TERMINALS"])
    verify_inference(pi, pool, ctx["T_train"], registry, ctx["TERMINALS"])
    expression = reconstruct_expression(pi, registry)
    tag = optimizer.algo.replace("*", "x").replace("~", "t")
    run_dir = os.path.join(cfg.run_dir_base, f"{unique_run_id}_{tag}_{seed}")
    save_run(run_dir, cfg, ctx["ae"], registry, pi, wrapper, operator,
             optimizer.algo, seed, expression)
    if verbose:
        print(f"[{optimizer.algo} seed {seed}] done: train_rmse="
              f"{elite.fitness:.4f} size={elite.size} -> {run_dir}")
    return (optimizer.algo, seed), elite


def evolve_freshpool(cfg, ctx, verbose=1):
    """Phases 4-5: (variant, ms_spec, stagnation_patience, seed) sweep, run
    concurrently across cfg.max_workers threads. Mirrors main_tabgpgo.py's
    evolve(), with cfg.stagnation_patience_values as an extra sweep axis
    (analogous to ms_hi_values) -- each combo gets its own cfg (via
    dataclasses.replace, same pattern as main/hpt_tabgpgo.py's grid) so
    concurrent jobs never share/mutate state, and its own algo label (which
    encodes the patience value, e.g. "SLIM+MIX_oms_pat5" vs
    "SLIM+MIX_oms_patnone" -- see FreshPoolSLIM.__init__) so sweep combos are
    distinguishable in the CSV without a separate manifest. Safe because each
    FreshPoolSLIM has its own reservoir/RNG and only ever touches the global
    random/np.random modules briefly, under a lock, in
    evolution_freshpool._TREE_GEN_LOCK."""
    os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
    unique_run_id = uuid.uuid1()
    jobs = [(dataclasses.replace(cfg, stagnation_patience=patience), variant, ms_spec, seed)
           for variant in cfg.variants
           for ms_spec in cfg.ms_hi_values
           for patience in cfg.stagnation_patience_values
           for seed in range(cfg.n_runs)]
    elites = {}
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = [pool.submit(_run_one_freshpool, job_cfg, variant, ms_spec, seed, ctx,
                               unique_run_id, verbose)
                  for job_cfg, variant, ms_spec, seed in jobs]
        for future in futures:
            key, elite = future.result()
            elites[key] = elite
    return elites, unique_run_id


def run_experiment_freshpool(cfg, verbose=1):
    ctx = prepare(cfg, verbose=bool(verbose), build_static_pool=False)
    elites, unique_run_id = evolve_freshpool(cfg, ctx, verbose=verbose)
    return ctx, elites, unique_run_id


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    config = build_config()
    if stage == "prepare":
        prepare(config, build_static_pool=False)
    elif stage == "eval-ae":
        evaluate_autoencoder(config, prepare(config, build_static_pool=False))
    elif stage == "evolve":
        evolve_freshpool(config, prepare(config, build_static_pool=False))
    elif stage == "all":
        run_experiment_freshpool(config)
    else:
        sys.exit(f"unknown stage '{stage}' (use: prepare | eval-ae | evolve | all)")
