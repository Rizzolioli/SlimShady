"""
TabGPGO experiment (linear-scaling variant): same fresh-pool pipeline as
main_tabgpgo_funcset.py, but instead of sweeping function/constant sets,
function_set and constant_set are held FIXED at the single best combo found
by that sweep -- full_extended / small_ints, the highest average zero-shot
val R^2 across the 6 real datasets (0.0162) -- and the sweep axis is ms_spec
(oms / 0.1 / 0.01) x whether linear scaling (LS) is applied.

Linear scaling (Keijzer 2003 / the classic GP-LS trick): for an individual's
raw output P(x_i) on the training set, the closed-form (a, b) minimizing
sum((t_i - (a + b*P(x_i)))^2) is computed via OLS (evaluators.fitness_
functions.linear_scaling), and RMSE/R^2 are then calculated on the SCALED
output a + b*P instead of raw P. This is guaranteed to never increase
training RMSE vs. the raw individual, and removes the burden of searching for
an overall additive/multiplicative constant from evolution itself -- GP is
then free to search purely for the shape of the target function.

Wired into tabgpgo/evolution_freshpool.py's FreshPoolSLIM as an optional
`use_ls` flag (default off, so every other experiment's behavior is
unchanged): when on, (a, b) are refit from the training set every time
fitness is computed (_evaluate, and _resync_elite for the elite each
generation) and frozen for validation scoring in elite_val_metrics -- (a, b)
come from the training set ONLY, never refit against validation/held-out
targets, so this stays a genuine held-out evaluation and not target leakage.

MS_SPECS sweeps the same ms_hi choices already explored in the freshpool ms x
patience sweep (oms plus two small fixed steps) -- patience is held fixed at
5 (TabGPGOConfig's own default) exactly as main_tabgpgo_funcset.py does.

Resumable the same way main_tabgpgo_funcset.py is: evolve_ls() skips any
(ms_spec, use_ls) combo that already has a completed run dir on disk
(elite.json present) before re-running it.

Usage:
  python main/main_tabgpgo_ls.py prepare   # phases 1-2 only (fill the cache)
  python main/main_tabgpgo_ls.py evolve    # phase 4-5 (reusing the cache, resumable)
  python main/main_tabgpgo_ls.py           # everything

Writes to main/log/tabgpgo_ls_results.csv / tabgpgo_ls_runs/ (same CSV
column schema as main_tabgpgo_funcset.py) and main/log/tabgpgo_ls_manifest.csv
(run_id, algo, ms_spec, use_ls, function_set, constant_set).
"""
import csv
import dataclasses
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_tabgpgo import evaluate_autoencoder, prepare
from main_tabgpgo_funcset import CONSTANT_SETS, FUNCTION_SETS, function_constant_set
from tabgpgo.config import ALGO_NAMES, REPO_ROOT, TabGPGOConfig
from tabgpgo.evolution_freshpool import FreshPoolSLIM, build_adapter
from tabgpgo.inference import load_run, reconstruct_expression, save_run, verify_inference

# Same HPT-winning combo main_tabgpgo_funcset.py uses.
HPT_OVERRIDES = dict(pop_size=200, tournament_size=2, p_inflate=0.5, n_elites=5)

VARIANT = ("mix", "mul")   # SLIM*MIX only
PATIENCE = 5               # fixed -- TabGPGOConfig's own default

# Best combo found by main_tabgpgo_funcset.py's 26 function-set x 2 constant-set
# sweep, by average zero-shot val R^2 across all 6 real datasets.
FUNCTION_SET_NAME = "full_extended"
CONSTANT_SET_NAME = "small_ints"

MS_SPECS = ["oms", 0.1, 0.01]
LS_OPTIONS = [False, True]

LS_LOG_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_ls_results.csv")
LS_RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_ls_runs")
LS_MANIFEST_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_ls_manifest.csv")


def build_config(**extra):
    return dataclasses.replace(
        TabGPGOConfig(), log_path=LS_LOG_PATH, variants=(VARIANT,),
        run_dir_base=LS_RUN_DIR_BASE, stagnation_patience=PATIENCE,
        **HPT_OVERRIDES, **extra)


def _base_algo_label(cfg, ms_spec):
    """Same label FreshPoolSLIM.__init__ computes for (variant, ms_spec,
    patience) -- see main_tabgpgo_funcset.py's identical helper."""
    use_oms = ms_spec == "oms"
    ms_hi = cfg.oms_bound if use_oms else ms_spec
    patience = cfg.stagnation_patience
    patience_label = "patnone" if not patience or patience <= 0 else f"pat{patience}"
    return f"{ALGO_NAMES[VARIANT]}_{'oms' if use_oms else f'ms{ms_hi:g}'}_{patience_label}"


def _combo_tag(cfg, ms_spec, use_ls):
    ls_label = "ls" if use_ls else "raw"
    algo = f"{_base_algo_label(cfg, ms_spec)}_fn-{FUNCTION_SET_NAME}_const-{CONSTANT_SET_NAME}_{ls_label}"
    return algo, algo.replace("*", "x").replace("~", "t")


def find_completed_run_dir(cfg, tag):
    """Same "done" check as main_tabgpgo_funcset.py: a run dir for this tag
    must exist AND hold elite.json (the artifact save_run() writes last)."""
    if not os.path.isdir(cfg.run_dir_base):
        return None
    suffix = f"_{tag}_0"
    for name in sorted(os.listdir(cfg.run_dir_base)):
        if name.endswith(suffix) and os.path.isfile(os.path.join(cfg.run_dir_base, name, "elite.json")):
            return os.path.join(cfg.run_dir_base, name)
    return None


def _run_one(cfg, ms_spec, use_ls, ctx, unique_run_id, verbose):
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]
    cfg = dataclasses.replace(cfg, p_c=p_c)
    wrapper, operator = VARIANT
    ls_label = "ls" if use_ls else "raw"
    with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
        optimizer = FreshPoolSLIM(cfg, VARIANT, ctx["T_train"], ctx["T_val"],
                                  ctx["y_target"], ctx["val_targets"], ctx["val_y_stats"],
                                  ctx["TERMINALS"], seed=0, ms_spec=ms_spec, use_ls=use_ls)
        optimizer.algo = f"{optimizer.algo}_fn-{FUNCTION_SET_NAME}_const-{CONSTANT_SET_NAME}_{ls_label}"
        elite = optimizer.solve(
            run_info=[optimizer.algo, unique_run_id, "synthetic_prior"],
            log_path=cfg.log_path, verbose=verbose)

        pi, registry, pool = build_adapter(elite, ctx["T_train"], ctx["TERMINALS"])
        verify_inference(pi, pool, ctx["T_train"], registry, ctx["TERMINALS"])
        expression = reconstruct_expression(pi, registry)
        tag = optimizer.algo.replace("*", "x").replace("~", "t")
        run_dir = os.path.join(cfg.run_dir_base, f"{unique_run_id}_{tag}_0")
        save_run(run_dir, cfg, ctx["ae"], registry, pi, wrapper, operator,
                 optimizer.algo, seed=0, expression=expression,
                 ls_a=elite.ls_a, ls_b=elite.ls_b)
    if verbose:
        print(f"[{optimizer.algo}] done: train_rmse={elite.fitness:.4f} "
              f"size={elite.size} ls_a={elite.ls_a:.4f} ls_b={elite.ls_b:.4f} -> {run_dir}")
    return optimizer.algo, elite


def evolve_ls(cfg, ctx, verbose=1):
    """Phase 4-5: (ms_spec, use_ls) sweep, run strictly sequentially (same
    global-FUNCTIONS/CONSTANTS constraint as evolve_funcset). Combos already
    completed by an earlier invocation are skipped -- see
    find_completed_run_dir()."""
    os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
    unique_run_id = uuid.uuid1()
    manifest_rows = []
    elites = {}
    for ms_spec in MS_SPECS:
        for use_ls in LS_OPTIONS:
            algo, tag = _combo_tag(cfg, ms_spec, use_ls)
            existing_dir = find_completed_run_dir(cfg, tag)
            if existing_dir is not None:
                if verbose:
                    print(f"[{algo}] already completed -> skipping ({os.path.basename(existing_dir)})")
                _, _, _, elite, _, _ = load_run(existing_dir)
                run_id = os.path.basename(existing_dir).split("_", 1)[0]
            else:
                algo, elite = _run_one(cfg, ms_spec, use_ls, ctx, unique_run_id, verbose)
                run_id = unique_run_id
            elites[(ms_spec, use_ls)] = elite
            manifest_rows.append({
                "run_id": run_id, "algo": algo, "ms_spec": ms_spec, "use_ls": use_ls,
                "function_set": FUNCTION_SET_NAME, "constant_set": CONSTANT_SET_NAME,
            })
    with open(LS_MANIFEST_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nwrote manifest -> {LS_MANIFEST_CSV}")
    return elites, unique_run_id


def run_experiment_ls(cfg, verbose=1):
    ctx = prepare(cfg, verbose=bool(verbose), build_static_pool=False)
    elites, unique_run_id = evolve_ls(cfg, ctx, verbose=verbose)
    return ctx, elites, unique_run_id


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    config = build_config()
    if stage == "prepare":
        prepare(config, build_static_pool=False)
    elif stage == "eval-ae":
        evaluate_autoencoder(config, prepare(config, build_static_pool=False))
    elif stage == "evolve":
        evolve_ls(config, prepare(config, build_static_pool=False))
    elif stage == "all":
        run_experiment_ls(config)
    else:
        sys.exit(f"unknown stage '{stage}' (use: prepare | eval-ae | evolve | all)")
