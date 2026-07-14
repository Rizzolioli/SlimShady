"""
TabGPGO experiment (function/constant-set variant): same fresh-pool pipeline
as main_tabgpgo_freshpool.py (see that module's docstring for why fresh-pool
exists), but instead of sweeping ms_hi x stagnation_patience, ms_hi and
patience are held FIXED (oms, patience=5 -- the best zero-shot combo found by
the earlier freshpool sweep for several datasets) and the sweep axis is the
GP function set and constant set used to build random trees.

`tabgpgo/tree_pool.py`'s FUNCTIONS/CONSTANTS are plain module-level dicts, not
config fields -- `generate_ramped_structures`/`evaluate_structure` (used for
tree generation AND for every semantics evaluation, training or inference)
resolve them as globals inside tree_pool.py's own namespace. So this script
monkeypatches `tabgpgo.tree_pool.FUNCTIONS`/`CONSTANTS` (module attributes,
not the bare names) via `function_constant_set()` below, immediately before
each combo's evolve+save, and restores the originals after -- this is safe
DESPITE evolution_freshpool.py's own concurrency (cfg.max_workers threads
within one FreshPoolSLIM.solve() call), because each combo here runs as a
single (seed=0) job with no other combo's job overlapping in time; combos
themselves run strictly sequentially, never concurrently with each other.

IMPORTANT for anyone reloading these runs later (e.g. for a held-out
fine-tuning eval like tabpfn_baseline/run_split_eval.py does for the ms x
patience sweep): `tabgpgo.inference.load_run` reconstructs a TabGPGOConfig
and encoder from disk, but it does NOT know which function/constant set was
active when a given elite's tree structures were generated -- FUNCTIONS/
CONSTANTS aren't TabGPGOConfig fields, so they aren't in config.json. Before
evaluating a saved elite from this sweep, re-apply the exact same
function_constant_set(...) context (looked up per run_id/algo in
FUNCTION_SET_MANIFEST_CSV) that produced it, or evaluate_structure will
either KeyError (a function name missing from whatever set happens to be
active) or -- if you ever rename sets to overlapping key names with different
semantics -- silently evaluate the wrong function.

Resumable: evolve_funcset() skips any (function_set, constant_set) combo that
already has a completed run dir on disk (elite.json present -- see
find_completed_run_dir()) before re-evolving it, so re-running this script
after a crash (or after adding new combos to FUNCTION_SETS/CONSTANT_SETS)
only evolves what's actually missing. A combo whose generation curve got
logged to the results CSV but crashed before elite.json was saved (e.g. the
KeyError structure_to_str() used to raise on a function outside its old
hardcoded _INFIX table -- now fixed with a generic function-call fallback)
is correctly NOT treated as done, since elite.json never got written.

Usage:
  python main/main_tabgpgo_funcset.py prepare   # phases 1-2 only (fill the cache)
  python main/main_tabgpgo_funcset.py evolve    # phase 4-5 (reusing the cache, resumable)
  python main/main_tabgpgo_funcset.py           # everything

Writes to main/log/tabgpgo_funcset_results.csv / tabgpgo_funcset_runs/ (same
CSV column schema as main_tabgpgo.py) and main/log/tabgpgo_funcset_manifest.csv
(run_id, algo, function_set, constant_set, p_c, functions, constants).
"""
import contextlib
import csv
import dataclasses
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_tabgpgo import evaluate_autoencoder, prepare
from tabgpgo import tree_pool
from tabgpgo.config import ALGO_NAMES, REPO_ROOT, TabGPGOConfig
from tabgpgo.evolution_freshpool import FreshPoolSLIM, build_adapter
from tabgpgo.inference import load_run, reconstruct_expression, save_run, verify_inference

# Same HPT-winning combo main_tabgpgo_freshpool.py uses (p_inflate/tournament_size
# match TabGPGOConfig's own defaults already -- stated explicitly for clarity
# and so this keeps working if those defaults ever change).
HPT_OVERRIDES = dict(pop_size=200, tournament_size=2, p_inflate=0.5, n_elites=5)

VARIANT = ("mix", "mul")   # SLIM*MIX only
MS_SPEC = "oms"            # fixed -- best zero-shot ms for several datasets in the
                           # ms x patience sweep (see the fresh-pool dashboard)
PATIENCE = 5               # fixed -- TabGPGOConfig's own default, stated explicitly

# {'add','subtract','multiply','divide'} (arity-2) + {'sin','cos','tan','log',
# 'sqrt','exp'} (arity-1, see tree_pool.EXTRA_FUNCTIONS) -- merged into one
# lookup so FUNCTION_SETS below can freely mix both families by name.
_BASE_FUNCTIONS = {**tree_pool.FUNCTIONS, **tree_pool.EXTRA_FUNCTIONS}

_ARITH = ["add", "subtract", "multiply", "divide"]

# Subsets/extensions of the current 4 arithmetic functions, now including
# arity-1 transcendental candidates (tree_pool.EXTRA_FUNCTIONS) -- tree
# generation/evaluation support arity-1 nodes generically (see
# evaluate_structure), so "testing different function sets" now also covers
# whether adding periodic (sin/cos/tan) or scale-sensitive (log/sqrt/exp)
# nonlinearities helps, not just which arithmetic primitives earn their keep.
FUNCTION_SETS = {

    # # --------------------------------------------------
    # # Baselines
    # # --------------------------------------------------
    #
    # "arith_full":
    #     _ARITH,
    #
    # "arith_base":
    #     ["add", "subtract"],
    #
    # "add_mult":
    #     ["add", "multiply"],
    #
    #
    # # --------------------------------------------------
    # # Trigonometric family
    # # --------------------------------------------------
    #
    # "arith_trig":
    #     _ARITH + [
    #         "sin",
    #         "cos"
    #     ],
    #
    # "arith_trig_tan":
    #     _ARITH + [
    #         "sin",
    #         "cos",
    #         "tan"
    #     ],


    # --------------------------------------------------
    # Log / power-law family
    # --------------------------------------------------

    "arith_log_sqrt":
        _ARITH + [
            "log",
            "sqrt"
        ],

    "arith_power":
        _ARITH + [
            "square",
            "cube",
            "sqrt"
        ],

    "arith_log_power":
        _ARITH + [
            "log",
            "sqrt",
            "square",
            "cube"
        ],


    # --------------------------------------------------
    # Exponential family
    # --------------------------------------------------

    "arith_exp":
        _ARITH + [
            "exp"
        ],

    "arith_exp_log":
        _ARITH + [
            "exp",
            "log"
        ],

    "arith_exp_decay":
        _ARITH + [
            "exp",
            "neg_exp"
        ],


    # --------------------------------------------------
    # ML activation family
    # --------------------------------------------------

    "arith_activation":
        _ARITH + [
            "tanh",
            "sigmoid",
            "softplus"
        ],

    "arith_tanh":
        _ARITH + [
            "tanh"
        ],


    # --------------------------------------------------
    # Rational / scientific regression family
    # --------------------------------------------------

    "arith_rational":
        _ARITH + [
            "reciprocal",
            "log",
            "sqrt"
        ],


    # --------------------------------------------------
    # Smooth nonlinear SR family
    # --------------------------------------------------

    "smooth_extended":
        _ARITH + [
            "sin",
            "cos",
            "tanh",
            "log",
            "sqrt",
            "square"
        ],


    # --------------------------------------------------
    # Physics / scientific discovery style
    # --------------------------------------------------

    "scientific_extended":
        _ARITH + [
            "sin",
            "cos",
            "exp",
            "log",
            "sqrt",
            "square",
            "cube"
        ],


    # --------------------------------------------------
    # Piecewise ML family
    # --------------------------------------------------

    "piecewise":
        _ARITH + [
            "abs",
            "maximum",
            "minimum"
        ],


    # --------------------------------------------------
    # Large search grammar
    # --------------------------------------------------

    "full_extended":
        _ARITH + [
            "sin",
            "cos",
            "tan",
            "log",
            "sqrt",
            "exp",
            "tanh",
            "square",
            "cube",
            "abs",
            "reciprocal"
        ],


    "full_extended_piecewise":
        _ARITH + [
            "sin",
            "cos",
            "tan",
            "log",
            "sqrt",
            "exp",
            "tanh",
            "square",
            "cube",
            "abs",
            "reciprocal",
            "maximum",
            "minimum"
        ],

# --------------------------------------------------
    # ML activation families
    # --------------------------------------------------

    "ml_tanh":
        _ARITH + [
            "tanh"
        ],

    "ml_sigmoid":
        _ARITH + [
            "sigmoid"
        ],

    "ml_softplus":
        _ARITH + [
            "softplus"
        ],

    "ml_relu":
        _ARITH + [
            "relu"
        ],


    # --------------------------------------------------
    # Combined ML nonlinearities
    # --------------------------------------------------

    "ml_activations":
        _ARITH + [
            "tanh",
            "sigmoid",
            "softplus",
            "relu"
        ],


    "ml_smooth":
        _ARITH + [
            "tanh",
            "softplus",
            "sigmoid",
            "abs"
        ],


    # --------------------------------------------------
    # ML + classical SR hybrid
    # --------------------------------------------------

    "trig_ml":
        _ARITH + [
            "sin",
            "cos",
            "tanh"
        ],


    "extended_ml":
        _ARITH + [
            "sin",
            "cos",
            "log",
            "sqrt",
            "tanh",
            "softplus"
        ],


    "extended_ml_exp":
        _ARITH + [
            "sin",
            "cos",
            "log",
            "sqrt",
            "exp",
            "tanh",
            "softplus"
        ],


    # --------------------------------------------------
    # Feature-engineering inspired SR
    # --------------------------------------------------

    "ml_features":
        _ARITH + [
            "square",
            "cube",
            "abs",
            "tanh",
            "log"
        ],


    "ml_features_extended":
        _ARITH + [
            "square",
            "cube",
            "abs",
            "tanh",
            "softplus",
            "log",
            "sqrt"
        ],


    # --------------------------------------------------
    # Large ML symbolic grammar
    # --------------------------------------------------

    "full_ml_extended":
        _ARITH + [
            "sin",
            "cos",
            "tan",
            "log",
            "sqrt",
            "exp",
            "tanh",
            "sigmoid",
            "softplus",
            "relu",
            "square",
            "cube",
            "abs"
        ],
}

# name -> (p_c, {const_name: lambda _: value}). "none" reproduces today's
# behavior (p_c=0.0 means constants are never drawn regardless of what's in
# the dict); every other entry turns constants on (p_c=0.15) so the constant
# set actually gets exercised.
CONSTANT_SETS = {
    "none": (0.0, {}),
    "small_ints": (0.15, {
        "constant_2": lambda _: 2.0, "constant_3": lambda _: 3.0,
        "constant_4": lambda _: 4.0, "constant_5": lambda _: 5.0,
        "constant__1": lambda _: -1.0,
    }),
    # "unit": (0.15, {
    #     "constant_1": lambda _: 1.0, "constant__1": lambda _: -1.0,
    #     "constant_2": lambda _: 2.0, "constant__2": lambda _: -2.0,
    # }),
    # "fractional": (0.15, {
    #     "constant_half": lambda _: 0.5, "constant_1": lambda _: 1.0,
    #     "constant_2": lambda _: 2.0,
    #     "constant__half": lambda _: -0.5, "constant__1": lambda _: -1.0,
    # }),
}

FUNCSET_LOG_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_funcset_results.csv")
FUNCSET_RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_funcset_runs")
FUNCSET_MANIFEST_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_funcset_manifest.csv")


@contextlib.contextmanager
def function_constant_set(function_names, constants):
    """Temporarily point tabgpgo.tree_pool's FUNCTIONS/CONSTANTS at a subset/
    replacement set, restoring the originals on exit. Must fully enclose
    everything that generates OR evaluates trees for the combo (evolve,
    verify_inference, save_run) -- see module docstring."""
    orig_functions, orig_constants = tree_pool.FUNCTIONS, tree_pool.CONSTANTS
    tree_pool.FUNCTIONS = {n: _BASE_FUNCTIONS[n] for n in function_names}
    tree_pool.CONSTANTS = constants
    try:
        yield
    finally:
        tree_pool.FUNCTIONS = orig_functions
        tree_pool.CONSTANTS = orig_constants


def build_config(**extra):
    return dataclasses.replace(
        TabGPGOConfig(), log_path=FUNCSET_LOG_PATH, variants=(VARIANT,),
        run_dir_base=FUNCSET_RUN_DIR_BASE, stagnation_patience=PATIENCE,
        **HPT_OVERRIDES, **extra)


def _base_algo_label(cfg):
    """Reproduce FreshPoolSLIM.__init__'s algo-label computation (variant +
    ms + patience) without constructing an optimizer -- MS_SPEC/VARIANT are
    fixed module constants here, so this is deterministic and stable across
    runs, letting _combo_tag() below name a combo's run dir/algo the exact
    same way _run_one()'s real FreshPoolSLIM instance would, purely from
    cfg -- no evolution needed just to compute a label."""
    use_oms = MS_SPEC == "oms"
    ms_hi = cfg.oms_bound if use_oms else MS_SPEC
    patience = cfg.stagnation_patience
    patience_label = "patnone" if not patience or patience <= 0 else f"pat{patience}"
    return f"{ALGO_NAMES[VARIANT]}_{'oms' if use_oms else f'ms{ms_hi:g}'}_{patience_label}"


def _combo_tag(cfg, fname, cname):
    algo = f"{_base_algo_label(cfg)}_fn-{fname}_const-{cname}"
    return algo, algo.replace("*", "x").replace("~", "t")


def find_completed_run_dir(cfg, tag):
    """A combo counts as done only if a run dir for its tag exists AND holds
    elite.json -- the artifact save_run() writes last, after
    reconstruct_expression() succeeds. This deliberately does NOT trust the
    results CSV alone: a combo can log all 200 generations' rows there and
    still never finish (e.g. the KeyError this function was added alongside,
    which crashed inside reconstruct_expression after solve() had already
    completed and logged) -- CSV rows without elite.json means "not done,"
    re-run it. run_id is NOT part of the match (only the tag suffix is),
    since every invocation mints a fresh uuid1() run_id, so a previous
    invocation's completed run dir must still be found under today's new id."""
    if not os.path.isdir(cfg.run_dir_base):
        return None
    suffix = f"_{tag}_0"
    for name in sorted(os.listdir(cfg.run_dir_base)):
        if name.endswith(suffix) and os.path.isfile(os.path.join(cfg.run_dir_base, name, "elite.json")):
            return os.path.join(cfg.run_dir_base, name)
    return None


def _run_one(cfg, fname, cname, ctx, unique_run_id, verbose):
    p_c, constants = CONSTANT_SETS[cname]
    cfg = dataclasses.replace(cfg, p_c=p_c)
    wrapper, operator = VARIANT
    with function_constant_set(FUNCTION_SETS[fname], constants):
        optimizer = FreshPoolSLIM(cfg, VARIANT, ctx["T_train"], ctx["T_val"],
                                  ctx["y_target"], ctx["val_targets"], ctx["val_y_stats"],
                                  ctx["TERMINALS"], seed=0, ms_spec=MS_SPEC)
        optimizer.algo = f"{optimizer.algo}_fn-{fname}_const-{cname}"
        elite = optimizer.solve(
            run_info=[optimizer.algo, unique_run_id, "synthetic_prior"],
            log_path=cfg.log_path, verbose=verbose)

        pi, registry, pool = build_adapter(elite, ctx["T_train"], ctx["TERMINALS"])
        verify_inference(pi, pool, ctx["T_train"], registry, ctx["TERMINALS"])
        expression = reconstruct_expression(pi, registry)
        tag = optimizer.algo.replace("*", "x").replace("~", "t")
        run_dir = os.path.join(cfg.run_dir_base, f"{unique_run_id}_{tag}_0")
        save_run(run_dir, cfg, ctx["ae"], registry, pi, wrapper, operator,
                 optimizer.algo, seed=0, expression=expression)
    if verbose:
        print(f"[{optimizer.algo}] done: train_rmse={elite.fitness:.4f} "
              f"size={elite.size} -> {run_dir}")
    return optimizer.algo, elite, constants


def evolve_funcset(cfg, ctx, verbose=1):
    """Phase 4-5: (function_set, constant_set) sweep, run strictly
    sequentially -- FUNCTIONS/CONSTANTS are shared global state, so unlike
    main_tabgpgo_freshpool.py's ThreadPoolExecutor sweep, combos here must
    never overlap in time (each combo's own solve() may still use
    cfg.max_workers internally; there's just one job per combo).

    Combos already completed by an earlier (possibly crashed-partway-through)
    invocation are skipped -- see find_completed_run_dir(). The manifest is
    rebuilt fresh each call from whichever combos ran vs. were skipped, so it
    always reflects every combo's true origin run_id, not just this call's."""
    os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
    unique_run_id = uuid.uuid1()
    manifest_rows = []
    elites = {}
    for fname in FUNCTION_SETS:
        for cname in CONSTANT_SETS:
            algo, tag = _combo_tag(cfg, fname, cname)
            existing_dir = find_completed_run_dir(cfg, tag)
            if existing_dir is not None:
                if verbose:
                    print(f"[{algo}] already completed -> skipping ({os.path.basename(existing_dir)})")
                _, _, _, elite, _, _ = load_run(existing_dir)
                run_id = os.path.basename(existing_dir).split("_", 1)[0]
                constants = CONSTANT_SETS[cname][1]
            else:
                algo, elite, constants = _run_one(cfg, fname, cname, ctx, unique_run_id, verbose)
                run_id = unique_run_id
            elites[(fname, cname)] = elite
            manifest_rows.append({
                "run_id": run_id, "algo": algo,
                "function_set": fname, "constant_set": cname,
                "p_c": CONSTANT_SETS[cname][0],
                "functions": ";".join(FUNCTION_SETS[fname]),
                "constants": json.dumps({k: v(None) for k, v in constants.items()}),
            })
    with open(FUNCSET_MANIFEST_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nwrote manifest -> {FUNCSET_MANIFEST_CSV}")
    return elites, unique_run_id


def run_experiment_funcset(cfg, verbose=1):
    ctx = prepare(cfg, verbose=bool(verbose), build_static_pool=False)
    elites, unique_run_id = evolve_funcset(cfg, ctx, verbose=verbose)
    return ctx, elites, unique_run_id


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    config = build_config()
    if stage == "prepare":
        prepare(config, build_static_pool=False)
    elif stage == "eval-ae":
        evaluate_autoencoder(config, prepare(config, build_static_pool=False))
    elif stage == "evolve":
        evolve_funcset(config, prepare(config, build_static_pool=False))
    elif stage == "all":
        run_experiment_funcset(config)
    else:
        sys.exit(f"unknown stage '{stage}' (use: prepare | eval-ae | evolve | all)")
