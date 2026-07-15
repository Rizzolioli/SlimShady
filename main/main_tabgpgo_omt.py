"""
TabGPGO experiment (Optimal Mutation Tree variant): same fresh-pool pipeline
as main_tabgpgo_latent.py, but instead of sweeping latent_dim,
function_set/constant_set/ms_spec/patience/latent_dim are all held FIXED at
the best combo found so far (full_extended / small_ints, ms=oms, patience=5,
latent_dim=512 -- the default, since the latent-dim sweep found no
consistent winner across sizes), and the sweep axis is the aggregation
operator: "sum" (SLIM+OT) vs. "mul" (SLIM*OT), both with Optimal Mutation
Tree (OMT) enabled for a fraction of inflate mutations (see OMT_FRAC below).

Optimal Mutation Tree (OMT): every sweep so far (function sets, ms values,
linear scaling, latent sizes) converged to the same near-zero zero-shot
ceiling. The linear-scaling experiment's near-zero fitted slope showed why:
OMS already finds the OPTIMAL STEP SIZE for whatever candidate tree it's
handed, but does nothing about WHICH tree -- and a single random draw from
the reservoir essentially never correlates with the residual. OMT attacks
that directly: instead of drawing one random tree per inflate (wrapped in
abs/sig1/sig2), it runs a small embedded GSGP search (FreshPoolSLIM.
_omt_search in tabgpgo/evolution_freshpool.py -- an actual nested
FreshPoolSLIM run, SLIM*MIX, targeting the residual the parent still needs
to correct instead of the real y_target: y_target - parent.aggregate for
sum, y_target/parent.aggregate - 1 for mul, same convention _optimal_ms
already uses) and adds the winning individual's semantics TO as `T + ms*TO`
(sum) / `T*(1 + ms*TO)` (mul) -- ms is still the usual OMS step size, just
applied to a searched-for TO instead of a random tree. No squashing wrapper
(abs/sigmoid) is applied to TO itself at all, matching the literal
"T + OMS*TO" formulation this was scoped against -- TO is a compact nested
FreshIndividual (head + blocks, semantics cached incrementally, no tree
bloat), not a single flat tree, so it's stored directly as an "omt" block's
structure1 (see FreshBlock) and evaluated via the same recursive dispatch
used for every other structure.

Since every block's wrapper is "omt" (no abs/sig1/sig2 at all), these two
variants are named SLIM+OT/SLIM*OT, not SLIM+MIX/SLIM*MIX -- calling them
"MIX" would be misleading despite sharing the sum/mul aggregation axis (see
ALGO_NAMES in tabgpgo/config.py).

cfg.omt_frac controls what fraction of inflate events use OMT vs. the
existing random-reservoir-tree path (0.0 disables OMT entirely, reproducing
every earlier experiment's exact behavior). Scaled down from the original
"every mutation" proposal after the first real run showed the nested-search
cost is substantial: OMT_FRAC=0.2 (1 in 5 inflate events) and
OMT_POP_SIZE=20 (down from 100) cut the number and size of nested searches
roughly 25x combined, while still exercising the mechanism regularly enough
to see whether it moves the needle at all. cfg.omt_gens (still 10) is the
inner GSGP search's own generation count.

IMPORTANT compute-cost note: nesting even a 20-individual/10-generation
SLIM*MIX run inside 1 in 5 inflate events is still meaningfully more
expensive per mutation than the existing single-random-draw path. n_gens is
cut to 50 (from the usual 200) for exactly this reason -- at pop_size=200/
n_elites=5/p_inflate=0.5, that's still on the order of ~5,000 inflate events
per variant, roughly 1,000 of which (at omt_frac=0.2) pay the nested-search
cost.

Resumable the same way main_tabgpgo_funcset.py / main_tabgpgo_ls.py /
main_tabgpgo_latent.py are: evolve_omt() skips any operator that already has
a completed run dir (elite.json present).

Usage:
  python main/main_tabgpgo_omt.py prepare   # phases 1-2 only (fill the cache)
  python main/main_tabgpgo_omt.py evolve    # phase 4-5 (reusing the cache, resumable)
  python main/main_tabgpgo_omt.py           # everything

Writes to main/log/tabgpgo_omt_results.csv / tabgpgo_omt_runs/ (same CSV
column schema as main_tabgpgo_funcset.py) and
main/log/tabgpgo_omt_manifest.csv (run_id, algo, operator, function_set,
constant_set, omt_frac, omt_pop_size, omt_gens).
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

# Same HPT-winning combo main_tabgpgo_funcset.py / main_tabgpgo_ls.py /
# main_tabgpgo_latent.py use.
HPT_OVERRIDES = dict(pop_size=200, tournament_size=2, p_inflate=0.5, n_elites=5)

WRAPPER = "omt"    # per-block wrapper choice for the (1 - omt_frac) fraction of
                   # inflate events (irrelevant here since omt_frac=1.0 means
                   # that fraction is never reached) -- "omt", not "mix", since
                   # no block ever uses abs/sig1/sig2 in this experiment; see
                   # ALGO_NAMES in tabgpgo/config.py for the SLIM+OT/SLIM*OT
                   # labels this produces.
OPERATORS_SWEPT = ["sum", "mul"]   # the two variants: SLIM+OT and SLIM*OT

MS_SPEC = "oms"    # fixed -- best zero-shot ms found by the earlier sweep
PATIENCE = 5       # fixed -- TabGPGOConfig's own default
N_GENS = 50        # cut from the usual 200 -- see module docstring's compute-cost note

# Best combo found by main_tabgpgo_funcset.py's 26 function-set x 2 constant-set
# sweep, by average zero-shot val R^2 across all 6 real datasets.
FUNCTION_SET_NAME = "full_extended"
CONSTANT_SET_NAME = "small_ints"

# Optimal Mutation Tree knobs -- see module docstring.
OMT_FRAC = 1
OMT_POP_SIZE = 20
OMT_GENS = 10

OMT_LOG_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_omt_results.csv")
OMT_RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_omt_runs")
OMT_MANIFEST_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_omt_manifest.csv")


def build_config(**extra):
    return dataclasses.replace(
        TabGPGOConfig(), log_path=OMT_LOG_PATH,
        run_dir_base=OMT_RUN_DIR_BASE, stagnation_patience=PATIENCE, n_gens=N_GENS,
        omt_frac=OMT_FRAC, omt_pop_size=OMT_POP_SIZE, omt_gens=OMT_GENS,
        **HPT_OVERRIDES, **extra)


def _base_algo_label(operator):
    """Same label FreshPoolSLIM.__init__ computes for (variant, MS_SPEC,
    PATIENCE) -- all fixed module constants here, so this needs no cfg."""
    return f"{ALGO_NAMES[(WRAPPER, operator)]}_oms_pat{PATIENCE}"


def _combo_tag(operator):
    # n_gens/omt_pop_size/omt_gens are embedded here -- not just cosmetic:
    # find_completed_run_dir matches PURELY on this tag, so if it didn't
    # encode these, changing any of them (e.g. to rerun with a bigger inner
    # search) would silently match and skip the OLD run instead of
    # launching a new one with the new parameters.
    algo = (f"{_base_algo_label(operator)}_fn-{FUNCTION_SET_NAME}_const-{CONSTANT_SET_NAME}_omt"
            f"_ogens{N_GENS}_ofrac{OMT_FRAC:g}_opop{OMT_POP_SIZE}_oigen{OMT_GENS}")
    return algo, algo.replace("*", "x").replace("~", "t")


def find_completed_run_dir(tag):
    """Same "done" check as the other tabgpgo sweep scripts: a run dir for
    this tag must exist under OMT_RUN_DIR_BASE AND hold elite.json."""
    if not os.path.isdir(OMT_RUN_DIR_BASE):
        return None
    suffix = f"_{tag}_0"
    for name in sorted(os.listdir(OMT_RUN_DIR_BASE)):
        if name.endswith(suffix) and os.path.isfile(os.path.join(OMT_RUN_DIR_BASE, name, "elite.json")):
            return os.path.join(OMT_RUN_DIR_BASE, name)
    return None


def _run_one(cfg, operator, ctx, unique_run_id, verbose):
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]
    cfg = dataclasses.replace(cfg, p_c=p_c)
    variant = (WRAPPER, operator)
    with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
        optimizer = FreshPoolSLIM(cfg, variant, ctx["T_train"], ctx["T_val"],
                                  ctx["y_target"], ctx["val_targets"], ctx["val_y_stats"],
                                  ctx["TERMINALS"], seed=0, ms_spec=MS_SPEC, use_ls=False)
        optimizer.algo = f"{optimizer.algo}_fn-{FUNCTION_SET_NAME}_const-{CONSTANT_SET_NAME}_omt"
        elite = optimizer.solve(
            run_info=[optimizer.algo, unique_run_id, "synthetic_prior"],
            log_path=cfg.log_path, verbose=verbose)

        pi, registry, pool = build_adapter(elite, ctx["T_train"], ctx["TERMINALS"])
        verify_inference(pi, pool, ctx["T_train"], registry, ctx["TERMINALS"])
        expression = reconstruct_expression(pi, registry)
        tag = optimizer.algo.replace("*", "x").replace("~", "t")
        run_dir = os.path.join(cfg.run_dir_base, f"{unique_run_id}_{tag}_0")
        save_run(run_dir, cfg, ctx["ae"], registry, pi, WRAPPER, operator,
                 optimizer.algo, seed=0, expression=expression)
    if verbose:
        print(f"[{optimizer.algo}] done: train_rmse={elite.fitness:.4f} "
              f"size={elite.size} -> {run_dir}")
    return optimizer.algo, elite


def evolve_omt(verbose=1):
    """operator sweep (sum vs. mul), run strictly sequentially
    (function_constant_set monkeypatches shared global state -- see
    main_tabgpgo_funcset.py's docstring). Both operators share one prepare()
    call's ctx, since neither varies function_set/constant_set/latent_dim.
    Already-completed operators are skipped -- see find_completed_run_dir()."""
    cfg = build_config()
    ctx = prepare(cfg, verbose=bool(verbose), build_static_pool=False)
    evaluate_autoencoder(cfg, ctx, verbose=bool(verbose))

    os.makedirs(os.path.dirname(OMT_LOG_PATH), exist_ok=True)
    unique_run_id = uuid.uuid1()
    manifest_rows = []
    elites = {}
    for operator in OPERATORS_SWEPT:
        algo, tag = _combo_tag(operator)
        existing_dir = find_completed_run_dir(tag)
        if existing_dir is not None:
            if verbose:
                print(f"[{algo}] already completed -> skipping ({os.path.basename(existing_dir)})")
            _, _, _, elite, _, _ = load_run(existing_dir)
            run_id = os.path.basename(existing_dir).split("_", 1)[0]
        else:
            algo, elite = _run_one(cfg, operator, ctx, unique_run_id, verbose)
            run_id = unique_run_id
        elites[operator] = elite
        manifest_rows.append({
            "run_id": run_id, "algo": algo, "operator": operator,
            "function_set": FUNCTION_SET_NAME, "constant_set": CONSTANT_SET_NAME,
            "omt_frac": OMT_FRAC, "omt_pop_size": OMT_POP_SIZE, "omt_gens": OMT_GENS,
        })
    with open(OMT_MANIFEST_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nwrote manifest -> {OMT_MANIFEST_CSV}")
    return elites, unique_run_id


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "prepare":
        prepare(build_config(), build_static_pool=False)
    elif stage == "evolve":
        evolve_omt()
    elif stage == "all":
        evolve_omt()
    else:
        sys.exit(f"unknown stage '{stage}' (use: prepare | evolve | all)")
