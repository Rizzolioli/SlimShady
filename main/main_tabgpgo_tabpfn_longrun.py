"""
TabGPGO experiment (TabPFN encoder, long run): a longer-horizon rerun of the
ms x rotation question from main_tabgpgo_tabpfn_rotate.py -- that 200-generation
sweep (switch_every in {1, 5, 10}) never beat the no-rotation baseline. This
tests whether that conclusion holds at 10x the generation budget, with a
simpler two-way comparison instead of a full switch_every sweep:

  - "norotate": the whole synthetic pool as ONE fixed training set for all
    2000 generations (dataset_chunks/dataset_switch_every left at their
    FreshPoolSLIM defaults -- identical behaviour to
    main_tabgpgo_tabpfn_encoder.py's ms=oms combo, just at a different ms
    and n_gens).
  - "rotate10": the active training data rotates to a fresh, never-repeated
    synthetic dataset (shuffled-cycle draw over the full pool) every 10
    generations, resetting the stagnation counter each switch -- see
    tabgpgo.evolution_freshpool.FreshPoolSLIM's dataset_chunks/
    dataset_switch_every args.

ms is swept over {"oms" (regularized Optimal Mutation Step, see
TensorSLIM._optimal_ms), 0.01, 1.0} x the two approaches above = 6 combos.
Function/constant set fixed to full_extended/small_ints (the established
best-known winner from main_tabgpgo_funcset.py's sweep) -- no sweep over
that dimension here. SLIM*MIX (wrapper="mix", operator="mul"), fresh pool,
OMT disabled, HPT-tuned hyperparameters (pop_size=200, patience=5), same as
every other tabgpgo experiment this session.

Encoder: reuses main_tabgpgo_tabpfn_rotate.py's own prepare_rotate/
_split_chunks/_cached_pool machinery verbatim (same cached TabPFN pool
bundle directory as every other tabgpgo_tabpfn_* script -- built once,
reused with zero extra TabPFN fitting if already on disk).

Because the two approaches log different column schemas (see
FreshPoolSLIM._log: rotating runs get 3 extra columns --
global_pool_rmse/global_pool_r2/active_dataset_idx -- non-rotating runs
don't), each approach gets its OWN results CSV so every row in a given file
has the same column count (torch/pandas positional read requires this):
  - main/log/tabgpgo_longrun_norotate_results.csv  (29 cols, no-rotation combos)
  - main/log/tabgpgo_longrun_rotate10_results.csv  (32 cols, rotate10 combos)
Both approaches share ONE manifest (main/log/tabgpgo_longrun_manifest.csv:
run_id, algo, ms, switch_every, function_set, constant_set) and ONE run-dir
tree (main/log/tabgpgo_longrun_runs/) since each combo's tag already embeds
ms/switch_every/n_gens and is therefore unique.

Resumable the same way every other tabgpgo sweep script is: evolve_longrun()
skips any combo that already has a completed run dir (elite.json present).

Usage:
  python main/main_tabgpgo_tabpfn_longrun.py prepare   # phase 1-2 only (build/cache the TabPFN pool)
  python main/main_tabgpgo_tabpfn_longrun.py evolve    # phase 4-5 (reusing the cache, resumable)
  python main/main_tabgpgo_tabpfn_longrun.py           # everything
"""
import csv
import dataclasses
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_tabgpgo_funcset import CONSTANT_SETS, FUNCTION_SETS, function_constant_set
from main_tabgpgo_tabpfn_rotate import _EncoderStub, _elite_fitness_only, prepare_rotate
from tabgpgo.config import ALGO_NAMES, REPO_ROOT, TabGPGOConfig
from tabgpgo.evolution_freshpool import FreshPoolSLIM, build_adapter
from tabgpgo.inference import reconstruct_expression, save_run, verify_inference

HPT_OVERRIDES = dict(pop_size=200, tournament_size=2, p_inflate=0.5, n_elites=5)

WRAPPER = "mix"
VARIANT = (WRAPPER, "mul")   # SLIM*MIX
PATIENCE = 5
N_GENS = 2000

FUNCTION_SET_NAME = "full_extended"
CONSTANT_SET_NAME = "small_ints"

MS_VALUES = ["oms", 0.01, 1.0]
SWITCH_EVERY_VALUES = [0, 10]   # 0 == "norotate" (fixed full pool the whole run)
COMBOS = [{"ms": ms, "switch_every": se}
         for ms in MS_VALUES for se in SWITCH_EVERY_VALUES]

# Same cache directory every tabgpgo_tabpfn_* script uses -- see
# main_tabgpgo_tabpfn_rotate.py's module docstring for why this is safe to share.
TABPFN_ARTIFACTS_DIR = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_tabpfn_artifacts")
NOROTATE_LOG_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_longrun_norotate_results.csv")
ROTATE10_LOG_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_longrun_rotate10_results.csv")
LONGRUN_RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_longrun_runs")
LONGRUN_MANIFEST_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_longrun_manifest.csv")


def build_config(**extra):
    return dataclasses.replace(
        TabGPGOConfig(), log_path=NOROTATE_LOG_PATH, variants=(VARIANT,),
        run_dir_base=LONGRUN_RUN_DIR_BASE, stagnation_patience=PATIENCE, n_gens=N_GENS,
        artifacts_dir=TABPFN_ARTIFACTS_DIR,
        **HPT_OVERRIDES, **extra)


def _base_algo_label(combo):
    ms_label = "oms" if combo["ms"] == "oms" else f"ms{combo['ms']:g}"
    return f"{ALGO_NAMES[VARIANT]}_{ms_label}_pat{PATIENCE}"


def _combo_tag(combo):
    approach = "norotate" if combo["switch_every"] == 0 else f"rotate{combo['switch_every']}"
    algo = (f"{_base_algo_label(combo)}_fn-{FUNCTION_SET_NAME}_const-{CONSTANT_SET_NAME}"
            f"_ogens{N_GENS}_{approach}")
    return algo, algo.replace("*", "x").replace("~", "t")


def find_completed_run_dir(tag):
    if not os.path.isdir(LONGRUN_RUN_DIR_BASE):
        return None
    suffix = f"_{tag}_0"
    for name in sorted(os.listdir(LONGRUN_RUN_DIR_BASE)):
        if name.endswith(suffix) and os.path.isfile(os.path.join(LONGRUN_RUN_DIR_BASE, name, "elite.json")):
            return os.path.join(LONGRUN_RUN_DIR_BASE, name)
    return None


def _run_one(cfg, combo, ctx, unique_run_id, verbose):
    algo, tag = _combo_tag(combo)
    rotating = combo["switch_every"] > 0
    rotate_kwargs = (dict(dataset_chunks=ctx["dataset_chunks"],
                          dataset_switch_every=combo["switch_every"])
                     if rotating else {})
    optimizer = FreshPoolSLIM(cfg, VARIANT, ctx["T_train"], ctx["T_val"],
                              ctx["y_target"], ctx["val_targets"], ctx["val_y_stats"],
                              ctx["TERMINALS"], seed=0, ms_spec=combo["ms"], use_ls=False,
                              **rotate_kwargs)
    optimizer.algo = algo
    log_path = ROTATE10_LOG_PATH if rotating else NOROTATE_LOG_PATH
    elite = optimizer.solve(
        run_info=[optimizer.algo, unique_run_id, "synthetic_prior"],
        log_path=log_path, verbose=verbose)

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


def evolve_longrun(verbose=1):
    cfg = build_config()
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]
    cfg = dataclasses.replace(cfg, p_c=p_c)

    os.makedirs(os.path.dirname(NOROTATE_LOG_PATH), exist_ok=True)
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
    with open(LONGRUN_MANIFEST_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nwrote manifest -> {LONGRUN_MANIFEST_CSV}")
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
        evolve_longrun()
    elif stage == "all":
        evolve_longrun()
    else:
        sys.exit(f"unknown stage '{stage}' (use: prepare | evolve | all)")
