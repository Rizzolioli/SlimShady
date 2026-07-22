"""
TabGPGO experiment (TabPFN encoder, no-rotation only, pop_size=500 variant):
same experiment as main_tabgpgo_tabpfn_norotate_20k.py (whole synthetic pool
as one fixed training set, ms in {oms, 0.0001, 0.01, 1.0}, no rotation), but
at pop_size=500 instead of 2000, kept at n_gens=5000.

Why a separate script/log file rather than just editing the 20k variant's
pop_size: main_tabgpgo_tabpfn_norotate_20k.py's own ms=oms run at
pop_size=2000 died mid-generation-7647 with no completed run/manifest --
almost certainly a RAM ceiling (each individual's cached aggregate is one
(500000,) float32 tensor, ~2MB; pop_size=2000 means ~4GB just for the live
population, before offspring temporarily coexist with the outgoing
population during a generation). pop_size=500 cuts that same footprint to
~1GB, a much safer margin -- but it's a DIFFERENT run, on different
hardware ("the other machine"), so it gets its own log/run-dir/manifest
rather than overwriting or mixing with the pop=2000 attempt's partial CSV
(which has a different implicit resource profile and shouldn't be treated
as the same experiment for comparison purposes).

Encoder: reuses main_tabgpgo_tabpfn_rotate.py's prepare_rotate/_cached_pool
machinery verbatim (same cached TabPFN pool bundle every tabgpgo_tabpfn_*
script shares).

Fixed for every combo: SLIM*MIX (wrapper="mix", operator="mul"), fresh pool
(FreshPoolSLIM), patience=5, pop_size=500, n_gens=5000, full_extended/
small_ints function/constant set, OMT disabled. tournament_size/p_inflate/
n_elites left at the same HPT-tuned values used everywhere else.

Resumable the same way every other tabgpgo sweep script is: evolve_norotate()
skips any ms combo that already has a completed run dir (elite.json present).

Usage:
  python main/main_tabgpgo_tabpfn_norotate_500pop.py prepare   # phase 1-2 only (build/cache the TabPFN pool)
  python main/main_tabgpgo_tabpfn_norotate_500pop.py evolve    # phase 4-5 (reusing the cache, resumable)
  python main/main_tabgpgo_tabpfn_norotate_500pop.py           # everything

Writes to main/log/tabgpgo_norotate500pop_results.csv /
tabgpgo_norotate500pop_runs/ and main/log/tabgpgo_norotate500pop_manifest.csv
(run_id, algo, ms, function_set, constant_set). CSV schema: same 29-column
schema as every other non-rotating tabgpgo sweep script.
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

HPT_OVERRIDES = dict(pop_size=500, tournament_size=2, p_inflate=0.5, n_elites=5)

WRAPPER = "mix"
VARIANT = (WRAPPER, "mul")   # SLIM*MIX
PATIENCE = 5
N_GENS = 5000

FUNCTION_SET_NAME = "full_extended"
CONSTANT_SET_NAME = "small_ints"

MS_VALUES = ["oms", 0.0001, 0.01, 1.0]
COMBOS = [{"ms": ms} for ms in MS_VALUES]

# Same cache directory every tabgpgo_tabpfn_* script uses -- see
# main_tabgpgo_tabpfn_rotate.py's module docstring for why this is safe to share.
TABPFN_ARTIFACTS_DIR = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_tabpfn_artifacts")
NOROTATE500_LOG_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_norotate500pop_results.csv")
NOROTATE500_RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_norotate500pop_runs")
NOROTATE500_MANIFEST_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_norotate500pop_manifest.csv")


def build_config(**extra):
    return dataclasses.replace(
        TabGPGOConfig(), log_path=NOROTATE500_LOG_PATH, variants=(VARIANT,),
        run_dir_base=NOROTATE500_RUN_DIR_BASE, stagnation_patience=PATIENCE, n_gens=N_GENS,
        artifacts_dir=TABPFN_ARTIFACTS_DIR,
        **HPT_OVERRIDES, **extra)


def _base_algo_label(combo):
    ms_label = "oms" if combo["ms"] == "oms" else f"ms{combo['ms']:g}"
    return f"{ALGO_NAMES[VARIANT]}_{ms_label}_pat{PATIENCE}"


def _combo_tag(combo):
    algo = (f"{_base_algo_label(combo)}_fn-{FUNCTION_SET_NAME}_const-{CONSTANT_SET_NAME}"
            f"_ogens{N_GENS}_pop{HPT_OVERRIDES['pop_size']}_norotate")
    return algo, algo.replace("*", "x").replace("~", "t")


def find_completed_run_dir(tag):
    if not os.path.isdir(NOROTATE500_RUN_DIR_BASE):
        return None
    suffix = f"_{tag}_0"
    for name in sorted(os.listdir(NOROTATE500_RUN_DIR_BASE)):
        if name.endswith(suffix) and os.path.isfile(os.path.join(NOROTATE500_RUN_DIR_BASE, name, "elite.json")):
            return os.path.join(NOROTATE500_RUN_DIR_BASE, name)
    return None


def _run_one(cfg, combo, ctx, unique_run_id, verbose):
    algo, tag = _combo_tag(combo)
    optimizer = FreshPoolSLIM(cfg, VARIANT, ctx["T_train"], ctx["T_val"],
                              ctx["y_target"], ctx["val_targets"], ctx["val_y_stats"],
                              ctx["TERMINALS"], seed=0, ms_spec=combo["ms"], use_ls=False)
    optimizer.algo = algo
    elite = optimizer.solve(
        run_info=[optimizer.algo, unique_run_id, "synthetic_prior"],
        log_path=NOROTATE500_LOG_PATH, verbose=verbose)

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


def evolve_norotate(verbose=1):
    cfg = build_config()
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]
    cfg = dataclasses.replace(cfg, p_c=p_c)

    os.makedirs(os.path.dirname(NOROTATE500_LOG_PATH), exist_ok=True)
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
                "function_set": FUNCTION_SET_NAME, "constant_set": CONSTANT_SET_NAME,
            })
    with open(NOROTATE500_MANIFEST_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nwrote manifest -> {NOROTATE500_MANIFEST_CSV}")
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
        evolve_norotate()
    elif stage == "all":
        evolve_norotate()
    else:
        sys.exit(f"unknown stage '{stage}' (use: prepare | evolve | all)")
