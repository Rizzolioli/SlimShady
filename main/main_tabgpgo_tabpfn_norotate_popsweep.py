"""
TabGPGO experiment (TabPFN encoder, no-rotation, pop_size sweep at
n_gens=5000): brackets pop_size=500's own result (already run --
main_tabgpgo_tabpfn_norotate_500pop.py) downward, testing whether a
SMALLER population at the SAME 5000-generation budget avoids the
overfitting-to-the-synthetic-pool regime that budget produced at pop=500 --
see that experiment's dashboard: train R2 climbed monotonically for the
whole 5000 generations while val R2 actively got WORSE with more training
(train<->val correlation flipped strongly NEGATIVE, -0.75 to -0.99 for most
combos, vs. strongly POSITIVE at the shorter 2000-generation/pop=200
budget, where val R2 plateaued instead of declining). Since elite block
count (the mechanical driver of overfitting capacity) grows roughly with
how many individuals get to attempt inflate mutations each generation, a
SMALLER population may grow blocks more slowly and stay in the
still-helps-transfer regime for more of the 5000-generation run.

Sweeps pop_size in {200, 100} (NOT re-including 500, already done) x
ms in {oms, 0.0001, 0.01, 1.0} = 8 combos, all n_gens=5000, no rotation.
Both pop sizes share ONE results/manifest/run-dir set (pop_size is embedded
in each combo's own tag, so there's no ambiguity or collision risk, and
comparing across pop sizes later just means reading one file instead of
two).

Timing estimate (from pop=500's own real per-combo wall-clock time, naive
linear scaling by pop_size ratio -- see conversation): ~57 min for the 4
pop=200 combos, ~28 min for the 4 pop=100 combos, ~85 min combined
sequential. Real times are likely somewhat lower since a smaller population
grows blocks (and therefore per-generation refold cost) more slowly.

Encoder: reuses main_tabgpgo_tabpfn_rotate.py's prepare_rotate/_cached_pool
machinery verbatim (same cached TabPFN pool bundle every tabgpgo_tabpfn_*
script shares) -- built ONCE regardless of pop_size, since pop_size doesn't
affect the synthetic pool itself.

Fixed for every combo: SLIM*MIX (wrapper="mix", operator="mul"), fresh pool
(FreshPoolSLIM), patience=5, n_gens=5000, full_extended/small_ints
function/constant set, OMT disabled. tournament_size/p_inflate/n_elites left
at the same HPT-tuned values used everywhere else.

Resumable the same way every other tabgpgo sweep script is: evolve_norotate()
skips any (pop_size, ms) combo that already has a completed run dir
(elite.json present).

Usage:
  python main/main_tabgpgo_tabpfn_norotate_popsweep.py prepare   # phase 1-2 only (build/cache the TabPFN pool)
  python main/main_tabgpgo_tabpfn_norotate_popsweep.py evolve    # phase 4-5 (reusing the cache, resumable)
  python main/main_tabgpgo_tabpfn_norotate_popsweep.py           # everything

Writes to main/log/tabgpgo_norotate_popsweep_results.csv /
tabgpgo_norotate_popsweep_runs/ and
main/log/tabgpgo_norotate_popsweep_manifest.csv (run_id, algo, pop_size, ms,
function_set, constant_set). CSV schema: same 29-column schema as every
other non-rotating tabgpgo sweep script.
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

# pop_size deliberately NOT in HPT_OVERRIDES -- it's a per-combo sweep value,
# applied via dataclasses.replace in build_config's caller (see COMBOS/_run_one).
HPT_OVERRIDES = dict(tournament_size=2, p_inflate=0.5, n_elites=5)

WRAPPER = "mix"
VARIANT = (WRAPPER, "mul")   # SLIM*MIX
PATIENCE = 5
N_GENS = 5000

FUNCTION_SET_NAME = "full_extended"
CONSTANT_SET_NAME = "small_ints"

POP_SIZES = [200, 100]
MS_VALUES = ["oms", 0.0001, 0.01, 1.0]
COMBOS = [{"pop_size": p, "ms": ms} for p in POP_SIZES for ms in MS_VALUES]

# Same cache directory every tabgpgo_tabpfn_* script uses -- see
# main_tabgpgo_tabpfn_rotate.py's module docstring for why this is safe to share.
TABPFN_ARTIFACTS_DIR = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_tabpfn_artifacts")
POPSWEEP_LOG_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_norotate_popsweep_results.csv")
POPSWEEP_RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_norotate_popsweep_runs")
POPSWEEP_MANIFEST_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_norotate_popsweep_manifest.csv")


def build_config(**extra):
    return dataclasses.replace(
        TabGPGOConfig(), log_path=POPSWEEP_LOG_PATH, variants=(VARIANT,),
        run_dir_base=POPSWEEP_RUN_DIR_BASE, stagnation_patience=PATIENCE, n_gens=N_GENS,
        artifacts_dir=TABPFN_ARTIFACTS_DIR,
        **HPT_OVERRIDES, **extra)


def _base_algo_label(combo):
    ms_label = "oms" if combo["ms"] == "oms" else f"ms{combo['ms']:g}"
    return f"{ALGO_NAMES[VARIANT]}_{ms_label}_pat{PATIENCE}"


def _combo_tag(combo):
    algo = (f"{_base_algo_label(combo)}_fn-{FUNCTION_SET_NAME}_const-{CONSTANT_SET_NAME}"
            f"_ogens{N_GENS}_pop{combo['pop_size']}_norotate")
    return algo, algo.replace("*", "x").replace("~", "t")


def find_completed_run_dir(tag):
    if not os.path.isdir(POPSWEEP_RUN_DIR_BASE):
        return None
    suffix = f"_{tag}_0"
    for name in sorted(os.listdir(POPSWEEP_RUN_DIR_BASE)):
        if name.endswith(suffix) and os.path.isfile(os.path.join(POPSWEEP_RUN_DIR_BASE, name, "elite.json")):
            return os.path.join(POPSWEEP_RUN_DIR_BASE, name)
    return None


def _run_one(base_cfg, combo, ctx, unique_run_id, verbose):
    algo, tag = _combo_tag(combo)
    cfg = dataclasses.replace(base_cfg, pop_size=combo["pop_size"])
    optimizer = FreshPoolSLIM(cfg, VARIANT, ctx["T_train"], ctx["T_val"],
                              ctx["y_target"], ctx["val_targets"], ctx["val_y_stats"],
                              ctx["TERMINALS"], seed=0, ms_spec=combo["ms"], use_ls=False)
    optimizer.algo = algo
    elite = optimizer.solve(
        run_info=[optimizer.algo, unique_run_id, "synthetic_prior"],
        log_path=POPSWEEP_LOG_PATH, verbose=verbose)

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
    base_cfg = build_config()
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]
    base_cfg = dataclasses.replace(base_cfg, p_c=p_c)

    os.makedirs(os.path.dirname(POPSWEEP_LOG_PATH), exist_ok=True)
    unique_run_id = uuid.uuid1()
    manifest_rows = []
    elites = {}
    with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
        ctx = prepare_rotate(base_cfg, verbose=bool(verbose))
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
                algo, elite = _run_one(base_cfg, combo, ctx, unique_run_id, verbose)
                elites[algo] = elite
                run_id = unique_run_id
            manifest_rows.append({
                "run_id": run_id, "algo": algo, "pop_size": combo["pop_size"], "ms": combo["ms"],
                "function_set": FUNCTION_SET_NAME, "constant_set": CONSTANT_SET_NAME,
            })
    with open(POPSWEEP_MANIFEST_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nwrote manifest -> {POPSWEEP_MANIFEST_CSV}")
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
