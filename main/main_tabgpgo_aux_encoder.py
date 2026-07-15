"""
TabGPGO experiment (target-aware autoencoder variant): tests whether adding
a lightweight auxiliary target-prediction head to the existing MLPAutoencoder
(tabgpgo/autoencoder.py) -- trained jointly with the usual reconstruction
loss -- changes the near-zero zero-shot ceiling every previous sweep this
session (function sets, ms, linear scaling, latent sizes, OMT) converged to
regardless of what else changed.

Motivation: the current MLPAutoencoder is trained purely to reconstruct X --
nothing in its training objective ever tells it to preserve target-relevant
structure. This adds cfg.ae_aux_weight * MSE(pred_head(encoder(x)), y) to the
autoencoder's own training loss (see tabgpgo/autoencoder.py::
train_autoencoder), so the encoder is now directly pushed to keep whatever
latent structure predicts the synthetic-prior target, not just whatever
reconstructs X. cfg.ae_aux_weight=0.0 (the TabGPGOConfig default) reproduces
the ORIGINAL unsupervised-only training exactly -- every other experiment
script's behavior/reproducibility is completely unaffected by this addition
(confirmed: main_tabgpgo.py's prepare() now always passes y_target into
train_autoencoder, but train_autoencoder only uses it when
cfg.ae_aux_weight > 0).

Sweeps ae_aux_weight in {0.0, 1.0}: a freshly-trained no-aux baseline and
the target-aware variant, both re-trained from scratch under their OWN
dedicated artifacts_dir (aux_0.0 / aux_1.0) -- never sharing the default
tabgpgo_artifacts cache, since the autoencoder itself differs between the
two and reusing a stale cache across different ae_aux_weight values would
silently give the wrong (or a shape-mismatched) encoder, exactly the same
reason main_tabgpgo_latent.py gives each latent_dim its own artifacts_dir.
Both runs use the established best-known GP config (full_extended/
small_ints, ms=oms, patience=5, HPT overrides, 200 generations) -- so the
ONLY thing that differs between them is whether the encoder ever saw y
during its own training.

ae_aux_weight=1.0 is a starting-point choice, not a tuned one: both X and y
are z-scored (unit variance) in this pipeline, so the reconstruction and
auxiliary losses sit on comparable natural scales, making 1:1 weighting a
reasonable first thing to try rather than an arbitrarily large or small
value.

Resumable the same way every other tabgpgo sweep script is: evolve_aux()
skips any ae_aux_weight value that already has a completed run dir
(elite.json present). Unlike main_tabgpgo_tabpfn_encoder.py, this script's
persisted encoder IS a real MLPAutoencoder, so tabgpgo.inference.load_run
works completely unmodified for the resumability skip path.

Usage:
  python main/main_tabgpgo_aux_encoder.py prepare   # phase 1-2 only (build/cache both autoencoders)
  python main/main_tabgpgo_aux_encoder.py evolve    # phase 4-5 (reusing the cache, resumable)
  python main/main_tabgpgo_aux_encoder.py           # everything

Writes to main/log/tabgpgo_aux_results.csv / tabgpgo_aux_runs/ (same CSV
column schema as main_tabgpgo_funcset.py) and
main/log/tabgpgo_aux_manifest.csv (run_id, algo, ae_aux_weight,
function_set, constant_set).
"""
import csv
import dataclasses
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_tabgpgo import prepare
from main_tabgpgo_funcset import CONSTANT_SETS, FUNCTION_SETS, function_constant_set
from tabgpgo.config import ALGO_NAMES, REPO_ROOT, TabGPGOConfig
from tabgpgo.evolution_freshpool import FreshPoolSLIM, build_adapter
from tabgpgo.inference import load_run, reconstruct_expression, save_run, verify_inference

HPT_OVERRIDES = dict(pop_size=200, tournament_size=2, p_inflate=0.5, n_elites=5)
VARIANT = ("mix", "mul")   # SLIM*MIX
MS_SPEC = "oms"
PATIENCE = 5
N_GENS = 200

# Best combo found by main_tabgpgo_funcset.py's 26 function-set x 2 constant-set
# sweep, by average zero-shot val R^2 across all 6 real datasets -- held
# fixed here since this experiment is testing the ENCODER, not the grammar.
FUNCTION_SET_NAME = "full_extended"
CONSTANT_SET_NAME = "small_ints"

AUX_WEIGHTS = [0.0, 1.0]   # 0.0: fresh no-aux baseline; 1.0: target-aware

AUX_ARTIFACTS_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_aux_artifacts")
AUX_LOG_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_aux_results.csv")
AUX_RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_aux_runs")
AUX_MANIFEST_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_aux_manifest.csv")


def build_config(ae_aux_weight, **extra):
    return dataclasses.replace(
        TabGPGOConfig(), log_path=AUX_LOG_PATH, variants=(VARIANT,),
        run_dir_base=AUX_RUN_DIR_BASE, stagnation_patience=PATIENCE, n_gens=N_GENS,
        ae_aux_weight=ae_aux_weight,
        artifacts_dir=os.path.join(AUX_ARTIFACTS_BASE, f"aux_{ae_aux_weight:g}"),
        **HPT_OVERRIDES, **extra)


def _base_algo_label():
    return f"{ALGO_NAMES[VARIANT]}_oms_pat{PATIENCE}"


def _combo_tag(ae_aux_weight):
    algo = (f"{_base_algo_label()}_fn-{FUNCTION_SET_NAME}_const-{CONSTANT_SET_NAME}"
            f"_ogens{N_GENS}_auxw{ae_aux_weight:g}")
    return algo, algo.replace("*", "x").replace("~", "t")


def find_completed_run_dir(tag):
    """Same "done" check as every other tabgpgo sweep script: a run dir for
    this tag must exist under AUX_RUN_DIR_BASE AND hold elite.json."""
    if not os.path.isdir(AUX_RUN_DIR_BASE):
        return None
    suffix = f"_{tag}_0"
    for name in sorted(os.listdir(AUX_RUN_DIR_BASE)):
        if name.endswith(suffix) and os.path.isfile(os.path.join(AUX_RUN_DIR_BASE, name, "elite.json")):
            return os.path.join(AUX_RUN_DIR_BASE, name)
    return None


def _run_one(ae_aux_weight, unique_run_id, verbose):
    cfg = build_config(ae_aux_weight)
    ctx = prepare(cfg, verbose=bool(verbose), build_static_pool=False)
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]
    cfg = dataclasses.replace(cfg, p_c=p_c)
    algo, tag = _combo_tag(ae_aux_weight)
    with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
        optimizer = FreshPoolSLIM(cfg, VARIANT, ctx["T_train"], ctx["T_val"],
                                  ctx["y_target"], ctx["val_targets"], ctx["val_y_stats"],
                                  ctx["TERMINALS"], seed=0, ms_spec=MS_SPEC, use_ls=False)
        optimizer.algo = algo
        elite = optimizer.solve(
            run_info=[optimizer.algo, unique_run_id, "synthetic_prior"],
            log_path=cfg.log_path, verbose=verbose)

        pi, registry, pool = build_adapter(elite, ctx["T_train"], ctx["TERMINALS"])
        verify_inference(pi, pool, ctx["T_train"], registry, ctx["TERMINALS"])
        expression = reconstruct_expression(pi, registry)
        run_dir = os.path.join(cfg.run_dir_base, f"{unique_run_id}_{tag}_0")
        save_run(run_dir, cfg, ctx["ae"], registry, pi, "mix", "mul",
                 optimizer.algo, seed=0, expression=expression)
    if verbose:
        print(f"[{optimizer.algo}] done: train_rmse={elite.fitness:.4f} "
              f"size={elite.size} -> {run_dir}")
    return optimizer.algo, elite


def evolve_aux(verbose=1):
    """ae_aux_weight sweep, run sequentially (function_constant_set
    monkeypatches shared global state -- see main_tabgpgo_funcset.py's
    docstring). Each value gets its own prepare() call (own artifacts_dir),
    so the two autoencoders never share a cache. Already-completed values
    are skipped -- see find_completed_run_dir()."""
    os.makedirs(os.path.dirname(AUX_LOG_PATH), exist_ok=True)
    unique_run_id = uuid.uuid1()
    manifest_rows = []
    elites = {}
    for ae_aux_weight in AUX_WEIGHTS:
        algo, tag = _combo_tag(ae_aux_weight)
        existing_dir = find_completed_run_dir(tag)
        if existing_dir is not None:
            if verbose:
                print(f"[{algo}] already completed -> skipping ({os.path.basename(existing_dir)})")
            _, _, _, elite, _, _ = load_run(existing_dir)
            run_id = os.path.basename(existing_dir).split("_", 1)[0]
        else:
            algo, elite = _run_one(ae_aux_weight, unique_run_id, verbose)
            run_id = unique_run_id
        elites[ae_aux_weight] = elite
        manifest_rows.append({
            "run_id": run_id, "algo": algo, "ae_aux_weight": ae_aux_weight,
            "function_set": FUNCTION_SET_NAME, "constant_set": CONSTANT_SET_NAME,
        })
    with open(AUX_MANIFEST_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nwrote manifest -> {AUX_MANIFEST_CSV}")
    return elites, unique_run_id


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "prepare":
        for w in AUX_WEIGHTS:
            prepare(build_config(w), build_static_pool=False)
    elif stage == "evolve":
        evolve_aux()
    elif stage == "all":
        evolve_aux()
    else:
        sys.exit(f"unknown stage '{stage}' (use: prepare | evolve | all)")
