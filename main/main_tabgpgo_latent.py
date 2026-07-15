"""
TabGPGO experiment (latent-dimension variant): same fresh-pool pipeline as
main_tabgpgo_ls.py, but instead of sweeping ms_spec/linear-scaling,
function_set/constant_set/ms_spec/patience are all held FIXED at the best
combo found so far (full_extended / small_ints, ms=oms, patience=5 -- see
main_tabgpgo_funcset.py / main_tabgpgo_ls.py), and the sweep axis is the
autoencoder's latent_dim (how many latent tokens the GP trees search over).

Motivation: across the funcset sweep (26 function sets) and the ms/linear-
scaling sweep, zero-shot val R^2 never got much above ~0 on any real dataset,
even though TabPFN gets R^2=0.94-0.99 on the SAME PCA-100 input features --
so the signal is provably present in the raw features, and the bottleneck is
somewhere in TabGPGO's own autoencoder -> latent-token -> random-tree
pipeline. latent_dim=512 (the default, an OVERCOMPLETE autoencoder -- bigger
than the 100-dim input) means a random tree's terminal set is 512 tokens
wide; if only a handful of those tokens actually carry target-relevant
structure, a single random tree draw has low odds of ever touching one. This
sweep tests both directions: smaller (bottlenecked, undercomplete) latent
spaces that might concentrate signal into fewer, more useful tokens, and
bigger (even more overcomplete) ones, to see whether performance is
monotonic, peaked, or flat in latent_dim -- flat across the whole range would
point away from "capacity/dimensionality" and back toward "the reconstruction
training objective itself doesn't preserve target-relevant structure",
which no latent_dim choice can fix.

Each latent_dim needs its OWN autoencoder (different weight shapes) and its
own T_train/T_val latent tokens, so -- unlike main_tabgpgo_funcset.py /
main_tabgpgo_ls.py, which share one prepare() call's ctx across their whole
sweep -- this script calls prepare() separately per latent_dim, each pointed
at its own cfg.artifacts_dir (main/log/tabgpgo_latent_artifacts/latent_<N>/)
so different-shaped cached tensors never collide. cfg.ae_hidden (the
autoencoder's single hidden-layer width) is held fixed at its default (256)
throughout -- only latent_dim varies.

Alongside the usual per-generation CSV log, this script also records each
latent_dim's autoencoder reconstruction R^2 (encoder+decoder, unsupervised --
see evaluate_autoencoder in main_tabgpgo.py) on the synthetic-prior training
data and every real validation dataset, in the manifest CSV. This is a cheap
diagnostic for whether a given latent_dim's autoencoder is even capable of
reconstructing real data at all (a necessary, though not sufficient,
condition for its latent tokens to carry real-dataset-relevant structure).

Resumable the same way main_tabgpgo_funcset.py / main_tabgpgo_ls.py are:
evolve_latent() skips any latent_dim that already has a completed run dir
(elite.json present) BEFORE calling prepare() for it -- so a crash partway
through this sweep never forces an already-finished latent_dim's autoencoder
to be retrained from scratch on resume (the expensive part of this
particular sweep, unlike the funcset/ms sweeps where prepare() itself was
shared and cheap after the first call).

Usage:
  python main/main_tabgpgo_latent.py prepare   # fill every latent_dim's cache (autoencoders only)
  python main/main_tabgpgo_latent.py evolve    # phase 4-5 (reusing caches, resumable)
  python main/main_tabgpgo_latent.py           # everything

Writes to main/log/tabgpgo_latent_results.csv / tabgpgo_latent_runs/ (same
CSV column schema as main_tabgpgo_funcset.py) and
main/log/tabgpgo_latent_manifest.csv (run_id, algo, latent_dim, function_set,
constant_set, ae_recon_r2_json).
"""
import csv
import dataclasses
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_tabgpgo import evaluate_autoencoder, prepare
from main_tabgpgo_funcset import CONSTANT_SETS, FUNCTION_SETS, function_constant_set
from tabgpgo.config import ALGO_NAMES, REPO_ROOT, TabGPGOConfig
from tabgpgo.evolution_freshpool import FreshPoolSLIM, build_adapter
from tabgpgo.inference import load_run, reconstruct_expression, save_run, verify_inference

# Same HPT-winning combo main_tabgpgo_funcset.py / main_tabgpgo_ls.py use.
HPT_OVERRIDES = dict(pop_size=200, tournament_size=2, p_inflate=0.5, n_elites=5)

VARIANT = ("mix", "mul")   # SLIM*MIX only
MS_SPEC = "oms"            # fixed -- best zero-shot ms found by the earlier sweep
PATIENCE = 5               # fixed -- TabGPGOConfig's own default

# Best combo found by main_tabgpgo_funcset.py's 26 function-set x 2 constant-set
# sweep, by average zero-shot val R^2 across all 6 real datasets.
FUNCTION_SET_NAME = "full_extended"
CONSTANT_SET_NAME = "small_ints"

# Default latent_dim is 512 (already overcomplete vs. the 100-dim input).
# Sweeps both directions: undercomplete bottlenecks that might concentrate
# signal into fewer tokens, and even-more-overcomplete spaces.
LATENT_DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096]

LATENT_LOG_PATH = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_latent_results.csv")
LATENT_RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_latent_runs")
LATENT_MANIFEST_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_latent_manifest.csv")
LATENT_ARTIFACTS_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_latent_artifacts")


def build_config(latent_dim, **extra):
    return dataclasses.replace(
        TabGPGOConfig(), log_path=LATENT_LOG_PATH, variants=(VARIANT,),
        run_dir_base=LATENT_RUN_DIR_BASE, stagnation_patience=PATIENCE,
        latent_dim=latent_dim,
        artifacts_dir=os.path.join(LATENT_ARTIFACTS_BASE, f"latent_{latent_dim}"),
        **HPT_OVERRIDES, **extra)


def _base_algo_label():
    """Same label FreshPoolSLIM.__init__ computes for (VARIANT, MS_SPEC,
    PATIENCE) -- all fixed module constants here (MS_SPEC is always "oms" in
    this script), so this needs no cfg."""
    ms_label = "oms" if MS_SPEC == "oms" else f"ms{MS_SPEC:g}"
    return f"{ALGO_NAMES[VARIANT]}_{ms_label}_pat{PATIENCE}"


def _combo_tag(latent_dim):
    algo = f"{_base_algo_label()}_fn-{FUNCTION_SET_NAME}_const-{CONSTANT_SET_NAME}_latent{latent_dim}"
    return algo, algo.replace("*", "x").replace("~", "t")


def find_completed_run_dir(tag):
    """Same "done" check as main_tabgpgo_funcset.py/main_tabgpgo_ls.py: a run
    dir for this tag must exist under LATENT_RUN_DIR_BASE AND hold
    elite.json. Checked BEFORE prepare() is ever called for a latent_dim, so
    a completed latent_dim's (expensive) autoencoder retrain is never
    repeated on resume."""
    if not os.path.isdir(LATENT_RUN_DIR_BASE):
        return None
    suffix = f"_{tag}_0"
    for name in sorted(os.listdir(LATENT_RUN_DIR_BASE)):
        if name.endswith(suffix) and os.path.isfile(os.path.join(LATENT_RUN_DIR_BASE, name, "elite.json")):
            return os.path.join(LATENT_RUN_DIR_BASE, name)
    return None


def _run_one(latent_dim, unique_run_id, verbose):
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]
    cfg = dataclasses.replace(build_config(latent_dim), p_c=p_c)
    wrapper, operator = VARIANT

    ctx = prepare(cfg, verbose=bool(verbose), build_static_pool=False)
    recon = evaluate_autoencoder(cfg, ctx, verbose=bool(verbose))
    recon_r2 = {name: stats["r2"] for name, stats in recon.items()}

    with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
        optimizer = FreshPoolSLIM(cfg, VARIANT, ctx["T_train"], ctx["T_val"],
                                  ctx["y_target"], ctx["val_targets"], ctx["val_y_stats"],
                                  ctx["TERMINALS"], seed=0, ms_spec=MS_SPEC, use_ls=False)
        optimizer.algo = f"{optimizer.algo}_fn-{FUNCTION_SET_NAME}_const-{CONSTANT_SET_NAME}_latent{latent_dim}"
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
              f"size={elite.size} ae_recon_r2={recon_r2} -> {run_dir}")
    return optimizer.algo, elite, recon_r2


def evolve_latent(verbose=1):
    """latent_dim sweep, run strictly sequentially (function_constant_set
    monkeypatches shared global state -- see main_tabgpgo_funcset.py's
    docstring). Combos already completed by an earlier invocation are
    skipped -- see find_completed_run_dir()."""
    os.makedirs(os.path.dirname(LATENT_LOG_PATH), exist_ok=True)
    unique_run_id = uuid.uuid1()
    manifest_rows = []
    elites = {}
    for latent_dim in LATENT_DIMS:
        algo, tag = _combo_tag(latent_dim)
        existing_dir = find_completed_run_dir(tag)
        if existing_dir is not None:
            if verbose:
                print(f"[{algo}] already completed -> skipping ({os.path.basename(existing_dir)})")
            _, _, _, elite, _, _ = load_run(existing_dir)
            run_id = os.path.basename(existing_dir).split("_", 1)[0]
            recon_r2 = None   # not recomputed on skip -- see manifest note below
        else:
            algo, elite, recon_r2 = _run_one(latent_dim, unique_run_id, verbose)
            run_id = unique_run_id
        elites[latent_dim] = elite
        manifest_rows.append({
            "run_id": run_id, "algo": algo, "latent_dim": latent_dim,
            "function_set": FUNCTION_SET_NAME, "constant_set": CONSTANT_SET_NAME,
            # None (blank in the CSV) on a skipped/resumed combo -- re-running
            # evaluate_autoencoder for an already-completed latent_dim would
            # mean loading/retraining that autoencoder again just for this
            # diagnostic, defeating the point of skipping it in the first place.
            "ae_recon_r2_json": json.dumps(recon_r2) if recon_r2 is not None else "",
        })
    with open(LATENT_MANIFEST_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nwrote manifest -> {LATENT_MANIFEST_CSV}")
    return elites, unique_run_id


def prepare_latent(verbose=1):
    """Fill every latent_dim's autoencoder/latent-token cache without
    evolving anything -- lets the (expensive) autoencoder retrains run
    ahead of time, e.g. overnight, independently of the evolve stage."""
    for latent_dim in LATENT_DIMS:
        p_c, _ = CONSTANT_SETS[CONSTANT_SET_NAME]
        cfg = dataclasses.replace(build_config(latent_dim), p_c=p_c)
        prepare(cfg, verbose=bool(verbose), build_static_pool=False)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "prepare":
        prepare_latent()
    elif stage == "evolve":
        evolve_latent()
    elif stage == "all":
        evolve_latent()
    else:
        sys.exit(f"unknown stage '{stage}' (use: prepare | evolve | all)")
