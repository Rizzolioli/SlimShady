"""
Shared logic for the TabGPGO initial-population convex-hull study (see
chull_init_local.py / chull_init_full.py for the two entry points).

Where does the convex hull of a freshly-INITIALIZED TabGPGO population sit
relative to the target?

Method (Boyd & Vandenberghe, 2004, adapted for GP): let n be the population
size and m the number of training samples, and let errors[i, j] be the signed
error of individual i on sample j. The global optimum (zero error on every
sample) lies inside conv({errors[i, :]}) iff the system
    a_1*errors[1,j] + ... + a_n*errors[n,j] = 0   for every j
    a_1 + ... + a_n = 1,  a_i >= 0
has a solution. Solving instead for the taxicab distance from the origin to
that hull (minimize sum(e_j + e~_j) s.t. the same system with a slack e_j -
e~_j added to each equation) gives a graded answer instead of a yes/no one.

utils/convexhull.py::distance_from_chull already implements exactly this LP,
but via cvxpy -- and cvxpy's compiled canonicalization backend
(cvxpy.cvxcore.python._cvxcore) fails to import in this environment (a
numpy 1.x/2.x ABI mismatch also breaking ECOS/SCS/OSQP; confirmed via direct
reproduction -- it is not specific to this script's inputs). That function's
`except: return np.inf` silently swallows the failure. Rather than touch the
shared utils/convexhull.py or the environment's cvxpy/numpy install, this
module solves the *identical* LP directly with scipy.optimize.linprog
(HiGHS) -- already a repo dependency, no environment changes, verified
against hand-checked convex-hull cases.
"""
import dataclasses
import os
import random
import time
import uuid

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog

from main_tabgpgo import prepare
from tabgpgo.config import TabGPGOConfig
from tabgpgo.tree_pool import evaluate_pool, generate_ramped_structures
from evaluators.fitness_functions import rmse, signed_errors
from utils.logger import logger

IN_HULL_TOLERANCE = 1e-6


def distance_from_chull(errors):
    """Taxicab distance from the origin (the global optimum -- zero error on
    every training sample) to conv({errors[i, :] for i in individuals}).
    `errors` is a torch tensor of shape (n_individuals, n_samples). Identical
    LP to utils/convexhull.py::distance_from_chull, solved with
    scipy.optimize.linprog(method="highs") instead of cvxpy (see module
    docstring for why). Returns np.inf if the solve fails to converge.
    """
    A = errors.numpy() if not isinstance(errors, np.ndarray) else errors  # (n, m)
    A = A.T  # (m, n): one row per sample, one column per individual
    m, n = A.shape

    # variables: a (n), e (m), e~ (m); minimize sum(e + e~)
    c = np.concatenate([np.zeros(n), np.ones(m), np.ones(m)])

    # m equality rows: A @ a + e - e~ = 0
    error_block = sparse.hstack(
        [sparse.csr_matrix(A), sparse.eye(m, format="csr"), -sparse.eye(m, format="csr")],
        format="csr")
    # 1 equality row: sum(a) = 1
    sum_row = sparse.hstack(
        [sparse.csr_matrix(np.ones((1, n))), sparse.csr_matrix((1, 2 * m))], format="csr")
    A_eq = sparse.vstack([error_block, sum_row], format="csr")
    b_eq = np.concatenate([np.zeros(m), [1.0]])

    bounds = [(0, None)] * (n + 2 * m)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return res.fun if res.success else np.inf


def build_config(log_dir, n_synth_datasets=None):
    """n_synth_datasets=None keeps TabGPGOConfig's own default (1000 x 500 =
    500,000 training rows, the full-scale run); pass a small int for a
    reduced synthetic prior."""
    overrides = {
        "artifacts_dir": os.path.join(log_dir, "artifacts"),
        "log_path": os.path.join(log_dir, "chull_init_results.csv"),
        "run_dir_base": os.path.join(log_dir, "runs"),
    }
    if n_synth_datasets is not None:
        overrides["n_synth_datasets"] = n_synth_datasets
    return dataclasses.replace(TabGPGOConfig(), **overrides)


def run_study(log_dir, n_synth_datasets=None, lp_sample_size=None, seed=0):
    """Build a gen-0 TabGPGO population on the synthetic prior, solve the
    convex-hull LP (optionally against a row-subsample for tractability),
    log one row to `<log_dir>/chull_init_results.csv`, and print a summary.

    n_synth_datasets=None -> TabGPGOConfig's default (500,000 rows).
    lp_sample_size=None -> no row-subsampling, solve against every row.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(log_dir, exist_ok=True)
    cfg = build_config(log_dir, n_synth_datasets)

    t0 = time.time()
    ctx = prepare(cfg, verbose=True, build_static_pool=False)
    T_train, y_target, TERMINALS = ctx["T_train"], ctx["y_target"], ctx["TERMINALS"]
    n_train_full = T_train.shape[0]

    # -- gen-0 population: one random tree per individual, exactly TabGPGO's
    # own initialization (see tabgpgo/tree_pool.py, used identically by
    # main_tabgpgo.py's static pool and by FreshPoolSLIM's gen-0 reservoir
    # draw) -----------------------------------------------------------------
    random.seed(cfg.data_seed)
    registry = generate_ramped_structures(cfg.pop_size, cfg.init_depth, cfg.p_c, TERMINALS)
    raw_semantics = evaluate_pool(registry, T_train, TERMINALS, cfg.get_pool_dtype())

    # -- optionally subsample training rows for LP tractability --------------
    if lp_sample_size is None:
        sample_size = n_train_full
        y_target_sub = y_target
        raw_semantics_sub = raw_semantics
    else:
        sample_size = min(lp_sample_size, n_train_full)
        idx = torch.randperm(n_train_full)[:sample_size]
        y_target_sub = y_target[idx]
        raw_semantics_sub = raw_semantics[:, idx]

    errors = signed_errors(y_target_sub, raw_semantics_sub)  # (pop_size, sample_size)
    chull_distance = distance_from_chull(errors)
    in_hull = bool(chull_distance < IN_HULL_TOLERANCE)

    per_individual_rmse = rmse(y_target_sub, raw_semantics_sub)  # (pop_size,)
    best_train_rmse = per_individual_rmse.min().item()
    mean_train_rmse = per_individual_rmse.mean().item()
    worst_train_rmse = per_individual_rmse.max().item()
    total_nodes = sum(entry["nodes"] for entry in registry)
    elapsed = time.time() - t0

    run_id = uuid.uuid1()
    log_path = cfg.log_path
    logger(
        log_path, generation=0, pop_val_fitness=best_train_rmse, timing=elapsed, nodes=total_nodes,
        additional_infos=[chull_distance, in_hull, best_train_rmse, mean_train_rmse,
                          worst_train_rmse, cfg.pop_size, n_train_full, sample_size],
        run_info=["chull_init", run_id, "synthetic_prior"], seed=seed,
    )

    print()
    print(f"population size:       {cfg.pop_size}")
    print(f"training rows (full):  {n_train_full}")
    print(f"training rows (LP):    {sample_size}")
    print(f"taxicab distance from target to population's error-hull: {chull_distance:.6f}")
    print(f"target inside hull:    {in_hull}")
    print(f"best/mean/worst individual train RMSE: "
          f"{best_train_rmse:.4f} / {mean_train_rmse:.4f} / {worst_train_rmse:.4f}")
    print(f"logged -> {log_path}  (run_id={run_id})")
