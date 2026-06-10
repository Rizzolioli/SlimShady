import csv
import os
import sys
import uuid
import time
import multiprocessing

# Ensure the project root is on the path when spawned as a subprocess
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

########################################################################################################################
# CONFIGURATION
########################################################################################################################

DATASETS = ["toxicity", "concrete", "instanbul", "ppb", "resid_build_sale_price", "energy"]
N_RUNS = 30
N_WORKERS = max(1, min(os.cpu_count() - 1, 8))

# (name, sig, two_trees, op, norm)
# norm: None = standard SLIM, "norm1" = SLIM-NORM1, "norm2" = SLIM-NORM2
VARIANTS = [
    # {"name": "SLIM+2SIG", "sig": True,  "two_trees": True,  "op": "sum", "norm": None},
    # {"name": "SLIM*2SIG", "sig": True,  "two_trees": True,  "op": "mul", "norm": None},
    # {"name": "SLIM*ABS",  "sig": False, "two_trees": False, "op": "mul", "norm": None},
    # {"name": "SLIM+ABS",  "sig": False, "two_trees": False, "op": "sum", "norm": None},
    # {"name": "SLIM*1SIG", "sig": True,  "two_trees": False, "op": "mul", "norm": None},
    # {"name": "SLIM+1SIG", "sig": True,  "two_trees": False, "op": "sum", "norm": None},
    # {"name": "SLIM+NORM1","sig": False, "two_trees": False, "op": "sum", "norm": "norm1"},
    # {"name": "SLIM*NORM1","sig": False, "two_trees": False, "op": "mul", "norm": "norm1"},
    # {"name": "SLIM+NORM2",   "sig": False, "two_trees": True,  "op": "sum", "norm": "norm2"},
    # {"name": "SLIM*NORM2",   "sig": False, "two_trees": True,  "op": "mul", "norm": "norm2"},
    {"name": "SLIM+NORMROB", "sig": False, "two_trees": True,  "op": "sum", "norm": "normrob"},
    {"name": "SLIM*NORMROB", "sig": False, "two_trees": True,  "op": "mul", "norm": "normrob"},
]

# Dataset-specific inflate probability
_DATASET_P_INFLATE = {
    "toxicity": 0.1,
    "concrete": 0.5,
}
_DEFAULT_P_INFLATE = 0.3

# Fixed output file names — accumulate across sessions
_GEN_LOG_NAME  = "results_normalized_generations.csv"
_SIMP_LOG_NAME = "results_normalized_simplification.csv"

# Generation log column header (matches logger() output at log=8)
_GEN_HEADER = [
    'algo', 'run_id', 'dataset', 'seed', 'generation',
    'train_fitness', 'timing', 'nodes',
    'test_fitness', 'nodes_count', 'm_phi', 'no', 'nnao', 'nnaoc', 'mae', 'r2', 'log_level',
]

########################################################################################################################
# WORKER  (module-level so multiprocessing can pickle it)
########################################################################################################################

def run_experiment_worker(dataset, variant_idx, seed, run_id_str,
                          gen_log_path, simp_log_path, gen_lock, simp_lock):
    """Run a single (dataset, variant, seed) experiment, writing directly to shared log files."""
    import numpy as np
    import torch

    from utils.utils import protected_div, get_terminals, get_best_min
    from evaluators.fitness_functions import rmse
    from algorithms.GP.operators.initializers import rhh
    from algorithms.SLIM_GSGP.operators.selection_algorithms import tournament_selection_min_slim
    from algorithms.SLIM_GSGP.operators.mutators import (
        inflate_mutation, deflate_mutation,
        inflate_mutation_normalized, inflate_mutation_norm1,
        inflate_mutation_normrob,
    )
    from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
    from datasets.data_loader import load_preloaded

    FUNCTIONS = {
        'add':      {'function': lambda x, y: torch.add(x, y),      'arity': 2},
        'subtract': {'function': lambda x, y: torch.sub(x, y),      'arity': 2},
        'multiply': {'function': lambda x, y: torch.mul(x, y),      'arity': 2},
        'divide':   {'function': lambda x, y: protected_div(x, y),  'arity': 2},
    }
    CONSTANTS = {
        'constant_2':  lambda x: torch.tensor(2.0),
        'constant_3':  lambda x: torch.tensor(3.0),
        'constant_4':  lambda x: torch.tensor(4.0),
        'constant_5':  lambda x: torch.tensor(5.0),
        'constant__1': lambda x: torch.tensor(-1.0),
    }

    variant = VARIANTS[variant_idx]

    X_train, y_train = load_preloaded(dataset, seed=seed + 1, training=True,  X_y=True)
    X_test,  y_test  = load_preloaded(dataset, seed=seed + 1, training=False, X_y=True)
    TERMINALS = get_terminals(dataset, seed + 1)

    ms_val = float(np.median(y_train.numpy()))
    ms = lambda: ms_val  # noqa: E731

    p_inflate = _DATASET_P_INFLATE.get(dataset, _DEFAULT_P_INFLATE)

    pi_init = {
        'init_pop_size': 100,
        'init_depth':    6,
        'FUNCTIONS':     FUNCTIONS,
        'TERMINALS':     TERMINALS,
        'CONSTANTS':     CONSTANTS,
        'p_c':           0,
    }

    norm = variant["norm"]
    if norm == "norm2":
        inflate_mutator = inflate_mutation_normalized(
            FUNCTIONS=FUNCTIONS, TERMINALS=TERMINALS, CONSTANTS=CONSTANTS,
            operator=variant["op"],
        )
    elif norm == "norm1":
        inflate_mutator = inflate_mutation_norm1(
            FUNCTIONS=FUNCTIONS, TERMINALS=TERMINALS, CONSTANTS=CONSTANTS,
            operator=variant["op"],
        )
    elif norm == "normrob":
        inflate_mutator = inflate_mutation_normrob(
            FUNCTIONS=FUNCTIONS, TERMINALS=TERMINALS, CONSTANTS=CONSTANTS,
            operator=variant["op"], scale='q99',
        )
    else:
        inflate_mutator = inflate_mutation(
            FUNCTIONS=FUNCTIONS, TERMINALS=TERMINALS, CONSTANTS=CONSTANTS,
            two_trees=variant["two_trees"],
            operator=variant["op"],
            sig=variant["sig"],
        )

    slim_params = {
        "initializer":     rhh,
        "selector":        tournament_selection_min_slim(2),
        "ms":              ms,
        "inflate_mutator": inflate_mutator,
        "deflate_mutator": deflate_mutation,
        "crossover":       None,
        "p_xo":            0,
        "p_m":             1,
        "pop_size":        100,
        "p_inflate":       p_inflate,
        "p_deflate":       1 - p_inflate,
        "copy_parent":     None,
        "operator":        variant["op"],
        "two_trees":       variant["two_trees"],
        "find_elit_func":  get_best_min,
        "settings_dict":   {"p_test": 0.2},
    }

    solve_params = {
        "elitism":           True,
        "log":               8,
        "verbose":           0,
        "test_elite":        True,
        "log_path":          gen_log_path,
        "run_info":          [variant["name"], run_id_str, dataset],
        "ffunction":         rmse,
        "n_iter":            2000,
        "max_depth":         None,
        "n_elites":          1,
        "reconstruct":       True,
        "simplify_elite":    True,
        "simplify_log_path": simp_log_path,
        "log_lock":          gen_lock,
        "simp_lock":         simp_lock,
    }

    optimizer = SLIM_GSGP(pi_init=pi_init, **slim_params, seed=seed)
    optimizer.solve(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        curr_dataset=f"load_{dataset}",
        **solve_params,
    )


########################################################################################################################
# MAIN
########################################################################################################################

if __name__ == "__main__":
    import pandas as pd

    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)

    gen_log_path  = os.path.join(log_dir, _GEN_LOG_NAME)
    simp_log_path = os.path.join(log_dir, _SIMP_LOG_NAME)

    # ── Skip detection: read completed (algo, dataset, seed) from simplification log ──
    completed_runs = set()
    if os.path.exists(simp_log_path):
        try:
            done_df = pd.read_csv(simp_log_path, usecols=['algo', 'dataset', 'seed'])
            for _, row in done_df.iterrows():
                completed_runs.add((str(row['algo']), str(row['dataset']), int(row['seed'])))
        except Exception as e:
            print(f"Warning: could not read existing simplification log ({e}); no runs skipped.")

    # ── Build task list, filtering already-completed runs ────────────────────────────
    run_id_str = str(uuid.uuid1())

    all_tasks = [
        (dataset, variant_idx, seed)
        for dataset     in DATASETS
        for variant_idx in range(len(VARIANTS))
        for seed        in range(N_RUNS)
    ]
    pending = [
        t for t in all_tasks
        if (VARIANTS[t[1]]['name'], t[0], t[2]) not in completed_runs
    ]
    skipped = len(all_tasks) - len(pending)
    total   = len(pending)
    print(f"Total runs: {len(all_tasks)} | Already done: {skipped} | To run: {total} | Workers: {N_WORKERS}")

    if total == 0:
        print("Nothing to do — all runs already completed.")
        raise SystemExit(0)

    # ── Ensure generation log has a header (only if file is new/empty) ───────────────
    if not os.path.exists(gen_log_path) or os.path.getsize(gen_log_path) == 0:
        with open(gen_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(_GEN_HEADER)

    # ── Ensure simplification log has a header (only if file is new/empty) ───────────
    from utils.logger import _SIMP_HEADER
    if not os.path.exists(simp_log_path) or os.path.getsize(simp_log_path) == 0:
        with open(simp_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(_SIMP_HEADER)

    # ── Create shared locks ───────────────────────────────────────────────────────────
    manager = multiprocessing.Manager()
    gen_lock  = manager.Lock()
    simp_lock = manager.Lock()

    tasks = [
        (dataset, variant_idx, seed, run_id_str, gen_log_path, simp_log_path, gen_lock, simp_lock)
        for dataset, variant_idx, seed in pending
    ]

    # ── Run experiments ───────────────────────────────────────────────────────────────
    t0 = time.time()
    completed = 0

    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        for _ in pool.starmap(run_experiment_worker, tasks):
            completed += 1
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - t0
                print(f"  {completed}/{total} done  ({elapsed/60:.1f} min elapsed)")

    print(f"\nAll done in {(time.time()-t0)/60:.1f} min")
    print(f"Generation log     -> {gen_log_path}")
    print(f"Simplification log -> {simp_log_path}")
