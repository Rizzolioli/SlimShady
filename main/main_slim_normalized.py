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
    {"name": "SLIM+2SIG", "sig": True,  "two_trees": True,  "op": "sum", "norm": None},
    {"name": "SLIM*2SIG", "sig": True,  "two_trees": True,  "op": "mul", "norm": None},
    {"name": "SLIM*ABS",  "sig": False, "two_trees": False, "op": "mul", "norm": None},
    {"name": "SLIM+ABS",  "sig": False, "two_trees": False, "op": "sum", "norm": None},
    {"name": "SLIM*1SIG", "sig": True,  "two_trees": False, "op": "mul", "norm": None},
    {"name": "SLIM+1SIG", "sig": True,  "two_trees": False, "op": "sum", "norm": None},
    {"name": "SLIM+NORM1","sig": False, "two_trees": False, "op": "sum", "norm": "norm1"},
    {"name": "SLIM*NORM1","sig": False, "two_trees": False, "op": "mul", "norm": "norm1"},
    {"name": "SLIM+NORM2","sig": False, "two_trees": True,  "op": "sum", "norm": "norm2"},
    {"name": "SLIM*NORM2","sig": False, "two_trees": True,  "op": "mul", "norm": "norm2"},
]

# Dataset-specific inflate probability
_DATASET_P_INFLATE = {
    "toxicity": 0.1,
    "concrete": 0.5,
}
_DEFAULT_P_INFLATE = 0.3

########################################################################################################################
# WORKER  (module-level so multiprocessing can pickle it)
########################################################################################################################

def run_experiment_worker(dataset, variant_idx, seed, run_id_str, log_dir):
    """Run a single (dataset, variant, seed) experiment and return the temp log path."""
    import numpy as np
    import torch
    import uuid as _uuid

    # Re-import everything locally so each subprocess is self-contained
    from utils.utils import protected_div, get_terminals, get_best_min
    from evaluators.fitness_functions import rmse
    from algorithms.GP.operators.initializers import rhh
    from algorithms.GSGP.operators.crossover_operators import geometric_crossover
    from algorithms.SLIM_GSGP.operators.selection_algorithms import tournament_selection_min_slim
    from algorithms.SLIM_GSGP.operators.mutators import (
        inflate_mutation, deflate_mutation,
        inflate_mutation_normalized, inflate_mutation_norm1,
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
    tmp_log = os.path.join(log_dir, f"tmp_{_uuid.uuid4().hex}.csv")

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
        "elitism":       True,
        "log":           8,
        "verbose":       0,
        "test_elite":    True,
        "log_path":      tmp_log,
        "run_info":      [variant["name"], run_id_str, dataset],
        "ffunction":     rmse,
        "n_iter":        2000,
        "max_depth":     None,
        "n_elites":      1,
        "reconstruct":   True,
        "simplify_elite": True,
    }

    optimizer = SLIM_GSGP(pi_init=pi_init, **slim_params, seed=seed)
    optimizer.solve(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        curr_dataset=f"load_{dataset}",
        **solve_params,
    )

    return tmp_log


########################################################################################################################
# MAIN
########################################################################################################################

if __name__ == "__main__":
    unique_run_id = uuid.uuid1()
    run_id_str = str(unique_run_id)

    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)

    tasks = [
        (dataset, variant_idx, seed, run_id_str, log_dir)
        for dataset    in DATASETS
        for variant_idx in range(len(VARIANTS))
        for seed       in range(N_RUNS)
    ]
    total = len(tasks)
    print(f"Total runs: {total}  |  Workers: {N_WORKERS}")

    t0 = time.time()
    completed = 0
    tmp_files = []

    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        for tmp_path in pool.starmap(run_experiment_worker, tasks):
            tmp_files.append(tmp_path)
            completed += 1
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - t0
                print(f"  {completed}/{total} done  ({elapsed/60:.1f} min elapsed)")

    # Merge temp files into a single results CSV
    final_log = os.path.join(log_dir, f"results_normalized_{unique_run_id}.csv")
    with open(final_log, "w", newline="") as outf:
        for fname in tmp_files:
            if os.path.exists(fname):
                with open(fname, "r") as inf:
                    outf.write(inf.read())
                os.remove(fname)

    print(f"\nAll done in {(time.time()-t0)/60:.1f} min")
    print(f"Results → {final_log}")
