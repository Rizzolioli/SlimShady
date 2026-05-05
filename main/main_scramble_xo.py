import os
import sys
import uuid
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# project root on path for worker processes (Windows spawn starts a fresh interpreter)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.logger import log_settings

########################################################################################################################

# CONFIGURATION

########################################################################################################################

N_JOBS = os.cpu_count()
n_runs = 5

# data_loaders = ["ppb"]
data_loaders = ["toxicity", "concrete", "instanbul", "ppb", "resid_build_sale_price", "energy"]

variants = [
    # (sig, ttrees, op, gsgp)
    # (True, True, "mul", False),   # SLIM*2SIG
    (True,  True,  "sum", False),   # SLIM+2SIG
    (False, False, "mul", False),   # SLIM*ABS
    (False, False, "sum", False),   # SLIM+ABS
    (True,  False, "mul", False),   # SLIM*1SIG
    (True,  False, "sum", False),   # SLIM+1SIG
]

_algo_names = {
    (True,  False, "mul", True):  "GSGP*1SIG",
    (False, False, "mul", True):  "GSGP*ABS",
    (True,  True,  "sum", True):  "GSGP",
    (True,  False, "mul", False): "SLIM*1SIG",
    (False, False, "mul", False): "SLIM*ABS",
    (True,  True,  "sum", False): "SLIM+2SIG",
    (True,  False, "sum", False): "SLIM+1SIG",
    (False, False, "sum", False): "SLIM+ABS",
}

head_xo_freq_values = [None, 10, 50, 100, 500]

# ms stored as (lo, hi) — closures are not picklable across spawn processes
_dataset_params = {
    "toxicity": {"p_inflate": 0.1, "ms_lo": 0.0, "ms_hi": 0.1},
    "concrete": {"p_inflate": 0.5, "ms_lo": 0.0, "ms_hi": 0.3},
    "other":    {"p_inflate": 0.3, "ms_lo": 0.0, "ms_hi": 1.0},
}

_LOG_PATH      = os.path.join(os.path.dirname(__file__), "log", "results_scramble_xo_05052026.csv")
_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "log", "settings.csv")

########################################################################################################################

# WORKER  (must be module-level so pickle can find it on Windows spawn)

########################################################################################################################

def run_one(task):
    (loader, sig, ttrees, op, head_xo_freq, seed,
     algo, unique_run_id, log_path, p_inflate, ms_lo, ms_hi, project_root) = task

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import time
    import torch
    from utils.utils import protected_div, get_best_min, get_terminals, generate_random_uniform
    from evaluators.fitness_functions import rmse
    from algorithms.GP.operators.initializers import rhh
    from algorithms.GSGP.operators.crossover_operators import geometric_crossover
    from algorithms.SLIM_GSGP.operators.selection_algorithms import tournament_selection_min_slim
    from algorithms.SLIM_GSGP.operators.mutators import inflate_mutation, deflate_mutation
    from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
    from datasets.data_loader import load_preloaded

    FUNCTIONS = {
        'add':      {'function': lambda x, y: torch.add(x, y), 'arity': 2},
        'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
        'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
        'divide':   {'function': lambda x, y: protected_div(x, y), 'arity': 2},
    }
    CONSTANTS = {
        'constant_2':  lambda x: torch.tensor(2).float(),
        'constant_3':  lambda x: torch.tensor(3).float(),
        'constant_4':  lambda x: torch.tensor(4).float(),
        'constant_5':  lambda x: torch.tensor(5).float(),
        'constant__1': lambda x: torch.tensor(-1).float(),
    }

    t0 = time.time()
    TERMINALS = get_terminals(loader, seed + 1)
    X_train, y_train = load_preloaded(loader, seed=seed + 1, training=True,  X_y=True)
    X_test,  y_test  = load_preloaded(loader, seed=seed + 1, training=False, X_y=True)

    pi_init = {
        'init_pop_size': 100,
        'init_depth':    6,
        'FUNCTIONS':     FUNCTIONS,
        'CONSTANTS':     CONSTANTS,
        'TERMINALS':     TERMINALS,
        'p_c':           0,
    }

    optimizer = SLIM_GSGP(
        pi_init=pi_init,
        initializer=rhh,
        selector=tournament_selection_min_slim(2),
        crossover=geometric_crossover,
        ms=generate_random_uniform(ms_lo, ms_hi),
        inflate_mutator=inflate_mutation(
            FUNCTIONS=FUNCTIONS, TERMINALS=TERMINALS, CONSTANTS=CONSTANTS,
            two_trees=ttrees, operator=op, sig=sig,
        ),
        deflate_mutator=deflate_mutation,
        p_xo=0, p_m=1,
        pop_size=100,
        settings_dict={"p_test": 0.2},
        find_elit_func=get_best_min,
        p_inflate=p_inflate,
        p_deflate=1 - p_inflate,
        copy_parent=None,
        operator=op,
        two_trees=ttrees,
        seed=seed,
    )
    optimizer.solve(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        curr_dataset=f"load_{loader}",
        run_info=[algo, unique_run_id, loader],
        elitism=True,
        log=8,
        verbose=1,
        test_elite=True,
        log_path=log_path,
        ffunction=rmse,
        n_iter=2000,
        max_depth=None,
        n_elites=1,
        reconstruct=True,
        head_xo_freq=head_xo_freq,
    )

    elapsed = time.time() - t0
    return (f"[{loader}] {algo} seed={seed} "
            f"train={float(optimizer.elite.fitness):.4f} "
            f"test={float(optimizer.elite.test_fitness):.4f} "
            f"time={elapsed:.1f}s")


########################################################################################################################

# ENTRY POINT

########################################################################################################################

if __name__ == '__main__':
    unique_run_id = uuid.uuid1()

    tasks = []
    for loader in data_loaders:
        dp = _dataset_params.get(loader, _dataset_params["other"])
        for (sig, ttrees, op, gsgp) in variants:
            algo_base = _algo_names[(sig, ttrees, op, gsgp)]
            for head_xo_freq in head_xo_freq_values:
                algo = f'{algo_base}_head_xo{head_xo_freq}'
                for seed in range(n_runs):
                    tasks.append((
                        loader, sig, ttrees, op, head_xo_freq, seed,
                        algo, unique_run_id, _LOG_PATH,
                        dp["p_inflate"], dp["ms_lo"], dp["ms_hi"],
                        _PROJECT_ROOT,
                    ))

    print(f"Submitting {len(tasks)} tasks on {N_JOBS} workers...")
    wall0 = time.time()

    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {executor.submit(run_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                print(f"[{i}/{len(tasks)}] {fut.result()}")
            except Exception as exc:
                print(f"[{i}/{len(tasks)}] FAILED [{t[0]}] {t[6]} seed={t[5]}: {exc}")

    print(f"\nAll runs complete — {time.time() - wall0:.1f}s total")

    log_settings(
        path=_SETTINGS_PATH,
        settings_dict=[
            {"n_runs": n_runs, "n_iter": 2000, "log": 8},
            {"pop_size": 100, "init_depth": 6, "p_xo": 0, "elitism": True},
            {"head_xo_freq_values": str(head_xo_freq_values),
             "variants": str([_algo_names[v] for v in variants])},
            {"data_loaders": str(data_loaders), "TERMINALS": "N/A"},
        ],
        unique_run_id=unique_run_id,
    )
