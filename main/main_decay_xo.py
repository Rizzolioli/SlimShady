"""
main_decay_xo.py
================
Experiment with cosine² p_xo decay schedule.

p_xo decays from 0.7 (gen 0) to 0.3 (gen n_iter) following:
    p_xo(t) = 0.3 + 0.4 * (0.5 * (1 + cos(pi * t / n_iter))) ** 2

Config: pop=500, n_iter=400, max_head_depth=17
Variants: SLIM+2SIG, SLIM*ABS, SLIM*1SIG
Datasets: all 6 benchmarks, 5 seeds

Log: level 9 (operator counts per gen + sem_gen companion file)

Log CSV columns (log=9):
  0:algo  1:run_id  2:dataset  3:seed  4:gen
  5:train_fit  6:timing  7:nodes  8:test_fit  9:nodes_count
  10:inflate_n  11:inflate_improved
  12:deflate_n  13:deflate_improved
  14:xo_n       15:xo_improved
  16:log_level
"""

import os
import sys
import uuid
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── CONFIG ────────────────────────────────────────────────────────────────────

N_JOBS = os.cpu_count()
n_runs = 5

data_loaders = ["toxicity", "concrete", "instanbul", "ppb",
                "resid_build_sale_price", "energy"]

variants = [
    (True,  True,  "sum", False),   # SLIM+2SIG
    (False, False, "mul", False),   # SLIM*ABS
    (True,  False, "mul", False),   # SLIM*1SIG
]

_algo_names = {
    (True,  True,  "sum", False): "SLIM+2SIG",
    (False, False, "mul", False): "SLIM*ABS",
    (True,  False, "mul", False): "SLIM*1SIG",
}

pop_size       = 500
n_iter         = 400
max_head_depth = 17

# cosine² schedule: 0.7 → 0.3
_PXO_HI = 0.7
_PXO_LO = 0.3

_dataset_params = {
    "toxicity": {"p_inflate": 0.1, "ms_lo": 0.0, "ms_hi": 0.1},
    "concrete": {"p_inflate": 0.5, "ms_lo": 0.0, "ms_hi": 0.3},
    "other":    {"p_inflate": 0.3, "ms_lo": 0.0, "ms_hi": 1.0},
}

_date     = time.strftime("%d%m%Y")
_LOG_PATH = os.path.join(os.path.dirname(__file__), "log",
                         f"results_decay_xo_{_date}.csv")

# ── COMPLETED-RUN DETECTION ───────────────────────────────────────────────────

def load_completed_runs(log_path):
    if not os.path.exists(log_path):
        return {}
    try:
        df = pd.read_csv(log_path, header=None, usecols=[0, 2, 3, 4])
        df.columns = ["algo", "loader", "seed", "gen"]
        df["seed"] = df["seed"].astype(int)
        return df.groupby(["algo", "loader", "seed"])["gen"].max().to_dict()
    except Exception:
        return {}

# ── WORKER ────────────────────────────────────────────────────────────────────

def run_one(task):
    (loader, sig, ttrees, op, seed,
     algo, unique_run_id, log_path,
     p_inflate, ms_lo, ms_hi, project_root,
     pxo_hi, pxo_lo) = task

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    sys.setrecursionlimit(10000)

    import math as _math
    import torch
    from utils.utils import protected_div, get_best_min, get_terminals, generate_random_uniform
    from evaluators.fitness_functions import rmse
    from algorithms.GP.operators.initializers import rhh
    from algorithms.GSGP.operators.crossover_operators import geometric_crossover
    from algorithms.SLIM_GSGP.operators.selection_algorithms import tournament_selection_min_slim
    from algorithms.SLIM_GSGP.operators.mutators import inflate_mutation, deflate_mutation
    from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
    from datasets.data_loader import load_preloaded

    def _cosine2_schedule(it, n_iter):
        cos_val = 0.5 * (1.0 + _math.cos(_math.pi * it / n_iter))
        return pxo_lo + (pxo_hi - pxo_lo) * cos_val ** 2

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

    t0       = time.time()
    TERMINALS = get_terminals(loader, seed + 1)
    X_train, y_train = load_preloaded(loader, seed=seed + 1, training=True,  X_y=True)
    X_test,  y_test  = load_preloaded(loader, seed=seed + 1, training=False, X_y=True)

    pi_init = {
        'init_pop_size': pop_size,
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
        p_xo=pxo_hi,          # initial value; schedule overrides from gen 1 onward
        p_m=1 - pxo_hi,
        pop_size=pop_size,
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
        log=9,
        verbose=1,
        test_elite=True,
        log_path=log_path,
        ffunction=rmse,
        n_iter=n_iter,
        max_depth=None,
        n_elites=1,
        reconstruct=True,
        head_xo_freq=None,
        max_head_depth=max_head_depth,
        p_xo_schedule=_cosine2_schedule,
    )

    elapsed = time.time() - t0
    return (f"[{loader}] {algo} seed={seed} "
            f"train={float(optimizer.elite.fitness):.4f} "
            f"test={float(optimizer.elite.test_fitness):.4f} "
            f"time={elapsed:.1f}s")

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # print schedule values for reference
    print("Cosine² p_xo schedule preview:")
    for t in [0, 50, 100, 200, 300, 400]:
        cos_val = 0.5 * (1.0 + math.cos(math.pi * t / n_iter))
        pxo = _PXO_LO + (_PXO_HI - _PXO_LO) * cos_val ** 2
        print(f"  gen {t:>3}: p_xo = {pxo:.3f}")
    print()

    unique_run_id = uuid.uuid1()

    completed = load_completed_runs(_LOG_PATH)
    if completed:
        print(f"Found {len(completed)} already-started run(s) - skipping where complete.")

    tasks = []
    for loader in data_loaders:
        dp = _dataset_params.get(loader, _dataset_params["other"])
        for (sig, ttrees, op, gsgp) in variants:
            algo_base = _algo_names[(sig, ttrees, op, gsgp)]
            algo = f'{algo_base}_pop{pop_size}_iter{n_iter}_decay_cos2'
            for seed in range(n_runs):
                if completed.get((algo, loader, seed), -1) >= n_iter:
                    print(f"  skip [{loader}] {algo} seed={seed}")
                    continue
                tasks.append((
                    loader, sig, ttrees, op, seed,
                    algo, unique_run_id, _LOG_PATH,
                    dp["p_inflate"], dp["ms_lo"], dp["ms_hi"],
                    _PROJECT_ROOT, _PXO_HI, _PXO_LO,
                ))

    print(f"Submitting {len(tasks)} tasks on {N_JOBS} workers...")
    wall0 = time.time()

    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {executor.submit(run_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                result = fut.result()
                print(f"[{i}/{len(tasks)}] {result}")
            except Exception as exc:
                t = futures[fut]
                print(f"[{i}/{len(tasks)}] FAILED {t[0]} {t[5]} seed={t[4]}: {exc}")

    print(f"\nDone in {time.time() - wall0:.1f}s")
    print(f"  Main log  -> {_LOG_PATH}")
    print(f"  Sem_gen   -> {_LOG_PATH[:-4]}_sem_gen.csv")
