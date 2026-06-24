"""
main_baselines.py — GP baseline runners for comparison with SLIM-GSGP.

Baselines (install with: python main/install_baselines.py)
---------
  gplearn      sklearn-compatible tree GP          pip install gplearn
  operon       state-of-the-art GP                 pip install operon-sklearn
  pygpgomea    GP-GOMEA (linkage-learning GP)       pip install pygpgomea
  pysr         PySR / SymbolicRegression.jl        pip install pysr  (+Julia)
  itea         ITEA (Interaction-Transformation EA) pip install itea-sklearn

Execution budget: pop=100 × 2000 generations = 200 000 evaluations (matches SLIM-GSGP).
Train/test splits: same pre-split tensors as main_slim_normalized.py.

After fitting, the final model expression is SymPy-simplified (cancel → simplify,
20-second thread timeout) and ell/m_phi before and after are recorded.

Output
------
  main/log/results_baselines.csv
  columns: algo, dataset, seed,
           test_rmse, test_mae, test_r2,
           m_phi_before, ell_before, no_before, nnao_before, nnaoc_before,
           m_phi_after,  ell_after,
           simp_ok, simp_time_s,
           runtime_s

Run from project root:
    python main/main_baselines.py
"""

import os
import sys
import csv
import time
import threading
import multiprocessing

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
DATASETS       = ["toxicity", "concrete", "instanbul", "ppb", "resid_build_sale_price", "energy"]
N_RUNS         = 30
N_WORKERS      = max(1, min(os.cpu_count() - 1, 8))
POP_SIZE       = 100
N_GENS         = 2000
SIMP_TIMEOUT_S = 20     # per-model simplification timeout (thread-based)

_LOG_NAME = "results_baselines.csv"
_HEADER   = [
    "algo", "dataset", "seed",
    "test_rmse", "test_mae", "test_r2",
    "m_phi_before", "ell_before", "no_before", "nnao_before", "nnaoc_before",
    "m_phi_after",  "ell_after",
    "simp_ok", "simp_time_s",
    "runtime_s",
]


# ── SymPy utilities ────────────────────────────────────────────────────────────

def _sympy_m_phi(expr):
    from utils.utils import sympy_m_phi
    return sympy_m_phi(expr)    # (m_phi, ell, no, nnao, nnaoc)


def _simplify_thread(expr, timeout=SIMP_TIMEOUT_S):
    """
    Run cancel → simplify in a daemon thread.
    Returns (simplified_expr, success).
    Thread-based (not subprocess) so it is safe to call from Pool workers.
    Caveat: if SymPy hangs, the thread lingers until the process exits (daemon).
    """
    import sympy as sp
    result = {"expr": None, "ok": False}

    def _worker():
        try:
            e = sp.cancel(expr)
            e = sp.simplify(e)
            result["expr"] = e
            result["ok"]   = True
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    return result["expr"], result["ok"]


def _apply_simp_filter(m_phi_b, ell_b, expr_simplified):
    """
    Analysis-side filter: if simplified expression is larger, revert.
    Returns (m_phi_after, ell_after, simp_ok).
    """
    if expr_simplified is None:
        return m_phi_b, ell_b, False
    try:
        m_a, ell_a, _, _, _ = _sympy_m_phi(expr_simplified)
        return max(m_a, m_phi_b), min(ell_a, ell_b), True
    except Exception:
        return m_phi_b, ell_b, False


# ── SymPy expression extractors (one per baseline) ────────────────────────────

def _gplearn_to_sympy(estimator, n_features):
    """
    Convert a gplearn program to a SymPy expression via eval.
    gplearn's __str__ produces prefix notation: add(mul(X0, X1), 1.0)
    """
    import sympy as sp
    syms = {f"X{i}": sp.Symbol(f"x{i}") for i in range(n_features)}
    ns = {
        **syms,
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,       # structural only — no protection needed
        "sqrt": sp.sqrt,
        "log":  lambda x: sp.log(sp.Abs(x)),
        "abs":  sp.Abs,
        "max":  sp.Max,
        "min":  sp.Min,
        "neg":  lambda x: -x,
        "inv":  lambda x: sp.Integer(1) / x,
        "sin":  sp.sin,
        "cos":  sp.cos,
        "tan":  sp.tan,
    }
    try:
        return eval(str(estimator._program), {"__builtins__": {}}, ns)
    except Exception:
        return None


def _gplearn_m_phi(estimator):
    """Compute M_phi directly from the flat program node list (no SymPy needed)."""
    try:
        from gplearn.functions import _Function as _GPF
        prog   = estimator._program.program
        ell    = len(prog)
        _ARITH = {"add", "sub", "mul", "div"}
        no     = sum(1 for n in prog if isinstance(n, _GPF))
        nnao   = sum(1 for n in prog if isinstance(n, _GPF) and n.name not in _ARITH)
        nnaoc  = 1 if nnao > 0 else 0
        m_phi  = 79.1 - 0.2 * ell - 0.5 * no - 3.4 * nnao - 4.5 * nnaoc
        return m_phi, ell, no, nnao, nnaoc
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def _operon_to_sympy(estimator):
    import sympy as sp
    try:
        expr_str = estimator.get_model_string(precision=16)
        return sp.sympify(expr_str)
    except Exception:
        return None


def _pysr_to_sympy(estimator):
    try:
        return estimator.sympy()
    except Exception:
        return None


def _itea_to_sympy(estimator):
    import sympy as sp
    try:
        return sp.sympify(str(estimator))
    except Exception:
        return None


def _gpgomea_to_sympy(estimator):
    import sympy as sp
    try:
        return sp.sympify(str(estimator))
    except Exception:
        return None


# ── Baseline factories ─────────────────────────────────────────────────────────
# Each factory returns (algo_name, make_fn, sympy_fn)
#   make_fn(seed)              -> unfitted estimator
#   sympy_fn(estimator, X_tr)  -> SymPy expr | None

def _register_gplearn():
    from gplearn.genetic import SymbolicRegressor

    def make(seed):
        return SymbolicRegressor(
            population_size=POP_SIZE,
            generations=N_GENS,
            tournament_size=2,
            function_set=("add", "sub", "mul", "div"),
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            metric="rmse",
            parsimony_coefficient=0.0,
            random_state=seed,
            n_jobs=1,
            verbose=0,
        )

    def sympy_fn(est, X_tr):
        return _gplearn_to_sympy(est, X_tr.shape[1])

    return ("GPLearn", make, sympy_fn)


def _register_operon():
    from operon.sklearn import SymbolicRegressor as OperonSR

    def make(seed):
        return OperonSR(
            allowed_symbols="add,sub,mul,div,square,cube,sqrt,cbrt,log,exp",
            population_size=POP_SIZE,
            max_generations=N_GENS,
            max_evaluations=POP_SIZE * N_GENS,
            random_state=seed,
            n_threads=1,
        )

    return ("Operon", make, lambda est, X: _operon_to_sympy(est))


def _register_gpgomea():
    from pygpgomea import GPGOMEARegressor

    def make(seed):
        return GPGOMEARegressor(
            budget=POP_SIZE * N_GENS,
            use_ims=True,
            random_state=seed,
            verbose=False,
        )

    return ("GP-GOMEA", make, lambda est, X: _gpgomea_to_sympy(est))


def _register_pysr():
    from pysr import PySRRegressor

    def make(seed):
        return PySRRegressor(
            niterations=N_GENS,
            populations=POP_SIZE,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[],
            verbosity=0,
            random_state=seed,
            deterministic=True,
            procs=0,
            multithreading=False,
        )

    return ("PySR", make, lambda est, X: _pysr_to_sympy(est))


def _register_itea():
    from itea.regression import ITEA_regressor

    def make(seed):
        return ITEA_regressor(
            gens=N_GENS,
            popsize=POP_SIZE,
            random_state=seed,
            verbose=False,
        )

    return ("ITEA", make, lambda est, X: _itea_to_sympy(est))


_FACTORIES = [
    ("gplearn",   _register_gplearn),
    ("operon",    _register_operon),
    ("pygpgomea", _register_gpgomea),
    ("pysr",      _register_pysr),
    ("itea",      _register_itea),
]


def _discover_baselines():
    available = []
    for lib, factory_fn in _FACTORIES:
        try:
            reg = factory_fn()
            available.append(reg)
            print(f"  [OK]  {reg[0]}")
        except ImportError:
            print(f"  [--]  {lib}  (not installed — run: python main/install_baselines.py)")
        except Exception as e:
            print(f"  [!!]  {lib}  ({e})")
    return available


# ── Per-task worker ────────────────────────────────────────────────────────────

def _run_one(algo_name, make_fn, sympy_fn, dataset, seed, log_path, lock):
    import numpy as np
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from datasets.data_loader import load_preloaded

    _EMPTY = [""] * (len(_HEADER) - 3)   # placeholders for all metric columns

    try:
        X_tr, y_tr = load_preloaded(dataset, seed=seed + 1, training=True,  X_y=True)
        X_te, y_te = load_preloaded(dataset, seed=seed + 1, training=False, X_y=True)
        X_tr = X_tr.numpy().astype(np.float64)
        y_tr = y_tr.numpy().astype(np.float64)
        X_te = X_te.numpy().astype(np.float64)
        y_te = y_te.numpy().astype(np.float64)

        estimator = make_fn(seed)
        t0        = time.time()
        estimator.fit(X_tr, y_tr)
        runtime   = time.time() - t0

        y_pred = estimator.predict(X_te)
        rmse   = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae    = float(mean_absolute_error(y_te, y_pred))
        r2     = float(r2_score(y_te, y_pred))

        # ── M_phi before simplification ───────────────────────────────────────
        if algo_name == "GPLearn":
            m_phi_b, ell_b, no_b, nnao_b, nnaoc_b = _gplearn_m_phi(estimator)
            expr = sympy_fn(estimator, X_tr)
        else:
            expr = sympy_fn(estimator, X_tr)
            if expr is not None:
                m_phi_b, ell_b, no_b, nnao_b, nnaoc_b = _sympy_m_phi(expr)
            else:
                m_phi_b = ell_b = no_b = nnao_b = nnaoc_b = np.nan

        # ── Simplification ────────────────────────────────────────────────────
        simp_ok   = False
        simp_time = 0.0
        m_phi_a   = m_phi_b
        ell_a     = ell_b

        if expr is not None and not (isinstance(ell_b, float) and np.isnan(ell_b)):
            t_s = time.time()
            simplified, simp_ok = _simplify_thread(expr, timeout=SIMP_TIMEOUT_S)
            simp_time = round(time.time() - t_s, 2)
            m_phi_a, ell_a, simp_ok = _apply_simp_filter(m_phi_b, ell_b, simplified)

        def _fmt(v):
            if isinstance(v, float) and np.isnan(v):
                return ""
            if isinstance(v, (int, np.integer)):
                return int(v)
            return round(float(v), 4) if v != "" else ""

        row = [
            algo_name, dataset, seed,
            round(rmse, 6), round(mae, 6), round(r2, 6),
            _fmt(m_phi_b), _fmt(ell_b), _fmt(no_b), _fmt(nnao_b), _fmt(nnaoc_b),
            _fmt(m_phi_a), _fmt(ell_a),
            int(simp_ok), simp_time,
            round(runtime, 2),
        ]

    except Exception as e:
        row = [algo_name, dataset, seed] + _EMPTY
        print(f"  ERROR {algo_name} {dataset} seed={seed}: {e}")

    with lock:
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Discovering available baselines …")
    baselines = _discover_baselines()

    if not baselines:
        print("No baselines available. Run: python main/install_baselines.py")
        sys.exit(1)

    log_dir  = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, _LOG_NAME)

    import pandas as pd
    completed = set()
    if os.path.exists(log_path):
        try:
            done = pd.read_csv(log_path, usecols=["algo", "dataset", "seed"])
            for _, r in done.iterrows():
                completed.add((str(r["algo"]), str(r["dataset"]), int(r["seed"])))
        except Exception:
            pass

    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(_HEADER)

    tasks = [
        (name, make_fn, sympy_fn, dataset, seed)
        for name, make_fn, sympy_fn in baselines
        for dataset in DATASETS
        for seed    in range(N_RUNS)
        if (name, dataset, seed) not in completed
    ]
    total   = len(tasks)
    skipped = len(baselines) * len(DATASETS) * N_RUNS - total
    print(f"\nTotal: {total}  |  Skipped: {skipped}  |  Workers: {N_WORKERS}")

    if total == 0:
        print("Nothing to do.")
        sys.exit(0)

    manager    = multiprocessing.Manager()
    lock       = manager.Lock()
    pool_tasks = [(n, mf, sf, ds, sd, log_path, lock)
                  for n, mf, sf, ds, sd in tasks]

    t0 = time.time()
    done_n = 0

    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        for _ in pool.starmap(_run_one, pool_tasks):
            done_n += 1
            if done_n % 10 == 0 or done_n == total:
                print(f"  {done_n}/{total}  ({(time.time()-t0)/60:.1f} min)")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min  →  {log_path}")
