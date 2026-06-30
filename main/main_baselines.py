"""
main_baselines.py — GP baseline runners for comparison with SLIM-GSGP.

Baselines (install with: python main/install_baselines.py)
---------
  gplearn      sklearn-compatible tree GP          pip install gplearn
  pyoperon     Operon — high-performance GP        pip install pyoperon
  pysr         PySR / SymbolicRegression.jl        pip install pysr  (+Julia)
  pygpgomea    GP-GOMEA (linkage-learning GP)       build from source (install_baselines.py)

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
N_WORKERS      = max(1, min(os.cpu_count() - 1, 10))
POP_SIZE       = 200
N_GENS         = 100
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

def _sympy_ell_no(expr):
    if expr.is_Atom:
        return 1, 0
    n_args = len(expr.args)
    if n_args == 0:
        return 1, 0
    child_ells, child_nos = zip(*[_sympy_ell_no(a) for a in expr.args])
    s_ell = sum(child_ells)
    s_no  = sum(child_nos)
    if n_args == 1:
        return 1 + s_ell, 1 + s_no
    ops_here = n_args - 1
    return ops_here + s_ell, ops_here + s_no


def _sympy_nnao(expr):
    import sympy as sp
    if expr.is_Atom:
        return 0
    is_nnao = 1 if expr.func is sp.exp else 0
    return is_nnao + sum(_sympy_nnao(a) for a in expr.args)


def _sympy_m_phi(expr):
    ell, no = _sympy_ell_no(expr)
    nnao    = _sympy_nnao(expr)
    nnaoc   = 1 if nnao > 0 else 0
    m_phi   = 79.1 - 0.2 * ell - 0.5 * no - 3.4 * nnao - 4.5 * nnaoc
    return m_phi, ell, no, nnao, nnaoc


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
    """
    pyoperon API: get_model_string(model) → infix string using X0, X1, ...
    Map to lowercase x0, x1, ... to match project convention.
    """
    import sympy as sp, re
    try:
        expr_str = estimator.get_model_string(estimator.model_)
        # X0 → x0, X1 → x1, etc.
        expr_str = re.sub(r'\bX(\d+)\b', r'x\1', expr_str)
        return sp.sympify(expr_str)
    except Exception:
        return None


def _pysr_to_sympy(estimator):
    try:
        return estimator.sympy()
    except Exception:
        return None


def _gpgomea_model_str(estimator):
    """Return the GP-GOMEA best-model string, or None if unavailable."""
    ea = getattr(estimator, "_ea", None)
    print(f"  [diag] _ea type={type(ea).__name__}", flush=True)

    # Probe all methods to distinguish int-return (C++ ptr ok) vs str-return (conversion bug)
    for obj_label, obj in [("_ea", ea), ("estimator", estimator)]:
        if obj is None:
            continue
        for method in ("get_n_nodes", "get_evaluations",
                        "get_model", "get_model_string", "get_progress_log"):
            fn = getattr(obj, method, None)
            if fn is None:
                continue
            try:
                val = fn()
                val_repr = repr(val)[:300] if not isinstance(val, str) else repr(val[:300])
                print(f"  [diag] {obj_label}.{method}() -> {type(val).__name__}: {val_repr}", flush=True)
                if method in ("get_model", "get_model_string"):
                    if isinstance(val, bytes):
                        val = val.decode()
                    if val and isinstance(val, str):
                        return val
                if method == "get_progress_log" and isinstance(val, str) and val.strip():
                    # Return last non-empty line as a possible model representation
                    lines = [l.strip() for l in val.splitlines() if l.strip()]
                    if lines:
                        print(f"  [diag] last log line: {lines[-1][:200]}", flush=True)
            except Exception as _e:
                print(f"  [diag] {obj_label}.{method}() raised {type(_e).__name__}: {_e}", flush=True)
    return None


def _gpgomea_mphi(estimator):
    """
    Compute M_phi for a fitted GPGOMEARegressor by counting nodes in the
    model string.  This avoids SymPy entirely (GP-GOMEA uses non-standard
    prefix notation that SymPy cannot parse).
    """
    import re
    model_str = _gpgomea_model_str(estimator)
    if not model_str:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Arithmetic operators (lower-cased for matching)
    _ARITH    = {"sum", "add", "mul", "sub", "div", "pdiv", "+", "-", "*", "/"}
    # Non-arithmetic operators
    _NON_ARITH = {"sqrt", "psqrt", "plog", "log", "log2", "log10",
                  "exp", "sin", "cos", "abs", "sigmoid", "psin", "pcos"}

    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|[+\-*/]', model_str)
    no    = sum(1 for t in tokens if t.lower() in _ARITH)
    nnao  = sum(1 for t in tokens if t.lower() in _NON_ARITH)
    nnaoc = 1 if nnao > 0 else 0

    n_vars   = len(re.findall(r'\b[Xx]\d+\b', model_str))
    n_consts = len(re.findall(
        r'(?<![A-Za-z_])\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?(?![A-Za-z_])',
        model_str))
    ell   = no + nnao + n_vars + n_consts
    m_phi = 79.1 - 0.2 * ell - 0.5 * no - 3.4 * nnao - 4.5 * nnaoc
    return m_phi, ell, no, nnao, nnaoc


def _gpgomea_to_sympy(estimator):
    """Try to convert the GP-GOMEA model string to a SymPy expression."""
    import sympy as sp, re
    model_str = _gpgomea_model_str(estimator)
    if not model_str:
        return None
    # Normalise variable names: X0 → x0
    model_str = re.sub(r'\bX(\d+)\b', r'x\1', model_str)
    try:
        return sp.sympify(model_str)
    except Exception:
        return None


# ── Baseline construction ──────────────────────────────────────────────────────
# Module-level functions so multiprocessing.Pool can pickle task arguments.
# Inner functions / lambdas cannot be pickled → never pass them to pool workers.

_ALGO_META = {
    # algo_key: (display_name, pip_package)
    # "gplearn":   ("GPLearn",  "gplearn"),
    # "pyoperon":  ("Operon",   "pyoperon"),
    "pygpgomea": ("GP-GOMEA", "pyGPGOMEA"),
    # "pysr":      ("PySR",     "pysr"),
}


def _make_estimator(algo_key, seed):
    """Construct an unfitted estimator for the given algo_key and seed."""
    if algo_key == "gplearn":
        from gplearn.genetic import SymbolicRegressor
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
    if algo_key == "pyoperon":
        from pyoperon.sklearn import SymbolicRegressor as OperonSR
        return OperonSR(
            allowed_symbols="add,sub,mul,div,square,sqrtabs,logabs,exp,constant,variable",
            population_size=POP_SIZE,
            generations=N_GENS,
            max_evaluations=POP_SIZE * N_GENS,
            n_threads=1,
            random_state=seed,
        )
    if algo_key == "pygpgomea":
        # Boost.Python < 1.79 bug on Python 3.10+: GPGOMEA.__init__ returns
        # NoneType instead of None, so slot_tp_init raises TypeError even though
        # the C++ ctor already ran.
        #
        # IMPORTANT: a Python subclass of GPGOMEA breaks Boost.Python's C++ pointer
        # extraction, so C++ methods like get_model() raise "error return without
        # exception set".  We therefore use a FACTORY CALLABLE instead of subclassing:
        # the factory creates a real gpgomea.GPGOMEA instance via __new__ + direct
        # __init__, catches the spurious TypeError, and returns the properly
        # initialised object.  Boost.Python can then extract the C++ pointer cleanly.
        try:
            import pyGPGOMEA as _pgpkg, os as _os
            _gp_dir = _os.path.dirname(_pgpkg.__file__)
            if _gp_dir not in sys.path:
                sys.path.insert(0, _gp_dir)
            import gpgomea as _gpmod

            if not getattr(_gpmod, "_init_patched", False):
                _Orig = _gpmod.GPGOMEA

                class _GPGOMEAFactory:
                    """Callable stored as gpgomea.GPGOMEA, returns real C++ instances.

                    Two Boost.Python < 1.79 bugs are fixed here:

                    1. __init__ NoneType bug: GPGOMEA.__init__ returns NoneType
                       instead of None, so slot_tp_init raises TypeError even
                       though the C++ ctor already ran successfully.  We bypass
                       type.__call__ entirely via __new__ + __init__ and swallow
                       that specific TypeError.

                    2. isinstance TypeError → SystemError: Boost.Python method
                       wrappers call isinstance(self, gpgomea.GPGOMEA) before
                       dispatching to C++.  After we store this factory *instance*
                       (not a type) as gpgomea.GPGOMEA, Python's isinstance raises
                       TypeError ("arg 2 must be a type"); Boost.Python clears that
                       error and returns NULL, giving SystemError on every C++ call.
                       __instancecheck__ defined here is found by Python at
                       type(factory_instance).__instancecheck__, making
                       isinstance(real_ea, factory) return True so dispatch works.
                    """

                    def __instancecheck__(self, instance):
                        # isinstance(x, gpgomea.GPGOMEA) where gpgomea.GPGOMEA is
                        # this factory instance: return True for real GPGOMEA objects.
                        return type(instance) is _Orig

                    def __call__(self, *args, **kwargs):
                        obj = _Orig.__new__(_Orig)
                        try:
                            _Orig.__init__(obj, *args, **kwargs)
                        except TypeError as _e:
                            if "should return None" not in str(_e):
                                raise
                        return obj

                _gpmod.GPGOMEA = _GPGOMEAFactory()
                _gpmod._init_patched = True

            # Overwrite the module-level 'GPGOMEA' binding captured at import time
            # by GPGOMEARegressor.py's "from gpgomea import GPGOMEA".
            import importlib as _il
            _reg_mod = _il.import_module("pyGPGOMEA.GPGOMEARegressor")
            if hasattr(_reg_mod, "GPGOMEA"):
                _reg_mod.GPGOMEA = _gpmod.GPGOMEA
        except Exception as _pe:
            print(f"  [warn] gpgomea patch failed: {_pe}", flush=True)

        from pyGPGOMEA import GPGOMEARegressor
        return GPGOMEARegressor(
            time=120,
            generations=-1,
            seed=seed,
            parallel=0,
        )
    if algo_key == "pysr":
        from pysr import PySRRegressor
        return PySRRegressor(
            niterations=N_GENS,
            populations=1,
            parallelism="serial",
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[],
            verbosity=0,
            random_state=seed,
            deterministic=True,
        )
    raise ValueError(f"Unknown algo_key: {algo_key!r}")


def _to_sympy_expr(algo_key, estimator, X_tr):
    """Extract a SymPy expression from a fitted estimator."""
    if algo_key == "gplearn":
        return _gplearn_to_sympy(estimator, X_tr.shape[1])
    if algo_key == "pyoperon":
        return _operon_to_sympy(estimator)
    if algo_key == "pygpgomea":
        return _gpgomea_to_sympy(estimator)
    if algo_key == "pysr":
        return _pysr_to_sympy(estimator)
    return None


def _discover_baselines():
    available = []
    for algo_key, (algo_name, lib) in _ALGO_META.items():
        try:
            # import-only check — do not construct (PySR triggers Julia init)
            if algo_key == "gplearn":
                from gplearn.genetic import SymbolicRegressor      # noqa: F401
            elif algo_key == "pyoperon":
                from pyoperon.sklearn import SymbolicRegressor     # noqa: F401
            elif algo_key == "pygpgomea":
                from pyGPGOMEA import GPGOMEARegressor             # noqa: F401
            elif algo_key == "pysr":
                from pysr import PySRRegressor                     # noqa: F401
            available.append((algo_key, algo_name))
            print(f"  [OK]  {algo_name}")
        except ImportError:
            print(f"  [--]  {lib}  (not installed — run: python main/install_baselines.py)")
        except Exception as e:
            print(f"  [!!]  {lib}  ({e})")
    return available


# ── Per-task worker ────────────────────────────────────────────────────────────

def _run_one(algo_key, algo_name, dataset, seed, log_path, lock):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    print(f"  START  {algo_name:<10s}  {dataset:<25s}  seed={seed}", flush=True)
    _EMPTY = [""] * (len(_HEADER) - 3)   # placeholders for all metric columns

    try:
        # Load pre-split CSVs directly to avoid importing torch (which requires
        # NumPy 1.x C extensions incompatible with NumPy 2.x in worker processes).
        _data_dir = os.path.join(_ROOT, "datasets", "pre_loaded_data")
        _s = seed + 1
        _df_tr = pd.read_csv(
            os.path.join(_data_dir, f"TRAINING_{_s}_{dataset.upper()}.txt"),
            sep=" ", header=None).iloc[:, :-1]
        _df_te = pd.read_csv(
            os.path.join(_data_dir, f"TEST_{_s}_{dataset.upper()}.txt"),
            sep=" ", header=None).iloc[:, :-1]
        X_tr = _df_tr.values[:, :-1].astype(np.float64)
        y_tr = _df_tr.values[:,  -1].astype(np.float64)
        X_te = _df_te.values[:, :-1].astype(np.float64)
        y_te = _df_te.values[:,  -1].astype(np.float64)

        estimator = _make_estimator(algo_key, seed)
        t0        = time.time()
        estimator.fit(X_tr, y_tr)
        runtime   = time.time() - t0
        if algo_name == "GP-GOMEA":
            print(f"  [diag] fit() took {runtime:.1f}s  _ea type={type(getattr(estimator,'_ea',None)).__name__}", flush=True)

        y_pred = np.asarray(estimator.predict(X_te), dtype=np.float64)
        y_pred = np.where(np.isfinite(y_pred), y_pred, np.nanmedian(y_tr))
        rmse   = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae    = float(mean_absolute_error(y_te, y_pred))
        r2     = float(r2_score(y_te, y_pred))

        # ── M_phi before simplification ───────────────────────────────────────
        if algo_name == "GPLearn":
            m_phi_b, ell_b, no_b, nnao_b, nnaoc_b = _gplearn_m_phi(estimator)
            expr = _to_sympy_expr(algo_key, estimator, X_tr)
        elif algo_name == "GP-GOMEA":
            # Count nodes directly from the C++ model string; SymPy parsing
            # attempted additionally for the simplification step.
            m_phi_b, ell_b, no_b, nnao_b, nnaoc_b = _gpgomea_mphi(estimator)
            if np.isnan(m_phi_b):
                raise RuntimeError(
                    "M_phi unavailable: _ea.get_model() returned nothing — "
                    "check that the GP-GOMEA C++ object exposes get_model()")
            expr = _to_sympy_expr(algo_key, estimator, X_tr)
            if expr is not None:
                # If SymPy did parse, prefer its counts (more accurate)
                mp, el, no_, nn, nnc = _sympy_m_phi(expr)
                if not np.isnan(mp):
                    m_phi_b, ell_b, no_b, nnao_b, nnaoc_b = mp, el, no_, nn, nnc
        else:
            expr = _to_sympy_expr(algo_key, estimator, X_tr)
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

        print(f"  DONE   {algo_name:<10s}  {dataset:<25s}  seed={seed}  rmse={rmse:.4f}  {runtime:.0f}s", flush=True)
    except Exception as e:
        row = [algo_name, dataset, seed] + _EMPTY
        print(f"  ERROR  {algo_name:<10s}  {dataset:<25s}  seed={seed}: {e}", flush=True)

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
            done = pd.read_csv(log_path,
                               usecols=["algo", "dataset", "seed",
                                        "test_rmse", "m_phi_before"])
            for _, r in done.iterrows():
                has_rmse = pd.notna(r["test_rmse"]) and str(r["test_rmse"]).strip() != ""
                has_mphi = pd.notna(r["m_phi_before"]) and str(r["m_phi_before"]).strip() != ""
                if not has_rmse:
                    continue
                # GP-GOMEA rows are only complete when M_phi was also recorded.
                if str(r["algo"]) == "GP-GOMEA" and not has_mphi:
                    continue
                completed.add((str(r["algo"]), str(r["dataset"]), int(r["seed"])))
        except Exception:
            pass

    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(_HEADER)

    tasks = [
        (algo_key, algo_name, dataset, seed)
        for algo_key, algo_name in baselines
        for dataset in DATASETS
        for seed    in range(N_RUNS)
        if (algo_name, dataset, seed) not in completed
    ]
    total   = len(tasks)
    skipped = len(baselines) * len(DATASETS) * N_RUNS - total
    print(f"\nTotal: {total}  |  Skipped: {skipped}  |  Workers: {N_WORKERS}")

    if total == 0:
        print("Nothing to do.")
        sys.exit(0)

    manager    = multiprocessing.Manager()
    lock       = manager.Lock()

    # GP-GOMEA's C++ Boost.Python bindings crash when initialized inside a
    # daemon Pool worker. Run each task in its own fresh non-daemonic Process
    # so the dynamic linker loads libboost_python cleanly every time.
    # We also set DYLD_LIBRARY_PATH (Mac) / LD_LIBRARY_PATH (Linux) so the
    # compiled .so finds the conda Boost/OpenMP libraries via RPATH fallback.
    if sys.platform == "darwin":
        _conda_lib = os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib")
        if _conda_lib and os.path.isdir(_conda_lib):
            _prev = os.environ.get("DYLD_LIBRARY_PATH", "")
            if _conda_lib not in _prev:
                os.environ["DYLD_LIBRARY_PATH"] = (
                    _conda_lib + (":" + _prev if _prev else ""))
    elif sys.platform.startswith("linux"):
        _conda_lib = os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib")
        if _conda_lib and os.path.isdir(_conda_lib):
            _prev = os.environ.get("LD_LIBRARY_PATH", "")
            if _conda_lib not in _prev:
                os.environ["LD_LIBRARY_PATH"] = (
                    _conda_lib + (":" + _prev if _prev else ""))
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    gomea_tasks = [(ak, an, ds, sd, log_path, lock)
                   for ak, an, ds, sd in tasks if ak == "pygpgomea"]
    pool_tasks  = [(ak, an, ds, sd, log_path, lock)
                   for ak, an, ds, sd in tasks if ak != "pygpgomea"]

    t0 = time.time()
    done_n = 0

    for args in gomea_tasks:
        # Non-daemonic Process: gets a fresh Python interpreter + clean
        # dynamic-library state, inheriting the env vars set above.
        p = multiprocessing.Process(target=_run_one, args=args)
        p.start()
        p.join(timeout=360)   # 6-minute hard cap per seed
        if p.is_alive():
            p.terminate()
            p.join()
        done_n += 1
        if done_n % 10 == 0 or done_n == total:
            print(f"  {done_n}/{total}  ({(time.time()-t0)/60:.1f} min)")

    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        for _ in pool.starmap(_run_one, pool_tasks):
            done_n += 1
            if done_n % 10 == 0 or done_n == total:
                print(f"  {done_n}/{total}  ({(time.time()-t0)/60:.1f} min)")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min  →  {log_path}")
