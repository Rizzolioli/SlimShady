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
    # Primary: model string captured from C-level stdout during fit().
    # Boost.Python's std::string→Python converter is broken in some builds
    # (get_model() raises SystemError), so we parse Terminate()'s output instead.
    captured = getattr(estimator, "_captured_model_str", None)
    if captured and captured.strip():
        return captured.strip()
    # Fallback: try get_model() in case the Boost.Python build supports it.
    import traceback
    ea = getattr(estimator, "_ea", None)
    for obj, label in [(estimator, "estimator"), (ea, "_ea")]:
        if obj is None:
            continue
        for method_name in ("get_model", "get_model_string"):
            fn = getattr(obj, method_name, None)
            if fn is None:
                continue
            try:
                val = fn()
                if isinstance(val, bytes):
                    val = val.decode()
                if not isinstance(val, str):
                    try:
                        val = str(val)
                    except Exception:
                        continue
                s = val.strip()
                if s and s.lower() not in ("none", "null", ""):
                    return s
            except Exception as exc:
                print(f"  [gpgomea] {label}.{method_name}() raised {type(exc).__name__}: {exc}",
                      flush=True)
                traceback.print_exc()
    return None


def _gpgomea_mphi(estimator):
    """
    Compute M_phi for a fitted GPGOMEARegressor by counting nodes in the
    model string.  Falls back to get_n_nodes() for ell when no string available.

    GP-GOMEA uses compound operator tokens: 'p/' (protected div) written as
    infix inside expressions like (X1p/X2), and function calls like plog(X0),
    sqrt(X0).  Normalise these before tokenising to avoid miscounting.
    """
    import re
    model_str = _gpgomea_model_str(estimator)
    if model_str:
        # Normalise compound GP-GOMEA operators so the tokeniser handles them:
        #   p/  → pdiv  (appears as "X1p/X2" in infix; replace before X→x subs)
        #   aq0.1 → aq01 , aq → aq  (analytic quotient variants — count as arith)
        #   ^2 → square  (square operator written as postfix)
        norm = re.sub(r'p/',   ' pdiv ',   model_str)
        norm = re.sub(r'aq0\.1', ' aq01 ', norm)
        norm = re.sub(r'\^2',  ' square ', norm)

        _ARITH     = {"sum", "add", "mul", "sub", "div", "pdiv",
                      "aq", "aq01", "+", "-", "*", "/"}
        _NON_ARITH = {"sqrt", "psqrt", "plog", "log", "log2", "log10",
                      "exp", "sin", "cos", "abs", "sigmoid", "psin", "pcos",
                      "square"}
        tokens   = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|[+\-*/]', norm)
        no       = sum(1 for t in tokens if t.lower() in _ARITH)
        nnao     = sum(1 for t in tokens if t.lower() in _NON_ARITH)
        nnaoc    = 1 if nnao > 0 else 0
        n_vars   = len(re.findall(r'\b[Xx]\d+\b', norm))
        n_consts = len(re.findall(
            r'(?<![A-Za-z_])\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?(?![A-Za-z_])',
            norm))
        ell   = no + nnao + n_vars + n_consts
        m_phi = 79.1 - 0.2 * ell - 0.5 * no - 3.4 * nnao - 4.5 * nnaoc
        return m_phi, ell, no, nnao, nnaoc

    # Fallback: use get_n_nodes() for ell only
    ea = getattr(estimator, "_ea", None)
    if ea is not None:
        try:
            n = int(ea.get_n_nodes())   # Boost.Python may return size_t, not int
            if n > 0:
                return np.nan, n, np.nan, np.nan, np.nan
        except Exception:
            pass
    return np.nan, np.nan, np.nan, np.nan, np.nan


def _parse_gpgomea_lisp(model_str):
    """
    Parse GP-GOMEA's Lisp-prefix notation — e.g. (+ X0 (* X1 2.5)) — into SymPy.
    Returns None if parsing fails.
    """
    import sympy as sp, re
    tokens = re.findall(r'\(|\)|[^\s()]+', model_str)
    idx = [0]

    _BIN = {
        '+': lambda a, b: a + b, 'add': lambda a, b: a + b,
        '-': lambda a, b: a - b, 'sub': lambda a, b: a - b,
        '*': lambda a, b: a * b, 'mul': lambda a, b: a * b,
        '/': lambda a, b: a / b, 'div': lambda a, b: a / b,
        'pdiv': lambda a, b: a / b,
    }
    _UN = {
        'sqrt':   lambda a: sp.sqrt(sp.Abs(a)),
        'psqrt':  lambda a: sp.sqrt(sp.Abs(a)),
        'log':    lambda a: sp.log(sp.Abs(a)),
        'plog':   lambda a: sp.log(sp.Abs(a)),
        'ln':     lambda a: sp.log(sp.Abs(a)),
        'exp':    sp.exp,
        'sin':    sp.sin,  'psin': sp.sin,
        'cos':    sp.cos,  'pcos': sp.cos,
        'abs':    sp.Abs,
        'neg':    lambda a: -a,
        'square': lambda a: a ** 2,
    }

    def _parse():
        if idx[0] >= len(tokens):
            raise ValueError("EOF")
        tok = tokens[idx[0]]; idx[0] += 1
        if tok != '(':
            m = re.match(r'^[Xx](\d+)$', tok)
            if m:
                return sp.Symbol(f'x{m.group(1)}')
            return sp.Float(tok)
        op = tokens[idx[0]]; idx[0] += 1
        args = []
        while idx[0] < len(tokens) and tokens[idx[0]] != ')':
            args.append(_parse())
        if idx[0] < len(tokens):
            idx[0] += 1
        op_l = op.lower()
        if op_l in _BIN and len(args) == 2:
            return _BIN[op_l](*args)
        if op_l in _UN and len(args) == 1:
            return _UN[op_l](*args)
        if op_l == 'sum':
            return sum(args)
        raise ValueError(f"Unknown: {op!r}/{len(args)} args")

    try:
        expr = _parse()
        return expr if idx[0] == len(tokens) else None
    except Exception:
        return None


def _gpgomea_to_sympy(estimator):
    """Try to convert the GP-GOMEA model string to a SymPy expression."""
    import sympy as sp, re
    model_str = _gpgomea_model_str(estimator)
    if not model_str:
        return None

    # Normalise GP-GOMEA infix operators BEFORE variable substitution so that
    # compound tokens like "X1p/X2" become "X1/X2" and not "x1p/x2".
    s = model_str
    s = s.replace('p/', '/')      # protected division: (X1p/X2) → (X1/X2)
    s = s.replace('^', '**')     # square: (X0)^2 → (X0)**2
    s = re.sub(r'\bX(\d+)\b', r'x\1', s)   # X0 → x0 etc.

    # Collect variable symbols referenced in the expression
    n_vars = max((int(m) for m in re.findall(r'x(\d+)', s)), default=-1) + 1
    local_dict = {f'x{i}': sp.Symbol(f'x{i}') for i in range(max(n_vars, 20))}
    # Protected log: plog(x) = log(|x|)
    local_dict['plog'] = lambda x: sp.log(sp.Abs(x))

    try:
        return sp.sympify(s, locals=local_dict)
    except Exception:
        pass
    # Fall back to custom Lisp-prefix parser (handles prefix notation if any)
    return _parse_gpgomea_lisp(model_str)


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
        # Boost.Python < 1.79 bug: calling GPGOMEA(hp_string) normally goes
        # through CPython's type_call → slot_tp_init, which raises TypeError
        # because Boost.Python's tp_init wrapper returns NoneType instead of None.
        # A ctypes patch to bypass this releases the GIL inside the C-level
        # tp_init callback, which causes a SIGSEGV when np::initialize() runs.
        #
        # Clean fix: bypass type_call entirely with __new__ + explicit __init__.
        #   • GPGOMEA.__new__(GPGOMEA)   — allocates the Python object
        #   • GPGOMEA.__init__(obj, hp)  — calls the Python-level __init__ dict
        #     entry (NOT slot_tp_init), which runs the C++ constructor without
        #     the return-type check → no TypeError, no GIL release, no SIGSEGV.
        #
        # We also set silent=False so Terminate()'s model expression is printed
        # to fd 1, which _run_one captures to get the model string (get_model()
        # raises SystemError due to a separate Boost.Python std::string bug).
        try:
            import pyGPGOMEA as _pgpkg
            _gp_dir = os.path.dirname(_pgpkg.__file__)
            if _gp_dir not in sys.path:
                sys.path.insert(0, _gp_dir)
            import gpgomea as _gpmod
            from pyGPGOMEA.GPGOMEARegressor import GPGOMEARegressor
        except Exception as _pe:
            raise RuntimeError(f"gpgomea import failed: {_pe}") from _pe

        _reg = GPGOMEARegressor.__new__(GPGOMEARegressor)
        for _k, _v in dict(
                time=120, generations=-1, evaluations=-1,
                prob="symbreg", multiobj=False, linearscaling=True,
                functions="+_*_-_p/_sqrt_plog", erc=True,
                classweights=False, gomea=True, gomfos="LT",
                subcross=0.5, submut=0.5, reproduction=0.0,
                sblibtype=False, sbrdo=0.0, sbagx=0.0,
                unifdepthvar=True, tournament=4, elitism=0,
                ims="5_1", syntuniqinit=1000, popsize=500,
                initmaxtreeheight=4, inittype=False,
                maxtreeheight=17, maxsize=1000,
                validation=False, coeffmut=False,
                gomcoeffmutstrat=False, batchsize=False,
                seed=seed, parallel=0, caching=False,
                silent=False,   # stdout kept open so fit() output is capturable
                logtofile=False,
        ).items():
            setattr(_reg, _k, _v)
        _hp = _reg._build_hyperparameters_string()
        _reg._ea = _gpmod.GPGOMEA.__new__(_gpmod.GPGOMEA)
        try:
            _gpmod.GPGOMEA.__init__(_reg._ea, _hp)
        except TypeError:
            pass  # Boost.Python bug: C++ constructor ran OK; ignore spurious error
        return _reg
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
        if algo_key == "pygpgomea":
            # GP-GOMEA's get_model() raises SystemError (Boost.Python std::string
            # bug).  Redirect fd 1 to a temp file so we can parse the model
            # expression that Terminate() prints: "Best solution found:\t<expr>".
            import tempfile as _tf
            _tmp_fd, _tmp_path = _tf.mkstemp(suffix=".txt", prefix="gpgomea_")
            try:
                _saved_fd1 = os.dup(1)
                os.dup2(_tmp_fd, 1)
                os.close(_tmp_fd)
                try:
                    t0      = time.time()
                    estimator.fit(X_tr, y_tr)
                    runtime = time.time() - t0
                    # Flush C stdio (synced with C++ cout by default) while
                    # fd 1 still points to the temp file, so Terminate()'s
                    # buffered output lands in the capture file, not the terminal.
                    import ctypes as _ct2
                    _ct2.CDLL(None).fflush(0)   # fflush(NULL)
                finally:
                    os.dup2(_saved_fd1, 1)
                    os.close(_saved_fd1)
                with open(_tmp_path, "r", errors="replace") as _gp_f:
                    _gp_out = _gp_f.read()
            finally:
                try:
                    os.unlink(_tmp_path)
                except OSError:
                    pass
            # Parse "Best solution found:\t<expr>" and linear scaling coefficients.
            _model_expr = None
            _lin_a = _lin_b = None
            for _ln in _gp_out.splitlines():
                if _model_expr is None and _ln.startswith("Best solution found:\t"):
                    _model_expr = _ln.split("\t", 1)[1].strip()
                elif _ln.startswith("Linear scaling coefficients:"):
                    for _tok in _ln.split("\t"):
                        if _tok.startswith("a="):
                            try: _lin_a = float(_tok[2:])
                            except ValueError: pass
                        elif _tok.startswith("b="):
                            try: _lin_b = float(_tok[2:])
                            except ValueError: pass
            if _model_expr is not None:
                if _lin_a is not None and _lin_b is not None:
                    # Reproduce get_model() format: std::to_string uses %.6f
                    _model_expr = f"{_lin_a:.6f}+{_lin_b:.6f}*({_model_expr})"
                estimator._captured_model_str = _model_expr
            else:
                estimator._captured_model_str = None
        else:
            t0      = time.time()
            estimator.fit(X_tr, y_tr)
            runtime = time.time() - t0

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
            # Token-based M_phi from the C++ model string (handles GP-GOMEA's
            # specific operators: p/, plog, sqrt, ^2).  SymPy is attempted only
            # for the simplification step; its node counts are NOT used to
            # override the token counts because _sympy_nnao only detects 'exp'.
            m_phi_b, ell_b, no_b, nnao_b, nnaoc_b = _gpgomea_mphi(estimator)
            try:
                expr = _to_sympy_expr(algo_key, estimator, X_tr)
            except Exception:
                expr = None
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
    import shutil
    completed = set()
    if os.path.exists(log_path):
        try:
            df_done = pd.read_csv(log_path)
            # Drop rows where RMSE is missing (crashed/empty runs).
            has_rmse_mask = (
                df_done["test_rmse"].notna() &
                (df_done["test_rmse"].astype(str).str.strip() != "")
            )
            n_empty = int((~has_rmse_mask).sum())
            df_done = df_done[has_rmse_mask].copy()
            # Among rows with RMSE, prefer the copy that also has m_phi_before.
            has_mphi_rank = (
                df_done["m_phi_before"].notna() &
                (df_done["m_phi_before"].astype(str).str.strip() != "")
            ).astype(int)
            df_done["_rank"] = has_mphi_rank
            df_done = (df_done
                .sort_values("_rank")
                .drop_duplicates(subset=["algo", "dataset", "seed"], keep="last")
                .drop(columns=["_rank"])
            )
            n_dup = int(has_rmse_mask.sum()) - len(df_done)
            if n_empty > 0 or n_dup > 0:
                shutil.copy(log_path, log_path + ".bak")
                df_done.to_csv(log_path, index=False)
                print(f"  CSV cleaned: removed {n_empty} empty rows, "
                      f"{n_dup} duplicate rows  (backup → {os.path.basename(log_path)}.bak)")
            for _, r in df_done.iterrows():
                has_rmse = pd.notna(r["test_rmse"]) and str(r["test_rmse"]).strip() != ""
                if not has_rmse:
                    continue
                # GP-GOMEA: re-run if m_phi_before was never computed.
                if str(r["algo"]) == "GP-GOMEA":
                    has_mphi = (pd.notna(r.get("m_phi_before", float("nan"))) and
                                str(r.get("m_phi_before", "")).strip() != "")
                    if not has_mphi:
                        continue
                completed.add((str(r["algo"]), str(r["dataset"]), int(r["seed"])))
        except Exception as exc:
            print(f"  Warning: could not read CSV ({exc}); starting fresh.")

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

    # Non-daemonic Processes for GP-GOMEA (Boost.Python needs a fresh interpreter).
    # Run up to N_WORKERS concurrently; hard-cap each seed at 6 minutes.
    _SEED_TIMEOUT = 360
    running = []   # list of (process, start_time, (algo_name, dataset, seed))

    def _reap(running, done_n):
        still = []
        for proc, t_start, meta in running:
            an, ds, sd = meta
            if proc.is_alive():
                if time.time() - t_start > _SEED_TIMEOUT:
                    proc.terminate()
                    proc.join()
                    done_n += 1
                    print(f"  TIMEOUT {an:<10s}  {ds:<25s}  seed={sd}", flush=True)
                    if done_n % 10 == 0 or done_n == total:
                        print(f"  {done_n}/{total}  ({(time.time()-t0)/60:.1f} min)")
                else:
                    still.append((proc, t_start, meta))
            else:
                proc.join()
                done_n += 1
                if proc.exitcode != 0:
                    print(f"  CRASH  {an:<10s}  {ds:<25s}  seed={sd}  "
                          f"exitcode={proc.exitcode}", flush=True)
                if done_n % 10 == 0 or done_n == total:
                    print(f"  {done_n}/{total}  ({(time.time()-t0)/60:.1f} min)")
        return still, done_n

    for args in gomea_tasks:
        while len(running) >= N_WORKERS:
            time.sleep(0.5)
            running, done_n = _reap(running, done_n)
        p = multiprocessing.Process(target=_run_one, args=args)
        p.start()
        ak, an, ds, sd, _lp, _lk = args
        running.append((p, time.time(), (an, ds, sd)))

    while running:
        time.sleep(0.5)
        running, done_n = _reap(running, done_n)

    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        for _ in pool.starmap(_run_one, pool_tasks):
            done_n += 1
            if done_n % 10 == 0 or done_n == total:
                print(f"  {done_n}/{total}  ({(time.time()-t0)/60:.1f} min)")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min  →  {log_path}")
