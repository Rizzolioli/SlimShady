"""
Smoke tests for all new modifications:
  - r2 fitness function
  - compute_m_phi
  - inflate_mutation_normalized (NORM2)
  - inflate_mutation_norm1 (NORM1)
  - slim_individual_to_sympy + sympy_m_phi
  - log level 8 + simplify_elite end-to-end (tiny run)
  - log_simplification / merge_simplification_logs
  - geometry study operators (math only, no matplotlib)

Run from the project root: python main/test_smoke.py
"""

import os
import sys
import tempfile
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import numpy as np

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def run(name, fn):
    try:
        fn()
        print(f"[{PASS}] {name}")
    except Exception as e:
        print(f"[{FAIL}] {name}")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# 1. r2
# ──────────────────────────────────────────────────────────────────────────────

def test_r2():
    from evaluators.fitness_functions import r2
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert abs(float(r2(y, y)) - 1.0) < 1e-5, "r2(y,y) should be 1"
    assert float(r2(y, torch.zeros_like(y))) < 0, "r2 with zero pred should be < 0"


# ──────────────────────────────────────────────────────────────────────────────
# 2. compute_m_phi on a trivial individual
# ──────────────────────────────────────────────────────────────────────────────

def test_compute_m_phi_trivial():
    from utils.utils import compute_m_phi
    from algorithms.GSGP.representations.tree import Tree
    from algorithms.SLIM_GSGP.representations.individual import Individual

    FUNCTIONS = {
        'add':      {'function': lambda x, y: torch.add(x, y), 'arity': 2},
        'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
        'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
        'divide':   {'function': lambda x, y: torch.div(x, y), 'arity': 2},
    }
    Tree.FUNCTIONS = FUNCTIONS
    Tree.TERMINALS = {'x0': 0}
    Tree.CONSTANTS = {}

    # Size-1 individual: single terminal 'x0'
    t = Tree(structure='x0', train_semantics=torch.ones(5), test_semantics=None, reconstruct=True)
    ind = Individual(collection=[t],
                     train_semantics=torch.ones(5).unsqueeze(0),
                     test_semantics=None,
                     reconstruct=True)

    m_phi, ell, no, nnao, nnaoc = compute_m_phi(ind, FUNCTIONS)
    assert ell == 1,  f"ell should be 1, got {ell}"
    assert no  == 0,  f"no should be 0,  got {no}"
    assert nnao == 0, f"nnao should be 0, got {nnao}"
    assert abs(m_phi - (79.1 - 0.2 * 1)) < 1e-5, f"m_phi wrong: {m_phi}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. NORM2 inflate mutator
# ──────────────────────────────────────────────────────────────────────────────

def _make_tiny_setup():
    """Returns (FUNCTIONS, TERMINALS, CONSTANTS, X_train, X_test, y_train, y_test, ind)."""
    from utils.utils import protected_div
    from algorithms.GP.operators.initializers import rhh
    from algorithms.GSGP.representations.tree import Tree
    from algorithms.GP.representations.tree import Tree as GP_Tree
    from algorithms.SLIM_GSGP.representations.individual import Individual

    FUNCTIONS = {
        'add':      {'function': lambda x, y: torch.add(x, y), 'arity': 2},
        'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
        'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
        'divide':   {'function': lambda x, y: protected_div(x, y), 'arity': 2},
    }
    TERMINALS = {'x0': 0, 'x1': 1}
    CONSTANTS = {
        'constant_2': lambda x: torch.tensor(2.0),
        'constant__1': lambda x: torch.tensor(-1.0),
    }
    # Both Tree classes must share FUNCTIONS/TERMINALS/CONSTANTS (mirrors SLIM_GSGP.__init__)
    Tree.FUNCTIONS    = FUNCTIONS
    Tree.TERMINALS    = TERMINALS
    Tree.CONSTANTS    = CONSTANTS
    GP_Tree.FUNCTIONS = FUNCTIONS
    GP_Tree.TERMINALS = TERMINALS
    GP_Tree.CONSTANTS = CONSTANTS

    torch.manual_seed(0)
    X_train = torch.randn(20, 2)
    X_test  = torch.randn(8,  2)
    y_train = torch.randn(20)
    y_test  = torch.randn(8)

    # Minimal initial individual: single terminal tree
    t = Tree(structure='x0',
             train_semantics=X_train[:, 0],
             test_semantics=X_test[:, 0],
             reconstruct=True)
    ind = Individual(collection=[t],
                     train_semantics=X_train[:, 0].unsqueeze(0),
                     test_semantics=X_test[:, 0].unsqueeze(0),
                     reconstruct=True)
    return FUNCTIONS, TERMINALS, CONSTANTS, X_train, X_test, y_train, y_test, ind


def test_inflate_norm2():
    from algorithms.SLIM_GSGP.operators.mutators import inflate_mutation_normalized
    FUNCTIONS, TERMINALS, CONSTANTS, X_train, X_test, _, _, ind = _make_tiny_setup()

    inflate = inflate_mutation_normalized(FUNCTIONS, TERMINALS, CONSTANTS, operator='sum')
    ms = 1.0
    offspring = inflate(ind, ms, X_train, max_depth=3, X_test=X_test)

    assert offspring.size == 2, f"Expected size 2, got {offspring.size}"
    assert offspring.train_semantics.shape[0] == 2
    assert offspring.test_semantics.shape[0]  == 2
    # alpha scales training diff to [-1,1]
    block = offspring.collection[1]
    alpha = block.structure[0].alpha
    tr1, tr2 = block.structure[1], block.structure[2]
    diff = tr1.train_semantics - tr2.train_semantics
    scaled = alpha * diff
    assert scaled.abs().max().item() <= 1.0 + 1e-5, \
        f"alpha*diff exceeds [-1,1] on training: max={scaled.abs().max()}"


def test_inflate_norm1():
    from algorithms.SLIM_GSGP.operators.mutators import inflate_mutation_norm1
    FUNCTIONS, TERMINALS, CONSTANTS, X_train, X_test, _, _, ind = _make_tiny_setup()

    inflate = inflate_mutation_norm1(FUNCTIONS, TERMINALS, CONSTANTS, operator='sum')
    ms = 1.0
    offspring = inflate(ind, ms, X_train, max_depth=3, X_test=X_test)

    assert offspring.size == 2, f"Expected size 2, got {offspring.size}"
    block = offspring.collection[1]
    variator = block.structure[0]
    assert hasattr(variator, 't_min'),   "t_min attribute missing on variator"
    assert hasattr(variator, 't_range'), "t_range attribute missing on variator"
    # normalised training output should be in [-1, 1]
    tr1 = block.structure[1]
    t_min   = variator.t_min
    t_range = variator.t_range
    norm = 2 * (tr1.train_semantics - t_min) / t_range - 1
    assert norm.abs().max().item() <= 1.0 + 1e-5, \
        f"NORM1 training output exceeds [-1,1]: max={norm.abs().max()}"


# ──────────────────────────────────────────────────────────────────────────────
# 3b. NORMROB inflate mutator
# ──────────────────────────────────────────────────────────────────────────────

def test_inflate_normrob():
    from algorithms.SLIM_GSGP.operators.mutators import inflate_mutation_normrob
    from utils.utils import protected_div
    from algorithms.GP.operators.initializers import rhh
    from algorithms.GSGP.representations.tree import Tree
    from algorithms.GP.representations.tree import Tree as GP_Tree
    from algorithms.SLIM_GSGP.representations.individual import Individual

    FUNCTIONS = {
        'add':      {'function': lambda x, y: torch.add(x, y), 'arity': 2},
        'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
        'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
        'divide':   {'function': lambda x, y: protected_div(x, y), 'arity': 2},
    }
    TERMINALS = {'x0': 0, 'x1': 1}
    CONSTANTS = {'constant_2': lambda x: torch.tensor(2.0), 'constant__1': lambda x: torch.tensor(-1.0)}
    Tree.FUNCTIONS = FUNCTIONS; Tree.TERMINALS = TERMINALS; Tree.CONSTANTS = CONSTANTS
    GP_Tree.FUNCTIONS = FUNCTIONS; GP_Tree.TERMINALS = TERMINALS; GP_Tree.CONSTANTS = CONSTANTS

    torch.manual_seed(42)
    # Use 200 samples so Q99 = 99th percentile is well-defined (not the max)
    X_train = torch.randn(200, 2)
    X_test  = torch.randn(50,  2)

    t = Tree(structure='x0', train_semantics=X_train[:, 0],
             test_semantics=X_test[:, 0], reconstruct=True)
    ind = Individual(collection=[t],
                     train_semantics=X_train[:, 0].unsqueeze(0),
                     test_semantics=X_test[:, 0].unsqueeze(0),
                     reconstruct=True)

    for sc in ('q99', 'iqr', 'mad'):
        inflate = inflate_mutation_normrob(FUNCTIONS, TERMINALS, CONSTANTS, operator='sum', scale=sc)
        offspring = inflate(ind, 1.0, X_train, max_depth=3, X_test=X_test)
        assert offspring.size == 2, f"[{sc}] Expected size 2, got {offspring.size}"
        block = offspring.collection[1]
        alpha = block.structure[0].alpha
        tr1, tr2 = block.structure[1], block.structure[2]
        diff = tr1.train_semantics - tr2.train_semantics
        # Q99 guarantees ≥99% of training steps in [-1,1]; IQR/MAD do not claim this
        if sc == 'q99':
            pct = (torch.abs(torch.tensor(alpha) * diff) <= 1.0 + 1e-5).float().mean().item()
            assert pct >= 0.99, f"[q99] Only {pct*100:.1f}% of training steps in [-1,1]"
        assert alpha > 0, f"[{sc}] alpha must be positive, got {alpha}"


# ──────────────────────────────────────────────────────────────────────────────
# 4. slim_individual_to_sympy + sympy_m_phi
# ──────────────────────────────────────────────────────────────────────────────

def test_sympy_conversion_trivial():
    import sympy as sp
    from utils.utils import slim_individual_to_sympy, sympy_m_phi
    from algorithms.GSGP.representations.tree import Tree
    from algorithms.SLIM_GSGP.representations.individual import Individual
    from utils.utils import protected_div

    FUNCTIONS = {
        'add':      {'function': lambda x, y: torch.add(x, y), 'arity': 2},
        'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
        'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
        'divide':   {'function': lambda x, y: protected_div(x, y), 'arity': 2},
    }
    TERMINALS = {'x0': 0, 'x1': 1}
    CONSTANTS = {'constant_2': lambda x: torch.tensor(2.0)}
    Tree.FUNCTIONS = FUNCTIONS
    Tree.TERMINALS = TERMINALS
    Tree.CONSTANTS = CONSTANTS

    # Individual with a simple tree: add(x0, x1)
    t = Tree(structure=('add', 'x0', 'x1'),
             train_semantics=torch.zeros(5),
             test_semantics=None,
             reconstruct=True)
    ind = Individual(collection=[t],
                     train_semantics=torch.zeros(5).unsqueeze(0),
                     test_semantics=None,
                     reconstruct=True)

    expr = slim_individual_to_sympy(ind, FUNCTIONS, TERMINALS, CONSTANTS, operator='sum')
    assert isinstance(expr, sp.Basic), "Expected a SymPy expression"
    x0, x1 = sp.Symbol('x0'), sp.Symbol('x1')
    assert expr.equals(x0 + x1), f"Expected x0+x1, got {expr}"

    m_phi, ell, no, nnao, nnaoc = sympy_m_phi(expr)
    # add(x0, x1): 1 op, 2 leaves → ell=3, no=1
    assert ell == 3, f"ell should be 3, got {ell}"
    assert no  == 1, f"no should be 1, got {no}"
    assert nnao == 0


def test_sympy_m_phi_simple():
    import sympy as sp
    from utils.utils import sympy_m_phi
    x = sp.Symbol('x')
    # Just a symbol: ell=1, no=0
    m, ell, no, nnao, nnaoc = sympy_m_phi(x)
    assert ell == 1 and no == 0 and nnao == 0
    assert abs(m - (79.1 - 0.2)) < 1e-5

    # exp(x): 1 op (exp, non-arith) + 1 leaf → ell=2, no=1, nnao=1, nnaoc=1
    m2, ell2, no2, nnao2, nnaoc2 = sympy_m_phi(sp.exp(x))
    assert ell2 == 2 and no2 == 1 and nnao2 == 1 and nnaoc2 == 1


# ──────────────────────────────────────────────────────────────────────────────
# 5. log_simplification + merge_simplification_logs
# ──────────────────────────────────────────────────────────────────────────────

def test_simplification_logger():
    import csv
    from utils.logger import log_simplification, merge_simplification_logs

    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = os.path.join(tmpdir, "simp1.csv")
        p2 = os.path.join(tmpdir, "simp2.csv")
        final = os.path.join(tmpdir, "final_simp.csv")

        log_simplification(p1, run_info=["SLIM+NORM2", "run-1", "concrete"], seed=0,
                           before_metrics=(10, 77.1, 4, 0, 0),
                           after_metrics=(6,  78.3, 2, 0, 0),
                           simplified_ok=True, simp_time=1.2,
                           test_rmse=0.5, test_mae=0.4, test_r2=0.9,
                           genotype_before='x0 + x1**2', genotype_after='x0 + x1**2')
        log_simplification(p2, run_info=["SLIM+2SIG", "run-1", "concrete"], seed=1,
                           before_metrics=(20, 73.0, 8, 2, 1),
                           after_metrics=(20, 73.0, 8, 2, 1),
                           simplified_ok=False, simp_time=60.0,
                           test_rmse=0.6, test_mae=0.5, test_r2=0.85,
                           genotype_before='x0*x1 + x2', genotype_after='x0*x1 + x2')

        merge_simplification_logs([p1, p2], final)

        assert os.path.exists(final), "merged file not created"
        with open(final) as f:
            rows = list(csv.reader(f))
        assert rows[0][0] == 'algo',    f"Header missing: {rows[0]}"
        assert len(rows) == 3,          f"Expected 3 rows (header+2), got {len(rows)}"
        assert rows[1][0] == 'SLIM+NORM2'
        assert rows[2][0] == 'SLIM+2SIG'
        assert not os.path.exists(p1), "temp file 1 not cleaned up"
        assert not os.path.exists(p2), "temp file 2 not cleaned up"


# ──────────────────────────────────────────────────────────────────────────────
# 6. End-to-end tiny SLIM run: log=8, simplify_elite=True
# ──────────────────────────────────────────────────────────────────────────────

def test_end_to_end_slim():
    import csv
    from utils.utils import protected_div, get_best_min
    from evaluators.fitness_functions import rmse
    from algorithms.GP.operators.initializers import rhh
    from algorithms.SLIM_GSGP.operators.selection_algorithms import tournament_selection_min_slim
    from algorithms.SLIM_GSGP.operators.mutators import inflate_mutation_norm1, deflate_mutation
    from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
    from algorithms.GSGP.representations.tree import Tree

    FUNCTIONS = {
        'add':      {'function': lambda x, y: torch.add(x, y), 'arity': 2},
        'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
        'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
        'divide':   {'function': lambda x, y: protected_div(x, y), 'arity': 2},
    }
    TERMINALS = {'x0': 0, 'x1': 1}
    CONSTANTS = {
        'constant_2':  lambda x: torch.tensor(2.0),
        'constant__1': lambda x: torch.tensor(-1.0),
    }

    torch.manual_seed(42)
    X_train = torch.randn(30, 2)
    X_test  = torch.randn(10, 2)
    y_train = torch.randn(30)
    y_test  = torch.randn(10)

    pi_init = {
        'init_pop_size': 10,
        'init_depth':    3,
        'FUNCTIONS': FUNCTIONS,
        'TERMINALS': TERMINALS,
        'CONSTANTS': CONSTANTS,
        'p_c': 0,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        gen_log  = os.path.join(tmpdir, "gen.csv")
        simp_log = os.path.join(tmpdir, "simp.csv")

        optimizer = SLIM_GSGP(
            pi_init=pi_init,
            initializer=rhh,
            selector=tournament_selection_min_slim(2),
            ms=lambda: 1.0,
            inflate_mutator=inflate_mutation_norm1(FUNCTIONS, TERMINALS, CONSTANTS, operator='sum'),
            deflate_mutator=deflate_mutation,
            crossover=None,
            p_xo=0, p_m=1,
            pop_size=10, p_inflate=0.5, p_deflate=0.5,
            copy_parent=None,
            operator='sum', two_trees=False,
            find_elit_func=get_best_min,
            settings_dict={"p_test": 0.2},
            seed=0,
        )
        optimizer.solve(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            curr_dataset="smoke_test",
            run_info=["SLIM+NORM1", "smoke", "synthetic"],
            n_iter=5,
            log=8, verbose=0, test_elite=True,
            log_path=gen_log,
            ffunction=rmse,
            max_depth=None, n_elites=1, reconstruct=True,
            simplify_elite=True,
            simplify_log_path=simp_log,
        )

        # Generation log: expect 6 rows (gen 0..5)
        assert os.path.exists(gen_log), "generation log not created"
        with open(gen_log) as f:
            gen_rows = list(csv.reader(f))
        assert len(gen_rows) == 6, f"Expected 6 gen rows, got {len(gen_rows)}"

        # Simplification log: expect 1 row, no header (raw temp file)
        assert os.path.exists(simp_log), "simplification log not created"
        with open(simp_log) as f:
            simp_rows = list(csv.reader(f))
        assert len(simp_rows) == 1, f"Expected 1 simp row, got {len(simp_rows)}"
        assert simp_rows[0][0] == 'SLIM+NORM1', f"run_info[0] wrong: {simp_rows[0][0]}"


# ──────────────────────────────────────────────────────────────────────────────
# 7. Geometry study operator math (no matplotlib)
# ──────────────────────────────────────────────────────────────────────────────

def test_geometry_operators():
    T = np.array([10.0, 10.0])

    def norm1_sum(r, ms):
        rmin, rmax = r.min(), r.max()
        rrange = max(rmax - rmin, 1e-8)
        return T + ms * (2 * (r - rmin) / rrange - 1)

    def norm2_sum(r1, r2, ms):
        diff = r1 - r2
        scale = max(abs(diff.min()), diff.max())
        alpha = 1.0 / scale if scale != 0.0 else 1.0
        return T + ms * alpha * diff

    np.random.seed(0)
    r  = np.random.uniform(-10, 10, size=2)
    r1 = np.random.uniform(-10, 10, size=2)
    r2 = np.random.uniform(-10, 10, size=2)

    out1 = norm1_sum(r,  ms=1.0)
    assert out1.shape == (2,), f"NORM1 output shape wrong: {out1.shape}"
    # normalised term in [-1,1], so output in [T-1, T+1] = [9,11]
    assert (out1 >= 9 - 1e-5).all() and (out1 <= 11 + 1e-5).all(), \
        f"NORM1 offspring out of expected range: {out1}"

    out2 = norm2_sum(r1, r2, ms=1.0)
    assert out2.shape == (2,), f"NORM2 output shape wrong: {out2.shape}"
    # alpha*diff in [-1,1], so offspring in [T-1, T+1]
    diff = r1 - r2
    scale = max(abs(diff.min()), diff.max())
    alpha = 1.0 / scale
    scaled = alpha * diff
    assert (np.abs(scaled) <= 1.0 + 1e-5).all(), \
        f"NORM2 alpha*diff exceeds [-1,1]: {scaled}"


# ──────────────────────────────────────────────────────────────────────────────
# 8. nested_nodes/depth calculators for new variators
# ──────────────────────────────────────────────────────────────────────────────

def test_tree_utils_new_variators():
    from algorithms.SLIM_GSGP.operators.mutators import (
        inflate_mutation_normalized, inflate_mutation_norm1,
    )
    FUNCTIONS, TERMINALS, CONSTANTS, X_train, X_test, _, _, ind = _make_tiny_setup()

    for norm_fn, name in [
        (inflate_mutation_normalized(FUNCTIONS, TERMINALS, CONSTANTS, 'sum'), 'NORM2-sum'),
        (inflate_mutation_normalized(FUNCTIONS, TERMINALS, CONSTANTS, 'mul'), 'NORM2-mul'),
        (inflate_mutation_norm1(FUNCTIONS, TERMINALS, CONSTANTS, 'sum'),      'NORM1-sum'),
        (inflate_mutation_norm1(FUNCTIONS, TERMINALS, CONSTANTS, 'mul'),      'NORM1-mul'),
    ]:
        offs = norm_fn(ind, 1.0, X_train, max_depth=3, X_test=X_test)
        assert offs.nodes_count > 0, f"{name}: nodes_count is 0"
        assert offs.depth > 0,       f"{name}: depth is 0"


# ──────────────────────────────────────────────────────────────────────────────
# Run all
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running smoke tests...\n")
    run("r2 function",                      test_r2)
    run("compute_m_phi trivial individual", test_compute_m_phi_trivial)
    run("NORM2 inflate (alpha scaling)",    test_inflate_norm2)
    run("NORM1 inflate (min-max scaling)",  test_inflate_norm1)
    run("NORMROB inflate (robust q99/iqr/mad)", test_inflate_normrob)
    run("SymPy conversion (trivial tree)",  test_sympy_conversion_trivial)
    run("sympy_m_phi (symbol + exp)",       test_sympy_m_phi_simple)
    run("log_simplification + merge",       test_simplification_logger)
    run("end-to-end SLIM log=8 + simplify", test_end_to_end_slim)
    run("geometry study operators (math)",  test_geometry_operators)
    run("nested_nodes/depth new variators", test_tree_utils_new_variators)
    print()
