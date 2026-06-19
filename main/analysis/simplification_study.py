"""
simplification_study.py — SymPy simplification effect on 10-inflate individuals.

For each SLIM variant (same list as mutation_mphi_study):
  1. Create N_TREES random single-block GP individuals.
  2. Apply the variant's inflate mutation INFLATE_STEPS times.
  3. Convert the resulting individual to a SymPy expression.
  4. Simplify with a per-individual timeout.
  5. Record nodes_count and M_phi before and after simplification
     (applying the analysis-side filter: if simplified is larger/worse, revert).

Tasks are parallelised with ThreadPoolExecutor. Each thread may spawn an
mp.Process for the SymPy timeout (threads are not daemonic — no restriction).

Output: tables printed to stdout + CSV in main/analysis/log/
Run from project root:
    python main/analysis/simplification_study.py
"""

import os
import sys
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import torch

from algorithms.GSGP.representations.tree import Tree
from algorithms.GP.representations.tree import Tree as GP_Tree
from algorithms.SLIM_GSGP.representations.individual import Individual
from algorithms.SLIM_GSGP.operators.mutators import (
    inflate_mutation,
    inflate_mutation_normalized,
    inflate_mutation_norm1,
    inflate_mutation_normrob,
    inflate_mutation_norm12,
    inflate_mutation_normfix,
)
from utils.utils import (
    get_random_tree, compute_m_phi, protected_div,
    slim_individual_to_sympy, sympy_m_phi,
)

# ── Configuration ──────────────────────────────────────────────────────────────

N_TREES       = 50
INFLATE_STEPS = 10
MAX_DEPTH     = 6
MS            = 1.0
N_FEAT        = 5
N_SAMPLES     = 100
SIMP_TIMEOUT  = 30        # seconds per individual
SEED          = 42
N_WORKERS     = max(1, min(os.cpu_count() - 1, 8))

LOG_DIR = os.path.join(os.path.dirname(__file__), "log")
os.makedirs(LOG_DIR, exist_ok=True)

# ── Variant registry (same as mutation_mphi_study) ────────────────────────────
VARIANTS = [
    ("SLIM+1SIG",    False, "sum", None,      "1-tree / sum"),
    ("SLIM+ABS",     False, "sum", None,      "1-tree / sum"),
    ("SLIM+NORM1",   False, "sum", "norm1",   "1-tree / sum"),
    ("SLIM+NORMFIX", False, "sum", "normfix", "1-tree / sum"),
    ("SLIM*1SIG",    False, "mul", None,      "1-tree / mul"),
    ("SLIM*ABS",     False, "mul", None,      "1-tree / mul"),
    ("SLIM*NORM1",   False, "mul", "norm1",   "1-tree / mul"),
    ("SLIM*NORMFIX", False, "mul", "normfix", "1-tree / mul"),
    ("SLIM+2SIG",    True,  "sum", None,      "2-tree / sum"),
    ("SLIM+NORM2",   True,  "sum", "norm2",   "2-tree / sum"),
    ("SLIM+NORMROB", True,  "sum", "normrob", "2-tree / sum"),
    ("SLIM+NORM12",  True,  "sum", "norm12",  "2-tree / sum"),
    ("SLIM*2SIG",    True,  "mul", None,      "2-tree / mul"),
    ("SLIM*NORM2",   True,  "mul", "norm2",   "2-tree / mul"),
    ("SLIM*NORMROB", True,  "mul", "normrob", "2-tree / mul"),
    ("SLIM*NORM12",  True,  "mul", "norm12",  "2-tree / mul"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_mutator(name, two_trees, op, norm, functions, terminals, constants, y_train=None):
    sig = name.endswith("SIG")
    if norm == "norm1":
        return inflate_mutation_norm1(functions, terminals, constants, operator=op)
    if norm == "norm2":
        return inflate_mutation_normalized(functions, terminals, constants, operator=op)
    if norm == "normrob":
        return inflate_mutation_normrob(functions, terminals, constants, operator=op, scale='q99')
    if norm == "norm12":
        return inflate_mutation_norm12(functions, terminals, constants, operator=op)
    if norm == "normfix":
        y_np = y_train.numpy() if y_train is not None else np.zeros(1)
        c_val = float(np.median(y_np))
        s_val = max(float((y_np.max() - y_np.min()) / 2), 1e-8)
        return inflate_mutation_normfix(functions, terminals, constants,
                                        operator=op, c=c_val, s=s_val)
    return inflate_mutation(functions, terminals, constants,
                            two_trees=two_trees, operator=op, sig=sig)


def _sympy_worker(expr_str, conn):
    """Run in a subprocess: expand → cancel → simplify, send result back via pipe."""
    import sympy as sp
    try:
        expr = sp.sympify(expr_str)
        # expr = sp.expand(expr)    # flatten / collect like terms
        expr = sp.cancel(expr)    # cancel common rational factors
        simplified = sp.simplify(expr)  # final heuristic pass on reduced expr
        conn.send(('ok', str(simplified)))
    except Exception as e:
        conn.send(('error', str(e)))
    finally:
        conn.close()


def simplify_with_timeout(expr, timeout=SIMP_TIMEOUT):
    """Simplify a SymPy expression in a subprocess with a hard timeout."""
    import sympy as sp
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=_sympy_worker, args=(str(expr), child_conn))
    p.start()
    child_conn.close()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None, False
    if parent_conn.poll():
        status, result = parent_conn.recv()
        if status == 'ok':
            try:
                return sp.sympify(result), True
            except Exception:
                return None, False
    return None, False


# ── Per-task worker (called by thread pool) ───────────────────────────────────

def _run_task(name, group, op, base_ind, ind_idx, mutator,
              functions, terminals, constants, X_train):
    """Inflate base_ind INFLATE_STEPS times, then SymPy-simplify."""
    ind = base_ind
    for _ in range(INFLATE_STEPS):
        ind = mutator(ind, MS, X_train,
                      max_depth=MAX_DEPTH, p_c=0,
                      X_test=None, reconstruct=True)

    m_b, ell_b, _, _, _ = compute_m_phi(ind, functions)
    nodes_before = ind.nodes_count

    try:
        expr_before = slim_individual_to_sympy(
            ind, functions, terminals, constants, operator=op)
    except Exception:
        return {
            'variant': name, 'group': group, 'ind_idx': ind_idx,
            'nodes_before': nodes_before, 'ell_before': ell_b, 'm_phi_before': m_b,
            'nodes_after': nodes_before, 'ell_after': ell_b, 'm_phi_after': m_b,
            'simplified_ok': False, 'simp_time_s': 0.0,
        }

    t0 = time.time()
    simplified, ok = simplify_with_timeout(expr_before, timeout=SIMP_TIMEOUT)
    simp_time = time.time() - t0

    if ok and simplified is not None:
        m_a, ell_a, _, _, _ = sympy_m_phi(simplified)
        ell_final   = min(ell_a, ell_b)
        m_phi_final = max(m_a, m_b)
    else:
        ell_final   = ell_b
        m_phi_final = m_b

    return {
        'variant':       name,
        'group':         group,
        'ind_idx':       ind_idx,
        'nodes_before':  nodes_before,
        'ell_before':    ell_b,
        'm_phi_before':  m_b,
        'nodes_after':   nodes_before - (ell_b - ell_final),
        'ell_after':     ell_final,
        'm_phi_after':   m_phi_final,
        'simplified_ok': ok,
        'simp_time_s':   round(simp_time, 2),
    }


# ── Main study ────────────────────────────────────────────────────────────────

def run_study():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    FUNCTIONS = {
        'add':      {'function': torch.add,     'arity': 2},
        'subtract': {'function': torch.sub,     'arity': 2},
        'multiply': {'function': torch.mul,     'arity': 2},
        'divide':   {'function': protected_div, 'arity': 2},
    }
    TERMINALS = {f'x{i}': i for i in range(N_FEAT)}
    CONSTANTS = {}

    Tree.FUNCTIONS = FUNCTIONS
    Tree.TERMINALS = TERMINALS
    Tree.CONSTANTS = CONSTANTS
    GP_Tree.FUNCTIONS = FUNCTIONS
    GP_Tree.TERMINALS = TERMINALS
    GP_Tree.CONSTANTS = CONSTANTS

    X_train = torch.randn(N_SAMPLES, N_FEAT)
    y_train = torch.randn(N_SAMPLES)

    # Generate N_TREES random base individuals (single block)
    print(f"Generating {N_TREES} random single-block individuals …", flush=True)
    base_individuals = []
    for _ in range(N_TREES):
        t = get_random_tree(MAX_DEPTH, FUNCTIONS, TERMINALS, CONSTANTS,
                            inputs=X_train, p_c=0, logistic=False)
        ind = Individual(collection=[t], train_semantics=None,
                         test_semantics=None, reconstruct=True)
        ind.calculate_semantics(X_train)
        base_individuals.append(ind)

    # Build all mutators upfront (one per variant)
    print("Building mutators …", flush=True)
    mutators = {
        name: _build_mutator(name, two_trees, op, norm,
                              FUNCTIONS, TERMINALS, CONSTANTS, y_train=y_train)
        for name, two_trees, op, norm, _ in VARIANTS
    }

    # Submit all (variant × individual) tasks to the thread pool
    total     = len(VARIANTS) * N_TREES
    counter   = [0]
    lock      = threading.Lock()
    records   = []
    t_start   = time.time()

    print(f"Submitting {total} tasks to {N_WORKERS} workers …", flush=True)

    futures = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        for name, two_trees, op, norm, group in VARIANTS:
            for i, base_ind in enumerate(base_individuals):
                fut = executor.submit(
                    _run_task,
                    name, group, op, base_ind, i, mutators[name],
                    FUNCTIONS, TERMINALS, CONSTANTS, X_train,
                )
                futures[fut] = name

        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            with lock:
                counter[0] += 1
                done = counter[0]
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t_start
                ok_so_far = sum(r['simplified_ok'] for r in records)
                print(f"  {done}/{total}  simp_ok={ok_so_far}  "
                      f"({elapsed/60:.1f} min)", flush=True)

    df = pd.DataFrame(records)

    # ── Summary table ─────────────────────────────────────────────────────────
    def _fmt(s):
        return f"{s.median():.1f} ({s.quantile(0.25):.1f}, {s.quantile(0.75):.1f})"

    rows = []
    for name, _, op, norm, group in VARIANTS:
        sub = df[df['variant'] == name]
        ok_rate = sub['simplified_ok'].mean() * 100
        rows.append({
            'Variant':             name,
            'Group':               group,
            'ell_before med(IQR)': _fmt(sub['ell_before']),
            'ell_after  med(IQR)': _fmt(sub['ell_after']),
            'Δell med':            round((sub['ell_after'] - sub['ell_before']).median(), 1),
            'Δm_phi med':          round((sub['m_phi_after'] - sub['m_phi_before']).median(), 2),
            'simp_ok%':            round(ok_rate, 1),
            'simp_time med(s)':    round(sub['simp_time_s'].median(), 1),
        })
    summary = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print(f"Simplification study — {INFLATE_STEPS} inflate steps, {N_TREES} individuals/variant, "
          f"{N_WORKERS} workers")
    print("=" * 80)
    try:
        from tabulate import tabulate
        print(tabulate(summary, headers='keys', tablefmt='simple', showindex=False))
    except ImportError:
        print(summary.to_string(index=False))

    raw_path     = os.path.join(LOG_DIR, "simplification_study_raw.csv")
    summary_path = os.path.join(LOG_DIR, "simplification_study_summary.csv")
    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nRaw data -> {raw_path}")
    print(f"Summary  -> {summary_path}")


if __name__ == "__main__":
    run_study()
