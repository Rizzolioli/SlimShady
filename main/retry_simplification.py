#!/usr/bin/env python
"""
Retry SymPy simplification for all rows where simplified_ok == 0.

Root-cause diagnosis: the original slim_gsgp.py worker called
_mp.Process(target=_sympy_simplify_worker, ...) where the target lives inside
slim_gsgp.py. On Windows (spawn method), the child process re-imports the
entire file, which fails due to project-level imports. That is why even
"x418 + x444" timed out — the subprocess never actually ran sp.simplify().

Fix here: a minimal worker defined at module-level that only imports sympy
and operates on plain strings, completely decoupled from the project.

Strategy:
  - Parse genotype_before with sp.sympify()
  - Run sp.simplify() with TIMEOUT_S timeout
  - Count nodes with sympy_m_phi (main process)
  - Update the CSV: ell_after, m_phi_after, no_after, nnao_after, nnaoc_after,
    genotype_after, simplified_ok, simp_time_s
"""
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.utils import sympy_m_phi

SIMP_LOG  = os.path.join(os.path.dirname(__file__), 'log',
                          'results_normalized_simplification.csv')
TIMEOUT_S = 120   # longer than the original 60 s
# ThreadPoolExecutor (not Pool) so threads can spawn subprocesses without
# the "daemonic processes cannot have children" restriction
N_WORKERS = max(1, min(os.cpu_count() - 1, 8))

# ── Worker — no project imports, only sympy ───────────────────────────────────
def _worker(expr_str, conn):
    """Subprocess: parse string → simplify → send result string (or None)."""
    import sympy as sp
    try:
        expr   = sp.sympify(expr_str, locals=None, convert_xor=False)
        result = sp.simplify(expr)
        conn.send(str(result))
    except Exception:
        conn.send(None)
    finally:
        conn.close()


def try_simplify(expr_str):
    """
    Attempt simplification of a SymPy expression string.
    Returns (simplified_str, elapsed_s) or (None, elapsed_s) on timeout/error.
    """
    parent_conn, child_conn = mp.Pipe(duplex=False)
    proc = mp.Process(target=_worker, args=(expr_str, child_conn))
    proc.start()
    child_conn.close()

    t0 = time.time()
    if parent_conn.poll(TIMEOUT_S):
        try:
            result = parent_conn.recv()
            elapsed = time.time() - t0
        except EOFError:
            result, elapsed = None, time.time() - t0
    else:
        proc.kill()
        result, elapsed = None, time.time() - t0

    proc.join()
    parent_conn.close()
    return result, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sympy as sp

    df = pd.read_csv(SIMP_LOG)
    pending_mask = (df['simplified_ok'] == 0) & (df['genotype_before'].str.len() > 0)
    pending = df[pending_mask].copy()

    print(f'Total rows: {len(df)}  |  Pending: {len(pending)}')
    print(f'Timeout per expression: {TIMEOUT_S}s  |  Workers: {N_WORKERS}\n')

    improved  = 0
    unchanged = 0
    failed    = 0
    done      = 0

    indices = pending.index.tolist()
    total   = len(indices)
    t_start = time.time()

    # ThreadPoolExecutor: threads (unlike Pool workers) are not daemonic,
    # so each thread is free to spawn a subprocess for SymPy isolation.
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        future_to_idx = {
            executor.submit(try_simplify, df.at[idx, 'genotype_before']): idx
            for idx in indices
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            done += 1
            try:
                result_str, elapsed = future.result()
            except Exception:
                failed += 1
                continue

            if result_str is None:
                failed += 1
                continue

            ell_b  = df.at[idx, 'ell_before']
            mphi_b = df.at[idx, 'm_phi_before']

            try:
                simplified = sp.sympify(result_str, convert_xor=False)
                mphi_a, ell_a, no_a, nnao_a, nnaoc_a = sympy_m_phi(simplified)
            except Exception:
                failed += 1
                continue

            df.at[idx, 'ell_after']      = int(ell_a)
            df.at[idx, 'm_phi_after']    = float(mphi_a)
            df.at[idx, 'no_after']       = int(no_a)
            df.at[idx, 'nnao_after']     = int(nnao_a)
            df.at[idx, 'nnaoc_after']    = int(nnaoc_a)
            df.at[idx, 'genotype_after'] = result_str
            df.at[idx, 'simplified_ok']  = 1
            df.at[idx, 'simp_time_s']    = round(elapsed, 4)

            if mphi_a >= mphi_b:
                improved += 1
            else:
                unchanged += 1

            if done % 50 == 0 or done == total:
                print(f'  {done}/{total}  improved={improved}  worse_sympy={unchanged}'
                      f'  failed/timeout={failed}  ({(time.time()-t_start)/60:.1f} min)')

    df.to_csv(SIMP_LOG, index=False)
    print(f'\nDone.  improved={improved}  worse_sympy={unchanged}  failed/timeout={failed}')
    print(f'CSV updated -> {SIMP_LOG}')
