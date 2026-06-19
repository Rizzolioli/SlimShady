"""
M_phi delta study: how much interpretability does each inflate mutation cost?

For each of the 10 SLIM inflate variants, generate N_TREES random single-block
individuals, apply one inflate, measure ΔM_phi = M_phi_after − M_phi_before.

M_phi is purely structural, so ms and normalization constants do not affect
the result. A small synthetic X_train is used only to satisfy NORM1/NORM2's
need to compute t_min / t_range / alpha at inflate time.

Output: 5 tables (4 grouped + 1 comparison) printed to stdout and saved as
CSV files in main/analysis/log/.
"""

import os
import sys

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
from utils.utils import get_random_tree, compute_m_phi, protected_div

# ── Configuration ─────────────────────────────────────────────────────────────

N_TREES   = 1000
N_FEAT    = 5
N_SAMPLES = 100
MAX_DEPTH = 6
MS        = 1.0   # scalar; does not affect M_phi (structural metric)
SEED      = 42

# ── Variant registry ──────────────────────────────────────────────────────────
#   (name, two_trees, operator, norm, group)
VARIANTS = [
    ("SLIM+1SIG",  False, "sum", None,    "1-tree / sum"),
    ("SLIM+ABS",   False, "sum", None,    "1-tree / sum"),
    ("SLIM+NORM1", False, "sum", "norm1", "1-tree / sum"),
    ("SLIM*1SIG",  False, "mul", None,    "1-tree / mul"),
    ("SLIM*ABS",   False, "mul", None,    "1-tree / mul"),
    ("SLIM*NORM1", False, "mul", "norm1", "1-tree / mul"),
    ("SLIM+2SIG",    True,  "sum", None,      "2-tree / sum"),
    ("SLIM+NORM2",   True,  "sum", "norm2",   "2-tree / sum"),
    ("SLIM+NORMROB", True,  "sum", "normrob", "2-tree / sum"),
    ("SLIM+NORM12",  True,  "sum", "norm12",   "2-tree / sum"),
    ("SLIM*2SIG",    True,  "mul", None,       "2-tree / mul"),
    ("SLIM*NORM2",   True,  "mul", "norm2",    "2-tree / mul"),
    ("SLIM*NORMROB", True,  "mul", "normrob",  "2-tree / mul"),
    ("SLIM*NORM12",  True,  "mul", "norm12",   "2-tree / mul"),
    ("SLIM+NORMFIX", False, "sum", "normfix",  "1-tree / sum"),
    ("SLIM*NORMFIX", False, "mul", "normfix",  "1-tree / mul"),
]

# Display order for the comparison table
GROUP_ORDER = ["1-tree / sum", "1-tree / mul", "2-tree / sum", "2-tree / mul"]

TABLE_TITLES = {
    "1-tree / sum": "Table 1 — 1-tree inflate, sum operator",
    "1-tree / mul": "Table 2 — 1-tree inflate, mul operator",
    "2-tree / sum": "Table 3 — 2-tree inflate, sum operator",
    "2-tree / mul": "Table 4 — 2-tree inflate, mul operator",
}


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


def _make_summary(sub_df):
    rows = []
    for variant in sub_df['variant'].unique():
        v = sub_df[sub_df['variant'] == variant]['dm_phi']
        e = sub_df[sub_df['variant'] == variant]['dell']
        n = sub_df[sub_df['variant'] == variant]['dno']
        a = sub_df[sub_df['variant'] == variant]['dnnao']
        rows.append({
            'Variant':       variant,
            'median dm_phi': round(v.median(), 3),
            'mean dm_phi':   round(v.mean(),   3),
            'std dm_phi':    round(v.std(),    3),
            'median dell':   round(e.median(), 1),
            'median dno':    round(n.median(), 1),
            'median dnnao':  round(a.median(), 1),
        })
    return pd.DataFrame(rows)


def _make_compact_summary(sub_df):
    """Compact table: median with IQR in parentheses for each metric."""
    rows = []
    for variant in sub_df['variant'].unique():
        v = sub_df[sub_df['variant'] == variant]['dm_phi']
        e = sub_df[sub_df['variant'] == variant]['dell']
        n = sub_df[sub_df['variant'] == variant]['dno']
        a = sub_df[sub_df['variant'] == variant]['dnnao']

        def _fmt(s, dec=2):
            med = s.median()
            q1  = s.quantile(0.25)
            q3  = s.quantile(0.75)
            return f"{med:.{dec}f} ({q1:.{dec}f}, {q3:.{dec}f})"

        rows.append({
            'Variant':               variant,
            'dm_phi med (IQR)':      _fmt(v, 2),
            'dell med (IQR)':        _fmt(e, 1),
            'dno med (IQR)':         _fmt(n, 1),
            'dnnao med (IQR)':       _fmt(a, 1),
        })
    return pd.DataFrame(rows)


def _all_pairwise(pivot, variant_order):
    """Compute raw Wilcoxon p-values for all n*(n-1)/2 pairs.

    Returns dict (vi, vj) -> raw_p  (vi precedes vj in variant_order).
    """
    from scipy.stats import wilcoxon
    pvals = {}
    n = len(variant_order)
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = variant_order[i], variant_order[j]
            diff = pivot[vi].values - pivot[vj].values
            if (diff == 0).all():
                pvals[(vi, vj)] = 1.0
            else:
                try:
                    _, p = wilcoxon(diff)
                    pvals[(vi, vj)] = float(p)
                except ValueError:
                    pvals[(vi, vj)] = 1.0
    return pvals


def _bonferroni(pvals_dict):
    """Return new dict with Bonferroni-corrected p-values (capped at 1.0)."""
    n = len(pvals_dict)
    return {k: min(v * n, 1.0) for k, v in pvals_dict.items()}


def _symbol(corrected_p, a_median, b_median, alpha=0.05):
    if corrected_p >= alpha:
        return '~'
    return '+' if a_median > b_median else '-'


def _build_matrices(pivot, variant_order, corrected):
    """Return (p_matrix, sym_matrix) as DataFrames — upper triangle only."""
    n = len(variant_order)
    p_cells   = {v: {v2: '' for v2 in variant_order} for v in variant_order}
    sym_cells = {v: {v2: '' for v2 in variant_order} for v in variant_order}
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = variant_order[i], variant_order[j]
            cp = corrected[(vi, vj)]
            sym = _symbol(cp, pivot[vi].median(), pivot[vj].median())
            p_cells[vi][vj]   = f"{cp:.3e}"
            sym_cells[vi][vj] = sym
    p_df   = pd.DataFrame(p_cells,   index=variant_order).T
    sym_df = pd.DataFrame(sym_cells, index=variant_order).T
    return p_df, sym_df


def _print_table(title, tbl_df, index=False):
    print(f"\n{'-' * 70}")
    print(title)
    print('-' * 70)
    try:
        from tabulate import tabulate
        print(tabulate(tbl_df, headers='keys', tablefmt='simple', showindex=index))
    except ImportError:
        print(tbl_df.to_string(index=index))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_study():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train = torch.randn(N_SAMPLES, N_FEAT)
    y_train = torch.randn(N_SAMPLES)   # synthetic; only c,s for NORMFIX depend on it

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

    # ── 1. Generate N_TREES single-block individuals ──────────────────────────
    print(f"Generating {N_TREES} random single-block individuals …", flush=True)
    individuals = []
    for _ in range(N_TREES):
        t = get_random_tree(MAX_DEPTH, FUNCTIONS, TERMINALS, CONSTANTS,
                            inputs=X_train, p_c=0, logistic=False)
        ind = Individual(collection=[t],
                         train_semantics=None,
                         test_semantics=None,
                         reconstruct=True)
        ind.calculate_semantics(X_train)
        individuals.append(ind)

    # ── 2. Apply each inflate variant, record ΔM_phi ─────────────────────────
    records = []
    for name, two_trees, op, norm, group in VARIANTS:
        print(f"  {name} …", flush=True)
        mutator = _build_mutator(name, two_trees, op, norm,
                                 FUNCTIONS, TERMINALS, CONSTANTS, y_train=y_train)
        for i, ind in enumerate(individuals):
            m_b, ell_b, no_b, nnao_b, nnaoc_b = compute_m_phi(ind, FUNCTIONS)
            offspring = mutator(ind, MS, X_train,
                                max_depth=MAX_DEPTH, p_c=0,
                                X_test=None, reconstruct=True)
            m_a, ell_a, no_a, nnao_a, nnaoc_a = compute_m_phi(offspring, FUNCTIONS)
            records.append({
                'ind_idx': i,
                'variant': name,
                'group':   group,
                'dm_phi':  m_a  - m_b,
                'dell':    ell_a - ell_b,
                'dno':     no_a  - no_b,
                'dnnao':   nnao_a - nnao_b,
                'dnnaoc':  nnaoc_a - nnaoc_b,
            })

    df = pd.DataFrame(records)

    # ── 3. Build and print tables ─────────────────────────────────────────────
    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)

    for group, title in TABLE_TITLES.items():
        sub  = df[df['group'] == group]
        tbl  = _make_summary(sub)
        _print_table(title, tbl)
        safe = group.replace(" / ", "_").replace(" ", "_")
        tbl.to_csv(os.path.join(log_dir, f"mutation_mphi_{safe}.csv"), index=False)

    # comparison table — all variants, sorted by group order
    df['_order'] = df['group'].map({g: i for i, g in enumerate(GROUP_ORDER)})
    df_sorted = df.sort_values('_order')
    all_tbl = _make_summary(df_sorted)
    _print_table("Table 5 — All variants comparison", all_tbl)
    all_tbl.to_csv(os.path.join(log_dir, "mutation_mphi_all.csv"), index=False)

    # compact tables: median + IQR only
    print("\n" + "=" * 70)
    print("COMPACT TABLES  (median  (Q1, Q3))")
    print("=" * 70)
    for group, title in TABLE_TITLES.items():
        sub  = df[df['group'] == group]
        ctbl = _make_compact_summary(sub)
        _print_table(title.replace("Table", "Compact Table"), ctbl)
        safe = group.replace(" / ", "_").replace(" ", "_")
        ctbl.to_csv(os.path.join(log_dir, f"mutation_mphi_compact_{safe}.csv"), index=False)

    compact_all = _make_compact_summary(df_sorted)
    _print_table("Compact Table 5 — All variants comparison", compact_all)
    compact_all.to_csv(os.path.join(log_dir, "mutation_mphi_compact_all.csv"), index=False)

    # ── 4. Statistical tests (Bonferroni-corrected Wilcoxon) ─────────────────
    variant_order = [v[0] for v in VARIANTS]
    pivot = df.pivot(index='ind_idx', columns='variant', values='dm_phi')[variant_order]

    raw_pvals  = _all_pairwise(pivot, variant_order)
    corrected  = _bonferroni(raw_pvals)          # 45 comparisons
    n_tests    = len(raw_pvals)
    best_variant = pivot.median().idxmax()       # least-negative median dm_phi

    p_matrix, sym_matrix = _build_matrices(pivot, variant_order, corrected)

    print("\n" + "=" * 70)
    print(f"PAIRWISE WILCOXON  (Bonferroni n={n_tests}; + row better, - worse, ~ similar)")
    print("=" * 70)
    _print_table("Symbol matrix (upper triangle)", sym_matrix, index=True)
    sym_matrix.to_csv(os.path.join(log_dir, "mutation_mphi_pairwise_symbols.csv"))

    _print_table("Corrected p-value matrix (upper triangle)", p_matrix, index=True)
    p_matrix.to_csv(os.path.join(log_dir, "mutation_mphi_pairwise_pvalues.csv"))

    # 4b. Compact all + vs-best column (using corrected p-values)
    print(f"\nBest variant (least interpretability cost): {best_variant}")
    vs_best = []
    for variant in compact_all['Variant']:
        if variant == best_variant:
            vs_best.append('ref')
        else:
            key = (variant, best_variant) if (variant, best_variant) in corrected \
                  else (best_variant, variant)
            cp  = corrected[key]
            vs_best.append(_symbol(cp, pivot[variant].median(),
                                   pivot[best_variant].median()))
    compact_all[f'vs {best_variant}'] = vs_best
    _print_table("Compact Table 5 + vs-best (Bonferroni)", compact_all)
    compact_all.to_csv(os.path.join(log_dir, "mutation_mphi_compact_all_stats.csv"), index=False)

    # simple stats CSV: Variant | dm_phi med (IQR) | vs-best
    # rebuilt from study data to avoid CSV-quoting issues with the user's file
    sym_map = dict(zip(compact_all['Variant'], compact_all[f'vs {best_variant}']))
    simple_path = os.path.join(log_dir, "mutation_mphi_compact_all_simple.csv")
    if os.path.exists(simple_path):
        # read with quoting=csv.QUOTE_ALL workaround: use engine='python' + proper sep
        import csv as _csv
        with open(simple_path, newline='') as f:
            reader = _csv.reader(f)
            rows = list(reader)
        header = rows[0]
        data_rows = rows[1:]
        # first field of each row is the variant name
        simple_records = []
        for row in data_rows:
            if not row:
                continue
            variant_name = row[0].strip()
            iqr_val = ','.join(row[1:]).strip()   # rejoin if comma split the IQR
            simple_records.append({
                'Variant': variant_name,
                header[1] if len(header) > 1 else 'dm_phi med (IQR)': iqr_val,
                f'vs {best_variant}': sym_map.get(variant_name, ''),
            })
        simple_df = pd.DataFrame(simple_records)
        simple_df.to_csv(
            os.path.join(log_dir, "mutation_mphi_compact_all_simple_stats.csv"),
            index=False, quoting=_csv.QUOTE_NONNUMERIC)
        print("Augmented simple table saved: mutation_mphi_compact_all_simple_stats.csv")

    print(f"CSVs saved to {log_dir}")


if __name__ == "__main__":
    run_study()
