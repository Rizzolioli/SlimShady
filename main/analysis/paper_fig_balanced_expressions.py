"""
paper_fig_balanced_expressions.py -- one representative elite expression per
dataset, chosen for the best balance between test RMSE and M_phi (not the
single best of either alone), rendered as proper LaTeX math via SymPy's
latex printer.

Selection: among all 480 runs for a dataset (16 variants x 30 seeds), rank by
test_rmse (ascending) and by M_phi (descending, analysis-side-filtered:
whichever of before/after simplification has the higher M_phi, falling back
to the raw pre-simplification genotype when SymPy never simplified that run
-- same convention as main/simplification_effect.py), then pick the run with
the lowest average of the two ranks. This is the single-run analogue of the
"middle" role used in paper_fig_pareto_baselines.py's variant-level
role-selection (best_rmse/best_mphi/middle), applied per individual run
instead of per variant median.

Run from the project root:
    python main/analysis/paper_fig_balanced_expressions.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MAIN = os.path.join(_ROOT, 'main')
for p in (_MAIN, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import sympy as sp

from paper_fig_common import DATASETS, DS_LABELS

SIMP_LOG = os.path.join(_MAIN, 'log', 'results_normalized_simplification.csv')
OUT_DIR  = os.path.join(_MAIN, 'paper_figures')


def pick_genotype(row):
    if row['simplified_ok'] == 1 and row['m_phi_after'] > row['m_phi_before']:
        return pd.Series({'m_phi_final': row['m_phi_after'], 'genotype_final': row['genotype_after'],
                           'ell_final': row['ell_after'], 'source': 'simplified'})
    return pd.Series({'m_phi_final': row['m_phi_before'], 'genotype_final': row['genotype_before'],
                       'ell_final': row['ell_before'], 'source': 'raw'})


def load_balanced_picks():
    df = pd.read_csv(SIMP_LOG)
    picked = df.apply(pick_genotype, axis=1)
    out = pd.concat([df[['algo', 'dataset', 'seed', 'test_rmse']], picked], axis=1)

    rows = []
    for ds in DATASETS:
        sub = out[out['dataset'] == ds].copy()
        sub['rmse_rank'] = sub['test_rmse'].rank()
        sub['mphi_rank'] = sub['m_phi_final'].rank(ascending=False)
        sub['avg_rank'] = (sub['rmse_rank'] + sub['mphi_rank']) / 2
        best = sub.loc[sub['avg_rank'].idxmin()]
        rows.append(best)
    return pd.DataFrame(rows)


def to_latex(genotype_str: str) -> str:
    expr = sp.sympify(genotype_str)
    return sp.latex(expr)


def write_tex(picks: pd.DataFrame, tex_path: str):
    lines = [
        '% Representative elite expressions, one per dataset, chosen for the best',
        '% balance of test RMSE and M_phi (lowest average of the two ranks among all',
        '% 480 runs per dataset). Requires \\usepackage{amsmath} for \\dfrac/\\left(\\right).',
        '',
    ]
    for _, row in picks.iterrows():
        ds_label = DS_LABELS[row['dataset']]
        latex_expr = to_latex(row['genotype_final'])
        lines += [
            r'\begin{table}[t]',
            r'\centering',
            r'\caption{' + ds_label + f": {row['algo']}, seed {int(row['seed'])} "
            r'-- test RMSE $= ' + f"{row['test_rmse']:.4f}" + r'$, M$_\phi = ' +
            f"{row['m_phi_final']:.2f}" + r'$ (' + row['source'] + r' expression, ' +
            f"{int(row['ell_final'])}" + r' nodes). Chosen as the single run, among all 480 '
            r'runs (16 variants $\times$ 30 seeds) for this dataset, with the lowest average '
            r'rank of test RMSE and M$_\phi$.}',
            r'\label{tab:balanced-expr-' + row['dataset'].replace('_', '-') + '}',
            r'\[',
            latex_expr,
            r'\]',
            r'\end{table}',
            '',
        ]
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines).rstrip() + '\n')
    print(f'  Saved -> {tex_path}')


if __name__ == '__main__':
    picks = load_balanced_picks()
    print(picks[['algo', 'dataset', 'seed', 'test_rmse', 'm_phi_final', 'ell_final', 'source']].to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    write_tex(picks, os.path.join(OUT_DIR, 'balanced_expressions.tex'))
