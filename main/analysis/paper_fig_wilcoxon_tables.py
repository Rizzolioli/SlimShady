"""
paper_fig_wilcoxon_tables.py — pairwise significance tables, all 16 SLIM
variants, for train RMSE / test RMSE / M_phi.

Reuses the exact paired-Wilcoxon + Bonferroni-correction pattern already
established in main/analysis/mutation_mphi_study.py (_all_pairwise /
_bonferroni / _symbol), but applied to the real benchmark results instead of
the synthetic single-mutation-step study, and with the symbol convention
normalized so '+' always means "row is significantly better" regardless of
whether the metric is lower-is-better (RMSE) or higher-is-better (M_phi).

Pairing: every one of the 16 variants is run on the identical 6 datasets x
30 seeds grid (main_slim_normalized.py), so for a fixed metric and dataset,
the value for (algo=A, seed=s) and (algo=B, seed=s) are a valid matched pair
-- same data split, same RNG seed, only the mutation operator differs.
Tests are run PER DATASET (30-seed paired sample), not pooled across
datasets -- each dataset gets its own significance table, since pooling
would mix distributions with very different scales/difficulty across
datasets into one test.

Data sources:
  - Train RMSE: final-generation train_fitness per (algo, dataset, seed),
    from results_normalized_generations.csv (same convention as
    paper_fig_summary_table.py).
  - Test RMSE / M_phi: results_normalized_simplification.csv, m_phi with the
    analysis-side max-filter applied (m_phi_after = max(m_phi_after,
    m_phi_before), CLAUDE.md convention).

Output: main/paper_figures/wilcoxon_tables.tex -- 18 16x16 upper-triangle
LaTeX tables (train RMSE / test RMSE / M_phi, x 6 datasets each).

Run from the project root:
    python main/analysis/paper_fig_wilcoxon_tables.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MAIN = os.path.join(_ROOT, 'main')
for p in (_MAIN, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
from scipy.stats import wilcoxon

from paper_fig_common import VARIANT_ORDER, variant_family, DATASETS, DS_LABELS

GEN_LOG  = os.path.join(_MAIN, 'log', 'results_normalized_generations.csv')
SIMP_LOG = os.path.join(_MAIN, 'log', 'results_normalized_simplification.csv')
OUT_DIR  = os.path.join(_MAIN, 'paper_figures')

ALPHA = 0.05

METRICS = [
    ('train_rmse', 'Train RMSE', False),
    ('test_rmse',  'Test RMSE',  False),
    ('m_phi',      r'M$_\phi$',  True),
]


def short_label(algo: str) -> str:
    """'SLIM+NORM12' -> '+NORM12', 'SLIM*2SIG' -> '*2SIG'."""
    op = '+' if '+' in algo else '*'
    return op + variant_family(algo)


def load_pivots():
    """Returns dict metric_key -> DataFrame, index=(dataset, seed), columns=VARIANT_ORDER."""
    print('Loading generation log for final-generation train RMSE (~1.1GB, may take a while)...', flush=True)
    gen = pd.read_csv(GEN_LOG, usecols=['algo', 'dataset', 'seed', 'generation', 'train_fitness'])
    final_idx = gen.groupby(['algo', 'dataset', 'seed'])['generation'].idxmax()
    gen_final = gen.loc[final_idx]
    train_pivot = gen_final.pivot_table(index=['dataset', 'seed'], columns='algo', values='train_fitness')

    simp = pd.read_csv(SIMP_LOG, usecols=['algo', 'dataset', 'seed', 'test_rmse', 'm_phi_before', 'm_phi_after'])
    simp['m_phi'] = simp[['m_phi_after', 'm_phi_before']].max(axis=1)
    test_pivot = simp.pivot_table(index=['dataset', 'seed'], columns='algo', values='test_rmse')
    mphi_pivot  = simp.pivot_table(index=['dataset', 'seed'], columns='algo', values='m_phi')

    pivots = {}
    for key, pivot in (('train_rmse', train_pivot), ('test_rmse', test_pivot), ('m_phi', mphi_pivot)):
        pivot = pivot[VARIANT_ORDER]
        n_before = len(pivot)
        pivot = pivot.dropna(how='any')
        if len(pivot) < n_before:
            print(f'  [{key}] dropped {n_before - len(pivot)} (dataset, seed) rows with missing values '
                  f'for at least one variant -- kept {len(pivot)} complete-case pairs')
        pivots[key] = pivot
    return pivots


def all_pairwise(pivot, variant_order):
    """Raw Wilcoxon signed-rank p-values for every variant pair. dict (vi, vj) -> p."""
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


def bonferroni(pvals_dict):
    n = len(pvals_dict)
    return {k: min(v * n, 1.0) for k, v in pvals_dict.items()}


def fmt_p(p):
    """1.00 when Bonferroni-capped, else 1-sig-fig scientific notation -- kept short (~5 chars)
    so the 16-column table still fits \\textwidth after \\resizebox without the text becoming
    illegibly small."""
    if p >= 0.999:
        return '1.00'
    return f'{p:.0e}'


def build_pvalue_matrix(pivot, variant_order):
    raw_pvals = all_pairwise(pivot, variant_order)
    corrected = bonferroni(raw_pvals)
    n = len(variant_order)
    cells = {v: {v2: '' for v2 in variant_order} for v in variant_order}
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = variant_order[i], variant_order[j]
            cp = corrected[(vi, vj)]
            text = fmt_p(cp)
            if cp < ALPHA:
                text = r'\textbf{' + text + '}'
            cells[vi][vj] = text
    return pd.DataFrame(cells, index=variant_order).T, len(raw_pvals)


CAPTION = (
    r"Pairwise Bonferroni-corrected Wilcoxon $p$-values for {metric_label} on {dataset_label}, "
    r"across all 16 SLIM variants, upper triangle only (row vs.\ column). Bold = significant at "
    r"$\alpha = {alpha}$."
)


def tex_table_block(p_df, variant_order, metric_key, metric_label, dataset_key, dataset_label):
    labels = [short_label(v) for v in variant_order]
    caption = CAPTION.format(metric_label=metric_label, dataset_label=dataset_label, alpha=ALPHA)
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{tab:wilcoxon-' + metric_key.replace('_', '-') + '-' + dataset_key.replace('_', '-') + '}',
        r'\setlength{\tabcolsep}{2.2pt}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{l' + 'c' * len(variant_order) + '}',
        r'\toprule',
        '& ' + ' & '.join(r'\rotatebox{90}{' + lab + '}' for lab in labels) + r' \\',
        r'\midrule',
    ]
    for i, vi in enumerate(variant_order):
        cells = [labels[i]]
        for j, vj in enumerate(variant_order):
            if j <= i:
                cells.append('')
            else:
                cells.append(p_df.loc[vi, vj])
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'}', r'\end{table}']
    return lines


def write_tex(pivots, tex_path):
    all_lines = []
    for metric_key, metric_label, higher_is_better in METRICS:
        pivot = pivots[metric_key]
        for ds in DATASETS:
            ds_pivot = pivot.loc[ds]  # 30-seed subset for this dataset only
            p_df, n_tests = build_pvalue_matrix(ds_pivot, VARIANT_ORDER)
            print(f'\n{metric_label} / {DS_LABELS[ds]} -- Bonferroni n={n_tests}')
            print(p_df.to_string())
            all_lines += tex_table_block(p_df, VARIANT_ORDER, metric_key, metric_label, ds, DS_LABELS[ds])
            all_lines.append('')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_lines).rstrip() + '\n')
    print(f'\n  Saved -> {tex_path}')


if __name__ == '__main__':
    pivots = load_pivots()
    os.makedirs(OUT_DIR, exist_ok=True)
    write_tex(pivots, os.path.join(OUT_DIR, 'wilcoxon_tables.tex'))
