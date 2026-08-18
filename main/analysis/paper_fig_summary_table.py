"""
paper_fig_summary_table.py -- final median performance table, all 16 SLIM
variants, one table PER DATASET, color-coded by relative column performance.

Layout (per dataset): 8 rows (one per mutation family, FAMILY_ORDER), two
horizontally joined 3-column blocks -- SLIM+ (sum) on the left, SLIM* (mul) on
the right -- each block = Train RMSE / Test RMSE / M_phi. This packs all 16
variants into one compact table (8 rows instead of 16) with the sum/mul pair
for a family directly comparable side by side.

Data sources / conventions (matching every other paper_fig_*.py script):
  - Train RMSE: median of the FINAL-generation train_fitness per
    (algo, dataset, seed), from results_normalized_generations.csv (this file
    lacks a post-hoc-simplified counterpart -- train fitness is always read
    off the live, unsimplified elite).
  - Test RMSE: median test_rmse per (algo, dataset, seed), from
    results_normalized_simplification.csv (final-generation elite; identical
    quantity to the generations log's final test_fitness, just cheaper to read
    since that file lacks the 1.1GB generation-by-generation history).
  - M_phi: median of the analysis-side-filtered post-simplification M_phi
    (m_phi_after = max(m_phi_after, m_phi_before), CLAUDE.md convention),
    from results_normalized_simplification.csv.
  - Each dataset's medians are taken over its own 30 seeds only (no pooling
    across datasets) -- one table per dataset, six tables total.

Color coding: within a given dataset's table, each of the 6 numeric columns
is colored independently (own min-max scale across the 8 family rows) with a
red(worst)-yellow-green(best) gradient -- lower is better for RMSE columns,
higher is better for M_phi columns. Requires \\usepackage[table]{xcolor} in
the LaTeX preamble for \\cellcolor.

Run from the project root:
    python main/analysis/paper_fig_summary_table.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MAIN = os.path.join(_ROOT, 'main')
for p in (_MAIN, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paper_fig_common import FAMILY_ORDER, DATASETS, DS_LABELS, save_all

GEN_LOG  = os.path.join(_MAIN, 'log', 'results_normalized_generations.csv')
SIMP_LOG = os.path.join(_MAIN, 'log', 'results_normalized_simplification.csv')
OUT_DIR  = os.path.join(_MAIN, 'paper_figures')

COLS = ['train_rmse', 'test_rmse', 'm_phi']
COL_LABELS = {'train_rmse': 'Train RMSE', 'test_rmse': 'Test RMSE', 'm_phi': r'M$_\phi$'}
HIGHER_IS_BETTER = {'train_rmse': False, 'test_rmse': False, 'm_phi': True}


def load_medians_per_dataset():
    """Returns DataFrame indexed by (algo, dataset) with columns train_rmse/test_rmse/m_phi,
    medians taken over each dataset's 30 seeds separately."""
    print('Loading generation log for final-generation train RMSE (~1.1GB, may take a while)...', flush=True)
    gen = pd.read_csv(GEN_LOG, usecols=['algo', 'dataset', 'seed', 'generation', 'train_fitness'])
    final_idx = gen.groupby(['algo', 'dataset', 'seed'])['generation'].idxmax()
    gen_final = gen.loc[final_idx]
    train_med = gen_final.groupby(['algo', 'dataset'])['train_fitness'].median().rename('train_rmse')

    simp = pd.read_csv(SIMP_LOG, usecols=['algo', 'dataset', 'seed', 'test_rmse', 'm_phi_before', 'm_phi_after'])
    simp['m_phi'] = simp[['m_phi_after', 'm_phi_before']].max(axis=1)
    test_med = simp.groupby(['algo', 'dataset'])['test_rmse'].median().rename('test_rmse')
    mphi_med = simp.groupby(['algo', 'dataset'])['m_phi'].median().rename('m_phi')

    table = pd.concat([train_med, test_med, mphi_med], axis=1)
    return table


def rag_color(value, col_values, higher_is_better):
    """Red(worst) -> yellow -> green(best) gradient, pastel-blended for legible text."""
    lo, hi = min(col_values), max(col_values)
    if hi == lo:
        frac = 0.5
    else:
        frac = (value - lo) / (hi - lo)
        if not higher_is_better:
            frac = 1 - frac
    red, yellow, green = (244, 67, 54), (255, 235, 59), (76, 175, 80)
    if frac < 0.5:
        t = frac / 0.5
        rgb = tuple(red[i] + t * (yellow[i] - red[i]) for i in range(3))
    else:
        t = (frac - 0.5) / 0.5
        rgb = tuple(yellow[i] + t * (green[i] - yellow[i]) for i in range(3))
    # blend 55% toward white to keep it pastel / text-legible
    rgb = tuple(int(round(c * 0.45 + 255 * 0.55)) for c in rgb)
    return '{:02X}{:02X}{:02X}'.format(*rgb)


def build_grid(medians: pd.DataFrame, dataset: str):
    """Returns dict: family -> {'sum': {col: val}, 'mul': {col: val}} for one dataset."""
    grid = {}
    for fam in FAMILY_ORDER:
        row = {'sum': {}, 'mul': {}}
        for op, key in (('sum', f'SLIM+{fam}'), ('mul', f'SLIM*{fam}')):
            for col in COLS:
                idx = (key, dataset)
                row[op][col] = medians.loc[idx, col] if idx in medians.index else float('nan')
        grid[fam] = row
    return grid


def fmt_value(v):
    """Adaptive precision: |v| < 1 (e.g. Istanbul's ~0.01 RMSE) needs 4 decimals or every
    row rounds to the same '0.01'/'0.02' and the column carries no information; larger
    magnitudes (RMSE in the tens-thousands range, M_phi) read fine at 2 decimals."""
    return f'{v:.4f}' if abs(v) < 1 else f'{v:.2f}'


def grid_col_values(grid: dict):
    return {
        (op, col): [grid[fam][op][col] for fam in FAMILY_ORDER if pd.notna(grid[fam][op][col])]
        for op in ('sum', 'mul') for col in COLS
    }


def tex_table_block(grid: dict, dataset: str, label_suffix: str):
    col_values = grid_col_values(grid)
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{' + DS_LABELS[dataset] + r': median train/test RMSE and M$_\phi$ per mutation '
        r'family, over the 30 seeds for this dataset. Left block: SLIM+ (sum); right block: '
        r'SLIM* (mul). Each column is color-coded red (worst) to green (best) relative to the '
        r'other 7 families in that same column (within this dataset); lower is better for '
        r'RMSE, higher is better for M$_\phi$. Requires \texttt{\textbackslash usepackage[table]\{xcolor\}}.}',
        r'\label{tab:summary-median-performance-' + label_suffix + '}',
        r'\begin{tabular}{l' + 'r' * len(COLS) + '|' + 'r' * len(COLS) + '}',
        r'\toprule',
        r'& \multicolumn{' + str(len(COLS)) + r'}{c|}{SLIM+ (sum)} & \multicolumn{' + str(len(COLS)) + r'}{c}{SLIM* (mul)} \\',
        'Family & ' + ' & '.join(COL_LABELS[c] for c in COLS) + ' & ' + ' & '.join(COL_LABELS[c] for c in COLS) + r' \\',
        r'\midrule',
    ]
    for fam in FAMILY_ORDER:
        cells = [fam]
        for op in ('sum', 'mul'):
            for col in COLS:
                v = grid[fam][op][col]
                if pd.isna(v):
                    cells.append('--')
                else:
                    hexcolor = rag_color(v, col_values[(op, col)], HIGHER_IS_BETTER[col])
                    cells.append(f'\\cellcolor[HTML]{{{hexcolor}}}{fmt_value(v)}')
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return lines


def write_tex(medians: pd.DataFrame, tex_path: str):
    all_lines = []
    for dataset in DATASETS:
        grid = build_grid(medians, dataset)
        all_lines += tex_table_block(grid, dataset, dataset.replace('_', '-'))
        all_lines.append('')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_lines).rstrip() + '\n')
    print(f'  Saved -> {tex_path}')


def render_preview(medians: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes_flat = axes.flatten()

    for ax, dataset in zip(axes_flat, DATASETS):
        grid = build_grid(medians, dataset)
        col_values = grid_col_values(grid)
        headers = ['Family'] + [f'{COL_LABELS[c]} (sum)' for c in COLS] + [f'{COL_LABELS[c]} (mul)' for c in COLS]
        cell_text, cell_colors = [], []
        for fam in FAMILY_ORDER:
            row_text, row_color = [fam], ['#FFFFFF']
            for op in ('sum', 'mul'):
                for col in COLS:
                    v = grid[fam][op][col]
                    if pd.isna(v):
                        row_text.append('--')
                        row_color.append('#FFFFFF')
                    else:
                        hexcolor = rag_color(v, col_values[(op, col)], HIGHER_IS_BETTER[col])
                        row_text.append(fmt_value(v))
                        row_color.append(f'#{hexcolor}')
            cell_text.append(row_text)
            cell_colors.append(row_color)

        ax.axis('off')
        ax.set_title(DS_LABELS[dataset], fontsize=11, fontweight='bold')
        tbl = ax.table(cellText=cell_text, cellColours=cell_colors, colLabels=headers,
                        loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.4)

    fig.suptitle('Median train/test RMSE and M$_\\phi$ per mutation family, per dataset (30 seeds each)\n'
                 'red = worst, green = best, within each column of each dataset\'s table', fontsize=12)
    fig.tight_layout()
    save_all(fig, out_dir, 'summary_median_table', tikz_ok=False)
    plt.close(fig)


if __name__ == '__main__':
    medians = load_medians_per_dataset()
    print(medians.to_string())

    os.makedirs(OUT_DIR, exist_ok=True)
    write_tex(medians, os.path.join(OUT_DIR, 'summary_median_table.tex'))
    render_preview(medians, OUT_DIR)
