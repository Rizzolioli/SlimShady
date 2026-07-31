"""
paper_fig_mphi_table.py — two-column M_phi summary table, all 16 variants.

Column A: M_phi added by a single inflate-mutation step (structural cost,
independent of ms), from main/analysis/log/mutation_mphi_all.csv
(mutation_mphi_study.py's output).

Column B: median M_phi delta from SymPy-simplifying the final elite
(m_phi_after - m_phi_before, analysis-side filter applied, restricted to
simplified_ok==1 runs), from main/log/results_normalized_simplification.csv —
same convention as main/simplification_effect.py.

matplot2tikz cannot convert matplotlib Table objects (verified: silently
drops all cell content), so the primary deliverable here is a hand-written
LaTeX booktabs table (mphi_table.tex); png/pdf are a matplotlib-rendered
preview of the same numbers, not a matplot2tikz export.

Run from the project root:
    python main/analysis/paper_fig_mphi_table.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paper_fig_common import VARIANT_ORDER, save_all

MUT_MPHI_CSV = os.path.join(_ROOT, 'main', 'analysis', 'log', 'mutation_mphi_all.csv')
SIMP_LOG     = os.path.join(_ROOT, 'main', 'log', 'results_normalized_simplification.csv')
OUT_DIR      = os.path.join(_ROOT, 'main', 'paper_figures')


def load_table():
    mut = pd.read_csv(MUT_MPHI_CSV, usecols=['Variant', 'median dm_phi'])
    mut = mut.rename(columns={'median dm_phi': 'dmphi_mutation'}).set_index('Variant')

    simp = pd.read_csv(SIMP_LOG, usecols=[
        'algo', 'm_phi_before', 'm_phi_after', 'simplified_ok',
    ])
    # analysis-side filter (CLAUDE.md / simplification_effect.py convention)
    simp['m_phi_after'] = simp[['m_phi_after', 'm_phi_before']].max(axis=1)
    simp['delta_m_phi'] = simp['m_phi_after'] - simp['m_phi_before']
    ok = simp[simp['simplified_ok'] == 1]
    simp_delta = ok.groupby('algo')['delta_m_phi'].median()
    simp_delta.name = 'dmphi_simplification'

    table = mut.join(simp_delta, how='left').reindex(VARIANT_ORDER)
    table.index.name = 'Variant'
    return table.reset_index()


def write_tex(table: pd.DataFrame, tex_path: str):
    lines = [
        r'\begin{tabular}{lrr}',
        r'\toprule',
        r'Variant & $\Delta M_\phi$ (mutation) & $\Delta M_\phi$ (simplification) \\',
        r'\midrule',
    ]
    for _, row in table.iterrows():
        variant = row['Variant'].replace('SLIM', 'SLIM')  # keep literal, escaping not needed
        a = f"{row['dmphi_mutation']:.2f}" if pd.notna(row['dmphi_mutation']) else '--'
        b = f"{row['dmphi_simplification']:.2f}" if pd.notna(row['dmphi_simplification']) else '--'
        lines.append(f'{variant} & {a} & {b} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved -> {tex_path}')


def render_preview(table: pd.DataFrame):
    cell_text = [
        [row['Variant'],
         f"{row['dmphi_mutation']:.2f}" if pd.notna(row['dmphi_mutation']) else '--',
         f"{row['dmphi_simplification']:.2f}" if pd.notna(row['dmphi_simplification']) else '--']
        for _, row in table.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(table) + 1))
    ax.axis('off')
    tbl = ax.table(
        cellText=cell_text,
        colLabels=['Variant', r'$\Delta M_\phi$ (mutation)', r'$\Delta M_\phi$ (simplification)'],
        loc='center', cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.3)
    fig.suptitle(r'M$_\phi$ cost per mutation vs. M$_\phi$ recovered by simplification', fontsize=11)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    table = load_table()
    print(table.to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    write_tex(table, os.path.join(OUT_DIR, 'mphi_table.tex'))

    fig = render_preview(table)
    save_all(fig, OUT_DIR, 'mphi_table', tikz_ok=False)
    plt.close(fig)
