"""
paper_fig_pareto_variants.py — accuracy/interpretability Pareto front,
all 16 SLIM variants, test RMSE vs M_phi, one panel per dataset (2x3 grid).

Adapts main/analysis/plot_pareto_mphi_rmse.py's pareto_front dominance
logic and per-dataset grid layout, but:
  - recolors points by mutation family (variant_color) instead of the
    original 4-group tree-arity/op scheme, so a variant has the same color
    here as in every other paper figure (geometry grid, evolution grids).
  - encodes sum('+')/mul('*') as marker shape (circle/square,
    variant_marker) in addition to color, so all SLIM* variants read as
    one shape regardless of family.
  - Pareto membership is shown by fill (filled = Pareto-optimal, hollow =
    dominated) instead of a connecting front line.

M_phi convention: post-simplification m_phi_after with the analysis-side
max-filter applied (same field used in paper_fig_mphi_table.py's column B
and the codebase's established default for "final" M_phi).

Run from the project root:
    python main/analysis/paper_fig_pareto_variants.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper_fig_common import VARIANT_ORDER, DATASETS, DS_LABELS, variant_color, variant_marker, save_all

SIMP_LOG = os.path.join(_ROOT, 'main', 'log', 'results_normalized_simplification.csv')
OUT_DIR  = os.path.join(_ROOT, 'main', 'paper_figures')


def pareto_front(points):
    """Boolean mask of Pareto-optimal points (max mphi, min rmse). Verbatim from
    main/analysis/plot_pareto_mphi_rmse.py."""
    n = len(points)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (points[j, 0] >= points[i, 0] and points[j, 1] <= points[i, 1] and
                    (points[j, 0] > points[i, 0] or points[j, 1] < points[i, 1])):
                dominated[i] = True
                break
    return ~dominated


def load_data():
    df = pd.read_csv(SIMP_LOG, usecols=['algo', 'dataset', 'm_phi_before', 'm_phi_after', 'test_rmse'],
                     on_bad_lines='skip')
    df['m_phi'] = df[['m_phi_before', 'm_phi_after']].max(axis=1)
    return df


def make_plot(df):
    agg = df.groupby(['algo', 'dataset'])[['m_phi', 'test_rmse']].median().reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, DATASETS):
        sub = agg[agg['dataset'] == ds].reset_index(drop=True)
        pts = sub[['m_phi', 'test_rmse']].values
        mask = pareto_front(pts)

        for i, row in sub.iterrows():
            color = variant_color(row['algo'])
            marker = variant_marker(row['algo'])
            on_front = bool(mask[i])
            # ax.plot (Line2D), not ax.scatter (PathCollection): matplot2tikz mis-converts
            # single-point PathCollections, exporting the raw marker-glyph path vertices
            # (~-0.5..0.5) as literal data coordinates instead of the point's real offset --
            # harmless on a normal-range axis, but overflows pgfplots' dimension registers
            # on the Istanbul panel (test_rmse range only ~0.0038 wide). Line2D markers are
            # exported correctly as `only marks` + the real (x, y) coordinate.
            # color= (the invisible line's own color, unused visually since linestyle='none')
            # is pinned to match markeredgecolor: matplot2tikz uses Line2D.get_color() as the
            # mark's default draw color for *filled* markers (mark options only overrides fill
            # for solid marks), so leaving it at matplotlib's auto-cycled default would give
            # filled markers a mismatched edge color in the exported .tex.
            ax.plot(row['m_phi'], row['test_rmse'], marker=marker, linestyle='none', color=color,
                    markerfacecolor=color if on_front else 'none',
                    markeredgecolor=color, markersize=9.5,
                    markeredgewidth=1.5 if on_front else 1.2, zorder=3)

        ax.set_title(DS_LABELS[ds], fontsize=11)
        ax.set_xlabel(r'M$_\phi$  (higher = more interpretable $\rightarrow$)', fontsize=8.5)
        ax.set_ylabel('Test RMSE  ($\\downarrow$ lower = better)', fontsize=8.5)
        ax.tick_params(labelsize=7)

    color_handles = [Line2D([0], [0], marker=variant_marker(a), color='w', markerfacecolor=variant_color(a),
                             markeredgecolor=variant_color(a), markersize=8, label=a)
                      for a in VARIANT_ORDER]
    shape_handles = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#888888',
               markeredgecolor='#888888', markersize=8, label='SLIM+ (sum)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#888888',
               markeredgecolor='#888888', markersize=8, label='SLIM* (mul)'),
    ]
    fill_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
               markeredgecolor='#888888', markersize=8, label='Filled = Pareto-optimal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#888888', markersize=8, label='Hollow = dominated'),
    ]
    fig.legend(handles=color_handles + shape_handles + fill_handles, loc='lower center',
               ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.12))

    fig.suptitle(r'Accuracy-interpretability Pareto front — all 16 SLIM variants'
                 '\nEach point = median over 30 seeds (post-simplification M$_\\phi$)',
                 fontsize=12)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    df = load_data()
    fig = make_plot(df)
    save_all(fig, OUT_DIR, 'pareto_variants')
    plt.close(fig)
