"""
paper_fig_pareto_baselines.py — SLIM vs. SOTA baselines Pareto front
(PySR excluded), test RMSE vs M_phi, one panel per dataset (2x3 grid).

Adapts main/plot_baselines_scatter_no_pysr.py's role-selection logic (pick
the 3 most representative SLIM points per dataset: best-RMSE, best-M_phi,
middle) and baseline styling (GPLearn=red diamond, Operon=purple plus,
GP-GOMEA=cyan X, from main/log/results_baselines.csv which has no PySR
rows in this comparison by construction). Two adaptations from the source
script:
  1. Role-selection candidate pool extended from the source script's stale
     10-variant ALL_SLIM list to the full current 16-variant set (VARIANT_ORDER)
     — the source script predates NORMROB/NORM12/NORMFIX.
  2. Axis convention aligned with paper_fig_pareto_variants.py for
     side-by-side reading in the paper: raw M_phi (x, higher=better) and
     raw test RMSE (y, lower=better), instead of the source script's
     negated M_phi / GPLearn-normalized RMSE axes.
All 3 chosen SLIM points share one fixed color (SLIM_COLOR) regardless of
variant/family, so SLIM reads as a single coherent group against the 3
distinctly-colored baselines; role (best-RMSE / best-M_phi / middle) is
encoded by marker shape instead. Pareto membership (across all 6 points in
a panel) is shown by fill (filled = Pareto-optimal, hollow = dominated)
rather than a connecting front line.

"Middle" role criterion: among the 14 variants NOT chosen as best-RMSE or
best-M_phi, the one with the lowest average of (RMSE rank, M_phi rank) —
i.e. the best all-round compromise between accuracy and interpretability
among the leftover candidates, not just the 3rd-best on either axis alone.

Run from the project root:
    python main/analysis/paper_fig_pareto_baselines.py
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
from matplotlib.lines import Line2D

from paper_fig_common import VARIANT_ORDER, DATASETS, DS_LABELS, save_all
from paper_fig_pareto_variants import pareto_front

SLIM_CSV     = os.path.join(_ROOT, 'main', 'log', 'results_normalized_simplification.csv')
BASELINE_CSV = os.path.join(_ROOT, 'main', 'log', 'results_baselines.csv')
OUT_DIR      = os.path.join(_ROOT, 'main', 'paper_figures')

SLIM_COLOR = '#34495e'  # single fixed color for all SLIM points in this figure

ROLE_MARKERS = {'best_rmse': 'o', 'best_mphi': 's', 'middle': '^'}
ROLE_LABELS  = {'best_rmse': 'Best RMSE', 'best_mphi': r'Best M$_\phi$', 'middle': 'Middle'}

BASELINE_STYLES = {
    'GPLearn':  dict(color='#d62728', marker='D'),
    'Operon':   dict(color='#9467bd', marker='P'),
    'GP-GOMEA': dict(color='#17becf', marker='X'),
}
BASELINE_KEYS = list(BASELINE_STYLES)


def load_combined():
    slim = pd.read_csv(SLIM_CSV, usecols=['algo', 'dataset', 'seed', 'm_phi_before', 'm_phi_after', 'test_rmse'])
    slim = slim[slim['algo'].isin(VARIANT_ORDER)].copy()
    slim['m_phi_after'] = slim[['m_phi_after', 'm_phi_before']].max(axis=1)

    bl = pd.read_csv(BASELINE_CSV, usecols=['algo', 'dataset', 'seed', 'test_rmse', 'm_phi_before', 'm_phi_after'])
    bl = bl.dropna(subset=['test_rmse'])
    gp_mask = bl['algo'] == 'GP-GOMEA'
    bl = pd.concat([bl[~gp_mask], bl[gp_mask & bl['m_phi_before'].notna()]], ignore_index=True)
    bl['m_phi_after'] = bl[['m_phi_after', 'm_phi_before']].max(axis=1)
    bl = bl[bl['algo'].isin(BASELINE_KEYS)].copy()

    cols = ['algo', 'dataset', 'seed', 'test_rmse', 'm_phi_after']
    combined = pd.concat([slim[cols], bl[cols]], ignore_index=True)
    combined = combined[combined['dataset'].isin(DATASETS)]
    med = (combined.groupby(['algo', 'dataset'])
                   .agg(m_phi=('m_phi_after', 'median'), test_rmse=('test_rmse', 'median'))
                   .reset_index())
    return med


def pick_roles(med, ds):
    sub = med[(med['algo'].isin(VARIANT_ORDER)) & (med['dataset'] == ds)].copy()
    sub['rmse_rank'] = sub['test_rmse'].rank()
    sub['mphi_rank'] = sub['m_phi'].rank(ascending=False)
    sub['avg_rank']  = (sub['rmse_rank'] + sub['mphi_rank']) / 2
    best_rmse = sub.loc[sub['rmse_rank'].idxmin()]
    best_mphi = sub.loc[sub['mphi_rank'].idxmin()]
    rest = sub[~sub['algo'].isin({best_rmse['algo'], best_mphi['algo']})]
    middle = rest.loc[rest['avg_rank'].idxmin()]
    return {'best_rmse': best_rmse, 'best_mphi': best_mphi, 'middle': middle}


def make_plot(med):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, DATASETS):
        roles = pick_roles(med, ds)
        base_rows = med[(med['algo'].isin(BASELINE_KEYS)) & (med['dataset'] == ds)]

        entries = []  # dicts: x, y, color, marker
        for role, row in roles.items():
            entries.append(dict(x=row['m_phi'], y=row['test_rmse'],
                                color=SLIM_COLOR, marker=ROLE_MARKERS[role]))
        for _, row in base_rows.iterrows():
            st = BASELINE_STYLES[row['algo']]
            entries.append(dict(x=row['m_phi'], y=row['test_rmse'],
                                color=st['color'], marker=st['marker']))

        pts = pd.DataFrame(entries)[['x', 'y']].values
        mask = pareto_front(pts)

        for e, on_front in zip(entries, mask):
            # ax.plot (Line2D), not ax.scatter (PathCollection): matplot2tikz mis-converts
            # single-point PathCollections, exporting the raw marker-glyph path vertices
            # (~-0.5..0.5) as literal data coordinates instead of the point's real offset --
            # overflows pgfplots' dimension registers on panels with a narrow axis range.
            # Line2D markers export correctly as `only marks` + the real (x, y) coordinate.
            # color= pinned to match markeredgecolor: matplot2tikz uses Line2D.get_color() as
            # the mark's default draw color for *filled* markers (mark options only overrides
            # fill for solid marks), so leaving it at matplotlib's auto-cycled default would
            # give filled markers a mismatched edge color in the exported .tex.
            ax.plot(e['x'], e['y'], marker=e['marker'], linestyle='none', color=e['color'],
                    markerfacecolor=e['color'] if on_front else 'none',
                    markeredgecolor=e['color'], markersize=11,
                    markeredgewidth=1.6 if on_front else 1.3, zorder=4)

        subtitle = (f"RMSE: {roles['best_rmse']['algo']}  |  "
                    f"M$_\\phi$: {roles['best_mphi']['algo']}  |  "
                    f"mid: {roles['middle']['algo']}")
        ax.set_title(f"{DS_LABELS[ds]}\n{subtitle}", fontsize=8.5)
        ax.set_xlabel(r'M$_\phi$  (higher = more interpretable $\rightarrow$)', fontsize=8.5)
        ax.set_ylabel('Test RMSE  ($\\downarrow$ lower = better)', fontsize=8.5)
        ax.tick_params(labelsize=7)

    role_handles = [
        Line2D([0], [0], marker=ROLE_MARKERS[r], color='w', markerfacecolor=SLIM_COLOR,
               markeredgecolor=SLIM_COLOR, markersize=9, label=f'SLIM – {ROLE_LABELS[r]}')
        for r in ROLE_MARKERS
    ]
    baseline_handles = [
        Line2D([0], [0], marker=BASELINE_STYLES[a]['marker'], color='w',
               markerfacecolor=BASELINE_STYLES[a]['color'],
               markeredgecolor=BASELINE_STYLES[a]['color'], markersize=9, label=a)
        for a in BASELINE_KEYS
    ]
    fill_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
               markeredgecolor='#888888', markersize=8, label='Filled = Pareto-optimal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#888888', markersize=8, label='Hollow = dominated'),
    ]
    fig.legend(handles=role_handles + baseline_handles + fill_handles,
               loc='lower center', ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle('SLIM vs. SOTA symbolic regression baselines (PySR excluded)\n'
                 'SLIM reduced to 3 representative variants/dataset (single color, '
                 'shape = role); medians over 30 seeds',
                 fontsize=12)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    med = load_combined()
    fig = make_plot(med)
    save_all(fig, OUT_DIR, 'pareto_baselines')
    plt.close(fig)
