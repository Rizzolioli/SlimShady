"""
paper_fig_evolution_mphi.py — M_phi evolution, all 16 variants, one panel
per dataset (2x3 grid). Companion to paper_fig_evolution_rmse.py.

Single solid line per variant (M_phi is a property of the generation's
elite, not a train/test fitness signal, so there's no dotted counterpart).
Median over 30 seeds, generations subsampled before aggregating, same
convention as paper_fig_evolution_rmse.py / plot_normalized_results.py.

Run from the project root:
    python main/analysis/paper_fig_evolution_mphi.py
"""
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper_fig_common import VARIANT_ORDER, DATASETS, DS_LABELS, variant_color, save_all

GEN_LOG = os.path.join(_ROOT, 'main', 'log', 'results_normalized_generations.csv')
OUT_DIR = os.path.join(_ROOT, 'main', 'paper_figures')

N_SAMPLE = 20


def load_agg():
    usecols = ['algo', 'dataset', 'seed', 'generation', 'm_phi']
    print('Loading generation log (this is a ~1.1GB file, may take a while)...', flush=True)
    t0 = time.time()
    df = pd.read_csv(GEN_LOG, usecols=usecols)
    print(f'  loaded {len(df):,} rows in {time.time()-t0:.1f}s', flush=True)

    df = df[df['generation'] % N_SAMPLE == 0]
    agg = df.groupby(['algo', 'dataset', 'generation'])['m_phi'].median().reset_index()
    return agg


def make_plot(agg):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, DATASETS):
        sub_ds = agg[agg['dataset'] == ds]
        for algo in VARIANT_ORDER:
            sub = sub_ds[sub_ds['algo'] == algo].sort_values('generation')
            if sub.empty:
                continue
            ax.plot(sub['generation'], sub['m_phi'], color=variant_color(algo), ls='-', lw=1.2)
        ax.set_title(DS_LABELS[ds], fontsize=11)
        ax.set_xlabel('Generation', fontsize=9)
        ax.set_ylabel(r'M$_\phi$', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.25, lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)

    variant_handles = [Line2D([0], [0], color=variant_color(a), lw=2, label=a) for a in VARIANT_ORDER]
    fig.legend(handles=variant_handles, loc='lower center', ncol=8,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.06),
               title='Variant (color)', frameon=True)

    fig.suptitle(r'M$_\phi$ evolution — all 16 SLIM variants (median, 30 seeds)',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    agg = load_agg()
    fig = make_plot(agg)
    save_all(fig, OUT_DIR, 'evolution_mphi_grid')
    plt.close(fig)
