#!/usr/bin/env python
"""
Two bidimensional scatter figures (last generation, 30 seeds):

  Fig 1: Test RMSE (y) vs Size/nodes_count (x)
  Fig 2: Test RMSE (y) vs M_phi (x)

Layout: 2×4 grid — 6 dataset panels + 1 "All datasets" panel + 1 legend panel.

Per-dataset panels:
  - Small transparent dots = individual seeds
  - Large opaque marker + IQR error bars = per-algo median
  - Log x-axis for Size (huge range across algos on some datasets)
  - Axis clipped to 5th-95th pct of seed values to suppress outliers

"All datasets" panel:
  - Per-(algo, dataset) median normalised within each dataset to [0,1]
    (0 = best algo on that dataset, 1 = worst)
  - Small dots = per-dataset normalised medians (6 per algo)
  - Large dot = median of the 6 normalised values per algo
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')
GEN_LOG = os.path.join(LOG_DIR, 'results_normalized_generations.csv')
OUT_DIR = os.path.join(LOG_DIR, 'figs')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
ALGOS = [
    'SLIM+2SIG',   'SLIM*2SIG',
    'SLIM+ABS',    'SLIM*ABS',
    'SLIM+1SIG',   'SLIM*1SIG',
    'SLIM+NORM1',  'SLIM*NORM1',
    'SLIM+NORM2',  'SLIM*NORM2',
    'SLIM+NORMROB','SLIM*NORMROB',
    'SLIM+NORM12', 'SLIM*NORM12',
]
DATASETS = ['concrete', 'energy', 'instanbul', 'ppb', 'resid_build_sale_price', 'toxicity']
DS_LABELS = {
    'concrete': 'Concrete', 'energy': 'Energy', 'instanbul': 'Istanbul',
    'ppb': 'PPB', 'resid_build_sale_price': 'Resid. Build', 'toxicity': 'Toxicity',
}
_PALETTE = [
    '#1f77b4','#aec7e8',
    '#d62728','#ff9896',
    '#2ca02c','#98df8a',
    '#9467bd','#c5b0d5',
    '#8c564b','#c49c94',
    '#ff7f0e','#ffbb78',
    '#e377c2','#f7b6d2',
]
ALGO_COLOR  = {a: c for a, c in zip(ALGOS, _PALETTE)}
ALGO_MARKER = {a: ('o' if '+' in a else '^') for a in ALGOS}

# ── Load last generation ──────────────────────────────────────────────────────
print('Loading ...', flush=True)
df = pd.read_csv(GEN_LOG,
                 usecols=['algo','dataset','seed','generation',
                          'test_fitness','nodes_count','m_phi'])
last_gen = int(df['generation'].max())
df = df[df['generation'] == last_gen].copy()
print(f'  {len(df)} rows at gen {last_gen}', flush=True)

# Per-(algo, dataset) medians
med = (df.groupby(['algo','dataset'])[['test_fitness','nodes_count','m_phi']]
         .median().reset_index())


def _normalise_for_all_panel(med_df, x_col, y_col, x_lower_better):
    """
    For each dataset: min-max normalise x and y across the 10 algo medians.
    x: 0=best, 1=worst  (lower_better → direct; higher_better → flip)
    y (test RMSE): 0=best (lowest), 1=worst (highest)
    Returns a DataFrame with columns [algo, dataset, x_norm, y_norm]
    """
    rows = []
    for ds in DATASETS:
        sub = med_df[med_df['dataset'] == ds]
        xv = sub[x_col].values
        yv = sub['test_fitness'].values
        rng_x = xv.max() - xv.min()
        rng_y = yv.max() - yv.min()
        for _, row in sub.iterrows():
            xn = ((row[x_col] - xv.min()) / rng_x) if rng_x else 0.5
            yn = ((row['test_fitness'] - yv.min()) / rng_y) if rng_y else 0.5
            if not x_lower_better:   # m_phi: higher=better → flip so 0=best
                xn = 1 - xn
            rows.append({'algo': row['algo'], 'dataset': ds,
                         'x_norm': xn, 'y_norm': yn})
    return pd.DataFrame(rows)


def make_figure(x_col, x_label, x_log, x_lower_better, out_name):
    fig = plt.figure(figsize=(18, 10))
    # 2×4 grid; last cell (1,3) used for legend
    axes = []
    for pos in range(7):
        r, c = divmod(pos, 4)
        axes.append(fig.add_subplot(2, 4, pos + 1))
    legend_ax = fig.add_subplot(2, 4, 8)
    legend_ax.axis('off')

    y_col = 'test_fitness'

    # ── Per-dataset panels ────────────────────────────────────────────────────
    for idx, (ds, ax) in enumerate(zip(DATASETS, axes[:6])):
        ds_df = df[df['dataset'] == ds]

        # clip axes to 5th–95th pct of seed data
        xvals = ds_df[x_col].replace([np.inf, -np.inf], np.nan).dropna()
        yvals = ds_df[y_col].replace([np.inf, -np.inf], np.nan).dropna()
        xlo, xhi = np.percentile(xvals, 2), np.percentile(xvals, 98)
        ylo, yhi = np.percentile(yvals, 2), np.percentile(yvals, 98)
        xmarg = (xhi - xlo) * 0.08 or xhi * 0.1
        ymarg = (yhi - ylo) * 0.08 or yhi * 0.1

        for algo in ALGOS:
            sub   = ds_df[ds_df['algo'] == algo]
            sx    = sub[x_col].values
            sy    = sub[y_col].values
            color = ALGO_COLOR[algo]
            mkr   = ALGO_MARKER[algo]

            # individual seed points
            ax.scatter(sx, sy, color=color, marker=mkr,
                       s=12, alpha=0.20, linewidths=0, zorder=2)

            # median + IQR error bars
            mx = np.median(sx);  q25x, q75x = np.percentile(sx, 25), np.percentile(sx, 75)
            my = np.median(sy);  q25y, q75y = np.percentile(sy, 25), np.percentile(sy, 75)
            ax.errorbar(mx, my,
                        xerr=[[mx - q25x], [q75x - mx]],
                        yerr=[[my - q25y], [q75y - my]],
                        fmt=mkr, color=color, markersize=8,
                        elinewidth=1.0, capsize=3, zorder=4)

        ax.set_title(DS_LABELS[ds], fontsize=10, pad=3)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel('Test RMSE', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2, lw=0.4)
        ax.spines[['top', 'right']].set_visible(False)
        if x_log:
            ax.set_xscale('log')
            ax.set_xlim(max(xlo * 0.7, 1), xhi * 1.4)
        else:
            ax.set_xlim(xlo - xmarg, xhi + xmarg)
        ax.set_ylim(ylo - ymarg, yhi + ymarg)

    # ── "All datasets" panel (normalised) ────────────────────────────────────
    ax_all = axes[6]
    norm_df = _normalise_for_all_panel(med, x_col, y_col, x_lower_better)

    for algo in ALGOS:
        sub   = norm_df[norm_df['algo'] == algo]
        color = ALGO_COLOR[algo]
        mkr   = ALGO_MARKER[algo]
        # one small dot per dataset
        ax_all.scatter(sub['x_norm'], sub['y_norm'],
                       color=color, marker=mkr, s=30, alpha=0.55,
                       linewidths=0.4, edgecolors='white', zorder=2)
        # overall median
        ax_all.scatter(sub['x_norm'].median(), sub['y_norm'].median(),
                       color=color, marker=mkr, s=120, alpha=1.0,
                       linewidths=0.6, edgecolors='black', zorder=5)

    # axis labels depend on direction
    x_dir = 'lower=better' if x_lower_better else 'higher=better'
    ax_all.set_xlabel(f'{x_label}  [normalised, {x_dir}  →  0=best]', fontsize=8)
    ax_all.set_ylabel('Test RMSE  [normalised, 0=best]', fontsize=8)
    ax_all.set_title('All datasets (normalised)', fontsize=10, pad=3)
    ax_all.set_xlim(-0.05, 1.05)
    ax_all.set_ylim(-0.05, 1.05)
    ax_all.tick_params(labelsize=7)
    ax_all.grid(alpha=0.2, lw=0.4)
    ax_all.spines[['top', 'right']].set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = []
    for algo in ALGOS:
        handles.append(
            Line2D([0], [0],
                   marker=ALGO_MARKER[algo], color=ALGO_COLOR[algo],
                   markerfacecolor=ALGO_COLOR[algo], markersize=8,
                   linewidth=0, label=algo)
        )
    handles += [
        Line2D([0], [0], marker='o', color='grey', markersize=6,
               linewidth=0, label='+  (sum operator)', alpha=0.7),
        Line2D([0], [0], marker='^', color='grey', markersize=6,
               linewidth=0, label='*  (product operator)', alpha=0.7),
    ]
    legend_ax.legend(handles=handles, loc='center', fontsize=8.5,
                     frameon=True, ncol=1)

    fig.suptitle(
        f'Test RMSE vs {x_label}  —  gen {last_gen}, 30 seeds '
        f'(small dots = seeds, large = median + IQR)',
        fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, out_name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {path}')


# ── Figure 1: RMSE vs Size ────────────────────────────────────────────────────
make_figure(
    x_col='nodes_count',
    x_label='Size (blocks)',
    x_log=True,            # log scale: size spans 1–12000+
    x_lower_better=True,
    out_name='scatter_rmse_vs_size.png',
)

# ── Figure 2: RMSE vs M_phi ───────────────────────────────────────────────────
make_figure(
    x_col='m_phi',
    x_label=r'M$_\phi$',
    x_log=False,
    x_lower_better=False,  # M_phi: higher = more interpretable
    out_name='scatter_rmse_vs_mphi.png',
)

print('Done.')
