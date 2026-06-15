#!/usr/bin/env python
"""
NORM variants comparison — results from main_slim_normalized.py.

Produces three figures in main/log/figs/:
  norm_curves.png      — median test-RMSE convergence curves per dataset
  norm_lastgen.png     — last-generation boxplots: Test RMSE, M_phi, Size
  norm_scatter.png     — median (Test RMSE vs M_phi) and (RMSE vs Size) per dataset

Run from the project root:
    python main/plot_norm_comparison.py
"""
import os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')
GEN_LOG = os.path.join(LOG_DIR, 'results_normalized_generations.csv')
OUT_DIR = os.path.join(LOG_DIR, 'figs')
os.makedirs(OUT_DIR, exist_ok=True)

# ── NORM variants only ────────────────────────────────────────────────────────
NORM_ALGOS = [
    'SLIM+NORM1',   'SLIM*NORM1',
    'SLIM+NORM2',   'SLIM*NORM2',
    'SLIM+NORMROB', 'SLIM*NORMROB',
    'SLIM+NORM12',  'SLIM*NORM12',
]

# Colours: light=+, dark=*; purple=NORM1, brown=NORM2, orange=NORMROB, pink=NORM12
_PALETTE = [
    '#c5b0d5', '#9467bd',   # NORM1
    '#c49c94', '#8c564b',   # NORM2
    '#ffbb78', '#ff7f0e',   # NORMROB
    '#f7b6d2', '#e377c2',   # NORM12
]
ALGO_COLOR = {a: c for a, c in zip(NORM_ALGOS, _PALETTE)}
ALGO_LS    = {a: ('-' if '+' in a else '--') for a in NORM_ALGOS}

DATASETS = ['concrete', 'energy', 'instanbul', 'ppb', 'resid_build_sale_price', 'toxicity']
DS_LABELS = {
    'concrete':               'Concrete',
    'energy':                 'Energy',
    'instanbul':              'Istanbul',
    'ppb':                    'PPB',
    'resid_build_sale_price': 'Resid. Build',
    'toxicity':               'Toxicity',
}

# ── Load & filter ─────────────────────────────────────────────────────────────
print('Loading …', flush=True)
df = pd.read_csv(GEN_LOG,
                 usecols=['algo', 'dataset', 'seed', 'generation',
                          'train_fitness', 'test_fitness', 'nodes_count', 'm_phi'])
df = df[df['algo'].isin(NORM_ALGOS)].copy()
print(f'  {len(df):,} rows | {df["algo"].nunique()} algos | '
      f'gens 0–{df["generation"].max()}', flush=True)

last_gen = int(df['generation'].max())
df_last  = df[df['generation'] == last_gen].copy()

STEP = max(1, last_gen // 100)   # plot ~100 points per curve

# ── Shared legend handles ─────────────────────────────────────────────────────
def _legend_handles():
    handles = []
    norm_types = [('NORM1','#c5b0d5','#9467bd'),
                  ('NORM2','#c49c94','#8c564b'),
                  ('NORMROB','#ffbb78','#ff7f0e'),
                  ('NORM12','#f7b6d2','#e377c2')]
    for label, c_plus, c_mul in norm_types:
        handles.append(Line2D([0],[0], color=c_plus, lw=2.0,
                               linestyle='-',  label=f'SLIM+{label}'))
        handles.append(Line2D([0],[0], color=c_mul,  lw=2.0,
                               linestyle='--', label=f'SLIM*{label}'))
    return handles


########################################################################
# Figure 1 — Convergence curves (test RMSE)
########################################################################

def plot_curves(save_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.suptitle('Test RMSE convergence — NORM variants  (median ± Q25/Q75, 30 seeds)',
                 fontsize=12)

    for ax, ds in zip(axes.flat, DATASETS):
        sub = df[df['dataset'] == ds]
        sub_s = sub[sub['generation'] % STEP == 0]

        for algo in NORM_ALGOS:
            g = sub_s[sub_s['algo'] == algo].groupby('generation')['test_fitness']
            med  = g.median()
            q25  = g.quantile(0.25)
            q75  = g.quantile(0.75)
            c    = ALGO_COLOR[algo]
            ls   = ALGO_LS[algo]
            ax.plot(med.index, med.values, color=c, lw=1.6, ls=ls)
            ax.fill_between(med.index, q25.values, q75.values,
                            color=c, alpha=0.12)

        # y-axis capped at 95th pct of all NORM data for readability
        yvals = sub_s['test_fitness'].replace([np.inf, -np.inf], np.nan).dropna()
        ax.set_ylim(0, np.percentile(yvals, 95) * 1.05)
        ax.set_title(DS_LABELS[ds], fontsize=10)
        ax.set_xlabel('Generation', fontsize=8)
        ax.set_ylabel('Test RMSE', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2, lw=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    fig.legend(handles=_legend_handles(), loc='lower center', ncol=4,
               fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {save_path}')


########################################################################
# Figure 2 — Last-gen boxplots: Test RMSE, M_phi, Size
########################################################################

def plot_lastgen(save_path):
    metrics = [
        ('test_fitness', 'Test RMSE',        False),
        ('m_phi',        r'M$_\phi$',         True),
        ('nodes_count',  'Size (blocks)',      False),
    ]
    n_ds = len(DATASETS)
    n_m  = len(metrics)

    fig, axes = plt.subplots(n_ds, n_m, figsize=(4.5 * n_m, 3.2 * n_ds), squeeze=False)
    fig.suptitle(f'Last generation (gen {last_gen}) distributions — NORM variants',
                 fontsize=12, y=1.002)

    xs = np.arange(len(NORM_ALGOS))
    xlabels = [a.replace('SLIM+', '+').replace('SLIM*', '*') for a in NORM_ALGOS]

    for row, ds in enumerate(DATASETS):
        sub = df_last[df_last['dataset'] == ds]
        for col, (metric, ylabel, higher_better) in enumerate(metrics):
            ax = axes[row][col]
            data = [sub[sub['algo'] == a][metric].dropna().values for a in NORM_ALGOS]

            bp = ax.boxplot(data, positions=xs, widths=0.55, patch_artist=True,
                            medianprops=dict(color='black', linewidth=1.8),
                            whiskerprops=dict(linewidth=0.8),
                            capprops=dict(linewidth=0.8),
                            flierprops=dict(marker='.', markersize=2, alpha=0.4),
                            boxprops=dict(linewidth=0.7))
            for patch, algo in zip(bp['boxes'], NORM_ALGOS):
                patch.set_facecolor(ALGO_COLOR[algo])
                patch.set_alpha(0.85)

            # y clip to 5th–95th pct
            flat = np.concatenate([d for d in data if len(d)])
            flat = flat[np.isfinite(flat)]
            if len(flat):
                lo, hi = np.percentile(flat, 5), np.percentile(flat, 95)
                pad = (hi - lo) * 0.12 or hi * 0.05
                ax.set_ylim(lo - pad, hi + pad)

            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels, rotation=40, ha='right', fontsize=7)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(axis='y', alpha=0.2, lw=0.4)
            ax.spines[['top', 'right']].set_visible(False)

            if row == 0:
                ax.set_title(ylabel, fontsize=10, pad=4)
            if col == 0:
                ax.text(-0.32, 0.5, DS_LABELS[ds], transform=ax.transAxes,
                        fontsize=9, va='center', ha='center', rotation=90)

    plt.tight_layout(rect=[0.04, 0, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {save_path}')


########################################################################
# Figure 3 — Scatter: Test RMSE vs M_phi  and  RMSE vs Size
########################################################################

def plot_scatter(save_path):
    fig, axes = plt.subplots(2, len(DATASETS) + 1,
                             figsize=(3.8 * (len(DATASETS) + 1), 7.5),
                             squeeze=False)

    panel_cfg = [
        ('m_phi',       r'M$_\phi$',   False, False),   # row 0
        ('nodes_count', 'Size (blocks)', True,  True),   # row 1
    ]

    # per-(algo,dataset) medians at last gen
    med = (df_last.groupby(['algo', 'dataset'])[['test_fitness', 'nodes_count', 'm_phi']]
                  .median().reset_index())

    for row, (xcol, xlabel, xlog, x_lower_better) in enumerate(panel_cfg):
        for col, ds in enumerate(DATASETS):
            ax = axes[row][col]
            ds_med  = med[med['dataset'] == ds]
            ds_full = df_last[df_last['dataset'] == ds]

            for algo in NORM_ALGOS:
                sub_m = ds_med[ds_med['algo'] == algo]
                sub_f = ds_full[ds_full['algo'] == algo]
                if sub_m.empty:
                    continue
                mx = sub_m[xcol].values[0]
                my = sub_m['test_fitness'].values[0]
                ex = [[mx - sub_f[xcol].quantile(0.25)],
                      [sub_f[xcol].quantile(0.75) - mx]]
                ey = [[my - sub_f['test_fitness'].quantile(0.25)],
                      [sub_f['test_fitness'].quantile(0.75) - my]]
                mkr = 'o' if '+' in algo else '^'
                ax.errorbar(mx, my, xerr=ex, yerr=ey,
                            fmt=mkr, color=ALGO_COLOR[algo], markersize=8,
                            elinewidth=0.9, capsize=3,
                            markeredgecolor='black', markeredgewidth=0.5, zorder=4)

            if xlog:
                ax.set_xscale('log')
            if row == 0:
                ax.set_title(DS_LABELS[ds], fontsize=10, pad=3)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel('Test RMSE', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2, lw=0.4)
            ax.spines[['top', 'right']].set_visible(False)

        # "All datasets" panel — normalised medians
        ax_all = axes[row][-1]
        rows_norm = []
        for ds in DATASETS:
            ds_med = med[med['dataset'] == ds]
            algos_ds = ds_med['algo'].values
            xv = ds_med[xcol].values
            yv = ds_med['test_fitness'].values
            rng_x = xv.max() - xv.min() if xv.max() != xv.min() else 1.0
            rng_y = yv.max() - yv.min() if yv.max() != yv.min() else 1.0
            for _, r in ds_med.iterrows():
                xn = (r[xcol] - xv.min()) / rng_x
                yn = (r['test_fitness'] - yv.min()) / rng_y
                if not x_lower_better:   # M_phi: higher=better → flip
                    xn = 1 - xn
                rows_norm.append({'algo': r['algo'], 'x_norm': xn, 'y_norm': yn})
        norm_df = pd.DataFrame(rows_norm)
        for algo in NORM_ALGOS:
            sub = norm_df[norm_df['algo'] == algo]
            if sub.empty:
                continue
            mkr = 'o' if '+' in algo else '^'
            ax_all.scatter(sub['x_norm'], sub['y_norm'],
                           color=ALGO_COLOR[algo], marker=mkr,
                           s=25, alpha=0.5, linewidths=0.3, edgecolors='white', zorder=2)
            ax_all.scatter(sub['x_norm'].median(), sub['y_norm'].median(),
                           color=ALGO_COLOR[algo], marker=mkr,
                           s=110, alpha=1.0, linewidths=0.6, edgecolors='black', zorder=5)

        x_dir = 'lower=best' if x_lower_better else 'higher=best'
        ax_all.set_xlabel(f'{xlabel}\n[normalised, {x_dir} -> 0]', fontsize=8)
        ax_all.set_ylabel('Test RMSE  [normalised, 0=best]', fontsize=8)
        if row == 0:
            ax_all.set_title('All datasets\n(normalised)', fontsize=10, pad=3)
        ax_all.set_xlim(-0.05, 1.05)
        ax_all.set_ylim(-0.05, 1.05)
        ax_all.tick_params(labelsize=7)
        ax_all.grid(alpha=0.2, lw=0.4)
        ax_all.spines[['top', 'right']].set_visible(False)

    fig.legend(handles=_legend_handles(), loc='lower center', ncol=4,
               fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('Test RMSE vs interpretability / size  (median ± IQR, last gen)',
                 fontsize=12, y=1.002)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {save_path}')


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    plot_curves(os.path.join(OUT_DIR, 'norm_curves.png'))
    plot_lastgen(os.path.join(OUT_DIR, 'norm_lastgen.png'))
    plot_scatter(os.path.join(OUT_DIR, 'norm_scatter.png'))
    print('Done.')
