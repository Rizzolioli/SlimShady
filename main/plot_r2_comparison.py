#!/usr/bin/env python
"""
plot_r2_comparison.py — compare R²-trained (neg_r2 fitness) vs RMSE-baseline.

Uses:
  log/results_normalized_generations.csv   (baseline: test_fitness = RMSE)
  log/results_r2_generations.csv           (R²-trained: test_fitness = neg_r2;
                                             rmse col is corrupt = log level,
                                             so RMSE is derived from r2 + std(y_test))

Produces in log/figs/:
  r2_comparison_scatter.png   — per-(algo,dataset) median scatter: baseline vs R²-trained
  r2_comparison_lastgen.png   — last-gen boxplots: 3 metrics × 6 datasets, all 14 algos
  r2_comparison_curves.png    — test-R² convergence curves (both objectives)

Run from project root:
    python main/plot_r2_comparison.py
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
OUT_DIR = os.path.join(LOG_DIR, 'figs')
os.makedirs(OUT_DIR, exist_ok=True)

BASE_LOG = os.path.join(LOG_DIR, 'results_normalized_generations.csv')
R2_LOG   = os.path.join(LOG_DIR, 'results_r2_generations.csv')

DATASETS = ['concrete', 'energy', 'instanbul', 'ppb', 'resid_build_sale_price', 'toxicity']
DS_LABELS = {
    'concrete':               'Concrete',
    'energy':                 'Energy',
    'instanbul':              'Istanbul',
    'ppb':                    'PPB',
    'resid_build_sale_price': 'Resid. Build',
    'toxicity':               'Toxicity',
}

ALGOS = [
    'SLIM+2SIG', 'SLIM*2SIG',
    'SLIM+1SIG', 'SLIM*1SIG',
    'SLIM+ABS',  'SLIM*ABS',
    'SLIM+NORM1',   'SLIM*NORM1',
    'SLIM+NORM2',   'SLIM*NORM2',
    'SLIM+NORMROB', 'SLIM*NORMROB',
    'SLIM+NORM12',  'SLIM*NORM12',
]
SHORT_LABELS = [a.replace('SLIM+', '+').replace('SLIM*', '*') for a in ALGOS]

# Palette: 7 families × light/dark (+ / *)
_FAM_COLORS = {
    '2SIG':    ('#aec7e8', '#1f77b4'),
    '1SIG':    ('#98df8a', '#2ca02c'),
    'ABS':     ('#ffbb78', '#ff7f0e'),
    'NORM1':   ('#c5b0d5', '#9467bd'),
    'NORM2':   ('#c49c94', '#8c564b'),
    'NORMROB': ('#f7b6d2', '#e377c2'),
    'NORM12':  ('#dbdb8d', '#bcbd22'),
}

def _algo_color(algo):
    for fam, (light, dark) in _FAM_COLORS.items():
        if fam in algo:
            return light if '+' in algo else dark
    return '#aaaaaa'

ALGO_COLOR = {a: _algo_color(a) for a in ALGOS}
ALGO_MARKER = {a: ('o' if '+' in a else '^') for a in ALGOS}

# ── Load y_test std per (dataset, seed) for RMSE derivation ──────────────────
print('Loading test-set stds for RMSE derivation…', flush=True)
from datasets.data_loader import load_preloaded

y_test_std = {}
for ds in DATASETS:
    for seed in range(30):
        _, y_test = load_preloaded(ds, seed=seed + 1, training=False, X_y=True)
        y_test_std[(ds, seed)] = float(y_test.std())

def _derive_rmse(df):
    """Derive RMSE from r2 col + std(y_test) per (dataset, seed)."""
    stds = df.apply(lambda r: y_test_std.get((r['dataset'], r['seed']), np.nan), axis=1)
    return stds * np.sqrt(np.maximum(1.0 - df['r2'], 0.0))

# ── Load experiment logs ──────────────────────────────────────────────────────
COLS_BASE = ['algo', 'dataset', 'seed', 'generation',
             'test_fitness', 'r2', 'mae', 'nodes_count']
COLS_R2   = ['algo', 'dataset', 'seed', 'generation',
             'test_fitness', 'r2', 'mae', 'nodes_count']

print('Loading baseline…', flush=True)
base = pd.read_csv(BASE_LOG, usecols=COLS_BASE)
base['rmse'] = base['test_fitness']          # test_fitness IS rmse in baseline
base['objective'] = 'RMSE'

print('Loading R²-trained…', flush=True)
r2e = pd.read_csv(R2_LOG, usecols=COLS_R2)
r2e['rmse'] = _derive_rmse(r2e)             # derived from r2 col + std(y_test)
r2e['objective'] = 'R²'

df_all = pd.concat([base, r2e], ignore_index=True)
df_all = df_all[df_all['algo'].isin(ALGOS)]

last_gen = int(df_all['generation'].max())
df_last  = df_all[df_all['generation'] == last_gen].copy()

print(f'  {len(df_all):,} rows total | last gen = {last_gen}', flush=True)

STEP = max(1, last_gen // 100)

# ── Helpers ───────────────────────────────────────────────────────────────────
BASE_STYLE = dict(linestyle='-',  alpha=0.9, lw=1.6)
R2_STYLE   = dict(linestyle='--', alpha=0.7, lw=1.6)

def _legend_objective():
    return [
        Line2D([0], [0], color='#555', lw=2, ls='-',  label='RMSE-trained (baseline)'),
        Line2D([0], [0], color='#555', lw=2, ls='--', label='R²-trained'),
    ]

def _legend_algos():
    handles = []
    for algo in ALGOS:
        sym = '●' if '+' in algo else '▲'
        handles.append(mpatches.Patch(color=ALGO_COLOR[algo],
                                      label=f'{sym} {algo}'))
    return handles


########################################################################
# Figure 1 — Scatter: baseline median vs R²-trained median
########################################################################

def plot_scatter(save_path):
    metrics = [
        ('rmse',        'Test RMSE',       True),   # lower_better
        ('r2',          'Test R²',         False),
        ('mae',         'Test MAE',        True),
        ('nodes_count', 'Size (blocks)',   True),
    ]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.2), squeeze=False)
    fig.suptitle('Baseline (x) vs R²-trained (y) — per (algo, dataset) median at last gen',
                 fontsize=11)

    med = (df_last.groupby(['algo', 'dataset', 'objective'])
           [['rmse', 'r2', 'mae', 'nodes_count']].median().reset_index())

    for col, (metric, label, lower_better) in enumerate(metrics):
        ax = axes[0][col]
        base_m = med[med['objective'] == 'RMSE']
        r2_m   = med[med['objective'] == 'R²']
        merged = base_m.merge(r2_m, on=['algo', 'dataset'], suffixes=('_base', '_r2'))

        all_vals = []
        for algo in ALGOS:
            sub = merged[merged['algo'] == algo]
            if sub.empty:
                continue
            xv = sub[f'{metric}_base'].values
            yv = sub[f'{metric}_r2'].values
            ax.scatter(xv, yv, color=ALGO_COLOR[algo],
                       marker=ALGO_MARKER[algo], s=55, alpha=0.75,
                       edgecolors='black', linewidths=0.4, zorder=4)
            all_vals.extend(xv.tolist())
            all_vals.extend(yv.tolist())

        # Diagonal
        finite = [v for v in all_vals if np.isfinite(v)]
        if finite:
            lo = np.percentile(finite, 2)
            hi = np.percentile(finite, 96)
            ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.4, zorder=1)
            pad = (hi - lo) * 0.05
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)

        win = 'lower-left wins' if lower_better else 'upper-right wins'
        ax.set_xlabel(f'Baseline {label}', fontsize=9)
        ax.set_ylabel(f'R²-trained {label}', fontsize=9)
        ax.set_title(f'{label}\n({win})', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2, lw=0.4)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_aspect('equal', adjustable='box')

    # Shared legend (algos)
    fig.legend(handles=_legend_algos(), loc='lower center', ncol=7,
               fontsize=7, frameon=True, bbox_to_anchor=(0.5, -0.14))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {save_path}')


########################################################################
# Figure 2 — Last-gen boxplots: 3 metrics × 6 datasets
########################################################################

def plot_lastgen(save_path):
    metrics = [
        ('rmse',        'Test RMSE',     True,  [0, 95]),
        ('r2',          'Test R²',       False, [5, 95]),
        ('nodes_count', 'Size (blocks)', True,  [0, 95]),
    ]
    n_ds = len(DATASETS)
    n_m  = len(metrics)
    n_obj = 2  # baseline, r2-trained
    xs_base = np.arange(len(ALGOS)) * (n_obj + 1)
    xs_r2   = xs_base + 1

    fig, axes = plt.subplots(n_m, n_ds,
                             figsize=(3.8 * n_ds, 3.8 * n_m), squeeze=False)
    fig.suptitle(f'Last generation (gen {last_gen}) — RMSE-baseline vs R²-trained',
                 fontsize=11, y=1.002)

    for row, (metric, ylabel, lower_better, pct_range) in enumerate(metrics):
        for col, ds in enumerate(DATASETS):
            ax = axes[row][col]
            sub = df_last[df_last['dataset'] == ds]
            sub_b = sub[sub['objective'] == 'RMSE']
            sub_r = sub[sub['objective'] == 'R²']

            bdata = [sub_b[sub_b['algo'] == a][metric].dropna().values for a in ALGOS]
            rdata = [sub_r[sub_r['algo'] == a][metric].dropna().values for a in ALGOS]

            bp_b = ax.boxplot(bdata, positions=xs_base, widths=0.45,
                              patch_artist=True,
                              medianprops=dict(color='black', lw=1.6),
                              whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7),
                              flierprops=dict(marker='.', ms=2, alpha=0.35),
                              boxprops=dict(lw=0.6))
            bp_r = ax.boxplot(rdata, positions=xs_r2, widths=0.45,
                              patch_artist=True,
                              medianprops=dict(color='#d62728', lw=1.6),
                              whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7),
                              flierprops=dict(marker='.', ms=2, alpha=0.35),
                              boxprops=dict(lw=0.6))

            for patch, algo in zip(bp_b['boxes'], ALGOS):
                patch.set_facecolor(ALGO_COLOR[algo])
                patch.set_alpha(0.85)
            for patch, algo in zip(bp_r['boxes'], ALGOS):
                patch.set_facecolor(ALGO_COLOR[algo])
                patch.set_alpha(0.45)
                patch.set_hatch('///')

            flat = np.concatenate([d for d in bdata + rdata if len(d)])
            flat = flat[np.isfinite(flat)]
            if len(flat):
                lo = np.percentile(flat, pct_range[0])
                hi = np.percentile(flat, pct_range[1])
                pad = (hi - lo) * 0.12 or abs(hi) * 0.05
                ax.set_ylim(lo - pad, hi + pad)

            tick_x = (xs_base + xs_r2) / 2
            ax.set_xticks(tick_x)
            ax.set_xticklabels(SHORT_LABELS, rotation=45, ha='right', fontsize=6)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(axis='y', alpha=0.2, lw=0.4)
            ax.spines[['top', 'right']].set_visible(False)

            if row == 0:
                ax.set_title(DS_LABELS[ds], fontsize=10, pad=3)
            if col == 0:
                ax.text(-0.30, 0.5, ylabel, transform=ax.transAxes,
                        fontsize=8, va='center', ha='center', rotation=90)

    legend_handles = [
        mpatches.Patch(facecolor='#888', alpha=0.85, label='RMSE-trained (baseline)'),
        mpatches.Patch(facecolor='#888', alpha=0.45, hatch='///', label='R²-trained'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {save_path}')


########################################################################
# Figure 3 — Test-R² convergence curves (both objectives)
########################################################################

def plot_curves(save_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.suptitle('Test R² convergence — RMSE-baseline (—) vs R²-trained (- -)\n'
                 'median ± Q25/Q75, 30 seeds', fontsize=11)

    for ax, ds in zip(axes.flat, DATASETS):
        for algo in ALGOS:
            c = ALGO_COLOR[algo]
            for obj, style in [('RMSE', BASE_STYLE), ('R²', R2_STYLE)]:
                sub = df_all[(df_all['dataset'] == ds) &
                             (df_all['algo']    == algo) &
                             (df_all['objective'] == obj)]
                sub_s = sub[sub['generation'] % STEP == 0]
                g = sub_s.groupby('generation')['r2']
                med = g.median()
                q25 = g.quantile(0.25)
                q75 = g.quantile(0.75)
                ax.plot(med.index, med.values, color=c, **style)
                ax.fill_between(med.index, q25.values, q75.values,
                                color=c, alpha=0.07)

        yvals = df_all[(df_all['dataset'] == ds) &
                       (df_all['generation'] % STEP == 0)]['r2'].replace(
                           [np.inf, -np.inf], np.nan).dropna()
        if len(yvals):
            lo = np.percentile(yvals, 5)
            hi = np.percentile(yvals, 97)
            ax.set_ylim(lo, hi)

        ax.set_title(DS_LABELS[ds], fontsize=10)
        ax.set_xlabel('Generation', fontsize=8)
        ax.set_ylabel('Test R²', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color='#aaa', lw=0.6, ls=':')
        ax.grid(alpha=0.2, lw=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    obj_handles = _legend_objective()
    algo_handles = [Line2D([0], [0], color=ALGO_COLOR[a], lw=2, label=a) for a in ALGOS]
    fig.legend(handles=obj_handles + algo_handles,
               loc='lower center', ncol=8, fontsize=7.5,
               frameon=True, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {save_path}')


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    plot_scatter(os.path.join(OUT_DIR, 'r2_comparison_scatter.png'))
    plot_lastgen(os.path.join(OUT_DIR, 'r2_comparison_lastgen.png'))
    plot_curves(os.path.join(OUT_DIR, 'r2_comparison_curves.png'))
    print('Done.')
