#!/usr/bin/env python
"""
Visualise results from main_slim_normalized.py.

Produces two figures saved to main/log/figs/:
  curves_normalized.png   — median + Q25/Q75 ribbon convergence curves (30 seeds)
  boxplots_normalized.png — last-generation distributions, y-axis capped at 95th pct

Metrics shown: Train RMSE, Test RMSE, Size (blocks), M_phi
Layout: 6 rows (datasets) x 4 cols (metrics) for both figures.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_DIR  = os.path.join(os.path.dirname(__file__), 'log')
GEN_LOG  = os.path.join(LOG_DIR, 'results_normalized_generations.csv')
OUT_DIR  = os.path.join(LOG_DIR, 'figs')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Layout constants ───────────────────────────────────────────────────────────
ALGOS = [
    'SLIM+2SIG', 'SLIM*2SIG',
    'SLIM+ABS',  'SLIM*ABS',
    'SLIM+1SIG', 'SLIM*1SIG',
    'SLIM+NORM1','SLIM*NORM1',
    'SLIM+NORM2','SLIM*NORM2',
]

DATASETS = ['concrete', 'energy', 'instanbul', 'ppb', 'resid_build_sale_price', 'toxicity']
DS_LABELS = {
    'concrete':              'Concrete',
    'energy':                'Energy',
    'instanbul':             'Istanbul',
    'ppb':                   'PPB',
    'resid_build_sale_price':'Resid. Build',
    'toxicity':              'Toxicity',
}

METRICS = [
    ('train_fitness', 'Train RMSE'),
    ('test_fitness',  'Test RMSE'),
    ('nodes_count',   'Size (blocks)'),
    ('m_phi',         r'M$_\phi$'),
]
METRIC_COLS = [m[0] for m in METRICS]

# Colours: paired (+/*) within each mutation family
_PALETTE = [
    '#1f77b4', '#aec7e8',   # 2SIG
    '#d62728', '#ff9896',   # ABS
    '#2ca02c', '#98df8a',   # 1SIG
    '#9467bd', '#c5b0d5',   # NORM1
    '#8c564b', '#c49c94',   # NORM2
]
ALGO_COLOR = {a: c for a, c in zip(ALGOS, _PALETTE)}
ALGO_LS    = {a: ('-' if '+' in a else '--') for a in ALGOS}

N_SAMPLE = 20   # plot every Nth generation to keep curves readable

# ── Load ──────────────────────────────────────────────────────────────────────
print('Loading generation log …', flush=True)
usecols = ['algo', 'dataset', 'seed', 'generation'] + METRIC_COLS
df = pd.read_csv(GEN_LOG, usecols=usecols)
print(f'  {len(df):,} rows | {df["algo"].nunique()} algos | '
      f'{df["dataset"].nunique()} datasets | {df["seed"].nunique()} seeds', flush=True)

# ── Aggregate for curves (median + Q25/Q75, robust to outliers) ───────────────
df_sample = df[df['generation'] % N_SAMPLE == 0].copy()

agg = (df_sample
       .groupby(['algo', 'dataset', 'generation'])[METRIC_COLS]
       .agg(['median',
             lambda x: x.quantile(0.25),
             lambda x: x.quantile(0.75)]))
agg.columns = [f'{m}_{s}' for m, s in agg.columns]
agg.columns = [c.replace('<lambda_0>', 'q25').replace('<lambda_1>', 'q75')
               for c in agg.columns]
agg = agg.reset_index()

# ── Last-generation data for boxplots ─────────────────────────────────────────
last_gen = int(df['generation'].max())
df_last  = df[df['generation'] == last_gen]
print(f'  Last generation: {last_gen}', flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Convergence curves
# ─────────────────────────────────────────────────────────────────────────────
print('Plotting convergence curves …', flush=True)
nrows = len(DATASETS)
ncols = len(METRICS)

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
fig.suptitle('SLIM-GSGP variants — convergence curves (median, 30 seeds)',
             fontsize=13, y=1.002)

for r, dataset in enumerate(DATASETS):
    ds_agg = agg[agg['dataset'] == dataset]
    for c, (col, label) in enumerate(METRICS):
        ax = axes[r][c]
        med_col = f'{col}_median'
        q25_col = f'{col}_q25'
        q75_col = f'{col}_q75'
        all_meds = []
        for algo in ALGOS:
            sub = ds_agg[ds_agg['algo'] == algo].sort_values('generation')
            if sub.empty:
                continue
            x   = sub['generation'].values
            med = sub[med_col].values
            ax.plot(x, med, color=ALGO_COLOR[algo], ls=ALGO_LS[algo],
                    lw=1.5, label=algo)
            all_meds.append(med)
        # ylim: 5th–95th pct of all median values so one diverging algo
        # doesn't collapse the visible range for the others
        if all_meds:
            combined = np.concatenate(all_meds)
            lo = float(np.percentile(combined, 5))
            hi = float(np.percentile(combined, 95))
            margin = (hi - lo) * 0.08 if hi != lo else abs(hi) * 0.1 + 1
            ax.set_ylim(lo - margin, hi + margin)
        if r == 0:
            ax.set_title(label, fontsize=11, pad=4)
        if c == 0:
            ax.set_ylabel(DS_LABELS[dataset], fontsize=10)
        if r == nrows - 1:
            ax.set_xlabel('Generation', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.25, lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)

legend_handles = [
    Line2D([0], [0], color=ALGO_COLOR[a], ls=ALGO_LS[a], lw=1.8, label=a)
    for a in ALGOS
]
fig.legend(handles=legend_handles, loc='lower center', ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.04), frameon=True,
           columnspacing=1.0, handlelength=2)
plt.tight_layout()
curves_path = os.path.join(OUT_DIR, 'curves_normalized.png')
fig.savefig(curves_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Saved -> {curves_path}')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Boxplots (last generation)
# ─────────────────────────────────────────────────────────────────────────────
print('Plotting last-generation boxplots …', flush=True)

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
fig2.suptitle(f'SLIM-GSGP variants — gen {last_gen} distribution (30 seeds)',
              fontsize=13, y=1.002)

tick_labels = [a.replace('SLIM', '') for a in ALGOS]

for r, dataset in enumerate(DATASETS):
    ds = df_last[df_last['dataset'] == dataset]
    for c, (col, label) in enumerate(METRICS):
        ax = axes2[r][c]
        data = [ds[ds['algo'] == a][col].dropna().values for a in ALGOS]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', lw=1.5),
            whiskerprops=dict(lw=0.7),
            capprops=dict(lw=0.7),
            whis=(5, 95),   # whiskers at 5th/95th percentile
        )
        for patch, algo in zip(bp['boxes'], ALGOS):
            patch.set_facecolor(ALGO_COLOR[algo])
            patch.set_alpha(0.72)
        # Cap y-axis at 5th–95th percentile of all data in this panel
        all_vals = np.concatenate([v for v in data if len(v)])
        if len(all_vals):
            lo = float(np.percentile(all_vals, 2))
            hi = float(np.percentile(all_vals, 98))
            margin = (hi - lo) * 0.08
            ax.set_ylim(lo - margin, hi + margin)
        ax.set_xticks(range(1, len(ALGOS) + 1))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)
        if r == 0:
            ax.set_title(label, fontsize=11, pad=4)
        if c == 0:
            ax.set_ylabel(DS_LABELS[dataset], fontsize=10)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', alpha=0.25, lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
boxplot_path = os.path.join(OUT_DIR, 'boxplots_normalized.png')
fig2.savefig(boxplot_path, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'  Saved -> {boxplot_path}')

print('\nDone.')
