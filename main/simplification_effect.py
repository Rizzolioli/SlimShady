#!/usr/bin/env python
"""
Visualise the effect of SymPy simplification on node count (ell) and M_phi.
Only runs where simplified_ok == 1 are included.

Produces:
  log/figs/simp_effect.png  — 3-panel figure
  prints a summary table
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')
SIMP_LOG = os.path.join(LOG_DIR, 'results_normalized_simplification.csv')
OUT_DIR  = os.path.join(LOG_DIR, 'figs')
os.makedirs(OUT_DIR, exist_ok=True)

ALGOS = [
    'SLIM+2SIG', 'SLIM*2SIG',
    'SLIM+ABS',  'SLIM*ABS',
    'SLIM+1SIG', 'SLIM*1SIG',
    'SLIM+NORM1','SLIM*NORM1',
    'SLIM+NORM2','SLIM*NORM2',
]
_PALETTE = [
    '#1f77b4', '#aec7e8',
    '#d62728', '#ff9896',
    '#2ca02c', '#98df8a',
    '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94',
]
ALGO_COLOR = {a: c for a, c in zip(ALGOS, _PALETTE)}

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(SIMP_LOG,
                 usecols=['algo', 'dataset', 'seed',
                           'ell_before', 'ell_after',
                           'm_phi_before', 'm_phi_after',
                           'simplified_ok'])
# When SymPy made things worse, treat it as the original (analysis-side filter)
df['ell_after']   = df[['ell_after',   'ell_before'  ]].min(axis=1)
df['m_phi_after'] = df[['m_phi_after', 'm_phi_before']].max(axis=1)

# Deltas computed AFTER the filter so they reflect the kept version
df['delta_ell']   = df['ell_after']   - df['ell_before']
df['delta_m_phi'] = df['m_phi_after'] - df['m_phi_before']
df['pct_ell']     = df['delta_ell'] / df['ell_before'] * 100

total_per_algo = df.groupby('algo').size()
ok = df[df['simplified_ok'] == 1].copy()
ok_per_algo    = ok.groupby('algo').size()
success_rate   = (ok_per_algo / total_per_algo * 100).reindex(ALGOS).fillna(0)

# ── Summary table ─────────────────────────────────────────────────────────────
summary = ok.groupby('algo').agg(
    n          =('ell_before', 'count'),
    ell_before =('ell_before', 'median'),
    ell_after  =('ell_after',  'median'),
    delta_ell  =('delta_ell',  'median'),
    pct_ell    =('pct_ell',    'median'),
    mphi_before=('m_phi_before','median'),
    mphi_after =('m_phi_after', 'median'),
    delta_mphi =('delta_m_phi', 'median'),
).reindex(ALGOS)

print('\nSimplification effect (successfully simplified runs only)')
print('='*80)
print(f'{"Algorithm":<14} {"n":>4}  {"ell_before":>10} {"ell_after":>9} '
      f'{"delta_ell":>9} {"pct_ell":>8}  '
      f'{"mphi_before":>11} {"mphi_after":>10} {"delta_mphi":>10}')
print('-'*80)
for algo in ALGOS:
    r = summary.loc[algo]
    if pd.isna(r['n']):
        print(f'{algo:<14}  {"—":>4}  {"(no successful simplifications)"}')
    else:
        print(f'{algo:<14} {int(r["n"]):>4}  '
              f'{r["ell_before"]:>10.0f} {r["ell_after"]:>9.0f} '
              f'{r["delta_ell"]:>+9.0f} {r["pct_ell"]:>+7.1f}%  '
              f'{r["mphi_before"]:>11.1f} {r["mphi_after"]:>10.1f} '
              f'{r["delta_mphi"]:>+10.1f}')
print('='*80)

# ── Figure ────────────────────────────────────────────────────────────────────
# Algorithms that have at least one successful simplification
has_ok = [a for a in ALGOS if a in ok['algo'].values]
x = np.arange(len(ALGOS))
x_ok = np.arange(len(has_ok))

fig, axes = plt.subplots(3, 1, figsize=(13, 13))
fig.suptitle('Effect of SymPy simplification on model complexity',
             fontsize=13, y=1.002)

# ── Panel 1: success rate ─────────────────────────────────────────────────────
ax = axes[0]
bars = ax.bar(x, success_rate.values,
              color=[ALGO_COLOR[a] for a in ALGOS],
              edgecolor='black', linewidth=0.5, width=0.6)
ax.set_xticks(x)
ax.set_xticklabels(ALGOS, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Success rate (%)', fontsize=10)
ax.set_title('Simplification success rate (60 s timeout, 30 seeds x 6 datasets = 180 runs/algo)',
             fontsize=10)
ax.set_ylim(0, max(success_rate.values) * 1.25)
for bar, rate in zip(bars, success_rate.values):
    if rate > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=8)
ax.grid(axis='y', alpha=0.3, lw=0.5)
ax.spines[['top', 'right']].set_visible(False)

# ── Panel 2: ell before vs after (only algos with successful runs) ─────────────
ax = axes[1]
bw = 0.35
for i, algo in enumerate(has_ok):
    sub = ok[ok['algo'] == algo]
    q25_b, med_b, q75_b = sub['ell_before'].quantile([0.25, 0.5, 0.75])
    q25_a, med_a, q75_a = sub['ell_after'].quantile([0.25, 0.5, 0.75])
    color = ALGO_COLOR[algo]
    # before bar (lighter)
    ax.bar(i - bw/2, med_b, bw,
           color=color, alpha=0.45, edgecolor='black', lw=0.6, label='_')
    ax.errorbar(i - bw/2, med_b, [[med_b - q25_b], [q75_b - med_b]],
                fmt='none', color='black', capsize=3, lw=0.8)
    # after bar (darker)
    ax.bar(i + bw/2, med_a, bw,
           color=color, alpha=0.9, edgecolor='black', lw=0.6, label='_')
    ax.errorbar(i + bw/2, med_a, [[med_a - q25_a], [q75_a - med_a]],
                fmt='none', color='black', capsize=3, lw=0.8)

ax.set_xticks(x_ok)
ax.set_xticklabels(has_ok, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Node count (ell)  [median + IQR]', fontsize=10)
ax.set_title('Node count before (light) vs after (dark) simplification', fontsize=10)
ax.legend(handles=[Patch(facecolor='grey', alpha=0.4, label='Before'),
                   Patch(facecolor='grey', alpha=0.9, label='After')],
          fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3, lw=0.5)
ax.spines[['top', 'right']].set_visible(False)

# ── Panel 3: M_phi before vs after ───────────────────────────────────────────
ax = axes[2]
for i, algo in enumerate(has_ok):
    sub = ok[ok['algo'] == algo]
    q25_b, med_b, q75_b = sub['m_phi_before'].quantile([0.25, 0.5, 0.75])
    q25_a, med_a, q75_a = sub['m_phi_after'].quantile([0.25, 0.5, 0.75])
    color = ALGO_COLOR[algo]
    ax.bar(i - bw/2, med_b, bw,
           color=color, alpha=0.45, edgecolor='black', lw=0.6)
    ax.errorbar(i - bw/2, med_b, [[med_b - q25_b], [q75_b - med_b]],
                fmt='none', color='black', capsize=3, lw=0.8)
    ax.bar(i + bw/2, med_a, bw,
           color=color, alpha=0.9, edgecolor='black', lw=0.6)
    ax.errorbar(i + bw/2, med_a, [[med_a - q25_a], [q75_a - med_a]],
                fmt='none', color='black', capsize=3, lw=0.8)

ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_xticks(x_ok)
ax.set_xticklabels(has_ok, rotation=30, ha='right', fontsize=9)
ax.set_ylabel(r'M$_\phi$  [median + IQR]', fontsize=10)
ax.set_title(r'M$_\phi$ before (light) vs after (dark) simplification', fontsize=10)
ax.legend(handles=[Patch(facecolor='grey', alpha=0.4, label='Before'),
                   Patch(facecolor='grey', alpha=0.9, label='After')],
          fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3, lw=0.5)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'simp_effect.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved -> {out_path}')
print('Done.')
