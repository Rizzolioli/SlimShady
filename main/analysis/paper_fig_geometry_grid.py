"""
paper_fig_geometry_grid.py — offspring-geometry grid, all 16 SLIM variants.

Single-condition version of main/geometry_study.py: fixed ms=1.0 and a single
interval r~U(-10,10) (instead of the 3-interval x 2-ms sweep), so each of the
16 named variants gets exactly one 2D offspring-cloud panel in a 4x4 grid.
Reuses the per-operator formulas and sampling helpers from geometry_study.py
verbatim; the SIG-family formulas are commented out there, so they're
reproduced here (same math) to complete the 16-variant set.

Run from the project root:
    python main/analysis/paper_fig_geometry_grid.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MAIN = os.path.join(_ROOT, 'main')
for p in (_MAIN, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from geometry_study import (
    T,
    norm1_sum, norm1_mul,
    norm2_sum, norm2_mul,
    normrob_sum, normrob_mul,
    norm12_sum, norm12_mul,
    normfix_sum, normfix_mul,
    slim_plus_abs, slim_mul_abs,
    sample_offspring_one_tree, sample_offspring_two_trees,
)
from paper_fig_common import VARIANT_ORDER, variant_color, save_all

OUT_DIR = os.path.join(_MAIN, 'paper_figures')

# ── SIG family: commented out in geometry_study.py, reproduced here verbatim ──

def slim_plus_2sig(r1, r2, ms, T=T):
    """SLIM+2SIG: T + ms * (sigmoid(r1) - sigmoid(r2))"""
    return T + ms * (1 / (1 + np.exp(-r1)) - 1 / (1 + np.exp(-r2)))

def slim_mul_2sig(r1, r2, ms, T=T):
    """SLIM*2SIG: T * (1 + ms * (sigmoid(r1) - sigmoid(r2)))"""
    return T * (1 + ms * (1 / (1 + np.exp(-r1)) - 1 / (1 + np.exp(-r2))))

def slim_plus_1sig(r, ms, T=T):
    """SLIM+1SIG: T + ms * (2*sigmoid(r) - 1)"""
    return T + ms * (2 / (1 + np.exp(-r)) - 1)

def slim_mul_1sig(r, ms, T=T):
    """SLIM*1SIG: T * (1 + ms * (2*sigmoid(r) - 1))"""
    return T * (1 + ms * (2 / (1 + np.exp(-r)) - 1))

# ── Single fixed condition ─────────────────────────────────────────────────
INTERVAL   = (-10, 10)
MS_VALUE   = 1.0
N_SAMPLES  = 1500

# variant name -> (two_trees, apply_fn)
_APPLY_FN = {
    'SLIM+2SIG':    (True,  slim_plus_2sig),
    'SLIM*2SIG':    (True,  slim_mul_2sig),
    'SLIM+ABS':     (False, slim_plus_abs),
    'SLIM*ABS':     (False, slim_mul_abs),
    'SLIM+1SIG':    (False, slim_plus_1sig),
    'SLIM*1SIG':    (False, slim_mul_1sig),
    'SLIM+NORM1':   (False, norm1_sum),
    'SLIM*NORM1':   (False, norm1_mul),
    'SLIM+NORM2':   (True,  norm2_sum),
    'SLIM*NORM2':   (True,  norm2_mul),
    'SLIM+NORMROB': (True,  normrob_sum),
    'SLIM*NORMROB': (True,  normrob_mul),
    'SLIM+NORM12':  (True,  norm12_sum),
    'SLIM*NORM12':  (True,  norm12_mul),
    'SLIM+NORMFIX': (False, normfix_sum),
    'SLIM*NORMFIX': (False, normfix_mul),
}


def make_plot():
    n = len(VARIANT_ORDER)
    ncols = 4
    nrows = -(-n // ncols)  # ceil

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
    low, high = INTERVAL
    ms_fn = lambda: MS_VALUE  # noqa: E731

    for idx, algo in enumerate(VARIANT_ORDER):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        two_trees, apply_fn = _APPLY_FN[algo]

        np.random.seed(42)
        if two_trees:
            pts = sample_offspring_two_trees(apply_fn, low, high, ms_fn, n=N_SAMPLES)
        else:
            pts = sample_offspring_one_tree(apply_fn, low, high, ms_fn, n=N_SAMPLES)

        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.35,
                   c=variant_color(algo), rasterized=True)
        ax.scatter(*T, s=60, c='red', marker='+', zorder=5, linewidths=1.5)

        ax.set_title(algo.replace('SLIM', ''), fontsize=9)
        ax.tick_params(labelsize=6)

    # hide any unused trailing cells
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis('off')

    fig.suptitle(
        f'Offspring distribution in 2D semantic space  (parent = (10, 10) ✚,  '
        f'r~U{INTERVAL},  ms={MS_VALUE})',
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    fig = make_plot()
    save_all(fig, OUT_DIR, 'geometry_grid')
    plt.close(fig)
