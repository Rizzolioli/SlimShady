"""
paper_fig_geometry_N_scaling.py -- offspring geometry for NORM1/NORM2/NORMROB/
NORM12 as the normalization statistic's auxiliary-vector size N grows.

Companion to paper_fig_geometry_grid.py. That figure's 4x4 grid uses N=2
(exactly 2 auxiliary random draws per sample -- one per plotted axis), which
is the minimum needed to produce a 2D point but is degenerate for any
operator that normalizes by a statistic (min/max, max(|diff|), Q99(|diff|))
computed over that same array: with only 2 elements, "the min" and "the max"
*are* the whole array, so the normalized output collapses to exact discrete
values (NORM1/NORM12) or a hollow ring (NORM2/NORMROB) -- see the geometry
grid figure. This companion figure instead sweeps N (standing in for "how
many training-instance semantics contribute to the normalization statistic",
i.e. approaching the real algorithm's full semantic vector) to show how each
operator's TRUE asymptotic geometry differs from that N=2 degenerate corner
case, and in particular to make NORM2 vs NORMROB's outlier-tolerance
distinction visible (invisible at N=2, where Q99 of 2 numbers ~= their max).

Same operator formulas as main/geometry_study.py (verbatim), generalized to
an N-element auxiliary vector; only the first 2 elements of the resulting
N-length output are kept as the plotted (x, y) offspring point per sample.

Run from the project root:
    python main/analysis/paper_fig_geometry_N_scaling.py
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

from paper_fig_common import FAMILY_PALETTE, save_all

OUT_DIR = os.path.join(_MAIN, 'paper_figures')

LOW, HIGH = -10, 10
T = 10.0
MS = 1.0
N_POINTS = 1500
N_VALUES = [2, 10, 50, 500]
SEED = 0


def norm1_sum(r, ms=MS, T=T):
    rmin, rmax = r.min(), r.max()
    rrange = max(rmax - rmin, 1e-8)
    return T + ms * (2 * (r - rmin) / rrange - 1)


def norm2_sum(r1, r2, ms=MS, T=T):
    diff = r1 - r2
    scale = max(abs(diff.min()), diff.max())
    alpha = 1.0 / scale if scale != 0.0 else 1.0
    return T + ms * alpha * diff


def normrob_sum(r1, r2, ms=MS, T=T):
    diff = r1 - r2
    denom = np.quantile(np.abs(diff), 0.99)
    alpha = 1.0 / denom if denom > 1e-10 else 1.0
    return T + ms * alpha * diff


def norm12_sum(r1, r2, ms=MS, T=T):
    n1 = 2 * (r1 - r1.min()) / max(r1.max() - r1.min(), 1e-8) - 1
    n2 = 2 * (r2 - r2.min()) / max(r2.max() - r2.min(), 1e-8) - 1
    return T + (ms / 2) * (n1 - n2)


def sample_one_tree(fn, n_instances, rng):
    pts = np.zeros((N_POINTS, 2))
    for i in range(N_POINTS):
        r = rng.uniform(LOW, HIGH, size=n_instances)
        pts[i] = fn(r)[:2]
    return pts


def sample_two_trees(fn, n_instances, rng):
    pts = np.zeros((N_POINTS, 2))
    for i in range(N_POINTS):
        r1 = rng.uniform(LOW, HIGH, size=n_instances)
        r2 = rng.uniform(LOW, HIGH, size=n_instances)
        pts[i] = fn(r1, r2)[:2]
    return pts


# (label, sampler, formula fn, family key into FAMILY_PALETTE -- dark shade used, matches "*" mul shade)
OPS = [
    ('NORM1',   sample_one_tree,  norm1_sum,   'NORM1'),
    ('NORM2',   sample_two_trees, norm2_sum,   'NORM2'),
    ('NORMROB', sample_two_trees, normrob_sum, 'NORMROB'),
    ('NORM12',  sample_two_trees, norm12_sum,  'NORM12'),
]


def make_plot():
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(OPS), len(N_VALUES),
                              figsize=(3.6 * len(N_VALUES), 3.6 * len(OPS)), squeeze=False)

    for row, (name, sampler, fn, family) in enumerate(OPS):
        color = FAMILY_PALETTE[family][1]  # dark shade
        for col, n_inst in enumerate(N_VALUES):
            ax = axes[row][col]
            pts = sampler(fn, n_inst, rng)
            ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.35, c=color, rasterized=True)
            ax.scatter(T, T, s=60, c='red', marker='+', zorder=5, linewidths=1.5)
            ax.set_xlim(8.7, 11.3)
            ax.set_ylim(8.7, 11.3)
            if row == 0:
                ax.set_title(f'N = {n_inst}', fontsize=11)
            if col == 0:
                ax.set_ylabel(name, fontsize=11, fontweight='bold')
            ax.tick_params(labelsize=6)

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    fig = make_plot()
    save_all(fig, OUT_DIR, 'geometry_norm_family_N_scaling')
    plt.close(fig)
