"""
Geometry study: offspring distribution in 2D semantic space.

A parent T = (10, 10) is fixed. 5000 neighbors are generated for each
mutation operator by drawing random tree outputs uniformly from an interval
and applying the operator formula directly (no actual GP trees needed).

Rows:    SLIM+NORM1, SLIM*NORM1, SLIM+NORM2, SLIM*NORM2
Columns: 6 conditions = 3 intervals × 2 ms settings
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_SAMPLES = 5000
T = np.array([10.0, 10.0])          # fixed parent semantics (2D)

INTERVALS  = [(-1, 1), (-10, 10), (-100, 100)]
MS_CONFIGS = [("ms=1.0", lambda: 1.0), ("ms~U(0,1)", lambda: np.random.uniform(0, 1))]

########################################################################################################################
# Operator implementations (pure numpy, applied element-wise over 2D vectors)
########################################################################################################################

def norm1_sum(r, ms, T=T):
    """SLIM+NORM1: T + ms * (2*(r - min)/(range) - 1)"""
    rmin, rmax = r.min(), r.max()
    rrange = max(rmax - rmin, 1e-8)
    normalised = 2 * (r - rmin) / rrange - 1
    return T + ms * normalised

def norm1_mul(r, ms, T=T):
    """SLIM*NORM1: T * (1 + ms * (2*(r - min)/(range) - 1))"""
    rmin, rmax = r.min(), r.max()
    rrange = max(rmax - rmin, 1e-8)
    normalised = 2 * (r - rmin) / rrange - 1
    return T * (1 + ms * normalised)

def norm2_sum(r1, r2, ms, T=T):
    """SLIM+NORM2: T + ms * alpha * (r1 - r2), alpha from training diff"""
    diff = r1 - r2
    scale = max(abs(diff.min()), diff.max())
    alpha = 1.0 / scale if scale != 0.0 else 1.0
    return T + ms * alpha * diff

def norm2_mul(r1, r2, ms, T=T):
    """SLIM*NORM2: T * (1 + ms * alpha * (r1 - r2))"""
    diff = r1 - r2
    scale = max(abs(diff.min()), diff.max())
    alpha = 1.0 / scale if scale != 0.0 else 1.0
    return T * (1 + ms * alpha * diff)

# ---  Commented-out baselines (uncomment to include in plot) ---
#
# def slim_plus_2sig(r1, r2, ms, T=T):
#     """SLIM+2SIG: T + ms * (sigmoid(r1) - sigmoid(r2))"""
#     return T + ms * (1/(1+np.exp(-r1)) - 1/(1+np.exp(-r2)))
#
# def slim_mul_2sig(r1, r2, ms, T=T):
#     """SLIM*2SIG: T * (1 + ms * (sigmoid(r1) - sigmoid(r2)))"""
#     return T * (1 + ms * (1/(1+np.exp(-r1)) - 1/(1+np.exp(-r2))))
#
# def slim_plus_1sig(r, ms, T=T):
#     """SLIM+1SIG: T + ms * (2*sigmoid(r) - 1)"""
#     return T + ms * (2/(1+np.exp(-r)) - 1)
#
# def slim_mul_1sig(r, ms, T=T):
#     """SLIM*1SIG: T * (1 + ms * (2*sigmoid(r) - 1))"""
#     return T * (1 + ms * (2/(1+np.exp(-r)) - 1))
#
# def slim_plus_abs(r, ms, T=T):
#     """SLIM+ABS: T + ms * (1 - 2/(1+|r|))"""
#     return T + ms * (1 - 2/(1 + np.abs(r)))
#
# def slim_mul_abs(r, ms, T=T):
#     """SLIM*ABS: T * (1 + ms * (1 - 2/(1+|r|)))"""
#     return T * (1 + ms * (1 - 2/(1 + np.abs(r))))

########################################################################################################################
# Active operators for the study
########################################################################################################################

# Each entry: (label, uses_two_trees, apply_fn)
OPERATORS = [
    ("SLIM+NORM1", False, norm1_sum),
    ("SLIM*NORM1", False, norm1_mul),
    ("SLIM+NORM2", True,  norm2_sum),
    ("SLIM*NORM2", True,  norm2_mul),
]

########################################################################################################################
# Sampling helpers
########################################################################################################################

def sample_offspring_one_tree(apply_fn, low, high, ms_fn, n=N_SAMPLES):
    """Sample N offspring for a one-tree operator."""
    offsprings = np.zeros((n, 2))
    for i in range(n):
        r  = np.random.uniform(low, high, size=2)
        ms = ms_fn()
        offsprings[i] = apply_fn(r, ms)
    return offsprings

def sample_offspring_two_trees(apply_fn, low, high, ms_fn, n=N_SAMPLES):
    """Sample N offspring for a two-tree operator."""
    offsprings = np.zeros((n, 2))
    for i in range(n):
        r1 = np.random.uniform(low, high, size=2)
        r2 = np.random.uniform(low, high, size=2)
        ms = ms_fn()
        offsprings[i] = apply_fn(r1, r2, ms)
    return offsprings

########################################################################################################################
# Plotting
########################################################################################################################

def make_plot(save_path):
    n_ops   = len(OPERATORS)
    n_conds = len(INTERVALS) * len(MS_CONFIGS)   # 6
    fig, axes = plt.subplots(n_ops, n_conds,
                             figsize=(3.5 * n_conds, 3.5 * n_ops),
                             squeeze=False)

    col = 0
    for (iv_low, iv_high) in INTERVALS:
        for (ms_label, ms_fn) in MS_CONFIGS:
            col_title = f"r∈[{iv_low},{iv_high}]\n{ms_label}"
            for row, (op_name, two_trees, apply_fn) in enumerate(OPERATORS):
                ax = axes[row][col]

                np.random.seed(42)
                if two_trees:
                    pts = sample_offspring_two_trees(apply_fn, iv_low, iv_high, ms_fn)
                else:
                    pts = sample_offspring_one_tree(apply_fn, iv_low, iv_high, ms_fn)

                ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.3, c="steelblue", rasterized=True)
                ax.scatter(*T, s=60, c="red", marker="+", zorder=5, linewidths=1.5)

                if row == 0:
                    ax.set_title(col_title, fontsize=8)
                if col == 0:
                    ax.set_ylabel(op_name, fontsize=9)

                ax.tick_params(labelsize=6)
            col += 1

    fig.suptitle("Offspring distribution in 2D semantic space  (parent = (10, 10)  ✚)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, "geometry_study.png")
    make_plot(save_path)
