"""
Geometry study (3D): offspring distribution in 3-instance semantic space.

Same operators and conditions as geometry_study.py, but each semantic vector
has 3 components so the offspring cloud is shown as a 3D scatter.  The parent
is fixed at T = (10, 10, 10).

Rows:    all 10 mutation operators
Columns: 6 conditions = 3 intervals x 2 ms settings
Output:  log/geometry_study_3d.png
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

N_SAMPLES = 2000                          # fewer than 2D; 3D rendering is heavier
T = np.array([10.0, 10.0, 10.0])         # fixed parent semantics (3D)
ELEV, AZIM = 22, -55                      # viewing angle shared across all subplots

INTERVALS  = [(-1, 1), (-10, 10), (-100, 100)]
MS_CONFIGS = [("ms=1.0", lambda: 1.0), ("ms~U(0,1)", lambda: np.random.uniform(0, 1))]

########################################################################################################################
# Operator implementations — element-wise, work for any vector length
########################################################################################################################

def norm1_sum(r, ms, T=T):
    rmin, rmax = r.min(), r.max()
    rrange = max(rmax - rmin, 1e-8)
    normalised = 2 * (r - rmin) / rrange - 1
    return T + ms * normalised

def norm1_mul(r, ms, T=T):
    rmin, rmax = r.min(), r.max()
    rrange = max(rmax - rmin, 1e-8)
    normalised = 2 * (r - rmin) / rrange - 1
    return T * (1 + ms * normalised)

def norm2_sum(r1, r2, ms, T=T):
    diff = r1 - r2
    scale = max(abs(diff.min()), diff.max())
    alpha = 1.0 / scale if scale != 0.0 else 1.0
    return T + ms * alpha * diff

def norm2_mul(r1, r2, ms, T=T):
    diff = r1 - r2
    scale = max(abs(diff.min()), diff.max())
    alpha = 1.0 / scale if scale != 0.0 else 1.0
    return T * (1 + ms * alpha * diff)

def normrob_sum(r1, r2, ms, T=T):
    diff = r1 - r2
    denom = np.quantile(np.abs(diff), 0.99)
    alpha = 1.0 / denom if denom > 1e-10 else 1.0
    return T + ms * alpha * diff

def normrob_mul(r1, r2, ms, T=T):
    diff = r1 - r2
    denom = np.quantile(np.abs(diff), 0.99)
    alpha = 1.0 / denom if denom > 1e-10 else 1.0
    return T * (1 + ms * alpha * diff)

def norm12_sum(r1, r2, ms, T=T):
    n1 = 2 * (r1 - r1.min()) / max(r1.max() - r1.min(), 1e-8) - 1
    n2 = 2 * (r2 - r2.min()) / max(r2.max() - r2.min(), 1e-8) - 1
    return T + (ms / 2) * (n1 - n2)

def norm12_mul(r1, r2, ms, T=T):
    n1 = 2 * (r1 - r1.min()) / max(r1.max() - r1.min(), 1e-8) - 1
    n2 = 2 * (r2 - r2.min()) / max(r2.max() - r2.min(), 1e-8) - 1
    return T * (1 + (ms / 2) * (n1 - n2))

def slim_plus_abs(r, ms, T=T):
    return T + ms * (1 - 2 / (1 + np.abs(r)))

def slim_mul_abs(r, ms, T=T):
    return T * (1 + ms * (1 - 2 / (1 + np.abs(r))))

########################################################################################################################
# Active operators
########################################################################################################################

OPERATORS = [
    ("SLIM+ABS",     False, slim_plus_abs),
    ("SLIM*ABS",     False, slim_mul_abs),
    ("SLIM+NORM1",   False, norm1_sum),
    ("SLIM*NORM1",   False, norm1_mul),
    ("SLIM+NORM2",   True,  norm2_sum),
    ("SLIM*NORM2",   True,  norm2_mul),
    ("SLIM+NORMROB", True,  normrob_sum),
    ("SLIM*NORMROB", True,  normrob_mul),
    ("SLIM+NORM12",  True,  norm12_sum),
    ("SLIM*NORM12",  True,  norm12_mul),
]

########################################################################################################################
# Sampling helpers
########################################################################################################################

def sample_one_tree(apply_fn, low, high, ms_fn, n=N_SAMPLES):
    out = np.zeros((n, 3))
    for i in range(n):
        r  = np.random.uniform(low, high, size=3)
        ms = ms_fn()
        out[i] = apply_fn(r, ms)
    return out

def sample_two_trees(apply_fn, low, high, ms_fn, n=N_SAMPLES):
    out = np.zeros((n, 3))
    for i in range(n):
        r1 = np.random.uniform(low, high, size=3)
        r2 = np.random.uniform(low, high, size=3)
        ms = ms_fn()
        out[i] = apply_fn(r1, r2, ms)
    return out

########################################################################################################################
# Plotting
########################################################################################################################

def make_plot(save_path):
    n_ops   = len(OPERATORS)
    n_conds = len(INTERVALS) * len(MS_CONFIGS)   # 6

    fig = plt.figure(figsize=(4.5 * n_conds, 4.2 * n_ops))
    fig.suptitle(
        "Offspring distribution in 3D semantic space  (parent = (10,10,10)  red dot)",
        fontsize=12, y=1.002,
    )

    subplot_idx = 1
    col_titles = []
    for iv_low, iv_high in INTERVALS:
        for ms_label, _ in MS_CONFIGS:
            col_titles.append(f"r in [{iv_low},{iv_high}]\n{ms_label}")

    for row, (op_name, two_trees, apply_fn) in enumerate(OPERATORS):
        col = 0
        for iv_low, iv_high in INTERVALS:
            for ms_label, ms_fn in MS_CONFIGS:
                ax = fig.add_subplot(n_ops, n_conds, subplot_idx, projection="3d")
                subplot_idx += 1

                np.random.seed(42)
                if two_trees:
                    pts = sample_two_trees(apply_fn, iv_low, iv_high, ms_fn)
                else:
                    pts = sample_one_tree(apply_fn, iv_low, iv_high, ms_fn)

                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    s=2, alpha=0.25, c="steelblue",
                    rasterized=True, depthshade=True,
                )
                ax.scatter(*T, s=60, c="red", marker="o", zorder=5)

                ax.view_init(elev=ELEV, azim=AZIM)
                ax.tick_params(labelsize=5, pad=0)
                ax.set_xlabel("s1", fontsize=6, labelpad=1)
                ax.set_ylabel("s2", fontsize=6, labelpad=1)
                ax.set_zlabel("s3", fontsize=6, labelpad=1)

                if row == 0:
                    ax.set_title(col_titles[col], fontsize=8, pad=4)
                if col == 0:
                    ax.text2D(-0.18, 0.5, op_name, transform=ax.transAxes,
                              fontsize=9, va="center", rotation=90)

                col += 1

    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(save_path, dpi=90, bbox_inches="tight")
    print(f"Saved -> {save_path}")


if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, "geometry_study_3d.png")
    make_plot(save_path)
