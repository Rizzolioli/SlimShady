"""
plot_pareto_mphi_rmse.py — 2-D Pareto front: test RMSE vs M_phi.

One subplot per dataset. Each point = median over 30 seeds for one variant.
Pareto-optimal points (lower RMSE + higher M_phi) are connected by a step line.

Run from project root:
    python main/analysis/plot_pareto_mphi_rmse.py
"""

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SIMP_LOG = os.path.join(_ROOT, "main", "log", "results_normalized_simplification.csv")
FIG_DIR  = os.path.join(_ROOT, "main", "analysis", "log", "figs")
os.makedirs(FIG_DIR, exist_ok=True)

DATASETS  = ["concrete", "energy", "instanbul", "ppb", "resid_build_sale_price", "toxicity"]
DS_LABELS = ["Concrete", "Energy", "Istanbul", "PPB", "Resid. BSP", "Toxicity"]

_GROUP = {
    "SLIM+1SIG":    "1-tree / sum", "SLIM+ABS":    "1-tree / sum",
    "SLIM+NORM1":   "1-tree / sum", "SLIM+NORMFIX": "1-tree / sum",
    "SLIM+2SIG":    "2-tree / sum", "SLIM+NORM2":  "2-tree / sum",
    "SLIM+NORMROB": "2-tree / sum", "SLIM+NORM12": "2-tree / sum",
    "SLIM*1SIG":    "1-tree / mul", "SLIM*ABS":    "1-tree / mul",
    "SLIM*NORM1":   "1-tree / mul", "SLIM*NORMFIX": "1-tree / mul",
    "SLIM*2SIG":    "2-tree / mul", "SLIM*NORM2":  "2-tree / mul",
    "SLIM*NORMROB": "2-tree / mul", "SLIM*NORM12": "2-tree / mul",
}
_CMAP = {
    "1-tree / sum": "#2196F3",
    "1-tree / mul": "#FF9800",
    "2-tree / sum": "#4CAF50",
    "2-tree / mul": "#9C27B0",
}


def pareto_front(points):
    """Return boolean mask of Pareto-optimal points (max mphi, min rmse)."""
    n = len(points)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has higher mphi AND lower/equal rmse (strictly better on one)
            if (points[j, 0] >= points[i, 0] and points[j, 1] <= points[i, 1] and
                    (points[j, 0] > points[i, 0] or points[j, 1] < points[i, 1])):
                dominated[i] = True
                break
    return ~dominated


def pareto_step_line(mphi, rmse):
    """Build (x, y) coords for a step-function Pareto front line."""
    # sort by mphi ascending
    order = np.argsort(mphi)
    mx, ry = mphi[order], rmse[order]
    # keep only the running minimum of rmse (as mphi increases)
    xs, ys = [], []
    min_r = np.inf
    for m, r in zip(mx, ry):
        if r < min_r:
            min_r = r
        xs.append(m)
        ys.append(min_r)
    return np.array(xs), np.array(ys)


def load_data():
    df = pd.read_csv(SIMP_LOG, on_bad_lines="skip")
    # analysis-side filter already applied in the log; take the post-simplification value
    df["m_phi"] = df[["m_phi_before", "m_phi_after"]].max(axis=1)
    df["group"] = df["algo"].map(_GROUP)
    return df


def load_data_after():
    """Same as load_data but use m_phi_after (post-simplification) as the interpretability axis."""
    df = pd.read_csv(SIMP_LOG, on_bad_lines="skip")
    df["m_phi"] = df["m_phi_after"]   # strictly post-simplification
    df["group"] = df["algo"].map(_GROUP)
    return df


def make_plot(df, save_path, title_suffix=""):
    # Aggregate: median per (algo, dataset)
    agg = (df.groupby(["algo", "dataset"])[["m_phi", "test_rmse"]]
             .median()
             .reset_index())

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax_idx, (ds, ds_label) in enumerate(zip(DATASETS, DS_LABELS)):
        ax = axes[ax_idx]
        sub = agg[agg["dataset"] == ds].copy()
        sub["group"] = sub["algo"].map(_GROUP)

        pts = sub[["m_phi", "test_rmse"]].values

        # Pareto front
        mask = pareto_front(pts)
        front = sub[mask]
        if len(front) > 1:
            xs, ys = pareto_step_line(front["m_phi"].values, front["test_rmse"].values)
            ax.plot(xs, ys, color="black", linewidth=1.2, linestyle="--",
                    zorder=1, alpha=0.6, label="Pareto front")

        # Scatter all variants
        for _, row in sub.iterrows():
            is_normfix = "NORMFIX" in row["algo"]
            color  = "#E91E63" if is_normfix else _CMAP[row["group"]]
            marker = "*"       if is_normfix else "o"
            size   = 160       if is_normfix else 60
            zorder = 5         if is_normfix else 3
            ax.scatter(row["m_phi"], row["test_rmse"],
                       c=color, marker=marker, s=size, zorder=zorder,
                       edgecolors="white", linewidths=0.4)

        # Label NORMFIX variants explicitly
        for _, row in sub[sub["algo"].str.contains("NORMFIX")].iterrows():
            ax.annotate(row["algo"],
                        xy=(row["m_phi"], row["test_rmse"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=6.5, color="#E91E63", zorder=6)

        ax.set_title(ds_label, fontsize=10)
        ax.set_xlabel("Mφ  (higher = more interpretable →)", fontsize=8)
        ax.set_ylabel("Test RMSE  (↓ lower = better)", fontsize=8)
        ax.tick_params(labelsize=7)

        # "better" corner label
        ax.text(0.02, 0.02, "← better →\n(↑ Mφ, ↓ RMSE)", transform=ax.transAxes,
                fontsize=6.5, color="grey", va="bottom", ha="left")

    # legend
    legend_patches = [mpatches.Patch(color=c, label=g) for g, c in _CMAP.items()]
    legend_patches.append(mpatches.Patch(color="#E91E63", label="NORMFIX"))
    legend_patches.append(plt.Line2D([0], [0], color="black", linestyle="--",
                                      linewidth=1.2, label="Pareto front"))
    fig.legend(handles=legend_patches, loc="lower center", ncol=6,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Accuracy–Interpretability Pareto front{title_suffix}\n"
                 "Each point = median over 30 seeds  |  NORMFIX ★ highlighted",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"Saved -> {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    df_before = load_data()
    make_plot(df_before, os.path.join(FIG_DIR, "pareto_mphi_rmse.png"))

    df_after = load_data_after()
    make_plot(df_after, os.path.join(FIG_DIR, "pareto_mphi_after_rmse.png"),
              title_suffix=" (Mφ after SymPy simplification)")
