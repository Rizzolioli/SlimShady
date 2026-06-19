"""
generate_op_stats_figs.py
=========================
Visualise per-generation operator improvement rates from a log=9 CSV.

For each operator (inflate, deflate, xo) and each generation we compute:
    improvement_rate = n_improved / n_applied   (NaN when n_applied == 0)

Output: one figure per dataset, rows = variants, cols = operators.
Each panel shows median improvement rate across seeds with IQR shading.

Additionally: per-operator application counts (how often each fires).

Outputs -> main/log/latex_final/op_stats/
    op_stats_rate_{dataset}.{png,tex}    -- improvement rates
    op_stats_count_{dataset}.{png,tex}   -- application counts
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import matplot2tikz as tikzplotlib
    _TIKZ = True
except ImportError:
    _TIKZ = False

_HERE    = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR = os.path.join(_HERE, "..", "log")
_OUT     = os.path.join(_LOG_DIR, "latex_final", "op_stats")

_VARIANTS = ["SLIM+2SIG", "SLIM*ABS", "SLIM*1SIG"]
_DATASETS = ["toxicity", "concrete", "instanbul", "ppb",
             "resid_build_sale_price", "energy"]

_OPS = ["inflate", "deflate", "xo"]
_OP_COLORS = {"inflate": "#e41a1c", "deflate": "#377eb8", "xo": "#4daf4a"}
_OP_LABELS = {"inflate": "Inflate", "deflate": "Deflate", "xo": "Head XO"}

_COLS = {
    0: "algo", 1: "run_id", 2: "dataset", 3: "seed", 4: "gen",
    5: "train_fit", 6: "timing", 7: "nodes", 8: "test_fit", 9: "nodes_count",
    10: "inflate_n", 11: "inflate_improved",
    12: "deflate_n", 13: "deflate_improved",
    14: "xo_n",     15: "xo_improved",
    16: "log_level",
}

_SMOOTH = 10   # rolling window for smoothing rates


def _find_log():
    pattern = os.path.join(_LOG_DIR, "results_op_stats_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No op_stats log found matching: {pattern}")
    return files[-1]   # most recent


def _load(path):
    df = pd.read_csv(path, header=None).rename(columns=_COLS)
    df["seed"] = df["seed"].astype(int)

    # keep only fully completed runs
    complete = df[df["gen"] == df["gen"].max()][["algo","dataset","seed"]].drop_duplicates()
    df = df.merge(complete, on=["algo","dataset","seed"])
    df = df.drop_duplicates(subset=["algo","dataset","seed","gen"], keep="last")

    # extract variant
    df["variant"] = df["algo"].str.extract(r'^(SLIM[+*]\w+)_pop')
    df = df[df["variant"].isin(_VARIANTS)]

    # compute rates (NaN when n==0)
    for op in _OPS:
        df[f"{op}_rate"] = df[f"{op}_improved"] / df[f"{op}_n"].replace(0, np.nan)

    return df


def _save_fig(fig, stem):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {os.path.basename(stem)}.png")
    if _TIKZ:
        try:
            tikzplotlib.save(stem + ".tex", figure=fig, strict=False)
            print(f"  Saved: {os.path.basename(stem)}.tex")
        except Exception as e:
            print(f"  [WARN] tikz skipped: {e}")
    plt.close(fig)


def _plot_median_iqr(ax, series_per_seed, color, label, smooth=_SMOOTH):
    """series_per_seed: dict {seed: pd.Series indexed by gen}"""
    if not series_per_seed:
        return
    # align on a common gen index
    df = pd.DataFrame(series_per_seed)
    if smooth > 1:
        df = df.rolling(smooth, min_periods=1).mean()
    med = df.median(axis=1)
    q25 = df.quantile(0.25, axis=1)
    q75 = df.quantile(0.75, axis=1)
    ax.plot(med.index, med.values, color=color, linewidth=1.4, label=label)
    ax.fill_between(med.index, q25.values, q75.values, color=color, alpha=0.15)
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)


def make_rate_figures(df):
    """One figure per dataset: rows=variants, cols=operators, y=improvement rate."""
    for dataset in _DATASETS:
        dset = df[df["dataset"] == dataset]
        if dset.empty:
            print(f"  [SKIP] {dataset} - no data")
            continue

        fig, axes = plt.subplots(
            len(_VARIANTS), len(_OPS),
            figsize=(13, 3.5 * len(_VARIANTS)), sharex=True,
        )
        fig.suptitle(f"Operator improvement rate — {dataset.replace('_', ' ')}",
                     fontsize=13, fontweight="bold", y=1.005)

        for ri, variant in enumerate(_VARIANTS):
            sub = dset[dset["variant"] == variant]
            axes[ri, 0].set_ylabel(variant, fontsize=9, fontweight="bold")

            for ci, op in enumerate(_OPS):
                ax = axes[ri, ci]
                if ri == 0:
                    ax.set_title(_OP_LABELS[op], fontsize=10, pad=3)
                if ri == len(_VARIANTS) - 1:
                    ax.set_xlabel("Generation", fontsize=9)

                rate_col = f"{op}_rate"
                series_per_seed = {
                    seed: grp.set_index("gen")[rate_col]
                    for seed, grp in sub.groupby("seed")
                }
                _plot_median_iqr(ax, series_per_seed, _OP_COLORS[op], _OP_LABELS[op])
                ax.set_ylim(-0.02, 1.02)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

        fig.tight_layout()
        _save_fig(fig, os.path.join(_OUT, f"op_stats_rate_{dataset}"))


def make_count_figures(df):
    """One figure per dataset: rows=variants, cols=operators, y=n_applied per gen."""
    for dataset in _DATASETS:
        dset = df[df["dataset"] == dataset]
        if dset.empty:
            continue

        fig, axes = plt.subplots(
            len(_VARIANTS), len(_OPS),
            figsize=(13, 3.5 * len(_VARIANTS)), sharex=True,
        )
        fig.suptitle(f"Operator application count — {dataset.replace('_', ' ')}",
                     fontsize=13, fontweight="bold", y=1.005)

        for ri, variant in enumerate(_VARIANTS):
            sub = dset[dset["variant"] == variant]
            axes[ri, 0].set_ylabel(variant, fontsize=9, fontweight="bold")

            for ci, op in enumerate(_OPS):
                ax = axes[ri, ci]
                if ri == 0:
                    ax.set_title(_OP_LABELS[op], fontsize=10, pad=3)
                if ri == len(_VARIANTS) - 1:
                    ax.set_xlabel("Generation", fontsize=9)

                count_col = f"{op}_n"
                series_per_seed = {
                    seed: grp.set_index("gen")[count_col]
                    for seed, grp in sub.groupby("seed")
                }
                _plot_median_iqr(ax, series_per_seed, _OP_COLORS[op], _OP_LABELS[op],
                                 smooth=5)

        fig.tight_layout()
        _save_fig(fig, os.path.join(_OUT, f"op_stats_count_{dataset}"))


def make_combined_figure(df):
    """Single summary figure: all operators on one axis per (variant, dataset)."""
    for dataset in _DATASETS:
        dset = df[df["dataset"] == dataset]
        if dset.empty:
            continue

        fig, axes = plt.subplots(
            len(_VARIANTS), 1,
            figsize=(8, 3.5 * len(_VARIANTS)), sharex=True,
        )
        fig.suptitle(f"Improvement rates — {dataset.replace('_', ' ')}",
                     fontsize=13, fontweight="bold", y=1.005)

        for ri, variant in enumerate(_VARIANTS):
            sub = dset[dset["variant"] == variant]
            ax = axes[ri]
            ax.set_ylabel(variant, fontsize=9, fontweight="bold")
            if ri == len(_VARIANTS) - 1:
                ax.set_xlabel("Generation", fontsize=9)

            for op in _OPS:
                series_per_seed = {
                    seed: grp.set_index("gen")[f"{op}_rate"]
                    for seed, grp in sub.groupby("seed")
                }
                _plot_median_iqr(ax, series_per_seed, _OP_COLORS[op], _OP_LABELS[op])

            ax.set_ylim(-0.02, 1.02)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

        legend_handles = [Line2D([0], [0], color=_OP_COLORS[op], linewidth=2,
                                 label=_OP_LABELS[op]) for op in _OPS]
        fig.legend(handles=legend_handles, loc="upper right", fontsize=9,
                   framealpha=0.85, bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        _save_fig(fig, os.path.join(_OUT, f"op_stats_combined_{dataset}"))


if __name__ == "__main__":
    log_path = _find_log()
    print(f"Loading: {log_path}")
    df = _load(log_path)
    print(f"  {len(df)} rows | variants: {sorted(df['variant'].unique())} "
          f"| datasets: {sorted(df['dataset'].unique())} "
          f"| seeds: {sorted(df['seed'].unique())}")

    os.makedirs(_OUT, exist_ok=True)
    print(f"\nOutput -> {_OUT}\n")

    print("-- Rate figures (one per dataset) ---")
    make_rate_figures(df)

    print("\n-- Count figures (one per dataset) --")
    make_count_figures(df)

    print("\n-- Combined figures (all operators per panel) --")
    make_combined_figure(df)

    print("\nDone.")
