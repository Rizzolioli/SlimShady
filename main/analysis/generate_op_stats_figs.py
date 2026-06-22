"""
generate_op_stats_figs.py
=========================
Visualise per-generation operator improvement rates from a log=9 CSV.

One figure per dataset:
  rows = 3 variants (SLIM+2SIG, SLIM*ABS, SLIM*1SIG)
  col 0 = improvement rate  (n_improved / n_applied, as %)
  col 1 = n applications per generation

Each panel shows 3 coloured lines (inflate / deflate / xo),
median over seeds with IQR shading.

X axis: generation  (secondary label: evaluations = gen × pop_size)

Outputs -> main/log/latex_final/op_stats/
    op_stats_{dataset}.{png,tex}
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

POP_SIZE = 500   # for evaluations label


def _find_log():
    pattern = os.path.join(_LOG_DIR, "results_op_stats_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No op_stats log found matching: {pattern}")
    return files[-1]


def _load(path):
    df = pd.read_csv(path, header=None).rename(columns=_COLS)
    df["seed"] = df["seed"].astype(int)
    df = df.drop_duplicates(subset=["algo", "dataset", "seed", "gen"], keep="last")
    df["variant"] = df["algo"].str.extract(r'^(SLIM[+*]\w+)_pop')
    df = df[df["variant"].isin(_VARIANTS)]
    # compute rates (NaN when n_applied == 0)
    for op in _OPS:
        df[f"{op}_rate"] = (df[f"{op}_improved"] / df[f"{op}_n"].replace(0, np.nan)) * 100
    return df


def _median_iqr(sub, col):
    """Return (gen_index, median, q25, q75) aggregated over seeds."""
    pivot = sub.pivot_table(index="gen", columns="seed", values=col)
    med = pivot.median(axis=1)
    q25 = pivot.quantile(0.25, axis=1)
    q75 = pivot.quantile(0.75, axis=1)
    return med, q25, q75


def _save_fig(fig, stem):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {os.path.basename(stem)}.png")
    if _TIKZ:
        try:
            tikzplotlib.save(stem + ".tex", figure=fig, strict=False)
            print(f"  Saved: {os.path.basename(stem)}.tex")
        except Exception as e:
            print(f"  [WARN] tikz: {e}")
    plt.close(fig)


def make_figures(df):
    for dataset in _DATASETS:
        dset = df[df["dataset"] == dataset]
        if dset.empty:
            print(f"  [SKIP] {dataset}")
            continue

        fig, axes = plt.subplots(
            len(_VARIANTS), 2,
            figsize=(13, 3.8 * len(_VARIANTS)),
            sharex=True,
        )
        fig.suptitle(f"Operator statistics — {dataset.replace('_', ' ')}",
                     fontsize=13, fontweight="bold", y=1.005)

        # column headers
        axes[0, 0].set_title("Improvement rate (%)", fontsize=10, pad=4)
        axes[0, 1].set_title("Applications per generation", fontsize=10, pad=4)

        for ri, variant in enumerate(_VARIANTS):
            sub = dset[dset["variant"] == variant]

            ax_rate  = axes[ri, 0]
            ax_count = axes[ri, 1]

            ax_rate.set_ylabel(variant, fontsize=9, fontweight="bold")

            for op in _OPS:
                color = _OP_COLORS[op]
                label = _OP_LABELS[op]

                # --- improvement rate ---
                med, q25, q75 = _median_iqr(sub, f"{op}_rate")
                ax_rate.plot(med.index, med.values, color=color, linewidth=1.4,
                             label=label)
                ax_rate.fill_between(med.index, q25.values, q75.values,
                                     color=color, alpha=0.15)

                # --- application count ---
                med_n, q25_n, q75_n = _median_iqr(sub, f"{op}_n")
                ax_count.plot(med_n.index, med_n.values, color=color, linewidth=1.4,
                              label=label)
                ax_count.fill_between(med_n.index, q25_n.values, q75_n.values,
                                      color=color, alpha=0.15)

            ax_rate.set_ylim(-2, 102)
            ax_rate.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax_rate.grid(True, linewidth=0.4, alpha=0.5)
            ax_rate.tick_params(labelsize=8)

            ax_count.set_ylim(bottom=0)
            ax_count.grid(True, linewidth=0.4, alpha=0.5)
            ax_count.tick_params(labelsize=8)

            if ri == len(_VARIANTS) - 1:
                for ax in (ax_rate, ax_count):
                    ax.set_xlabel("Generation", fontsize=9)
                    # add secondary evaluations ticks
                    max_gen = sub["gen"].max()
                    ax2 = ax.twiny()
                    ax2.set_xlim(ax.get_xlim())
                    tick_gens = ax.get_xticks()
                    ax2.set_xticks([])
                    ax2.set_visible(False)   # keep clean, gen is clear enough

        legend_handles = [Line2D([0], [0], color=_OP_COLORS[op], linewidth=2,
                                 label=_OP_LABELS[op]) for op in _OPS]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=[0, 0.03, 1, 1])

        _save_fig(fig, os.path.join(_OUT, f"op_stats_{dataset}"))


if __name__ == "__main__":
    log_path = _find_log()
    print(f"Loading: {log_path}")
    df = _load(log_path)
    print(f"  {len(df)} rows | algos: {sorted(df['variant'].unique())}")

    os.makedirs(_OUT, exist_ok=True)
    print(f"\nOutput -> {_OUT}\n")

    make_figures(df)
    print("\nDone.")
