"""
Visualize evolution of train/test RMSE and elite node count for results_head_size_07052026.csv.

Experiment sweeps:
  - 5 SLIM variants
  - max_head_depth ∈ [5, 17, 25]
  - head_xo_freq ∈ [50, 500]
  - 6 datasets, 5 seeds, 2000 generations

One figure per dataset (6 total).
Grid: 5 variants × 3 metrics (train RMSE, test RMSE, elite nodes).
Lines: colour = max_head_depth, linestyle = xo_freq.
Mean ± 1 std over seeds; SMOOTH-generation rolling mean applied.

Output: main/log/figs/head_size_{dataset}.png
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

########################################################################################################################
# CONFIG
########################################################################################################################

_HERE    = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(_HERE, "..", "log", "results_head_size_07052026.csv")
OUT_DIR  = os.path.join(_HERE, "..", "log", "figs")
N_ITER   = 2000
SMOOTH   = 50

VARIANTS   = ["SLIM+2SIG", "SLIM+1SIG", "SLIM+ABS", "SLIM*1SIG", "SLIM*ABS"]
HD_VALUES  = ["5", "17", "25"]
XO_VALUES  = ["50", "500"]

# colour by head depth, linestyle by xo_freq
COLORS = {
    "5":  "#e41a1c",   # red
    "17": "#377eb8",   # blue
    "25": "#4daf4a",   # green
}
LINES = {
    "50":  "-",
    "500": "--",
}
ALPHAS = {
    "50":  0.9,
    "500": 0.75,
}

########################################################################################################################
# LOAD & FILTER
########################################################################################################################

cols = {0: "algo", 1: "run_id", 2: "dataset", 3: "seed", 4: "gen",
        5: "train", 6: "timing", 7: "nodes", 8: "test", 9: "nodes_count", 10: "log"}

df = pd.read_csv(LOG_PATH, header=None).rename(columns=cols)

# keep only complete runs
complete_keys = df[df["gen"] == N_ITER][["algo", "dataset", "seed"]].drop_duplicates()
df = df.merge(complete_keys, on=["algo", "dataset", "seed"])

# deduplicate restarted runs — keep most recent per (algo, dataset, seed, gen)
df = df.drop_duplicates(subset=["algo", "dataset", "seed", "gen"], keep="last")

# parse variant, max_head_depth, xo_freq from algo name
df["variant"] = df["algo"].str.extract(r'^(SLIM[+*]\w+)_hd')
df["hd"]      = df["algo"].str.extract(r'_hd(\d+)_xo')
df["xo_freq"] = df["algo"].str.extract(r'_xo(\d+)$')

n_complete = df[["algo", "dataset", "seed"]].drop_duplicates().shape[0]
print(f"Loaded {len(df):,} rows | {n_complete} complete runs | "
      f"{df['dataset'].nunique()} datasets | "
      f"hd values: {sorted(df['hd'].dropna().unique())} | "
      f"xo_freq values: {sorted(df['xo_freq'].dropna().unique())}")

########################################################################################################################
# PLOT
########################################################################################################################

os.makedirs(OUT_DIR, exist_ok=True)

METRICS = [
    ("train",       "Train RMSE",  "%.0f"),
    ("test",        "Test RMSE",   "%.0f"),
    ("nodes_count", "Elite nodes", "%.0f"),
]

for dataset in sorted(df["dataset"].unique()):
    dset = df[df["dataset"] == dataset]

    fig, axes = plt.subplots(
        nrows=len(VARIANTS), ncols=len(METRICS),
        figsize=(20, 3.5 * len(VARIANTS)),
        sharex=True,
    )
    fig.suptitle(f"Head-size XO experiment — Dataset: {dataset}",
                 fontsize=14, fontweight="bold", y=1.005)

    for row, variant in enumerate(VARIANTS):
        sub = dset[dset["variant"] == variant]

        for col, (metric, ylabel, fmt) in enumerate(METRICS):
            ax = axes[row, col]

            for hd in HD_VALUES:
                for xo in XO_VALUES:
                    fdata = (sub[(sub["hd"] == hd) & (sub["xo_freq"] == xo)]
                             .sort_values(["seed", "gen"]))
                    if fdata.empty:
                        continue

                    pivot = fdata.pivot_table(index="gen", columns="seed", values=metric)
                    if SMOOTH > 1:
                        pivot = pivot.rolling(SMOOTH, min_periods=1).mean()

                    mean = pivot.mean(axis=1)
                    std  = pivot.std(axis=1)
                    color = COLORS[hd]
                    ls    = LINES[xo]

                    ax.plot(mean.index, mean.values,
                            color=color, linestyle=ls,
                            linewidth=1.3, alpha=ALPHAS[xo],
                            label=f"hd={hd} xo={xo}")
                    ax.fill_between(mean.index,
                                    mean.values - std.values,
                                    mean.values + std.values,
                                    color=color, alpha=0.10)

            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(variant, fontsize=10, pad=3)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
            ax.tick_params(labelsize=8)
            ax.grid(True, linewidth=0.4, alpha=0.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("Generation", fontsize=9)

    # shared legend — colour = head depth, linestyle = xo_freq
    legend_elements = []
    for hd in HD_VALUES:
        legend_elements.append(
            Line2D([0], [0], color=COLORS[hd], linewidth=2, label=f"max_depth={hd}")
        )
    legend_elements.append(Line2D([0], [0], color="gray", linewidth=0, label=""))
    for xo in XO_VALUES:
        legend_elements.append(
            Line2D([0], [0], color="black", linestyle=LINES[xo],
                   linewidth=1.5, label=f"XO every {xo} gen")
        )
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9,
               framealpha=0.85, ncol=1,
               bbox_to_anchor=(1.0, 1.0))

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"head_size_{dataset}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

print("Done.")
