"""
Visualize results from results_prob_xo_12052026.csv.

Experiment: head XO (max_head_depth=17) applied per-individual with probability p_xo ∈ [0.0, 0.3, 0.5, 0.7].
p_xo=0.0 is the standard SLIM baseline (no XO).

One figure per dataset (6 total).
Grid: 5 variants × 3 metrics (train RMSE, test RMSE, elite nodes).
One line per p_xo value; mean ± 1 std over 5 seeds, SMOOTH-generation rolling mean.

Output: main/log/figs/prob_xo_{dataset}.png
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

########################################################################################################################
# CONFIG
########################################################################################################################

_HERE    = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(_HERE, "..", "log", "results_prob_xo_12052026.csv")
OUT_DIR  = os.path.join(_HERE, "..", "log", "figs")
N_ITER   = 2000
SMOOTH   = 50

VARIANTS  = ["SLIM+2SIG", "SLIM+1SIG", "SLIM+ABS", "SLIM*1SIG", "SLIM*ABS"]
PXO_VALUES = ["0.0", "0.3", "0.5", "0.7"]

COLORS = {
    "0.0": "#555555",   # gray  — standard SLIM (no XO)
    "0.3": "#4daf4a",   # green
    "0.5": "#377eb8",   # blue
    "0.7": "#e41a1c",   # red
}
LABELS = {
    "0.0": "p_xo=0.0  (standard SLIM)",
    "0.3": "p_xo=0.3",
    "0.5": "p_xo=0.5",
    "0.7": "p_xo=0.7",
}

########################################################################################################################
# LOAD & FILTER
########################################################################################################################

cols = {0: "algo", 1: "run_id", 2: "dataset", 3: "seed", 4: "gen",
        5: "train", 6: "timing", 7: "nodes", 8: "test", 9: "nodes_count", 10: "log"}

df = pd.read_csv(LOG_PATH, header=None).rename(columns=cols)

complete_keys = df[df["gen"] == N_ITER][["algo", "dataset", "seed"]].drop_duplicates()
df = df.merge(complete_keys, on=["algo", "dataset", "seed"])
df = df.drop_duplicates(subset=["algo", "dataset", "seed", "gen"], keep="last")

df["variant"] = df["algo"].str.extract(r'^(SLIM[+*]\w+)_pxo')
df["pxo"]     = df["algo"].str.extract(r'_pxo([0-9.]+)$')

n_complete = df[["algo", "dataset", "seed"]].drop_duplicates().shape[0]
print(f"Loaded {len(df):,} rows | {n_complete} complete runs | "
      f"{df['dataset'].nunique()} datasets | "
      f"p_xo values: {sorted(df['pxo'].dropna().unique())}")

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
    fig.suptitle(f"Prob-XO (max_depth=17) — Dataset: {dataset}",
                 fontsize=14, fontweight="bold", y=1.005)

    for row, variant in enumerate(VARIANTS):
        sub = dset[dset["variant"] == variant]

        for col, (metric, ylabel, fmt) in enumerate(METRICS):
            ax = axes[row, col]

            for pxo in PXO_VALUES:
                fdata = sub[sub["pxo"] == pxo].sort_values(["seed", "gen"])
                if fdata.empty:
                    continue

                pivot = fdata.pivot_table(index="gen", columns="seed", values=metric)
                if SMOOTH > 1:
                    pivot = pivot.rolling(SMOOTH, min_periods=1).mean()

                mean = pivot.mean(axis=1)
                std  = pivot.std(axis=1)

                ax.plot(mean.index, mean.values,
                        color=COLORS[pxo], linewidth=1.5,
                        label=LABELS[pxo])
                ax.fill_between(mean.index,
                                mean.values - std.values,
                                mean.values + std.values,
                                color=COLORS[pxo], alpha=0.12)

            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(variant, fontsize=10, pad=3)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
            ax.tick_params(labelsize=8)
            ax.grid(True, linewidth=0.4, alpha=0.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("Generation", fontsize=9)

    legend_elements = [
        Line2D([0], [0], color=COLORS[p], linewidth=2, label=LABELS[p])
        for p in PXO_VALUES
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=10,
               framealpha=0.85, bbox_to_anchor=(1.0, 1.0))

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"prob_xo_{dataset}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

print("Done.")
