"""
Visualize evolution of train/test RMSE and elite node count for results_scramble_xo_05052026.csv.

One figure per dataset (6 total), grid: 5 variants × 3 (train, test, nodes).
Each subplot shows one line per head_xo_freq, mean ± 1 std over seeds.
Only complete runs (generation == n_iter) are included.

Output: main/log/figs/scramble_xo_{dataset}.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

########################################################################################################################
# CONFIG
########################################################################################################################

LOG_PATH  = os.path.join(os.path.dirname(__file__), "log", "results_scramble_xo_05052026.csv")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "log", "figs")
N_ITER    = 2000
SMOOTH    = 50    # rolling-mean window (generations); set to 1 to disable

VARIANTS = ["SLIM+2SIG", "SLIM+1SIG", "SLIM+ABS", "SLIM*1SIG", "SLIM*ABS"]
FREQS    = ["None", "10", "50", "100", "500"]

COLORS = {
    "None": "#555555",
    "10":   "#e41a1c",
    "50":   "#ff7f00",
    "100":  "#377eb8",
    "500":  "#4daf4a",
}
LABELS = {
    "None": "No XO",
    "10":   "XO every 10",
    "50":   "XO every 50",
    "100":  "XO every 100",
    "500":  "XO every 500",
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

# deduplicate restarted runs — keep most recent (last in file) per (algo, dataset, seed, gen)
df = df.drop_duplicates(subset=["algo", "dataset", "seed", "gen"], keep="last")

# parse variant and xo_freq from algo column
df["variant"] = df["algo"].str.extract(r'^(SLIM[+*]\w+)_head_xo')
df["xo_freq"] = df["algo"].str.extract(r'_head_xo(\w+)$')

print(f"Loaded {len(df):,} rows | "
      f"{df[['algo','dataset','seed']].drop_duplicates().shape[0]} complete runs | "
      f"{df['dataset'].nunique()} datasets")

########################################################################################################################
# PLOT
########################################################################################################################

os.makedirs(OUT_DIR, exist_ok=True)

for dataset in sorted(df["dataset"].unique()):
    dset = df[df["dataset"] == dataset]

    METRICS = [
        ("train",       "Train RMSE",       "%.0f"),
        ("test",        "Test RMSE",        "%.0f"),
        ("nodes_count", "Elite nodes",      "%.0f"),
    ]

    fig, axes = plt.subplots(
        nrows=len(VARIANTS), ncols=len(METRICS),
        figsize=(20, 3.5 * len(VARIANTS)),
        sharex=True,
    )
    fig.suptitle(f"Dataset: {dataset}", fontsize=14, fontweight="bold", y=1.005)

    for row, variant in enumerate(VARIANTS):
        for col, (metric, ylabel, fmt) in enumerate(METRICS):
            ax = axes[row, col]
            sub = dset[dset["variant"] == variant]

            for freq in FREQS:
                fdata = sub[sub["xo_freq"] == freq].sort_values(["seed", "gen"])
                if fdata.empty:
                    continue

                pivot = fdata.pivot_table(index="gen", columns="seed", values=metric)
                if SMOOTH > 1:
                    pivot = pivot.rolling(SMOOTH, min_periods=1).mean()

                mean = pivot.mean(axis=1)
                std  = pivot.std(axis=1)

                ax.plot(mean.index, mean.values,
                        color=COLORS[freq], label=LABELS[freq], linewidth=1.2)
                ax.fill_between(mean.index,
                                mean.values - std.values,
                                mean.values + std.values,
                                color=COLORS[freq], alpha=0.15)

            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(variant, fontsize=10, pad=3)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
            ax.tick_params(labelsize=8)
            ax.grid(True, linewidth=0.4, alpha=0.5)

            if row == 0 and col == len(METRICS) - 1:
                ax.legend(fontsize=8, loc="upper left", framealpha=0.8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Generation", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"scramble_xo_{dataset}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

print("Done.")
