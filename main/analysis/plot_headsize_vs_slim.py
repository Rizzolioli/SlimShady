"""
Compare best head-XO config (hd=17, xo_freq=500) from results_head_size_07052026.csv
against standard SLIM (no head XO, xo_freq=None) from results_scramble_xo_05052026.csv.

One figure per dataset (6 total).
Grid: 5 variants × 3 metrics (train RMSE, test RMSE, elite nodes).
Two lines per subplot: hd17_xo500 (blue) vs standard SLIM (gray).
Mean ± 1 std over seeds, SMOOTH-generation rolling mean.

Output: main/log/figs/headsize_vs_slim_{dataset}.png
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

_HERE     = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR  = os.path.join(_HERE, "..", "log")

HEAD_SIZE_CSV  = os.path.join(_LOG_DIR, "results_head_size_07052026.csv")
SCRAMBLE_CSV   = os.path.join(_LOG_DIR, "results_scramble_xo_05052026.csv")
OUT_DIR        = os.path.join(_LOG_DIR, "figs")

N_ITER   = 2000
SMOOTH   = 50

VARIANTS = ["SLIM+2SIG", "SLIM+1SIG", "SLIM+ABS", "SLIM*1SIG", "SLIM*ABS"]

COLORS = {
    "hd17_xo500": "#377eb8",   # blue
    "standard":   "#555555",   # gray
}
LABELS = {
    "hd17_xo500": "Head XO  hd=17, xo=500",
    "standard":   "Standard SLIM (no head XO)",
}

########################################################################################################################
# HELPERS
########################################################################################################################

_COL_MAP = {0: "algo", 1: "run_id", 2: "dataset", 3: "seed", 4: "gen",
            5: "train", 6: "timing", 7: "nodes", 8: "test", 9: "nodes_count", 10: "log"}


def load_and_filter(path, n_iter):
    df = pd.read_csv(path, header=None).rename(columns=_COL_MAP)
    complete_keys = df[df["gen"] == n_iter][["algo", "dataset", "seed"]].drop_duplicates()
    df = df.merge(complete_keys, on=["algo", "dataset", "seed"])
    df = df.drop_duplicates(subset=["algo", "dataset", "seed", "gen"], keep="last")
    return df


########################################################################################################################
# LOAD
########################################################################################################################

# --- head size file: keep only hd=17, xo=500 ---
hs = load_and_filter(HEAD_SIZE_CSV, N_ITER)
hs["variant"] = hs["algo"].str.extract(r'^(SLIM[+*]\w+)_hd')
hs["hd"]      = hs["algo"].str.extract(r'_hd(\d+)_xo')
hs["xo_freq"] = hs["algo"].str.extract(r'_xo(\d+)$')
hs = hs[(hs["hd"] == "17") & (hs["xo_freq"] == "500")].copy()
hs["config"] = "hd17_xo500"

# --- scramble_xo file: keep only xo_freq=None (standard SLIM) ---
sc = load_and_filter(SCRAMBLE_CSV, N_ITER)
sc["variant"] = sc["algo"].str.extract(r'^(SLIM[+*]\w+)_head_xo')
sc["xo_freq"] = sc["algo"].str.extract(r'_head_xo(\w+)$')
sc = sc[sc["xo_freq"] == "None"].copy()
sc["config"] = "standard"

combined = pd.concat([hs, sc], ignore_index=True)

datasets = sorted(combined["dataset"].unique())
print(f"Head-XO (hd17, xo500): {hs[['variant','dataset','seed']].drop_duplicates().shape[0]} runs")
print(f"Standard SLIM:          {sc[['variant','dataset','seed']].drop_duplicates().shape[0]} runs")
print(f"Datasets: {datasets}")

########################################################################################################################
# PLOT
########################################################################################################################

os.makedirs(OUT_DIR, exist_ok=True)

METRICS = [
    ("train",       "Train RMSE",  "%.0f"),
    ("test",        "Test RMSE",   "%.0f"),
    ("nodes_count", "Elite nodes", "%.0f"),
]

for dataset in datasets:
    dset = combined[combined["dataset"] == dataset]

    fig, axes = plt.subplots(
        nrows=len(VARIANTS), ncols=len(METRICS),
        figsize=(20, 3.5 * len(VARIANTS)),
        sharex=True,
    )
    fig.suptitle(f"Head XO (hd=17, xo=500) vs Standard SLIM — Dataset: {dataset}",
                 fontsize=13, fontweight="bold", y=1.005)

    for row, variant in enumerate(VARIANTS):
        sub = dset[dset["variant"] == variant]

        for col, (metric, ylabel, fmt) in enumerate(METRICS):
            ax = axes[row, col]

            for cfg in ("standard", "hd17_xo500"):
                fdata = sub[sub["config"] == cfg].sort_values(["seed", "gen"])
                if fdata.empty:
                    continue

                pivot = fdata.pivot_table(index="gen", columns="seed", values=metric)
                if SMOOTH > 1:
                    pivot = pivot.rolling(SMOOTH, min_periods=1).mean()

                mean = pivot.mean(axis=1)
                std  = pivot.std(axis=1)

                ax.plot(mean.index, mean.values,
                        color=COLORS[cfg], linewidth=1.5,
                        label=LABELS[cfg])
                ax.fill_between(mean.index,
                                mean.values - std.values,
                                mean.values + std.values,
                                color=COLORS[cfg], alpha=0.15)

            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(variant, fontsize=10, pad=3)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
            ax.tick_params(labelsize=8)
            ax.grid(True, linewidth=0.4, alpha=0.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("Generation", fontsize=9)

    legend_elements = [
        Line2D([0], [0], color=COLORS["standard"],   linewidth=2, label=LABELS["standard"]),
        Line2D([0], [0], color=COLORS["hd17_xo500"], linewidth=2, label=LABELS["hd17_xo500"]),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=10,
               framealpha=0.85, bbox_to_anchor=(1.0, 1.0))

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"headsize_vs_slim_{dataset}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

print("Done.")
