"""
Visualize results from results_pop_xo_15052026.csv.

Experiment: p_xo=0.7 (head XO, max_head_depth=17) with different population/generation configs,
all sharing the same ~200K evaluation budget.
  pop=200,  n_iter=1000
  pop=500,  n_iter=400
  pop=1000, n_iter=200

Baseline (p_xo=0.0, pop=100, n_iter=2000) pulled from results_prob_xo_12052026.csv.

X-axis: evaluations = generation × pop_size  (budget-normalised).
One figure per dataset (6 total).
Grid: 5 variants × 3 metrics (train RMSE, test RMSE, elite nodes).
One line per config; mean ± 1 std over 5 seeds, 50-pt rolling mean on the eval axis.

Output: main/log/figs/pop_xo_{dataset}.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

########################################################################################################################
# CONFIG
########################################################################################################################

_HERE       = os.path.dirname(os.path.abspath(__file__))
LOG_POP     = os.path.join(_HERE, "..", "log", "results_pop_xo_15052026.csv")
LOG_BASE    = os.path.join(_HERE, "..", "log", "results_prob_xo_12052026.csv")
OUT_DIR     = os.path.join(_HERE, "..", "log", "figs")

BUDGET      = 200_000   # total evaluations budget
EVAL_STEP   = 1_000     # resample grid step (evaluations)
SMOOTH      = 10        # rolling window on resampled grid

VARIANTS = ["SLIM+2SIG", "SLIM+1SIG", "SLIM+ABS", "SLIM*1SIG", "SLIM*ABS"]

CONFIGS = [
    ("baseline", 100,  2000, 0.0),   # from prob_xo log
    ("pop200",   200,  1000, 0.7),
    ("pop500",   500,   400, 0.7),
    ("pop1000", 1000,   200, 0.7),
]

COLORS = {
    "baseline": "#555555",   # gray
    "pop200":   "#4daf4a",   # green
    "pop500":   "#377eb8",   # blue
    "pop1000":  "#e41a1c",   # red
}
LABELS = {
    "baseline": "Baseline  p_xo=0.0  pop=100  iter=2000",
    "pop200":   "p_xo=0.7  pop=200   iter=1000",
    "pop500":   "p_xo=0.7  pop=500   iter=400",
    "pop1000":  "p_xo=0.7  pop=1000  iter=200",
}

########################################################################################################################
# LOAD DATA
########################################################################################################################

_cols = {0: "algo", 1: "run_id", 2: "dataset", 3: "seed", 4: "gen",
         5: "train", 6: "timing", 7: "nodes", 8: "test", 9: "nodes_count", 10: "log"}

# --- pop_xo results (p_xo=0.7 only) ---
df_pop = pd.read_csv(LOG_POP, header=None).rename(columns=_cols)
df_pop["seed"] = df_pop["seed"].astype(int)

df_pop["variant"] = df_pop["algo"].str.extract(r"^(SLIM[+*]\w+)_pop")
df_pop["pop"]     = df_pop["algo"].str.extract(r"_pop(\d+)_").astype(int)
df_pop["n_iter"]  = df_pop["algo"].str.extract(r"_iter(\d+)_").astype(int)
df_pop["pxo"]     = df_pop["algo"].str.extract(r"_pxo([0-9.]+)")
df_pop["cfg"]     = df_pop["pop"].map({200: "pop200", 500: "pop500", 1000: "pop1000"})
df_pop["evals"]   = df_pop["gen"] * df_pop["pop"]

# Keep only complete runs
complete_mask = df_pop.apply(lambda r: r["gen"] == r["n_iter"], axis=1)
complete_keys = df_pop[complete_mask][["algo", "dataset", "seed"]].drop_duplicates()
df_pop = df_pop.merge(complete_keys, on=["algo", "dataset", "seed"])
df_pop = df_pop.drop_duplicates(subset=["algo", "dataset", "seed", "gen"], keep="last")

# --- baseline (p_xo=0.0 from prob_xo log, pop=100 n_iter=2000) ---
df_b = pd.read_csv(LOG_BASE, header=None).rename(columns=_cols)
df_b["seed"] = df_b["seed"].astype(int)
df_b = df_b[df_b["algo"].str.endswith("_pxo0.0")].copy()
df_b["variant"] = df_b["algo"].str.extract(r"^(SLIM[+*]\w+)_pxo")
df_b["pop"]     = 100
df_b["n_iter"]  = 2000
df_b["cfg"]     = "baseline"
df_b["evals"]   = df_b["gen"] * 100

complete_b = df_b[df_b["gen"] == 2000][["algo", "dataset", "seed"]].drop_duplicates()
df_b = df_b.merge(complete_b, on=["algo", "dataset", "seed"])
df_b = df_b.drop_duplicates(subset=["algo", "dataset", "seed", "gen"], keep="last")

# Unified frame
df = pd.concat([df_pop, df_b], ignore_index=True)

n_complete = df[["cfg", "variant", "dataset", "seed"]].drop_duplicates().shape[0]
print(f"Loaded {len(df):,} rows | {n_complete} complete (cfg, variant, dataset, seed) combos")
print(f"Datasets: {sorted(df['dataset'].unique())}")

########################################################################################################################
# RESAMPLE TO COMMON EVAL GRID
# Each config has a different generation spacing; we interpolate every EVAL_STEP
# evaluations so all configs share the same x-axis.
########################################################################################################################

EVAL_GRID = np.arange(0, BUDGET + 1, EVAL_STEP)

METRICS = [
    ("train",       "Train RMSE",  "%.0f"),
    ("test",        "Test RMSE",   "%.0f"),
    ("nodes_count", "Elite nodes", "%.0f"),
]

def resample_series(sub_seed, eval_col, metric, grid):
    """Linear interpolation of a single-seed metric onto the eval grid."""
    s = sub_seed.sort_values(eval_col)
    return np.interp(grid, s[eval_col].values, s[metric].values,
                     left=np.nan, right=np.nan)

########################################################################################################################
# PLOT
########################################################################################################################

os.makedirs(OUT_DIR, exist_ok=True)

for dataset in sorted(df["dataset"].unique()):
    dset = df[df["dataset"] == dataset]

    fig, axes = plt.subplots(
        nrows=len(VARIANTS), ncols=len(METRICS),
        figsize=(20, 3.5 * len(VARIANTS)),
        sharex=True,
    )
    fig.suptitle(f"Pop/Iter sweep (p_xo=0.7, max_depth=17) — Dataset: {dataset}",
                 fontsize=14, fontweight="bold", y=1.005)

    for row, variant in enumerate(VARIANTS):
        sub_var = dset[dset["variant"] == variant]

        for col, (metric, ylabel, fmt) in enumerate(METRICS):
            ax = axes[row, col]

            for cfg_name, pop, n_iter, pxo in CONFIGS:
                sub_cfg = sub_var[sub_var["cfg"] == cfg_name]
                if sub_cfg.empty:
                    continue

                # Resample each seed onto EVAL_GRID
                resampled = []
                for seed, grp in sub_cfg.groupby("seed"):
                    rs = resample_series(grp, "evals", metric, EVAL_GRID)
                    resampled.append(rs)

                if not resampled:
                    continue

                mat = np.vstack(resampled)           # (n_seeds, n_grid)

                # Rolling smooth along eval axis
                df_mat = pd.DataFrame(mat.T)
                if SMOOTH > 1:
                    df_mat = df_mat.rolling(SMOOTH, min_periods=1).mean()

                mean = df_mat.mean(axis=1).values
                std  = df_mat.std(axis=1).values

                ax.plot(EVAL_GRID, mean,
                        color=COLORS[cfg_name], linewidth=1.5,
                        label=LABELS[cfg_name])
                ax.fill_between(EVAL_GRID,
                                mean - std, mean + std,
                                color=COLORS[cfg_name], alpha=0.12)

            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(variant, fontsize=10, pad=3)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))
            ax.tick_params(labelsize=8)
            ax.grid(True, linewidth=0.4, alpha=0.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("Evaluations", fontsize=9)

    legend_elements = [
        Line2D([0], [0], color=COLORS[c], linewidth=2, label=LABELS[c])
        for c, *_ in CONFIGS
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9,
               framealpha=0.85, bbox_to_anchor=(1.0, 1.0))

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"pop_xo_{dataset}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

print("Done.")
