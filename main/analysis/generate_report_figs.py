"""
generate_report_figs.py
=======================
Generate PNG + TikZ figures for all head-XO experiments.
Filtered to three SLIM variants: SLIM+2SIG, SLIM*ABS, SLIM*1SIG.

Requirements
------------
    pip install matplot2tikz

    matplot2tikz is the maintained successor to the abandoned tikzplotlib.
    It fixes the `common_textification` ImportError that tikzplotlib >= 0.11
    has with matplotlib 3.7+.

Outputs
-------
main/log/latex/
  evolution/
    head_size_<dataset>.{png,tex}
    headsize_vs_slim_<dataset>.{png,tex}
    scramble_xo_<dataset>.{png,tex}
    prob_xo_<dataset>.{png,tex}
    pop_xo_<dataset>.{png,tex}
  stns/
    <dataset>/
      <dataset>_<layout>_<node_size>_stn.{png,tex}

Include in LaTeX:
    \\usepackage{tikz}
    \\input{figure.tex}
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import networkx as nx
import matplot2tikz as tikzplotlib

# ── PATH SETUP ────────────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR  = os.path.join(_HERE, "..", "log")
_OUT_ROOT = os.path.join(_LOG_DIR, "latex")
_STN_ROOT = os.path.join(_LOG_DIR, "stns")

sys.path.insert(0, _HERE)   # so stn_plot imports resolve when run from any cwd

# ── SHARED CONFIG ─────────────────────────────────────────────────────────────

VARIANTS = ["SLIM+2SIG", "SLIM*ABS", "SLIM*1SIG"]
DATASETS = ["toxicity", "concrete", "instanbul", "ppb",
            "resid_build_sale_price", "energy"]

N_ITER = 2000
SMOOTH = 50

_COLS = {0: "algo", 1: "run_id", 2: "dataset", 3: "seed", 4: "gen",
         5: "train", 6: "timing", 7: "nodes",  8: "test", 9: "nodes_count", 10: "log"}

METRICS = [
    ("train",       "Train RMSE",  "%.0f"),
    ("test",        "Test RMSE",   "%.0f"),
    ("nodes_count", "Elite nodes", "%.0f"),
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _load_csv(path, n_iter=N_ITER):
    df = pd.read_csv(path, header=None).rename(columns=_COLS)
    df["seed"] = df["seed"].astype(int)
    complete = df[df["gen"] == n_iter][["algo", "dataset", "seed"]].drop_duplicates()
    df = df.merge(complete, on=["algo", "dataset", "seed"])
    df = df.drop_duplicates(subset=["algo", "dataset", "seed", "gen"], keep="last")
    return df


def _save_fig(fig, stem):
    """Save *fig* as <stem>.png and <stem>.tex (TikZ) then close it."""
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    png_path = stem + ".png"
    tex_path = stem + ".tex"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    tikzplotlib.save(tex_path, figure=fig, strict=False)
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {tex_path}")


def _make_grid(nrows, title):
    """Create a (nrows × 3-metric) subplot grid."""
    fig, axes = plt.subplots(
        nrows=nrows, ncols=len(METRICS),
        figsize=(14, 3.5 * nrows),
        sharex=True,
    )
    if nrows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.005)
    return fig, axes


def _plot_lines(ax, groups, colors, metric, fmt):
    """
    groups : dict {key: DataFrame with columns gen, seed, <metric>}
    Draws mean ± 1-std lines on *ax* for each key, applying SMOOTH rolling mean.
    """
    for key, fdata in groups.items():
        if fdata.empty:
            continue
        pivot = fdata.pivot_table(index="gen", columns="seed", values=metric)
        if SMOOTH > 1:
            pivot = pivot.rolling(SMOOTH, min_periods=1).mean()
        mean = pivot.mean(axis=1)
        std  = pivot.std(axis=1)
        ax.plot(mean.index, mean.values, color=colors[key], linewidth=1.5)
        ax.fill_between(mean.index, mean - std, mean + std,
                        color=colors[key], alpha=0.12)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)


def _legend_lines(colors, labels):
    return [Line2D([0], [0], color=c, linewidth=2, label=labels[k])
            for k, c in colors.items()]


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 0 — Head-size sweep
# ══════════════════════════════════════════════════════════════════════════════

def plot_head_size():
    print("\n── Head-size sweep ──────────────────────────────────────────────")
    log_path = os.path.join(_LOG_DIR, "results_head_size_07052026.csv")
    out_dir  = os.path.join(_OUT_ROOT, "evolution")

    HD_VALUES = ["5", "17", "25"]
    XO_VALUES = ["50", "500"]

    COLORS = {"5": "#e41a1c", "17": "#377eb8", "25": "#4daf4a"}
    LINES  = {"50": "-",      "500": "--"}
    ALPHAS = {"50": 0.9,      "500": 0.75}

    df = _load_csv(log_path)
    df["variant"] = df["algo"].str.extract(r'^(SLIM[+*]\w+)_hd')
    df["hd"]      = df["algo"].str.extract(r'_hd(\d+)_xo')
    df["xo_freq"] = df["algo"].str.extract(r'_xo(\d+)$')
    df = df[df["variant"].isin(VARIANTS)]

    for dataset in DATASETS:
        dset = df[df["dataset"] == dataset]
        if dset.empty:
            print(f"  [SKIP] {dataset} — no data")
            continue

        fig, axes = _make_grid(len(VARIANTS),
                               f"Head-size XO sweep — {dataset}")

        for row, variant in enumerate(VARIANTS):
            sub = dset[dset["variant"] == variant]
            for col, (metric, ylabel, fmt) in enumerate(METRICS):
                ax = axes[row, col]
                ax.set_title(variant, fontsize=10, pad=3)
                ax.set_ylabel(ylabel, fontsize=9)

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
                        ax.plot(mean.index, mean.values,
                                color=COLORS[hd], linestyle=LINES[xo],
                                linewidth=1.3, alpha=ALPHAS[xo])
                        ax.fill_between(mean.index, mean - std, mean + std,
                                        color=COLORS[hd], alpha=0.10)
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
                ax.tick_params(labelsize=8)
                ax.grid(True, linewidth=0.4, alpha=0.5)

        for ax in axes[-1, :]:
            ax.set_xlabel("Generation", fontsize=9)

        legend_elems = ([Line2D([0], [0], color=COLORS[h], linewidth=2,
                                label=f"max_depth={h}") for h in HD_VALUES]
                        + [Line2D([0], [0], color="k", linestyle=LINES[x],
                                  linewidth=1.5, label=f"XO every {x} gen")
                           for x in XO_VALUES])
        fig.legend(handles=legend_elems, loc="upper right", fontsize=9,
                   framealpha=0.85, bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()

        _save_fig(fig, os.path.join(out_dir, f"head_size_{dataset}"))


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 0b — Best head-XO config vs standard SLIM
# ══════════════════════════════════════════════════════════════════════════════

def plot_headsize_vs_slim():
    print("\n── Head-XO vs standard SLIM ─────────────────────────────────────")
    out_dir = os.path.join(_OUT_ROOT, "evolution")

    COLORS = {"hd17_xo500": "#377eb8", "standard": "#555555"}
    LABELS = {"hd17_xo500": "Head XO  hd=17, xo=500",
              "standard":   "Standard SLIM (no head XO)"}

    hs = _load_csv(os.path.join(_LOG_DIR, "results_head_size_07052026.csv"))
    hs["variant"] = hs["algo"].str.extract(r'^(SLIM[+*]\w+)_hd')
    hs["hd"]      = hs["algo"].str.extract(r'_hd(\d+)_xo')
    hs["xo_freq"] = hs["algo"].str.extract(r'_xo(\d+)$')
    hs = hs[(hs["hd"] == "17") & (hs["xo_freq"] == "500")].copy()
    hs["config"]  = "hd17_xo500"

    sc = _load_csv(os.path.join(_LOG_DIR, "results_scramble_xo_05052026.csv"))
    sc["variant"] = sc["algo"].str.extract(r'^(SLIM[+*]\w+)_head_xo')
    sc["xo_freq"] = sc["algo"].str.extract(r'_head_xo(\w+)$')
    sc = sc[sc["xo_freq"] == "None"].copy()
    sc["config"]  = "standard"

    df = pd.concat([hs, sc], ignore_index=True)
    df = df[df["variant"].isin(VARIANTS)]

    for dataset in DATASETS:
        dset = df[df["dataset"] == dataset]
        if dset.empty:
            print(f"  [SKIP] {dataset} — no data")
            continue

        fig, axes = _make_grid(len(VARIANTS),
                               f"Head XO (hd=17, xo=500) vs Standard SLIM — {dataset}")

        for row, variant in enumerate(VARIANTS):
            sub = dset[dset["variant"] == variant]
            for col, (metric, ylabel, fmt) in enumerate(METRICS):
                ax = axes[row, col]
                ax.set_title(variant, fontsize=10, pad=3)
                ax.set_ylabel(ylabel, fontsize=9)
                groups = {cfg: sub[sub["config"] == cfg].sort_values(["seed", "gen"])
                          for cfg in ("standard", "hd17_xo500")}
                _plot_lines(ax, groups, COLORS, metric, fmt)

        for ax in axes[-1, :]:
            ax.set_xlabel("Generation", fontsize=9)

        fig.legend(handles=_legend_lines(COLORS, LABELS),
                   loc="upper right", fontsize=10,
                   framealpha=0.85, bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()

        _save_fig(fig, os.path.join(out_dir, f"headsize_vs_slim_{dataset}"))


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Scramble / periodic XO frequency
# ══════════════════════════════════════════════════════════════════════════════

def plot_scramble_xo():
    print("\n── Scramble XO (frequency sweep) ───────────────────────────────")
    log_path = os.path.join(_LOG_DIR, "results_scramble_xo_05052026.csv")
    out_dir  = os.path.join(_OUT_ROOT, "evolution")

    FREQS  = ["None", "10", "50", "100", "500"]
    COLORS = {"None": "#555555", "10": "#e41a1c", "50": "#ff7f00",
              "100": "#377eb8",  "500": "#4daf4a"}
    LABELS = {"None": "No XO",      "10":  "XO every 10",
              "50":  "XO every 50", "100": "XO every 100",
              "500": "XO every 500"}

    df = _load_csv(log_path)
    df["variant"] = df["algo"].str.extract(r'^(SLIM[+*]\w+)_head_xo')
    df["xo_freq"] = df["algo"].str.extract(r'_head_xo(\w+)$')
    df = df[df["variant"].isin(VARIANTS)]

    for dataset in DATASETS:
        dset = df[df["dataset"] == dataset]
        if dset.empty:
            print(f"  [SKIP] {dataset} — no data")
            continue

        fig, axes = _make_grid(len(VARIANTS), f"Scramble XO (frequency) — {dataset}")

        for row, variant in enumerate(VARIANTS):
            sub = dset[dset["variant"] == variant]
            for col, (metric, ylabel, fmt) in enumerate(METRICS):
                ax = axes[row, col]
                ax.set_title(variant, fontsize=10, pad=3)
                ax.set_ylabel(ylabel, fontsize=9)
                groups = {f: sub[sub["xo_freq"] == f].sort_values(["seed", "gen"])
                          for f in FREQS}
                _plot_lines(ax, groups, COLORS, metric, fmt)

        for ax in axes[-1, :]:
            ax.set_xlabel("Generation", fontsize=9)

        fig.legend(handles=_legend_lines(COLORS, LABELS),
                   loc="upper right", fontsize=9,
                   framealpha=0.85, bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()

        _save_fig(fig, os.path.join(out_dir, f"scramble_xo_{dataset}"))


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Probabilistic XO
# ══════════════════════════════════════════════════════════════════════════════

def plot_prob_xo():
    print("\n── Probabilistic XO ─────────────────────────────────────────────")
    log_path = os.path.join(_LOG_DIR, "results_prob_xo_12052026.csv")
    out_dir  = os.path.join(_OUT_ROOT, "evolution")

    PXO_VALUES = ["0.0", "0.3", "0.5", "0.7"]
    COLORS = {"0.0": "#555555", "0.3": "#4daf4a", "0.5": "#377eb8", "0.7": "#e41a1c"}
    LABELS = {"0.0": "p_xo=0.0 (standard SLIM)",
              "0.3": "p_xo=0.3", "0.5": "p_xo=0.5", "0.7": "p_xo=0.7"}

    df = _load_csv(log_path)
    df["variant"] = df["algo"].str.extract(r'^(SLIM[+*]\w+)_pxo')
    df["pxo"]     = df["algo"].str.extract(r'_pxo([0-9.]+)$')
    df = df[df["variant"].isin(VARIANTS)]

    for dataset in DATASETS:
        dset = df[df["dataset"] == dataset]
        if dset.empty:
            print(f"  [SKIP] {dataset} — no data")
            continue

        fig, axes = _make_grid(len(VARIANTS),
                               f"Prob-XO (max_depth=17) — {dataset}")

        for row, variant in enumerate(VARIANTS):
            sub = dset[dset["variant"] == variant]
            for col, (metric, ylabel, fmt) in enumerate(METRICS):
                ax = axes[row, col]
                ax.set_title(variant, fontsize=10, pad=3)
                ax.set_ylabel(ylabel, fontsize=9)
                groups = {p: sub[sub["pxo"] == p].sort_values(["seed", "gen"])
                          for p in PXO_VALUES}
                _plot_lines(ax, groups, COLORS, metric, fmt)

        for ax in axes[-1, :]:
            ax.set_xlabel("Generation", fontsize=9)

        fig.legend(handles=_legend_lines(COLORS, LABELS),
                   loc="upper right", fontsize=10,
                   framealpha=0.85, bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()

        _save_fig(fig, os.path.join(out_dir, f"prob_xo_{dataset}"))


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Population / budget allocation with XO
# ══════════════════════════════════════════════════════════════════════════════

def plot_pop_xo():
    print("\n── Pop/budget sweep ─────────────────────────────────────────────")
    log_pop  = os.path.join(_LOG_DIR, "results_pop_xo_15052026.csv")
    log_base = os.path.join(_LOG_DIR, "results_prob_xo_12052026.csv")
    out_dir  = os.path.join(_OUT_ROOT, "evolution")

    BUDGET    = 200_000
    EVAL_STEP = 1_000
    SMOOTH_POP = 10
    EVAL_GRID = np.arange(0, BUDGET + 1, EVAL_STEP)

    CONFIGS = [("baseline", 100, 2000, 0.0),
               ("pop200",   200, 1000, 0.7),
               ("pop500",   500,  400, 0.7),
               ("pop1000", 1000,  200, 0.7)]
    COLORS  = {"baseline": "#555555", "pop200": "#4daf4a",
               "pop500":   "#377eb8", "pop1000": "#e41a1c"}
    LABELS  = {"baseline": "Baseline  p_xo=0.0  pop=100  iter=2000",
               "pop200":   "p_xo=0.7  pop=200   iter=1000",
               "pop500":   "p_xo=0.7  pop=500   iter=400",
               "pop1000":  "p_xo=0.7  pop=1000  iter=200"}

    # --- pop_xo log ---
    dp = pd.read_csv(log_pop, header=None).rename(columns=_COLS)
    dp["seed"]    = dp["seed"].astype(int)
    dp["variant"] = dp["algo"].str.extract(r"^(SLIM[+*]\w+)_pop")
    dp["pop"]     = dp["algo"].str.extract(r"_pop(\d+)_").astype(int)
    dp["n_iter"]  = dp["algo"].str.extract(r"_iter(\d+)_").astype(int)
    dp["cfg"]     = dp["pop"].map({200: "pop200", 500: "pop500", 1000: "pop1000"})
    dp["evals"]   = dp["gen"] * dp["pop"]
    complete_mask = dp.apply(lambda r: r["gen"] == r["n_iter"], axis=1)
    complete_keys = dp[complete_mask][["algo", "dataset", "seed"]].drop_duplicates()
    dp = dp.merge(complete_keys, on=["algo", "dataset", "seed"])
    dp = dp.drop_duplicates(subset=["algo", "dataset", "seed", "gen"], keep="last")

    # --- baseline from prob_xo log (p_xo=0.0) ---
    db = _load_csv(log_base)
    db = db[db["algo"].str.endswith("_pxo0.0")].copy()
    db["variant"] = db["algo"].str.extract(r"^(SLIM[+*]\w+)_pxo")
    db["pop"]     = 100
    db["cfg"]     = "baseline"
    db["evals"]   = db["gen"] * 100

    df = pd.concat([dp, db], ignore_index=True)
    df = df[df["variant"].isin(VARIANTS)]

    def _resample(sub_seed, eval_col, metric):
        s = sub_seed.sort_values(eval_col)
        return np.interp(EVAL_GRID, s[eval_col].values, s[metric].values,
                         left=np.nan, right=np.nan)

    for dataset in DATASETS:
        dset = df[df["dataset"] == dataset]
        if dset.empty:
            print(f"  [SKIP] {dataset} — no data")
            continue

        fig, axes = _make_grid(len(VARIANTS),
                               f"Pop/Iter sweep (p_xo=0.7, max_depth=17) — {dataset}")

        for row, variant in enumerate(VARIANTS):
            sub_var = dset[dset["variant"] == variant]

            for col, (metric, ylabel, fmt) in enumerate(METRICS):
                ax = axes[row, col]
                ax.set_title(variant, fontsize=10, pad=3)
                ax.set_ylabel(ylabel, fontsize=9)

                for cfg_name, *_ in CONFIGS:
                    sub_cfg = sub_var[sub_var["cfg"] == cfg_name]
                    if sub_cfg.empty:
                        continue
                    resampled = [_resample(grp, "evals", metric)
                                 for _, grp in sub_cfg.groupby("seed")]
                    if not resampled:
                        continue
                    mat = np.vstack(resampled)
                    df_mat = pd.DataFrame(mat.T)
                    if SMOOTH_POP > 1:
                        df_mat = df_mat.rolling(SMOOTH_POP, min_periods=1).mean()
                    mean = df_mat.mean(axis=1).values
                    std  = df_mat.std(axis=1).values
                    ax.plot(EVAL_GRID, mean, color=COLORS[cfg_name], linewidth=1.5)
                    ax.fill_between(EVAL_GRID, mean - std, mean + std,
                                    color=COLORS[cfg_name], alpha=0.12)

                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))
                ax.tick_params(labelsize=8)
                ax.grid(True, linewidth=0.4, alpha=0.5)

        for ax in axes[-1, :]:
            ax.set_xlabel("Evaluations", fontsize=9)

        fig.legend(handles=_legend_lines(COLORS, LABELS),
                   loc="upper right", fontsize=9,
                   framealpha=0.85, bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()

        _save_fig(fig, os.path.join(out_dir, f"pop_xo_{dataset}"))


# ══════════════════════════════════════════════════════════════════════════════
# STN PLOTS
# (imports from stn_plot.py in the same directory)
# ══════════════════════════════════════════════════════════════════════════════

def plot_stns():
    print("\n── STN plots ────────────────────────────────────────────────────")

    from stn_plot import plot_stn, _get_layout, _NODE, FSIZE

    out_root = os.path.join(_OUT_ROOT, "stns")

    STN_JOBS = [
        # (layout, node_sizes, ncols, x_attr, y_attr, models)
        ("stress",  ("tree", "node"), 3, "TreeSize", "Fitness",
         ("genotype", "hypercube", "clustering")),
        ("fitness", ("tree", "node"), 3, "TreeSize", "Fitness",
         ("genotype", "hypercube", "clustering")),
        ("bivar",   ("tree", "node"), 2, "TreeSize", "Fitness",
         ("genotype", "hypercube")),
        ("bivar",   ("tree", "node"), 1, "ClusterID", "Fitness",
         ("clustering",)),
    ]

    for benchmark in DATASETS:
        infolder  = os.path.join(_STN_ROOT, benchmark)
        outfolder = os.path.join(out_root,  benchmark)

        if not os.path.isdir(infolder):
            print(f"  [SKIP] {benchmark} — STN folder not found: {infolder}")
            continue

        print(f"\n  {benchmark}")

        for layout, node_sizes, ncols, x_attr, y_attr, models in STN_JOBS:
            pkls = sorted(f for f in os.listdir(infolder)
                          if f.endswith(".pkl") and any(m in f for m in models))
            if not pkls:
                print(f"    [SKIP] no PKL files for models={models}")
                continue

            graphs = []
            for p in pkls:
                with open(os.path.join(infolder, p), "rb") as fh:
                    d = pickle.load(fh)
                    if any(v in d["alg"] for v in VARIANTS):
                        graphs.append(d)

            if not graphs:
                print(f"    [SKIP] no graphs matched VARIANTS for layout={layout}")
                continue

            # global scaling limits
            all_fit, all_tree, all_count, all_edge = [], [], [], []
            all_x, all_y = [], []
            for d in graphs:
                G = d["G"]
                all_fit.extend(nx.get_node_attributes(G, "Fitness").values())
                all_tree.extend(nx.get_node_attributes(G, "TreeSize").values())
                all_count.extend(nx.get_node_attributes(G, "Count").values())
                all_edge.extend(nx.get_edge_attributes(G, "Count").values() or [1])
                if layout == "bivar":
                    all_x.extend(nx.get_node_attributes(G, x_attr).values())
                    all_y.extend(nx.get_node_attributes(G, y_attr).values())

            tree_range  = (min(all_tree),  max(all_tree))
            count_range = (min(all_count), max(all_count))
            edge_range  = (min(all_edge),  max(all_edge))
            x_limits    = (min(all_x), max(all_x)) if all_x else None
            y_limits    = (min(all_y), max(all_y)) if all_y else None

            # compute layouts once, reuse across node_size variants
            pos_list = [_get_layout(d["G"], layout, x_attr=x_attr, y_attr=y_attr)
                        for d in graphs]

            n     = len(graphs)
            ncols = min(ncols, n)
            nrows = (n + ncols - 1) // ncols
            w     = 4.5 * ncols + 1.8
            h     = 4.2 * nrows + 0.6

            model_tag = "" if len(models) == 3 else "_" + "+".join(models)

            for node_size in node_sizes:
                size_range = tree_range if node_size == "tree" else count_range
                attr       = "TreeSize" if node_size == "tree" else "Count"

                fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False)

                for idx, (d, pos) in enumerate(zip(graphs, pos_list)):
                    r, c = divmod(idx, ncols)
                    plot_stn(axes[r][c], d["G"], d["model"], d["alg"],
                             layout=layout, node_size_attr=attr,
                             size_range=size_range, edge_range=edge_range,
                             x_attr=x_attr, y_attr=y_attr,
                             x_limits=x_limits, y_limits=y_limits,
                             pos=pos)

                for idx in range(n, nrows * ncols):
                    r, c = divmod(idx, ncols)
                    axes[r][c].set_visible(False)

                legend_handles = [
                    mpatches.Patch(color=_NODE["Start"][0],  label="Start"),
                    mpatches.Patch(color=_NODE["Medium"][0], label="Medium", alpha=0.45),
                    mpatches.Patch(color=_NODE["End"][0],    label="End"),
                    mpatches.Patch(color=_NODE["Best"][0],   label="Best"),
                ]
                fig.legend(handles=legend_handles, loc="center right",
                           fontsize=FSIZE, framealpha=0.9,
                           bbox_to_anchor=(1.0, 0.5))
                fig.suptitle(
                    f"STN — {benchmark}  |  layout={layout}  node={node_size}",
                    fontsize=FSIZE + 1, fontweight="bold")
                fig.tight_layout(rect=[0, 0, 0.88, 0.97])

                stem = os.path.join(
                    outfolder,
                    f"{benchmark}_{layout}{model_tag}_{node_size}_stn"
                )
                _save_fig(fig, stem)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(_OUT_ROOT, exist_ok=True)

    plot_head_size()
    plot_headsize_vs_slim()
    plot_scramble_xo()
    plot_prob_xo()
    plot_pop_xo()
    plot_stns()

    print(f"\nAll figures written to: {_OUT_ROOT}")
