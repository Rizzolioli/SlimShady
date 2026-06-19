"""
generate_final_figs.py
======================
Single script that produces all publication-ready figures and tables into
main/log/latex_final/.

Sections
--------
1. Periodic head XO  (xo_freq = None, 50, 500)  — one figure per dataset
2. Probabilistic XO + depth cap (hd ∈ {5,10,17}, p_xo ∈ {0.3,0.7}) — one figure per dataset
3. SLIM*ABS test RMSE per dataset                — one figure per dataset
4. p_xo summary table (p_xo = 0.0, 0.3, 0.7)
5. STN comparison grids (from generate_stn_grids.py)

Output layout
-------------
main/log/latex_final/
  periodic_xo/
    periodic_xo_<dataset>.{png,tex}       (6 files, one per dataset)
  prob_xo/
    prob_xo_<dataset>.{png,tex}           (6 files, one per dataset)
    slim_abs_test_<dataset>.{png,tex}     (6 files)
  tables/
    prob_xo_table.csv
  stns/
    <dataset>_stn_<stem>.{png,tex}        (requires pre-built pkl in main/log/stns/)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

try:
    import matplot2tikz as tikzplotlib
    _TIKZ = True
except ImportError:
    _TIKZ = False
    print("[WARN] matplot2tikz not found - .tex files will be skipped.")

# ── PATH SETUP ────────────────────────────────────────────────────────────────

_HERE    = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR = os.path.join(_HERE, "..", "log")
_OUT     = os.path.join(_LOG_DIR, "latex_final")

sys.path.insert(0, _HERE)   # so generate_stn_grids imports resolve

# ── SHARED CONFIG ─────────────────────────────────────────────────────────────

_VARIANTS = ["SLIM+2SIG", "SLIM*ABS", "SLIM*1SIG"]
_DATASETS = ["toxicity", "concrete", "instanbul", "ppb",
             "resid_build_sale_price", "energy"]

# (column in df, human label, filename key)
_METRICS = [
    ("train",       "Train RMSE",  "train"),
    ("test",        "Test RMSE",   "test"),
    ("nodes_count", "Node count",  "nodes"),
]

_SMOOTH = 50

_COLS = {0: "algo", 1: "run_id", 2: "dataset", 3: "seed", 4: "gen",
         5: "train", 6: "timing", 7: "nodes",   8: "test", 9: "nodes_count", 10: "log"}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _load_csv(path, n_iter=2000):
    """Load log CSV; keep only trajectories that reached n_iter; deduplicate."""
    df = pd.read_csv(path, header=None).rename(columns=_COLS)
    df["seed"] = df["seed"].astype(int)
    complete = df[df["gen"] == n_iter][["algo", "dataset", "seed"]].drop_duplicates()
    df = df.merge(complete, on=["algo", "dataset", "seed"])
    df = df.drop_duplicates(subset=["algo", "dataset", "seed", "gen"], keep="last")
    return df


def _load_depth_cap():
    """Load depth_cap CSV with UUID-based dedup; return df with variant/p_xo/hd columns."""
    from uuid import UUID
    df = pd.read_csv(os.path.join(_LOG_DIR, "results_depth_cap_new.csv"),
                     header=None).rename(columns=_COLS)
    df["seed"]    = df["seed"].astype(int)
    df["variant"] = df["algo"].str.extract(r"^(SLIM[+*]\w+)_pxo")
    df["p_xo"]    = pd.to_numeric(df["algo"].str.extract(r"_pxo([0-9.]+)_")[0],
                                   errors="coerce")
    df["hd"]      = pd.to_numeric(df["algo"].str.extract(r"_hd(\d+)$")[0],
                                   errors="coerce")
    df = df[df["hd"].isin([5, 10, 17]) & df["p_xo"].isin([0.3, 0.7])].copy()
    df["hd"]  = df["hd"].astype(int)
    df["p_xo"] = df["p_xo"].astype(float)

    # Keep only the latest batch per (algo, dataset, seed) via UUID timestamp
    df["_uuid_t"] = df["run_id"].map(lambda r: UUID(r).time)
    latest = (df.groupby(["algo", "dataset", "seed"])["_uuid_t"]
                .max().reset_index().rename(columns={"_uuid_t": "_latest"}))
    df = df.merge(latest, on=["algo", "dataset", "seed"])
    df = df[df["_uuid_t"] == df["_latest"]].drop(columns=["_uuid_t", "_latest"])

    return df[df["variant"].isin(_VARIANTS)].copy()


def _last_gen(df):
    """Keep only the final-generation row per (algo, dataset, seed)."""
    last = (df.groupby(["algo", "dataset", "seed"])["gen"]
              .max().reset_index().rename(columns={"gen": "last_gen"}))
    df = df.merge(last, on=["algo", "dataset", "seed"])
    return df[df["gen"] == df["last_gen"]].copy()


def _stats(df):
    """Median + IQR (Q75-Q25) over seeds for train, test, nodes_count."""
    def iqr(x):
        return x.quantile(0.75) - x.quantile(0.25)
    return (df.groupby(["algo", "dataset"])
              .agg(
                  train_med=("train",       "median"),
                  train_iqr=("train",       iqr),
                  test_med =("test",        "median"),
                  test_iqr =("test",        iqr),
                  size_med =("nodes_count", "median"),
                  size_iqr =("nodes_count", iqr),
                  n_seeds  =("seed",        "nunique"),
              )
              .reset_index())


def _save_fig(fig, stem):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {os.path.basename(stem)}.png")
    if _TIKZ:
        try:
            tikzplotlib.save(stem + ".tex", figure=fig, strict=False)
            print(f"  Saved: {os.path.basename(stem)}.tex")
        except Exception as e:
            print(f"  [WARN] tikz skipped for {os.path.basename(stem)}: {e}")
    plt.close(fig)


def _plot_line(ax, fdata, metric, color, label, no_shade=False, linestyle="-"):
    """Plot mean +/- std convergence line; skip fill_between when no_shade=True."""
    if fdata.empty:
        return
    pivot = fdata.pivot_table(index="gen", columns="seed", values=metric)
    if _SMOOTH > 1:
        pivot = pivot.rolling(_SMOOTH, min_periods=1).mean()
    mean = pivot.mean(axis=1)
    std  = pivot.std(axis=1)
    ax.plot(mean.index, mean.values, color=color, linewidth=1.3,
            label=label, linestyle=linestyle)
    if not no_shade:
        ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.12)
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)


def _make_grid(title):
    """Create a (n_variants x n_metrics) subplot grid for one dataset."""
    fig, axes = plt.subplots(
        nrows=len(_VARIANTS), ncols=len(_METRICS),
        figsize=(14, 3.5 * len(_VARIANTS)),
        sharex=True,
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.005)
    for ri, variant in enumerate(_VARIANTS):
        axes[ri, 0].set_ylabel(variant, fontsize=9, fontweight="bold")
    for ci, (_, metric_label, _) in enumerate(_METRICS):
        axes[0, ci].set_title(metric_label, fontsize=10, pad=3)
    for ci in range(len(_METRICS)):
        axes[-1, ci].set_xlabel("Generation", fontsize=9)
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Periodic XO  (xo_freq in {None, 50, 500})
# One figure per dataset; rows = variants, cols = metrics.
# Baseline (xoNone) as black line without shading.
# ══════════════════════════════════════════════════════════════════════════════

def section_periodic_xo():
    print("\n-- Periodic XO --------------------------------------------------")
    out_dir = os.path.join(_OUT, "periodic_xo")

    FREQS  = ["None", "50", "500"]
    COLORS = {"None": "#000000", "50": "#ff7f00", "500": "#377eb8"}
    LABELS = {"None": "No head XO (baseline)",
              "50":   "XO every 50 gen",
              "500":  "XO every 500 gen"}

    df = _load_csv(os.path.join(_LOG_DIR, "results_scramble_xo_05052026.csv"))
    df["variant"] = df["algo"].str.extract(r'^(SLIM[+*]\w+)_head_xo')
    df["xo_freq"] = df["algo"].str.extract(r'_head_xo(\w+)$')
    df = df[df["variant"].isin(_VARIANTS) & df["xo_freq"].isin(FREQS)]

    for dataset in _DATASETS:
        dset = df[df["dataset"] == dataset]
        if dset.empty:
            print(f"  [SKIP] {dataset} - no data")
            continue

        fig, axes = _make_grid(f"Periodic head XO — {dataset.replace('_', ' ')}")

        for ri, variant in enumerate(_VARIANTS):
            sub = dset[dset["variant"] == variant]
            for ci, (metric_col, _, _) in enumerate(_METRICS):
                ax = axes[ri, ci]
                for freq in FREQS:
                    _plot_line(
                        ax,
                        sub[sub["xo_freq"] == freq].sort_values(["seed", "gen"]),
                        metric_col,
                        color=COLORS[freq],
                        label=LABELS[freq],
                        no_shade=(freq == "None"),
                    )

        legend_handles = [Line2D([0], [0], color=COLORS[f], linewidth=2, label=LABELS[f])
                          for f in FREQS]
        fig.legend(handles=legend_handles, loc="upper right", fontsize=9,
                   framealpha=0.85, bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()

        _save_fig(fig, os.path.join(out_dir, f"periodic_xo_{dataset}"))

    print(f"  -> {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Probabilistic XO + depth cap
# Baseline (p_xo=0.0) from prob_xo, black no shading.
# Treatment: depth_cap data with hd in {5,10,17}, p_xo in {0.3,0.7}.
# Color = hd, linestyle = p_xo.  One figure per dataset.
# ══════════════════════════════════════════════════════════════════════════════

def section_prob_xo():
    print("\n-- Probabilistic XO + depth cap ---------------------------------")
    out_dir = os.path.join(_OUT, "prob_xo")

    HD_COLORS  = {5: "#e41a1c", 10: "#ff7f00", 17: "#377eb8"}
    HD_LABELS  = {5: "hd=5",    10: "hd=10",   17: "hd=17"}
    PXO_STYLES = {0.3: "--", 0.7: "-"}
    PXO_LABELS = {0.3: "p_xo=0.3", 0.7: "p_xo=0.7"}

    # Baseline from prob_xo (p_xo=0.0)
    db = _load_csv(os.path.join(_LOG_DIR, "results_prob_xo_12052026.csv"))
    db["variant"] = db["algo"].str.extract(r'^(SLIM[+*]\w+)_pxo')
    db["pxo_str"] = db["algo"].str.extract(r'_pxo([0-9.]+)$')
    db = db[db["variant"].isin(_VARIANTS) & (db["pxo_str"] == "0.0")]

    # Depth-cap treatment data
    dc = _load_depth_cap()

    for dataset in _DATASETS:
        db_dset = db[db["dataset"] == dataset]
        dc_dset = dc[dc["dataset"] == dataset]

        fig, axes = _make_grid(f"Probabilistic head XO — {dataset.replace('_', ' ')}")

        for ri, variant in enumerate(_VARIANTS):
            sub_base = db_dset[db_dset["variant"] == variant]
            sub_dc   = dc_dset[dc_dset["variant"] == variant]

            for ci, (metric_col, _, _) in enumerate(_METRICS):
                ax = axes[ri, ci]

                # Baseline: black, no shade
                _plot_line(ax, sub_base.sort_values(["seed", "gen"]), metric_col,
                           color="#000000", label="p_xo=0.0 (baseline)", no_shade=True)

                # hd / p_xo variants
                for hd in [5, 10, 17]:
                    for pxo in [0.3, 0.7]:
                        fdata = sub_dc[(sub_dc["hd"] == hd) & (sub_dc["p_xo"] == pxo)]
                        _plot_line(ax, fdata.sort_values(["seed", "gen"]), metric_col,
                                   color=HD_COLORS[hd],
                                   label=f"{HD_LABELS[hd]}, {PXO_LABELS[pxo]}",
                                   linestyle=PXO_STYLES[pxo])

        legend_elems = [
            Line2D([0], [0], color="#000000", linewidth=2, label="p_xo=0.0 (baseline)"),
        ]
        for hd in [5, 10, 17]:
            for pxo in [0.3, 0.7]:
                legend_elems.append(
                    Line2D([0], [0], color=HD_COLORS[hd], linestyle=PXO_STYLES[pxo],
                           linewidth=1.5, label=f"{HD_LABELS[hd]}, {PXO_LABELS[pxo]}")
                )
        fig.legend(handles=legend_elems, loc="upper right", fontsize=8,
                   framealpha=0.85, bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()

        _save_fig(fig, os.path.join(out_dir, f"prob_xo_{dataset}"))

    # ── SLIM*ABS test RMSE, one single-panel image per dataset ──────────────
    print("  SLIM*ABS per-dataset test RMSE:")

    pxo_df = _load_csv(os.path.join(_LOG_DIR, "results_prob_xo_12052026.csv"))
    pxo_df["variant"] = pxo_df["algo"].str.extract(r'^(SLIM[+*]\w+)_pxo')
    pxo_df["pxo"]     = pxo_df["algo"].str.extract(r'_pxo([0-9.]+)$')
    PXO    = ["0.0", "0.3", "0.7"]
    PXO_COLORS = {"0.0": "#000000", "0.3": "#4daf4a", "0.7": "#e41a1c"}
    PXO_L      = {"0.0": "p_xo=0.0 (baseline)", "0.3": "p_xo=0.3", "0.7": "p_xo=0.7"}
    sub_abs = pxo_df[(pxo_df["variant"] == "SLIM*ABS") & pxo_df["pxo"].isin(PXO)]

    for dataset in _DATASETS:
        dset = sub_abs[sub_abs["dataset"] == dataset]
        if dset.empty:
            print(f"    [SKIP] {dataset} - no data")
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"SLIM*ABS - Test RMSE - {dataset.replace('_', ' ')}",
                     fontsize=11, fontweight="bold")
        for pxo in PXO:
            _plot_line(ax, dset[dset["pxo"] == pxo].sort_values(["seed", "gen"]),
                       "test", color=PXO_COLORS[pxo], label=PXO_L[pxo],
                       no_shade=(pxo == "0.0"))
        ax.set_xlabel("Generation", fontsize=9)
        ax.set_ylabel("Test RMSE", fontsize=9)
        ax.legend(fontsize=9, framealpha=0.9)
        fig.tight_layout()
        _save_fig(fig, os.path.join(out_dir, f"slim_abs_test_{dataset.replace('_', '-')}"))

    print(f"  -> {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — p_xo summary table  (p_xo in {0.0, 0.3, 0.7})
# ══════════════════════════════════════════════════════════════════════════════

def section_table():
    print("\n-- p_xo summary table -------------------------------------------")
    out_dir = os.path.join(_OUT, "tables")
    os.makedirs(out_dir, exist_ok=True)

    df = _load_csv(os.path.join(_LOG_DIR, "results_prob_xo_12052026.csv"))
    df = _last_gen(df)
    df["variant"] = df["algo"].str.extract(r'^(SLIM[+*]\w+)_pxo')
    df["p_xo"]    = df["algo"].str.extract(r'_pxo([0-9.]+)$')
    df = df[df["variant"].isin(_VARIANTS) & df["p_xo"].isin(["0.0", "0.3", "0.7"])]

    stats = _stats(df)
    meta  = df[["algo", "variant", "p_xo"]].drop_duplicates("algo")
    stats = stats.merge(meta, on="algo")

    csv_path = os.path.join(out_dir, "prob_xo_table.csv")
    stats.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  CSV  -> {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STN comparison grids
# ══════════════════════════════════════════════════════════════════════════════

def section_stns():
    print("\n-- STN grids ----------------------------------------------------")
    out_dir = os.path.join(_OUT, "stns")

    try:
        from generate_stn_grids import make_stn_grid, FIGURE_JOBS, DATASETS as _STN_DS
    except ImportError as e:
        print(f"  [SKIP] cannot import generate_stn_grids: {e}")
        return

    for benchmark in _STN_DS:
        print(f"\n  {benchmark}")
        for stem_suffix, layout, model, size_attr, x_attr, y_attr in FIGURE_JOBS:
            print(f"    [{stem_suffix}]")
            make_stn_grid(benchmark, layout, model, size_attr, x_attr, y_attr,
                          stem_suffix, out_root=out_dir)

    print(f"\n  -> {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(_OUT, exist_ok=True)
    print(f"Output root: {_OUT}\n")

    section_periodic_xo()
    section_prob_xo()
    section_table()
    section_stns()

    print(f"\n\nDone. All outputs written to: {_OUT}")
