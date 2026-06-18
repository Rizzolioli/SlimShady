"""
generate_final_figs.py
======================
Single script that produces all publication-ready figures and tables into
main/log/latex_final/.

Sections
--------
1. Periodic head XO  (xo_freq = None, 50, 500)  — one figure per metric
2. Probabilistic XO  (p_xo = 0.0, 0.3, 0.7)    — one figure per metric
3. SLIM*ABS test RMSE per dataset                — one figure per dataset
4. p_xo summary table (p_xo = 0.0, 0.3, 0.7)
5. STN comparison grids (from generate_stn_grids.py)

Output layout
-------------
main/log/latex_final/
  periodic_xo/
    periodic_xo_train.{png,tex}
    periodic_xo_test.{png,tex}
    periodic_xo_nodes.{png,tex}
  prob_xo/
    prob_xo_train.{png,tex}
    prob_xo_test.{png,tex}
    prob_xo_nodes.{png,tex}
    slim_abs_test_<dataset>.{png,tex}   (6 files)
  tables/
    prob_xo_table.csv
    prob_xo_table.tex
  stns/
    <dataset>_stn_<stem>.{png,tex}      (requires pre-built pkl in main/log/stns/)
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


def _last_gen(df):
    """Keep only the final-generation row per (algo, dataset, seed)."""
    last = (df.groupby(["algo", "dataset", "seed"])["gen"]
              .max().reset_index().rename(columns={"gen": "last_gen"}))
    df = df.merge(last, on=["algo", "dataset", "seed"])
    return df[df["gen"] == df["last_gen"]].copy()


def _stats(df):
    """Median + IQR (Q75−Q25) over seeds for train, test, nodes_count."""
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


def _plot_line(ax, fdata, metric, color, label, no_shade=False):
    """Plot mean ± std convergence line; skip fill_between when no_shade=True."""
    if fdata.empty:
        return
    pivot = fdata.pivot_table(index="gen", columns="seed", values=metric)
    if _SMOOTH > 1:
        pivot = pivot.rolling(_SMOOTH, min_periods=1).mean()
    mean = pivot.mean(axis=1)
    std  = pivot.std(axis=1)
    ax.plot(mean.index, mean.values, color=color, linewidth=1.3, label=label)
    if not no_shade:
        ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.12)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)


def _decorate_grid(axes, variants, datasets):
    """Dataset column headers, variant row labels, x-label on bottom row only."""
    nrows, ncols = axes.shape
    for ci, ds in enumerate(datasets):
        axes[0, ci].set_title(ds.replace("_", " "), fontsize=8, fontweight="bold")
    for ri, var in enumerate(variants):
        axes[ri, 0].set_ylabel(var, fontsize=9, fontweight="bold")
    for ri in range(nrows):
        for ci in range(1, ncols):
            axes[ri, ci].set_ylabel("")
    for ci in range(ncols):
        axes[nrows - 1, ci].set_xlabel("Generation", fontsize=8)
    for ri in range(nrows - 1):
        for ci in range(ncols):
            axes[ri, ci].set_xlabel("")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Periodic XO  (xo_freq ∈ {None, 50, 500})
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

    for metric_col, metric_label, metric_key in _METRICS:
        fig, axes = plt.subplots(
            len(_VARIANTS), len(_DATASETS),
            figsize=(22, 10.5), sharex=True,
            gridspec_kw={"hspace": 0.40, "wspace": 0.30},
        )
        fig.suptitle(f"Periodic head XO — {metric_label}",
                     fontsize=12, fontweight="bold", y=1.01)

        for ri, variant in enumerate(_VARIANTS):
            for ci, dataset in enumerate(_DATASETS):
                ax  = axes[ri, ci]
                sub = df[(df["variant"] == variant) & (df["dataset"] == dataset)]
                for freq in FREQS:
                    _plot_line(
                        ax,
                        sub[sub["xo_freq"] == freq].sort_values(["seed", "gen"]),
                        metric_col,
                        color=COLORS[freq],
                        label=LABELS[freq],
                        no_shade=(freq == "None"),
                    )

        _decorate_grid(axes, _VARIANTS, _DATASETS)

        legend_handles = [Line2D([0], [0], color=COLORS[f], linewidth=2, label=LABELS[f])
                          for f in FREQS]
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=len(FREQS), fontsize=9, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=[0, 0.04, 1, 1])

        _save_fig(fig, os.path.join(out_dir, f"periodic_xo_{metric_key}"))

    print(f"  -> {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Probabilistic XO  (p_xo ∈ {0.0, 0.3, 0.7})
#           + SLIM*ABS test RMSE per dataset
# ══════════════════════════════════════════════════════════════════════════════

def section_prob_xo():
    print("\n-- Probabilistic XO ---------------------------------------------")
    out_dir = os.path.join(_OUT, "prob_xo")

    PXO    = ["0.0", "0.3", "0.7"]
    COLORS = {"0.0": "#000000", "0.3": "#4daf4a", "0.7": "#e41a1c"}
    LABELS = {"0.0": "p_xo=0.0 (baseline)", "0.3": "p_xo=0.3", "0.7": "p_xo=0.7"}

    df = _load_csv(os.path.join(_LOG_DIR, "results_prob_xo_12052026.csv"))
    df["variant"] = df["algo"].str.extract(r'^(SLIM[+*]\w+)_pxo')
    df["pxo"]     = df["algo"].str.extract(r'_pxo([0-9.]+)$')
    df = df[df["variant"].isin(_VARIANTS) & df["pxo"].isin(PXO)]

    # ── 2a: one image per metric ─────────────────────────────────────────────
    for metric_col, metric_label, metric_key in _METRICS:
        fig, axes = plt.subplots(
            len(_VARIANTS), len(_DATASETS),
            figsize=(22, 10.5), sharex=True,
            gridspec_kw={"hspace": 0.40, "wspace": 0.30},
        )
        fig.suptitle(f"Probabilistic head XO — {metric_label}",
                     fontsize=12, fontweight="bold", y=1.01)

        for ri, variant in enumerate(_VARIANTS):
            for ci, dataset in enumerate(_DATASETS):
                ax  = axes[ri, ci]
                sub = df[(df["variant"] == variant) & (df["dataset"] == dataset)]
                for pxo in PXO:
                    _plot_line(
                        ax,
                        sub[sub["pxo"] == pxo].sort_values(["seed", "gen"]),
                        metric_col,
                        color=COLORS[pxo],
                        label=LABELS[pxo],
                        no_shade=(pxo == "0.0"),
                    )

        _decorate_grid(axes, _VARIANTS, _DATASETS)

        legend_handles = [Line2D([0], [0], color=COLORS[p], linewidth=2, label=LABELS[p])
                          for p in PXO]
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=len(PXO), fontsize=9, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=[0, 0.04, 1, 1])

        _save_fig(fig, os.path.join(out_dir, f"prob_xo_{metric_key}"))

    # ── 2b: SLIM*ABS test RMSE, one image per dataset ────────────────────────
    print("  SLIM*ABS per-dataset test RMSE:"  )
    sub_abs = df[df["variant"] == "SLIM*ABS"]

    for dataset in _DATASETS:
        dset = sub_abs[sub_abs["dataset"] == dataset]
        if dset.empty:
            print(f"    [SKIP] {dataset} - no data")
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"SLIM*ABS — Test RMSE — {dataset.replace('_', ' ')}",
                     fontsize=11, fontweight="bold")

        for pxo in PXO:
            _plot_line(
                ax,
                dset[dset["pxo"] == pxo].sort_values(["seed", "gen"]),
                "test",
                color=COLORS[pxo],
                label=LABELS[pxo],
                no_shade=(pxo == "0.0"),
            )

        ax.set_xlabel("Generation", fontsize=9)
        ax.set_ylabel("Test RMSE", fontsize=9)
        ax.legend(fontsize=9, framealpha=0.9)
        fig.tight_layout()

        _save_fig(fig, os.path.join(out_dir, f"slim_abs_test_{dataset.replace('_', '-')}"))

    print(f"  -> {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — p_xo summary table  (p_xo ∈ {0.0, 0.3, 0.7})
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_cell(med, iqr, decimals=2):
    fmt = f"{{:.{decimals}f}}"
    return (fmt + " ({})").format(med, fmt.format(iqr))


def _to_latex(df_stats, caption, label, config_cols):
    n = len(config_cols)
    metric_header = (
        r"\multicolumn{2}{c}{Train RMSE} & "
        r"\multicolumn{2}{c}{Test RMSE} & "
        r"\multicolumn{2}{c}{Model Size}"
    )
    sub_header  = r"Med & IQR & Med & IQR & Med & IQR"
    col_spec    = "ll" + "l" * n + "rrrrrr"
    config_head = " & ".join(c.replace("_", " ").title() for c in config_cols)

    def _header_row():
        return (r"Dataset & Variant & " +
                (config_head + " & " if config_head else "") +
                metric_header + r" \\")

    def _sub_row():
        return r" & & " + (" & " * n) + sub_header + r" \\"

    def _cmidrules():
        return (
            r"\cmidrule(lr){" + str(3 + n) + "-" + str(4 + n) + r"}"
            r"\cmidrule(lr){" + str(5 + n) + "-" + str(6 + n) + r"}"
            r"\cmidrule(lr){" + str(7 + n) + "-" + str(8 + n) + r"}"
        )

    lines = [
        r"\begin{longtable}{" + col_spec + "}",
        r"\caption{" + caption + r"} \label{" + label + r"} \\",
        r"\toprule", _header_row(), _cmidrules(), _sub_row(),
        r"\midrule", r"\endfirsthead",
        r"\toprule", _header_row(), _sub_row(),
        r"\midrule", r"\endhead",
        r"\bottomrule", r"\endfoot",
    ]

    prev_ds = None
    for _, row in df_stats.sort_values(["dataset", "variant"] + config_cols).iterrows():
        if prev_ds is not None and row["dataset"] != prev_ds:
            lines.append(r"\midrule")
        ds = row["dataset"] if row["dataset"] != prev_ds else ""
        prev_ds = row["dataset"]
        cfg  = " & ".join(str(row[c]) for c in config_cols)
        row_str = f"{ds} & {row['variant']}"
        if cfg:
            row_str += f" & {cfg}"
        row_str += (f" & {_fmt_cell(row['train_med'], row['train_iqr'])}"
                    f" & {_fmt_cell(row['test_med'],  row['test_iqr'])}"
                    f" & {_fmt_cell(row['size_med'],  row['size_iqr'], decimals=0)} \\\\")
        lines.append(row_str)

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


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

    tex = _to_latex(
        stats,
        caption=(r"Probabilistic head XO (max\_depth=17): "
                 r"median (IQR) at final generation. "
                 r"p\_xo=0.0 is the standard SLIM-GSGP baseline."),
        label="tab:prob_xo_final",
        config_cols=["p_xo"],
    )
    tex_path = os.path.join(out_dir, "prob_xo_table.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"  TEX  -> {tex_path}")


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
