"""
generate_depth_cap_plots.py
===========================
Convergence curves and final-gen summary for the depth-cap experiment
(results_depth_cap_2.csv).  Compares max_head_depth in {5, 17, 25} with
p_xo=0.7, 3 variants, 6 datasets.

Outputs -> main/log/latex/depth_cap/
  depth_cap_{variant}_convergence.png/.tex   (one figure per variant,
                                              6-panel grid across datasets)
  depth_cap_summary_table.csv / .tex
"""

import os
import sys
import re
from uuid import UUID
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    import matplot2tikz as tikzplotlib
    _TIKZ = True
except ImportError:
    _TIKZ = False

# ── PATHS ─────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG  = os.path.join(_HERE, "..", "log")
_OUT  = os.path.join(_LOG, "latex", "depth_cap")
os.makedirs(_OUT, exist_ok=True)

LOG_CSV = os.path.join(_LOG, "results_depth_cap_new.csv")

# ── CONFIG ─────────────────────────────────────────────────────────────────────

_COLS = {0:"algo",1:"run_id",2:"dataset",3:"seed",4:"gen",
         5:"train",6:"timing",7:"nodes",8:"test",9:"nodes_count",10:"log"}

DATASETS  = ["toxicity", "concrete", "instanbul", "ppb",
             "resid_build_sale_price", "energy"]
VARIANTS  = ["SLIM+2SIG", "SLIM*ABS", "SLIM*1SIG"]
HD_KEEP   = [5, 10, 17]
PXO_KEEP  = [0.3, 0.7]

# Colors per hd value; line style distinguishes p_xo
HD_COLORS  = {5: "#e41a1c", 10: "#ff7f00", 17: "#377eb8"}
HD_LABELS  = {5: "hd=5",    10: "hd=10",   17: "hd=17"}
PXO_STYLES = {0.3: "--", 0.7: "-"}
PXO_LABELS = {0.3: "p_xo=0.3", 0.7: "p_xo=0.7"}

# ── DATA LOAD ─────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(LOG_CSV, header=None).rename(columns=_COLS)
    df["seed"] = df["seed"].astype(int)
    df["variant"] = df["algo"].str.extract(r"^(SLIM[+*]\w+)_pxo")
    df["p_xo"]    = pd.to_numeric(df["algo"].str.extract(r"_pxo([0-9.]+)_")[0],
                                   errors="coerce")
    df["hd"]      = pd.to_numeric(df["algo"].str.extract(r"_hd(\d+)$")[0],
                                   errors="coerce")

    df = df[df["hd"].isin(HD_KEEP) & df["p_xo"].isin(PXO_KEEP)].copy()
    df["hd"]  = df["hd"].astype(int)

    # For groups with multiple batches keep only the latest run (UUID1 timestamp).
    df["_uuid_time"] = df["run_id"].map(lambda r: UUID(r).time)
    latest = (df.groupby(["algo", "dataset", "seed"])["_uuid_time"]
                .max().reset_index().rename(columns={"_uuid_time": "_latest"}))
    df = df.merge(latest, on=["algo", "dataset", "seed"])
    df = df[df["_uuid_time"] == df["_latest"]].drop(columns=["_uuid_time", "_latest"])

    return df


# ── CONVERGENCE CURVES ────────────────────────────────────────────────────────

def _median_iqr(sub, metric):
    """Return (gen, median, q25, q75) aggregated over seeds."""
    agg = (sub.groupby("gen")[metric]
             .agg(["median",
                   lambda x: x.quantile(0.25),
                   lambda x: x.quantile(0.75)])
             .reset_index())
    agg.columns = ["gen", "med", "q25", "q75"]
    return agg


def plot_convergence(df, variant, metric="test", ax=None, dataset=None):
    """
    One curve per (hd, p_xo) combo: colour = hd, linestyle = p_xo.
    Legend drawn on the first panel only.
    """
    sub = df[(df["variant"] == variant) & (df["dataset"] == dataset)]
    for hd in HD_KEEP:
        for p_xo in PXO_KEEP:
            s = sub[(sub["hd"] == hd) & (sub["p_xo"] == p_xo)]
            if s.empty:
                continue
            agg = _median_iqr(s, metric)
            c  = HD_COLORS[hd]
            ls = PXO_STYLES[p_xo]
            ax.plot(agg["gen"], agg["med"], color=c, linestyle=ls,
                    linewidth=1.4,
                    label=f"{HD_LABELS[hd]}, {PXO_LABELS[p_xo]}")
            ax.fill_between(agg["gen"], agg["q25"], agg["q75"],
                            color=c, alpha=0.10)
    ax.set_title(dataset, fontsize=8)
    ax.tick_params(labelsize=7)


def make_convergence_figures(df):
    for variant in VARIANTS:
        safe_v = re.sub(r'[^A-Za-z0-9_\-]', '_', variant)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
        fig.suptitle(f"{variant} — test RMSE convergence  (colour=hd, style=p_xo)",
                     fontsize=12, fontweight="bold", y=1.01)

        for idx, dataset in enumerate(DATASETS):
            ax = axes[idx // 3][idx % 3]
            plot_convergence(df, variant, metric="test", ax=ax, dataset=dataset)
            if idx == 0:
                ax.legend(fontsize=6, framealpha=0.7, ncol=2)

        for row in axes:
            row[0].set_ylabel("Test RMSE (median)", fontsize=8)
        for ax in axes[1]:
            ax.set_xlabel("Generation", fontsize=8)

        fig.tight_layout()
        stem = os.path.join(_OUT, f"depth_cap_{safe_v}_convergence")
        fig.savefig(stem + ".png", dpi=150, bbox_inches="tight")
        if _TIKZ:
            try:
                tikzplotlib.save(stem + ".tex", figure=fig, strict=False)
            except Exception as e:
                print(f"  tikz warning ({variant}): {e}")
        plt.close(fig)
        print(f"  Saved: depth_cap_{safe_v}_convergence.png")


# ── SIZE-OVER-TIME FIGURE ─────────────────────────────────────────────────────

def make_size_figures(df):
    """Median model size (nodes_count) over generations per variant."""
    for variant in VARIANTS:
        safe_v = re.sub(r'[^A-Za-z0-9_\-]', '_', variant)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
        fig.suptitle(f"{variant} — model size (nodes)  (colour=hd, style=p_xo)",
                     fontsize=12, fontweight="bold", y=1.01)

        for idx, dataset in enumerate(DATASETS):
            ax = axes[idx // 3][idx % 3]
            sub = df[(df["variant"] == variant) & (df["dataset"] == dataset)]
            for hd in HD_KEEP:
                for p_xo in PXO_KEEP:
                    s = sub[(sub["hd"] == hd) & (sub["p_xo"] == p_xo)]
                    if s.empty:
                        continue
                    agg = _median_iqr(s, "nodes_count")
                    c  = HD_COLORS[hd]
                    ls = PXO_STYLES[p_xo]
                    ax.plot(agg["gen"], agg["med"], color=c, linestyle=ls,
                            linewidth=1.4,
                            label=f"{HD_LABELS[hd]}, {PXO_LABELS[p_xo]}")
                    ax.fill_between(agg["gen"], agg["q25"], agg["q75"],
                                    color=c, alpha=0.10)
            ax.set_title(dataset, fontsize=8)
            ax.tick_params(labelsize=7)
            if idx == 0:
                ax.legend(fontsize=6, framealpha=0.7, ncol=2)

        for row in axes:
            row[0].set_ylabel("Model size (nodes, median)", fontsize=8)
        for ax in axes[1]:
            ax.set_xlabel("Generation", fontsize=8)

        fig.tight_layout()
        stem = os.path.join(_OUT, f"depth_cap_{safe_v}_size")
        fig.savefig(stem + ".png", dpi=150, bbox_inches="tight")
        if _TIKZ:
            try:
                tikzplotlib.save(stem + ".tex", figure=fig, strict=False)
            except Exception as e:
                print(f"  tikz warning ({variant}): {e}")
        plt.close(fig)
        print(f"  Saved: depth_cap_{safe_v}_size.png")


# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────

def make_summary_table(df):
    def iqr(x):
        return x.quantile(0.75) - x.quantile(0.25)

    last = (df.groupby(["algo","dataset","seed"])["gen"]
              .max().reset_index().rename(columns={"gen":"last_gen"}))
    df_last = df.merge(last, on=["algo","dataset","seed"])
    df_last  = df_last[df_last["gen"] == df_last["last_gen"]]

    agg = (df_last.groupby(["variant","p_xo","hd","dataset"])
                  .agg(
                      train_med  =("train",       "median"),
                      train_iqr  =("train",       iqr),
                      test_med   =("test",        "median"),
                      test_iqr   =("test",        iqr),
                      size_med   =("nodes_count", "median"),
                      size_iqr   =("nodes_count", iqr),
                      n_seeds    =("seed",        "nunique"),
                  ).reset_index())

    def fmt(m, q, d=3): return f"{m:.{d}f} ({q:.{d}f})"

    agg["Train RMSE"]   = [fmt(r.train_med, r.train_iqr) for _, r in agg.iterrows()]
    agg["Test RMSE"]    = [fmt(r.test_med,  r.test_iqr)  for _, r in agg.iterrows()]
    agg["Model Size"]   = [fmt(r.size_med,  r.size_iqr, d=0) for _, r in agg.iterrows()]

    csv_path = os.path.join(_OUT, "depth_cap_summary_table.csv")
    agg.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved: depth_cap_summary_table.csv")

    # LaTeX
    col_spec = "llllrrrrrr"
    lines = [
        r"\begin{longtable}{" + col_spec + "}",
        r"\caption{Depth-cap sweep: median (IQR) at final generation}"
        r" \label{tab:depth_cap} \\",
        r"\toprule",
        r"Variant & Dataset & p\_xo & max\_hd & "
        r"\multicolumn{2}{c}{Train RMSE} & "
        r"\multicolumn{2}{c}{Test RMSE} & "
        r"\multicolumn{2}{c}{Model Size} \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r" & & & & Med & IQR & Med & IQR & Med & IQR \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Variant & Dataset & p\_xo & max\_hd & "
        r"\multicolumn{2}{c}{Train RMSE} & "
        r"\multicolumn{2}{c}{Test RMSE} & "
        r"\multicolumn{2}{c}{Model Size} \\",
        r" & & & & Med & IQR & Med & IQR & Med & IQR \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    prev_key = None
    for _, r in agg.sort_values(["variant","dataset","p_xo","hd"]).iterrows():
        key = (r["variant"], r["dataset"])
        if prev_key is not None and key != prev_key:
            lines.append(r"\midrule")
        prev_key = key
        lines.append(
            f"{r['variant']} & {r['dataset']} & {r['p_xo']} & {r['hd']} & "
            f"{r['train_med']:.3f} & {r['train_iqr']:.3f} & "
            f"{r['test_med']:.3f} & {r['test_iqr']:.3f} & "
            f"{r['size_med']:.0f} & {r['size_iqr']:.0f} \\\\"
        )
    lines.append(r"\end{longtable}")

    tex_path = os.path.join(_OUT, "depth_cap_summary_table.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved: depth_cap_summary_table.tex")

    return agg


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {LOG_CSV} ...")
    df = load_data()
    print(f"  {len(df)} rows, {df['algo'].nunique()} algos, "
          f"{df['dataset'].nunique()} datasets, {df['seed'].nunique()} seeds")

    print("\n-- Convergence figures (test RMSE)")
    make_convergence_figures(df)

    print("\n-- Size figures (nodes_count)")
    make_size_figures(df)

    print("\n-- Summary table")
    summary = make_summary_table(df)

    print(f"\nDone. Output -> {_OUT}")

    # Quick console summary: mean test RMSE at last gen per hd, pooled across datasets
    last = (df.groupby(["algo","dataset","seed"])["gen"].max().reset_index()
              .rename(columns={"gen":"last_gen"}))
    df_last = df.merge(last, on=["algo","dataset","seed"])
    df_last  = df_last[df_last["gen"] == df_last["last_gen"]]
    print("\nPooled median test RMSE at gen 2000 (across all datasets & variants):")
    print(df_last.groupby(["p_xo","hd"])["test"].median().unstack("hd").to_string())
    print("\nPooled median model size at gen 2000:")
    print(df_last.groupby(["p_xo","hd"])["nodes_count"].median().unstack("hd").to_string())
