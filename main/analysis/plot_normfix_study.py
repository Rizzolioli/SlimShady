"""
plot_normfix_study.py — NORMFIX comparison + simplification study visualisation.

Three figures saved to main/analysis/log/figs/:
  normfix_rmse.png          — last-gen test RMSE, all 16 variants × 6 datasets
  normfix_mphi.png          — last-gen M_phi (analysis-side max(before,after))
  normfix_simplification.png — simp_ok%, ell_before, simp_time from simplification study

Data sources:
  main/log/results_normalized_simplification.csv  (final-gen, all 16 variants)
  main/analysis/log/simplification_study_raw.csv  (inflate-10 study)

Run from project root:
    python main/analysis/plot_normfix_study.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Paths ─────────────────────────────────────────────────────────────────────
SIMP_LOG  = os.path.join(_ROOT, "main", "log", "results_normalized_simplification.csv")
STUDY_RAW = os.path.join(_ROOT, "main", "analysis", "log", "simplification_study_raw.csv")
FIG_DIR   = os.path.join(_ROOT, "main", "analysis", "log", "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Variant ordering and styling ───────────────────────────────────────────────
VARIANT_ORDER = [
    "SLIM+1SIG", "SLIM+ABS",  "SLIM+NORM1",   "SLIM+NORMFIX",
    "SLIM+2SIG", "SLIM+NORM2","SLIM+NORMROB",  "SLIM+NORM12",
    "SLIM*1SIG", "SLIM*ABS",  "SLIM*NORM1",    "SLIM*NORMFIX",
    "SLIM*2SIG", "SLIM*NORM2","SLIM*NORMROB",  "SLIM*NORM12",
]

DATASETS = ["concrete", "energy", "instanbul", "ppb",
            "resid_build_sale_price", "toxicity"]
DS_LABELS = ["Concrete", "Energy", "Istanbul", "PPB",
             "Resid. BSP", "Toxicity"]

# colour per group
_CMAP = {
    "1-tree / sum": "#2196F3",   # blue
    "1-tree / mul": "#FF9800",   # orange
    "2-tree / sum": "#4CAF50",   # green
    "2-tree / mul": "#9C27B0",   # purple
}
_GROUP = {
    "SLIM+1SIG":    "1-tree / sum", "SLIM+ABS":    "1-tree / sum",
    "SLIM+NORM1":   "1-tree / sum", "SLIM+NORMFIX":"1-tree / sum",
    "SLIM+2SIG":    "2-tree / sum", "SLIM+NORM2":  "2-tree / sum",
    "SLIM+NORMROB": "2-tree / sum", "SLIM+NORM12": "2-tree / sum",
    "SLIM*1SIG":    "1-tree / mul", "SLIM*ABS":    "1-tree / mul",
    "SLIM*NORM1":   "1-tree / mul", "SLIM*NORMFIX":"1-tree / mul",
    "SLIM*2SIG":    "2-tree / mul", "SLIM*NORM2":  "2-tree / mul",
    "SLIM*NORMROB": "2-tree / mul", "SLIM*NORM12": "2-tree / mul",
}

def _bar_color(variant):
    base = _CMAP[_GROUP[variant]]
    return base if "NORMFIX" not in variant else "#E91E63"   # highlight NORMFIX pink


# ── Load data ─────────────────────────────────────────────────────────────────

def load_simp_log():
    df = pd.read_csv(SIMP_LOG, on_bad_lines="skip")
    df["m_phi"] = df[["m_phi_before", "m_phi_after"]].max(axis=1)   # analysis-side filter
    return df

def load_study_raw():
    return pd.read_csv(STUDY_RAW)


# ── Figure 1: last-gen test RMSE ──────────────────────────────────────────────

def plot_rmse(df, save_path):
    med = (df.groupby(["algo", "dataset"])["test_rmse"]
             .median()
             .reset_index()
             .pivot(index="algo", columns="dataset", values="test_rmse")
             .reindex(VARIANT_ORDER))

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 7), sharey=False)
    fig.suptitle("Last-generation test RMSE (median over 30 seeds)\nNORMFIX highlighted in pink",
                 fontsize=12)

    for col, (ds, ds_label) in enumerate(zip(DATASETS, DS_LABELS)):
        ax = axes[col]
        vals = med[ds]
        colors = [_bar_color(v) for v in VARIANT_ORDER]
        bars = ax.barh(range(len(VARIANT_ORDER)), vals, color=colors, edgecolor="white",
                       linewidth=0.4)
        ax.set_yticks(range(len(VARIANT_ORDER)))
        ax.set_yticklabels(VARIANT_ORDER if col == 0 else [], fontsize=7.5)
        ax.set_xlabel("Test RMSE", fontsize=8)
        ax.set_title(ds_label, fontsize=9)
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=7)

        # annotate best per group
        best_val = vals.min()
        for i, v in enumerate(VARIANT_ORDER):
            if abs(vals[v] - best_val) < 1e-6:
                ax.text(vals[v] * 1.01, i, f"★", va="center", fontsize=7, color="black")

    legend_patches = [mpatches.Patch(color=c, label=g) for g, c in _CMAP.items()]
    legend_patches.append(mpatches.Patch(color="#E91E63", label="NORMFIX"))
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  saved -> {save_path}")
    plt.close(fig)


# ── Figure 2: last-gen M_phi ──────────────────────────────────────────────────

def plot_mphi(df, save_path):
    med = (df.groupby(["algo", "dataset"])["m_phi"]
             .median()
             .reset_index()
             .pivot(index="algo", columns="dataset", values="m_phi")
             .reindex(VARIANT_ORDER))

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 7), sharey=False)
    fig.suptitle("Last-generation Mφ (analysis-side max, median over 30 seeds)\nHigher = more interpretable   |   NORMFIX in pink",
                 fontsize=12)

    for col, (ds, ds_label) in enumerate(zip(DATASETS, DS_LABELS)):
        ax = axes[col]
        vals = med[ds]
        colors = [_bar_color(v) for v in VARIANT_ORDER]
        # diverging: negative values use a lighter shade
        ax.barh(range(len(VARIANT_ORDER)), vals, color=colors, edgecolor="white",
                linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_yticks(range(len(VARIANT_ORDER)))
        ax.set_yticklabels(VARIANT_ORDER if col == 0 else [], fontsize=7.5)
        ax.set_xlabel("Mφ", fontsize=8)
        ax.set_title(ds_label, fontsize=9)
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=7)

        # annotate best
        best_val = vals.max()
        for i, v in enumerate(VARIANT_ORDER):
            if abs(vals[v] - best_val) < 1e-3:
                ax.text(max(vals[v], 0) + abs(vals.max()) * 0.01, i,
                        "★", va="center", fontsize=7, color="black")

    legend_patches = [mpatches.Patch(color=c, label=g) for g, c in _CMAP.items()]
    legend_patches.append(mpatches.Patch(color="#E91E63", label="NORMFIX"))
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  saved -> {save_path}")
    plt.close(fig)


# ── Figure 3: simplification study ───────────────────────────────────────────

def plot_simplification_study(raw, save_path):
    # Compute simp_ok% and median ell_before per variant (in VARIANT_ORDER)
    stats = []
    for v in VARIANT_ORDER:
        sub = raw[raw["variant"] == v]
        stats.append({
            "variant":     v,
            "simp_ok_pct": sub["simplified_ok"].mean() * 100,
            "ell_med":     sub["ell_before"].median(),
            "ell_q1":      sub["ell_before"].quantile(0.25),
            "ell_q3":      sub["ell_before"].quantile(0.75),
            "time_med":    sub[sub["simplified_ok"]]["simp_time_s"].median()
                           if sub["simplified_ok"].any() else 0.0,
        })
    st = pd.DataFrame(stats)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Simplification study: 10 inflate steps per individual (50 individuals/variant)",
                 fontsize=11)

    colors = [_bar_color(v) for v in VARIANT_ORDER]
    y = range(len(VARIANT_ORDER))

    # Panel A — simp_ok%
    ax = axes[0]
    ax.barh(y, st["simp_ok_pct"], color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(VARIANT_ORDER, fontsize=8)
    ax.set_xlabel("Simplification success rate (%)", fontsize=9)
    ax.set_title("A — SymPy success rate", fontsize=9)
    ax.axvline(50, color="grey", linewidth=0.7, linestyle=":")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    for i, row in st.iterrows():
        ax.text(row["simp_ok_pct"] + 1.5, i, f"{row['simp_ok_pct']:.0f}%",
                va="center", fontsize=7)

    # Panel B — ell_before (node count before simplification)
    ax = axes[1]
    xerr_lo = st["ell_med"] - st["ell_q1"]
    xerr_hi = st["ell_q3"] - st["ell_med"]
    ax.barh(y, st["ell_med"], xerr=[xerr_lo, xerr_hi],
            color=colors, edgecolor="white", linewidth=0.4,
            error_kw=dict(ecolor="black", capsize=2, linewidth=0.8))
    ax.set_yticks(y)
    ax.set_yticklabels([], fontsize=8)
    ax.set_xlabel("Node count before simplification (median ± IQR)", fontsize=9)
    ax.set_title("B — Expression size (ell)", fontsize=9)
    ax.invert_yaxis()

    # Panel C — median simp time for successful cases
    ax = axes[2]
    ax.barh(y, st["time_med"], color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([], fontsize=8)
    ax.set_xlabel("Median simplification time (s, successful only)", fontsize=9)
    ax.set_title("C — Simplification time", fontsize=9)
    ax.invert_yaxis()

    legend_patches = [mpatches.Patch(color=c, label=g) for g, c in _CMAP.items()]
    legend_patches.append(mpatches.Patch(color="#E91E63", label="NORMFIX"))
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  saved -> {save_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data …")
    df   = load_simp_log()
    raw  = load_study_raw()

    print("Plotting …")
    plot_rmse(df,  os.path.join(FIG_DIR, "normfix_rmse.png"))
    plot_mphi(df,  os.path.join(FIG_DIR, "normfix_mphi.png"))
    plot_simplification_study(raw, os.path.join(FIG_DIR, "normfix_simplification.png"))
    print("Done.")
