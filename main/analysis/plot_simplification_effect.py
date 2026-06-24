"""
plot_simplification_effect.py — How SymPy simplification changes evolved individuals.

Three panels:
  A) Simplification outcome per variant: success rate, fraction where ell actually
     decreased, fraction where SymPy expanded (reverted by analysis-side filter).
  B) Scatter ell_before vs ell_after (raw, log scale) for variants that ever succeeded.
     Points above the diagonal = SymPy expanded the expression (reverted by filter).
     Points below = genuine reduction.
  C) Box/strip of Δell (filtered) for the 3 variants where ell was truly reduced.

Data: main/log/results_normalized_simplification.csv

Run from project root:
    python main/analysis/plot_simplification_effect.py
"""

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SIMP_LOG = os.path.join(_ROOT, "main", "log", "results_normalized_simplification.csv")
FIG_DIR  = os.path.join(_ROOT, "main", "analysis", "log", "figs")
os.makedirs(FIG_DIR, exist_ok=True)

VARIANT_ORDER = [
    "SLIM+1SIG", "SLIM+ABS",  "SLIM+NORM1",   "SLIM+NORMFIX",
    "SLIM+2SIG", "SLIM+NORM2","SLIM+NORMROB",  "SLIM+NORM12",
    "SLIM*1SIG", "SLIM*ABS",  "SLIM*NORM1",    "SLIM*NORMFIX",
    "SLIM*2SIG", "SLIM*NORM2","SLIM*NORMROB",  "SLIM*NORM12",
]
_GROUP = {
    "SLIM+1SIG":    "1-tree / sum", "SLIM+ABS":    "1-tree / sum",
    "SLIM+NORM1":   "1-tree / sum", "SLIM+NORMFIX": "1-tree / sum",
    "SLIM+2SIG":    "2-tree / sum", "SLIM+NORM2":  "2-tree / sum",
    "SLIM+NORMROB": "2-tree / sum", "SLIM+NORM12": "2-tree / sum",
    "SLIM*1SIG":    "1-tree / mul", "SLIM*ABS":    "1-tree / mul",
    "SLIM*NORM1":   "1-tree / mul", "SLIM*NORMFIX": "1-tree / mul",
    "SLIM*2SIG":    "2-tree / mul", "SLIM*NORM2":  "2-tree / mul",
    "SLIM*NORMROB": "2-tree / mul", "SLIM*NORM12": "2-tree / mul",
}
_CMAP = {
    "1-tree / sum": "#2196F3",
    "1-tree / mul": "#FF9800",
    "2-tree / sum": "#4CAF50",
    "2-tree / mul": "#9C27B0",
}
SCATTER_COLORS = {
    "SLIM*NORM1": "#FF9800", "SLIM+NORM1": "#2196F3",
    "SLIM*ABS":   "#9C27B0", "SLIM*1SIG":  "#E91E63",
    "SLIM*NORM2": "#795548", "SLIM+NORM2": "#009688",
}


def load():
    df = pd.read_csv(SIMP_LOG, on_bad_lines="skip")
    df["simp_ok"]   = df["simplified_ok"].astype(int)
    # filtered values (analysis-side)
    df["ell_f"]     = df[["ell_after", "ell_before"]].min(axis=1)
    df["mphi_f"]    = df[["m_phi_after", "m_phi_before"]].max(axis=1)
    df["delta_ell"] = df["ell_f"]  - df["ell_before"]    # ≤ 0 after filter
    df["delta_mphi"]= df["mphi_f"] - df["m_phi_before"]  # ≥ 0 after filter
    return df


def make_plot(df, save_path):
    has_ok   = df[df["simp_ok"] == 1]["algo"].unique()
    has_redu = df[df["delta_ell"] < 0]["algo"].unique()

    fig = plt.figure(figsize=(18, 7))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.8, 1.0], wspace=0.35)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # ── Panel A: stacked bars — outcome per variant ───────────────────────────
    stats = []
    n_total = df.groupby("algo")["simp_ok"].count()
    for v in VARIANT_ORDER:
        sub  = df[df["algo"] == v]
        n    = len(sub)
        ok   = sub["simp_ok"].sum()
        redu = (sub["delta_ell"] < 0).sum()
        exp  = ok - redu                     # simplified but expression grew (reverted)
        fail = n - ok
        stats.append({"v": v,
                      "failed":    fail / n * 100,
                      "expanded":  exp  / n * 100,
                      "reduced":   redu / n * 100})
    st = pd.DataFrame(stats).set_index("v").reindex(VARIANT_ORDER)

    y  = np.arange(len(VARIANT_ORDER))
    h  = 0.65
    ax_a.barh(y, st["failed"],   height=h, color="#BDBDBD", label="Not simplified")
    ax_a.barh(y, st["expanded"], height=h, left=st["failed"],
              color="#FF8A65", label="Simplified but larger (reverted)")
    ax_a.barh(y, st["reduced"],  height=h, left=st["failed"] + st["expanded"],
              color="#43A047", label="Simplified & smaller")

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(VARIANT_ORDER, fontsize=8)
    ax_a.set_xlabel("% of runs  (30 seeds × 6 datasets = 180 total)", fontsize=8.5)
    ax_a.set_title("A — Simplification outcome", fontsize=10)
    ax_a.set_xlim(0, 100)
    ax_a.invert_yaxis()
    ax_a.legend(fontsize=7.5, loc="lower right")

    # ── Panel B: ell_before vs ell_after (raw, log scale) ────────────────────
    sub_ok = df[(df["algo"].isin(has_ok)) & (df["simp_ok"] == 1)].copy()

    # raw ell_after (before filter) shows what SymPy actually produced
    lim_min = 1
    lim_max = max(sub_ok["ell_before"].max(), sub_ok["ell_after"].max()) * 1.1
    ref     = np.array([lim_min, lim_max])

    ax_b.plot(ref, ref, "k--", linewidth=0.8, alpha=0.5, label="no change")
    ax_b.fill_between(ref, ref, lim_max, alpha=0.04, color="red")   # expansion zone
    ax_b.fill_between(ref, lim_min, ref, alpha=0.06, color="green") # reduction zone

    for algo, grp in sub_ok.groupby("algo"):
        c = SCATTER_COLORS.get(algo, "#607D8B")
        ax_b.scatter(grp["ell_before"], grp["ell_after"],
                     c=c, s=30, alpha=0.7, label=algo,
                     edgecolors="white", linewidths=0.3)

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("ell before simplification", fontsize=8.5)
    ax_b.set_ylabel("ell after simplification (raw)", fontsize=8.5)
    ax_b.set_title("B — Scatter ell before vs after\n(log scale; above diagonal = SymPy expanded → reverted)",
                   fontsize=8.5)
    ax_b.legend(fontsize=7.5, markerscale=1.2)

    ax_b.text(0.97, 0.03, "▼ reduction zone", transform=ax_b.transAxes,
              ha="right", va="bottom", fontsize=7, color="green", alpha=0.8)
    ax_b.text(0.97, 0.97, "▲ expansion zone", transform=ax_b.transAxes,
              ha="right", va="top",    fontsize=7, color="red",   alpha=0.8)

    # ── Panel C: Δell distribution for the 3 variants with real reduction ─────
    reduced_df = df[(df["algo"].isin(has_redu)) & (df["delta_ell"] < 0)]
    algo_list  = sorted(has_redu,
                        key=lambda a: reduced_df[reduced_df["algo"]==a]["delta_ell"].median())

    positions  = np.arange(len(algo_list))
    for i, algo in enumerate(algo_list):
        vals = reduced_df[reduced_df["algo"] == algo]["delta_ell"].values
        c    = SCATTER_COLORS.get(algo, "#607D8B")
        bp   = ax_c.boxplot(vals, positions=[i], widths=0.5, vert=True,
                            patch_artist=True, notch=False,
                            boxprops=dict(facecolor=c, alpha=0.7),
                            medianprops=dict(color="black", linewidth=1.5),
                            whiskerprops=dict(linewidth=0.8),
                            capprops=dict(linewidth=0.8),
                            flierprops=dict(marker="o", markersize=3, alpha=0.5))
        # annotate n
        ax_c.text(i, vals.min() - 2, f"n={len(vals)}", ha="center", fontsize=7.5)

    ax_c.set_xticks(positions)
    ax_c.set_xticklabels(algo_list, rotation=15, ha="right", fontsize=8)
    ax_c.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax_c.set_ylabel("Δell  (ell_after − ell_before, filtered)", fontsize=8.5)
    ax_c.set_title("C — Node reduction\n(runs where ell actually decreased)", fontsize=9)

    fig.suptitle("SymPy simplification effect on evolved SLIM individuals\n"
                 "30 seeds × 6 datasets × 16 variants  (180 runs each)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"Saved -> {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    df = load()
    make_plot(df, os.path.join(FIG_DIR, "simplification_effect.png"))

    # Also print a compact summary table
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    n = 180  # 30 seeds × 6 datasets
    rows = []
    for v in VARIANT_ORDER:
        sub  = df[df["algo"] == v]
        ok   = int(sub["simp_ok"].sum())
        redu = int((sub["delta_ell"] < 0).sum())
        sub_redu = sub[sub["delta_ell"] < 0]
        rows.append({
            "Variant":        v,
            "simp_ok":        f"{ok}/{n}  ({ok/n*100:.0f}%)",
            "ell_reduced":    f"{redu}/{n}  ({redu/n*100:.0f}%)",
            "med_dell":       round(sub_redu["delta_ell"].median(),  1) if redu else 0,
            "med_dmphi":      round(sub_redu["delta_mphi"].median(), 2) if redu else 0,
            "med_ell_before": round(sub["ell_before"].median(), 0),
        })
    tbl = pd.DataFrame(rows)
    try:
        from tabulate import tabulate
        print(tabulate(tbl, headers="keys", tablefmt="simple", showindex=False))
    except ImportError:
        print(tbl.to_string(index=False))
