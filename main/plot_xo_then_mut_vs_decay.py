"""
plot_xo_then_mut_vs_decay.py

Compare XO-then-Mutation vs Cosine-Decay-XO experiments.
Both run 3 SLIM variants (SLIM*1SIG, SLIM*ABS, SLIM+2SIG) × 6 datasets × 5 seeds × 400 gens.

Outputs (main/log/figs/):
  xo_vs_decay_convergence.png   — test-RMSE convergence curves per dataset
  xo_vs_decay_finalgen.png      — final-gen test-RMSE boxplots per dataset
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────────
_LOG  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
_FIGS = os.path.join(_LOG, "figs")
os.makedirs(_FIGS, exist_ok=True)

_COLS = ["algo","run_id","dataset","seed","generation",
         "train_fitness","timing","nodes","test_fitness",
         "nodes_count","m_phi","no","nnao","nnaoc","mae","r2","log_level"]

# ── load ──────────────────────────────────────────────────────────────────────
df_xo  = pd.read_csv(os.path.join(_LOG, "results_xo_then_mut_01072026.csv"),
                     header=None, names=_COLS)
df_dec = pd.read_csv(os.path.join(_LOG, "results_decay_xo_22062026.csv"),
                     header=None, names=_COLS)

df_xo ["condition"] = "XO→Mut"
df_dec["condition"] = "Cosine Decay XO"

# strip the config suffix from algo name → base variant
def _base(algo):
    return algo.split("_")[0]

for df in (df_xo, df_dec):
    df["variant"] = df["algo"].map(_base)

combined = pd.concat([df_xo, df_dec], ignore_index=True)

# ── style ──────────────────────────────────────────────────────────────────────
VARIANTS = ["SLIM*1SIG", "SLIM*ABS", "SLIM+2SIG"]
VARIANT_COLORS = {
    "SLIM*1SIG": "#1f77b4",
    "SLIM*ABS":  "#ff7f0e",
    "SLIM+2SIG": "#2ca02c",
}
COND_STYLE = {
    "XO→Mut":         dict(linestyle="-",  linewidth=1.6, alpha=0.9),
    "Cosine Decay XO": dict(linestyle="--", linewidth=1.6, alpha=0.9),
}
DATASETS = ["concrete", "energy", "instanbul", "ppb", "resid_build_sale_price", "toxicity"]
DS_LABELS = {
    "concrete":               "Concrete",
    "energy":                 "Energy",
    "instanbul":              "Istanbul",
    "ppb":                    "PPB",
    "resid_build_sale_price": "Resid. Build.",
    "toxicity":               "Toxicity",
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONVERGENCE CURVES
# ══════════════════════════════════════════════════════════════════════════════
# Median test RMSE per (variant, condition, dataset, generation)
med_conv = (combined
            .groupby(["variant","condition","dataset","generation"])["test_fitness"]
            .median()
            .reset_index())

fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
axes = axes.flatten()

for i, ds in enumerate(DATASETS):
    ax = axes[i]
    sub = med_conv[med_conv["dataset"] == ds]
    for variant in VARIANTS:
        color = VARIANT_COLORS[variant]
        for cond, cs in COND_STYLE.items():
            row = sub[(sub["variant"] == variant) & (sub["condition"] == cond)]
            if row.empty:
                continue
            ax.plot(row["generation"], row["test_fitness"],
                    color=color, label=f"{variant} | {cond}", **cs)
    ax.set_title(DS_LABELS[ds], fontsize=10, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=8)
    ax.set_ylabel("Test RMSE (median)", fontsize=8)
    ax.tick_params(labelsize=7)

# legend: variant color × condition linestyle
legend_handles = []
for variant in VARIANTS:
    for cond, cs in COND_STYLE.items():
        legend_handles.append(
            plt.Line2D([0],[0], color=VARIANT_COLORS[variant],
                       linestyle=cs["linestyle"], linewidth=1.8,
                       label=f"{variant}  [{cond}]")
        )
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Test RMSE Convergence — XO→Mut  vs  Cosine Decay XO\n"
             "Solid = XO→Mut  |  Dashed = Cosine Decay XO  |  Median over 5 seeds",
             fontsize=11)
out1 = os.path.join(_FIGS, "xo_vs_decay_convergence.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. FINAL-GENERATION BOXPLOTS
# ══════════════════════════════════════════════════════════════════════════════
final = combined[combined["generation"] == combined["generation"].max()].copy()

CONDITIONS = ["XO→Mut", "Cosine Decay XO"]
COND_COLORS = {"XO→Mut": "#4472C4", "Cosine Decay XO": "#ED7D31"}

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
axes2 = axes2.flatten()

n_variants = len(VARIANTS)
n_conds    = len(CONDITIONS)
group_w    = 0.8          # total width per variant group
box_w      = group_w / n_conds * 0.85

for i, ds in enumerate(DATASETS):
    ax = axes2[i]
    sub = final[final["dataset"] == ds]

    for vi, variant in enumerate(VARIANTS):
        x_center = vi
        for ci, cond in enumerate(CONDITIONS):
            vals = sub[(sub["variant"] == variant) & (sub["condition"] == cond)]["test_fitness"].dropna()
            if vals.empty:
                continue
            x_pos = x_center + (ci - (n_conds - 1) / 2) * (group_w / n_conds)
            bp = ax.boxplot(vals, positions=[x_pos], widths=box_w,
                            patch_artist=True, notch=False,
                            medianprops=dict(color="k", linewidth=1.5),
                            whiskerprops=dict(linewidth=1),
                            capprops=dict(linewidth=1),
                            flierprops=dict(marker=".", markersize=4, alpha=0.5))
            bp["boxes"][0].set_facecolor(COND_COLORS[cond])
            bp["boxes"][0].set_alpha(0.75)

    ax.set_xticks(range(n_variants))
    ax.set_xticklabels(VARIANTS, fontsize=7, rotation=15)
    ax.set_title(DS_LABELS[ds], fontsize=10, fontweight="bold")
    ax.set_ylabel("Test RMSE", fontsize=8)
    ax.tick_params(labelsize=7)

# legend
from matplotlib.patches import Patch
legend2 = [Patch(facecolor=COND_COLORS[c], alpha=0.75, label=c) for c in CONDITIONS]
fig2.legend(handles=legend2, loc="lower center", ncol=2,
            fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.04))
fig2.suptitle("Final-Generation Test RMSE — XO→Mut  vs  Cosine Decay XO\n"
              "5 seeds per variant × dataset",
              fontsize=11)
out2 = os.path.join(_FIGS, "xo_vs_decay_finalgen.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")
