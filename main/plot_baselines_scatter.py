"""
plot_baselines_scatter.py

2-D scatter: -M_phi (model complexity) vs normalised test RMSE.
Only median values (over 30 seeds) are plotted.

X axis : -M_phi  →  higher = more complex / less interpretable
Y axis : test RMSE / per-dataset GPLearn median RMSE
         (< 1  =  better than GPLearn)

Per dataset, three SLIM variants are selected:
  - best RMSE : lowest median RMSE
  - best M_phi: highest median M_phi (most interpretable)
  - middle     : best average rank on (RMSE rank + M_phi rank),
                 excluding the two already selected above

Run from project root:
    python main/plot_baselines_scatter.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG  = os.path.join(_ROOT, "main", "log")
_FIGS = os.path.join(_ROOT, "main", "log", "figs")
os.makedirs(_FIGS, exist_ok=True)

BASELINE_CSV = os.path.join(_LOG, "results_baselines.csv")
SLIM_CSV     = os.path.join(_LOG, "results_normalized_simplification.csv")
OUT_PANEL    = os.path.join(_FIGS, "baselines_scatter_panel.png")
OUT_AGG      = os.path.join(_FIGS, "baselines_scatter_agg.png")
OUT_PANEL_R2 = os.path.join(_FIGS, "baselines_scatter_panel_r2.png")
OUT_AGG_R2   = os.path.join(_FIGS, "baselines_scatter_agg_r2.png")

ALL_SLIM = ["SLIM+2SIG","SLIM*2SIG","SLIM+1SIG","SLIM*1SIG",
            "SLIM+ABS","SLIM*ABS","SLIM+NORM1","SLIM*NORM1","SLIM+NORM2","SLIM*NORM2"]

ROLE_STYLES = {
    "best_rmse": dict(label="Best RMSE",       color="#1f77b4", marker="o"),
    "best_mphi": dict(label="Best M$_\\phi$",  color="#2ca02c", marker="s"),
    "middle":    dict(label="Middle",           color="#ff7f0e", marker="^"),
}

# baselines (scatter points)
BASELINE_STYLES = {
    "GPLearn":  dict(label="GPLearn",  color="#d62728", marker="D"),
    "Operon":   dict(label="Operon",   color="#9467bd", marker="P"),
    "GP-GOMEA": dict(label="GP-GOMEA", color="#17becf", marker="X"),
    "PySR":     dict(label="PySR",     color="#8c564b", marker="v"),
}
BASELINE_KEYS = list(BASELINE_STYLES)

DATASETS = ["concrete", "energy", "instanbul", "ppb", "resid_build_sale_price", "toxicity"]
DS_LABELS = {
    "concrete":               "Concrete",
    "energy":                 "Energy",
    "instanbul":              "Istanbul",
    "ppb":                    "PPB",
    "resid_build_sale_price": "Resid. Build.",
    "toxicity":               "Toxicity",
}

# ── load SLIM ──────────────────────────────────────────────────────────────────
slim_raw = pd.read_csv(SLIM_CSV)
slim_raw = slim_raw[slim_raw["algo"].isin(ALL_SLIM)].copy()
slim_raw["ell_after"]   = slim_raw[["ell_after",   "ell_before"  ]].min(axis=1)
slim_raw["m_phi_after"] = slim_raw[["m_phi_after", "m_phi_before"]].max(axis=1)

# ── load baselines ─────────────────────────────────────────────────────────────
bl_raw = pd.read_csv(BASELINE_CSV)
bl_raw = bl_raw.dropna(subset=["test_rmse"])
# GP-GOMEA may have duplicate rows from earlier runs without M_phi; keep only
# the rows where m_phi_before was successfully computed.
gp_mask = bl_raw["algo"] == "GP-GOMEA"
bl_raw  = pd.concat([
    bl_raw[~gp_mask],
    bl_raw[gp_mask & bl_raw["m_phi_before"].notna()],
], ignore_index=True)
bl_raw["m_phi_after"] = bl_raw[["m_phi_after", "m_phi_before"]].max(axis=1)
bl_raw = bl_raw[bl_raw["algo"].isin(BASELINE_KEYS)].copy()

# ── combine scatter-eligible data & normalise ──────────────────────────────────
cols = ["algo", "dataset", "seed", "test_rmse", "m_phi_after"]
combined = pd.concat([slim_raw[cols], bl_raw[cols]], ignore_index=True)
combined = combined[combined["dataset"].isin(DATASETS)]

ref = (combined[combined["algo"] == "GPLearn"]
       .groupby("dataset")["test_rmse"].median())
combined["rmse_norm"] = combined["test_rmse"] / combined["dataset"].map(ref)

# ── median per (algo, dataset) ─────────────────────────────────────────────────
med = (combined
       .groupby(["algo", "dataset"])
       .agg(m_phi=("m_phi_after", "median"),
            rmse_norm=("rmse_norm", "median"))
       .reset_index())
med["neg_mphi"] = -med["m_phi"]

# ── pick three SLIM roles per dataset ─────────────────────────────────────────
def pick_roles(ds):
    sub = med[(med["algo"].isin(ALL_SLIM)) & (med["dataset"] == ds)].copy()
    sub["rmse_rank"] = sub["rmse_norm"].rank()
    sub["mphi_rank"] = sub["m_phi"].rank(ascending=False)
    sub["avg_rank"]  = (sub["rmse_rank"] + sub["mphi_rank"]) / 2

    best_rmse = sub.loc[sub["rmse_rank"].idxmin()]
    best_mphi = sub.loc[sub["mphi_rank"].idxmin()]
    rest      = sub[~sub["algo"].isin({best_rmse["algo"], best_mphi["algo"]})]
    middle    = rest.loc[rest["avg_rank"].idxmin()]
    return {"best_rmse": best_rmse, "best_mphi": best_mphi, "middle": middle}

roles_by_ds = {ds: pick_roles(ds) for ds in DATASETS}

# ── PANEL PLOT ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
axes = axes.flatten()

for i, ds in enumerate(DATASETS):
    ax    = axes[i]
    roles = roles_by_ds[ds]

    # three SLIM points
    for role, row in roles.items():
        st = ROLE_STYLES[role]
        ax.scatter(row["neg_mphi"], row["rmse_norm"],
                   color=st["color"], marker=st["marker"],
                   s=110, linewidths=0.8, edgecolors="k", zorder=4)
        ax.annotate(row["algo"],
                    xy=(row["neg_mphi"], row["rmse_norm"]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=6.5, color=st["color"])

    # baseline scatter points
    for _, row in med[(med["algo"].isin(BASELINE_KEYS)) &
                      (med["dataset"] == ds)].iterrows():
        st = BASELINE_STYLES[row["algo"]]
        ax.scatter(row["neg_mphi"], row["rmse_norm"],
                   color=st["color"], marker=st["marker"],
                   s=90, linewidths=0.7, edgecolors="k", zorder=3)
        ax.annotate(st["label"],
                    xy=(row["neg_mphi"], row["rmse_norm"]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=6.5, color=st["color"])

    r = roles
    subtitle = (f"RMSE: {r['best_rmse']['algo']}  |  "
                f"Mφ: {r['best_mphi']['algo']}  |  "
                f"mid: {r['middle']['algo']}")
    ax.set_title(f"{DS_LABELS[ds]}\n{subtitle}", fontsize=8, fontweight="bold")
    ax.set_xlabel("−M$_\\phi$  (higher = more complex)", fontsize=8)
    ax.set_ylabel("RMSE / GPLearn median", fontsize=8)
    ax.tick_params(labelsize=7)

# shared legend
slim_handles = [
    plt.Line2D([0],[0], marker=ROLE_STYLES[r]["marker"],
               color=ROLE_STYLES[r]["color"], linestyle="None",
               markersize=8, markeredgecolor="k", markeredgewidth=0.5,
               label=f"SLIM – {ROLE_STYLES[r]['label']}")
    for r in ROLE_STYLES
]
bl_handles = [
    plt.Line2D([0],[0], marker=BASELINE_STYLES[a]["marker"],
               color=BASELINE_STYLES[a]["color"], linestyle="None",
               markersize=8, markeredgecolor="k", markeredgewidth=0.5,
               label=a)
    for a in BASELINE_KEYS
]
fig.legend(handles=slim_handles + bl_handles,
           loc="lower center", ncol=7,
           fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.04))

fig.suptitle("−M$_\\phi$ vs Normalised Test RMSE — medians over 30 seeds\n"
             "RMSE normalised by per-dataset GPLearn median  (< 1 = better than GPLearn)",
             fontsize=10)
plt.savefig(OUT_PANEL, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_PANEL}")

# ── AGGREGATE PLOT ─────────────────────────────────────────────────────────────
agg_rows = []
for ds, roles in roles_by_ds.items():
    for role, row in roles.items():
        agg_rows.append({"role": role, "neg_mphi": row["neg_mphi"],
                         "rmse_norm": row["rmse_norm"]})
agg_slim = (pd.DataFrame(agg_rows)
              .groupby("role")
              .agg(neg_mphi=("neg_mphi","median"),
                   rmse_norm=("rmse_norm","median"))
              .reset_index())

bl_agg = (med[med["algo"].isin(BASELINE_KEYS)]
          .groupby("algo")
          .agg(neg_mphi=("neg_mphi","median"),
               rmse_norm=("rmse_norm","median"))
          .reset_index())

fig2, ax2 = plt.subplots(figsize=(7, 5), constrained_layout=True)

for _, row in agg_slim.iterrows():
    st = ROLE_STYLES[row["role"]]
    ax2.scatter(row["neg_mphi"], row["rmse_norm"],
                color=st["color"], marker=st["marker"],
                s=130, linewidths=0.8, edgecolors="k", zorder=4)
    ax2.annotate(f"SLIM\n({st['label']})",
                 xy=(row["neg_mphi"], row["rmse_norm"]),
                 xytext=(6, 4), textcoords="offset points",
                 fontsize=8, color=st["color"])

for _, row in bl_agg.iterrows():
    st = BASELINE_STYLES[row["algo"]]
    ax2.scatter(row["neg_mphi"], row["rmse_norm"],
                color=st["color"], marker=st["marker"],
                s=120, linewidths=0.8, edgecolors="k", zorder=3)
    ax2.annotate(st["label"],
                 xy=(row["neg_mphi"], row["rmse_norm"]),
                 xytext=(6, 4), textcoords="offset points",
                 fontsize=8, color=st["color"])

ax2.set_xlabel("−M$_\\phi$  (higher = more complex / less interpretable)", fontsize=10)
ax2.set_ylabel("RMSE / GPLearn median per dataset\n(< 1  =  better than GPLearn)", fontsize=9)
ax2.set_title("Interpretability vs Performance — medians across 6 datasets × 30 seeds",
              fontsize=10)
ax2.tick_params(labelsize=8)

handles2 = [
    plt.Line2D([0],[0], marker=ROLE_STYLES[r]["marker"],
               color=ROLE_STYLES[r]["color"], linestyle="None",
               markersize=9, markeredgecolor="k", markeredgewidth=0.5,
               label=f"SLIM – {ROLE_STYLES[r]['label']}")
    for r in ROLE_STYLES
] + [
    plt.Line2D([0],[0], marker=BASELINE_STYLES[a]["marker"],
               color=BASELINE_STYLES[a]["color"], linestyle="None",
               markersize=9, markeredgecolor="k", markeredgewidth=0.5,
               label=a)
    for a in BASELINE_KEYS
]
ax2.legend(handles=handles2, fontsize=8.5, loc="best")

plt.savefig(OUT_AGG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_AGG}")

# ══════════════════════════════════════════════════════════════════════════════
# R² SCATTER PLOTS  (no normalisation needed — higher is better)
# ══════════════════════════════════════════════════════════════════════════════

# ── build R² medians ──────────────────────────────────────────────────────────
r2_cols = ["algo", "dataset", "seed", "test_r2", "m_phi_after"]

slim_r2 = slim_raw[["algo", "dataset", "seed", "test_r2", "m_phi_after"]].copy()
bl_r2   = bl_raw  [["algo", "dataset", "seed", "test_r2", "m_phi_after"]].copy()

combined_r2 = pd.concat([slim_r2, bl_r2], ignore_index=True)
combined_r2 = combined_r2[combined_r2["dataset"].isin(DATASETS)]

med_r2 = (combined_r2
          .groupby(["algo", "dataset"])
          .agg(m_phi=("m_phi_after", "median"),
               r2=("test_r2", "median"))
          .reset_index())
med_r2["neg_mphi"] = -med_r2["m_phi"]

# ── pick roles per dataset using R² ───────────────────────────────────────────
ROLE_STYLES_R2 = {
    "best_r2":   dict(label="Best R²",        color="#1f77b4", marker="o"),
    "best_mphi": dict(label="Best M$_\\phi$", color="#2ca02c", marker="s"),
    "middle":    dict(label="Middle",          color="#ff7f0e", marker="^"),
}

def pick_roles_r2(ds):
    sub = med_r2[(med_r2["algo"].isin(ALL_SLIM)) & (med_r2["dataset"] == ds)].copy()
    sub["r2_rank"]   = sub["r2"].rank(ascending=False)   # rank 1 = highest R²
    sub["mphi_rank"] = sub["m_phi"].rank(ascending=False)
    sub["avg_rank"]  = (sub["r2_rank"] + sub["mphi_rank"]) / 2

    best_r2   = sub.loc[sub["r2_rank"].idxmin()]
    best_mphi = sub.loc[sub["mphi_rank"].idxmin()]
    rest      = sub[~sub["algo"].isin({best_r2["algo"], best_mphi["algo"]})]
    middle    = rest.loc[rest["avg_rank"].idxmin()]
    return {"best_r2": best_r2, "best_mphi": best_mphi, "middle": middle}

roles_r2_by_ds = {ds: pick_roles_r2(ds) for ds in DATASETS}

# ── PANEL PLOT (R²) ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
axes = axes.flatten()

for i, ds in enumerate(DATASETS):
    ax    = axes[i]
    roles = roles_r2_by_ds[ds]

    for role, row in roles.items():
        st = ROLE_STYLES_R2[role]
        ax.scatter(row["m_phi"], row["r2"],
                   color=st["color"], marker=st["marker"],
                   s=110, linewidths=0.8, edgecolors="k", zorder=4)
        ax.annotate(row["algo"],
                    xy=(row["m_phi"], row["r2"]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=6.5, color=st["color"])

    for _, row in med_r2[(med_r2["algo"].isin(BASELINE_KEYS)) &
                         (med_r2["dataset"] == ds)].iterrows():
        st = BASELINE_STYLES[row["algo"]]
        ax.scatter(row["m_phi"], row["r2"],
                   color=st["color"], marker=st["marker"],
                   s=90, linewidths=0.7, edgecolors="k", zorder=3)
        ax.annotate(st["label"],
                    xy=(row["m_phi"], row["r2"]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=6.5, color=st["color"])

    r = roles
    subtitle = (f"R²: {r['best_r2']['algo']}  |  "
                f"Mφ: {r['best_mphi']['algo']}  |  "
                f"mid: {r['middle']['algo']}")
    ax.set_title(f"{DS_LABELS[ds]}\n{subtitle}", fontsize=8, fontweight="bold")
    ax.set_xlabel("M$_\\phi$  (higher = more interpretable)", fontsize=8)
    ax.set_ylabel("Test R²  (higher = better)", fontsize=8)
    ax.tick_params(labelsize=7)

slim_handles_r2 = [
    plt.Line2D([0],[0], marker=ROLE_STYLES_R2[r]["marker"],
               color=ROLE_STYLES_R2[r]["color"], linestyle="None",
               markersize=8, markeredgecolor="k", markeredgewidth=0.5,
               label=f"SLIM – {ROLE_STYLES_R2[r]['label']}")
    for r in ROLE_STYLES_R2
]
bl_handles_r2 = [
    plt.Line2D([0],[0], marker=BASELINE_STYLES[a]["marker"],
               color=BASELINE_STYLES[a]["color"], linestyle="None",
               markersize=8, markeredgecolor="k", markeredgewidth=0.5,
               label=a)
    for a in BASELINE_KEYS
]
fig.legend(handles=slim_handles_r2 + bl_handles_r2,
           loc="lower center", ncol=7,
           fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("M$_\\phi$ vs Test R² — medians over 30 seeds  (both axes: higher = better)",
             fontsize=10)
plt.savefig(OUT_PANEL_R2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_PANEL_R2}")

# ── AGGREGATE PLOT (R²) ───────────────────────────────────────────────────────
agg_r2_rows = []
for ds, roles in roles_r2_by_ds.items():
    for role, row in roles.items():
        agg_r2_rows.append({"role": role, "m_phi": row["m_phi"], "r2": row["r2"]})
agg_slim_r2 = (pd.DataFrame(agg_r2_rows)
               .groupby("role")
               .agg(m_phi=("m_phi", "median"),
                    r2=("r2", "median"))
               .reset_index())

bl_agg_r2 = (med_r2[med_r2["algo"].isin(BASELINE_KEYS)]
             .groupby("algo")
             .agg(m_phi=("m_phi", "median"),
                  r2=("r2", "median"))
             .reset_index())

fig3, ax3 = plt.subplots(figsize=(7, 5), constrained_layout=True)

for _, row in agg_slim_r2.iterrows():
    st = ROLE_STYLES_R2[row["role"]]
    ax3.scatter(row["m_phi"], row["r2"],
                color=st["color"], marker=st["marker"],
                s=130, linewidths=0.8, edgecolors="k", zorder=4)
    ax3.annotate(f"SLIM\n({st['label']})",
                 xy=(row["m_phi"], row["r2"]),
                 xytext=(6, 4), textcoords="offset points",
                 fontsize=8, color=st["color"])

for _, row in bl_agg_r2.iterrows():
    st = BASELINE_STYLES[row["algo"]]
    ax3.scatter(row["m_phi"], row["r2"],
                color=st["color"], marker=st["marker"],
                s=120, linewidths=0.8, edgecolors="k", zorder=3)
    ax3.annotate(st["label"],
                 xy=(row["m_phi"], row["r2"]),
                 xytext=(6, 4), textcoords="offset points",
                 fontsize=8, color=st["color"])

ax3.set_xlabel("M$_\\phi$  (higher = more interpretable)", fontsize=10)
ax3.set_ylabel("Test R²  (higher = better)", fontsize=9)
ax3.set_title("M$_\\phi$ vs R² — medians across 6 datasets × 30 seeds  (both axes: higher = better)",
              fontsize=10)
ax3.tick_params(labelsize=8)

handles3 = [
    plt.Line2D([0],[0], marker=ROLE_STYLES_R2[r]["marker"],
               color=ROLE_STYLES_R2[r]["color"], linestyle="None",
               markersize=9, markeredgecolor="k", markeredgewidth=0.5,
               label=f"SLIM – {ROLE_STYLES_R2[r]['label']}")
    for r in ROLE_STYLES_R2
] + [
    plt.Line2D([0],[0], marker=BASELINE_STYLES[a]["marker"],
               color=BASELINE_STYLES[a]["color"], linestyle="None",
               markersize=9, markeredgecolor="k", markeredgewidth=0.5,
               label=a)
    for a in BASELINE_KEYS
]
ax3.legend(handles=handles3, fontsize=8.5, loc="best")

plt.savefig(OUT_AGG_R2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_AGG_R2}")
