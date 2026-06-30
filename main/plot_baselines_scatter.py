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

GP-GOMEA is included as a horizontal dashed line (RMSE only) because M_phi
was not extracted from its C++ expression tree during the run.

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

ALL_SLIM = ["SLIM+2SIG","SLIM*2SIG","SLIM+1SIG","SLIM*1SIG",
            "SLIM+ABS","SLIM*ABS","SLIM+NORM1","SLIM*NORM1","SLIM+NORM2","SLIM*NORM2"]

ROLE_STYLES = {
    "best_rmse": dict(label="Best RMSE",       color="#1f77b4", marker="o"),
    "best_mphi": dict(label="Best M$_\\phi$",  color="#2ca02c", marker="s"),
    "middle":    dict(label="Middle",           color="#ff7f0e", marker="^"),
}

# baselines with M_phi (scatter points)
BASELINE_STYLES = {
    "GPLearn": dict(label="GPLearn", color="#d62728", marker="D"),
    "Operon":  dict(label="Operon",  color="#9467bd", marker="P"),
    "PySR":    dict(label="PySR",    color="#8c564b", marker="X"),
}
BASELINE_KEYS = list(BASELINE_STYLES)

# GP-GOMEA: RMSE available, M_phi not — shown as horizontal reference line
GPGOMEA_COLOR = "#17becf"

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
bl_raw["m_phi_after"] = bl_raw[["m_phi_after", "m_phi_before"]].max(axis=1)

gp_gomea_raw = bl_raw[bl_raw["algo"] == "GP-GOMEA"].copy()
bl_raw       = bl_raw[bl_raw["algo"].isin(BASELINE_KEYS)].copy()

# ── combine scatter-eligible data & normalise ──────────────────────────────────
cols = ["algo", "dataset", "seed", "test_rmse", "m_phi_after"]
combined = pd.concat([slim_raw[cols], bl_raw[cols]], ignore_index=True)
combined = combined[combined["dataset"].isin(DATASETS)]

ref = (combined[combined["algo"] == "GPLearn"]
       .groupby("dataset")["test_rmse"].median())
combined["rmse_norm"] = combined["test_rmse"] / combined["dataset"].map(ref)

# normalise GP-GOMEA with the same per-dataset reference
gp_gomea_raw = gp_gomea_raw[gp_gomea_raw["dataset"].isin(DATASETS)].copy()
gp_gomea_raw["rmse_norm"] = gp_gomea_raw["test_rmse"] / gp_gomea_raw["dataset"].map(ref)

# ── median per (algo, dataset) ─────────────────────────────────────────────────
med = (combined
       .groupby(["algo", "dataset"])
       .agg(m_phi=("m_phi_after", "median"),
            rmse_norm=("rmse_norm", "median"))
       .reset_index())
med["neg_mphi"] = -med["m_phi"]

gp_gomea_med = (gp_gomea_raw
                .groupby("dataset")["rmse_norm"]
                .median()
                .reindex(DATASETS))

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

    # GP-GOMEA: horizontal dashed line (RMSE only, no M_phi position)
    gp_y = gp_gomea_med.get(ds, np.nan)
    if not np.isnan(gp_y):
        ax.axhline(gp_y, color=GPGOMEA_COLOR, linestyle="--",
                   linewidth=1.4, zorder=2, alpha=0.85)
        ax.annotate("GP-GOMEA", xy=(1, gp_y), xycoords=("axes fraction", "data"),
                    xytext=(-4, 3), textcoords="offset points",
                    fontsize=6.5, color=GPGOMEA_COLOR, ha="right", va="bottom")

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

    # baselines with M_phi
    for _, row in med[(med["algo"].isin(BASELINE_KEYS)) &
                      (med["dataset"] == ds)].iterrows():
        st = BASELINE_STYLES[row["algo"]]
        ax.scatter(row["neg_mphi"], row["rmse_norm"],
                   color=st["color"], marker=st["marker"],
                   s=90, linewidths=0.7, edgecolors="k", zorder=3)

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
gp_handle = [plt.Line2D([0],[0], color=GPGOMEA_COLOR, linestyle="--",
                         linewidth=1.5, label="GP-GOMEA (M$_\\phi$ N/A)")]
fig.legend(handles=slim_handles + bl_handles + gp_handle,
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

gp_gomea_agg_rmse = gp_gomea_med.median()

fig2, ax2 = plt.subplots(figsize=(7, 5), constrained_layout=True)

# GP-GOMEA horizontal reference line
ax2.axhline(gp_gomea_agg_rmse, color=GPGOMEA_COLOR, linestyle="--",
            linewidth=1.5, zorder=2, alpha=0.85)
ax2.annotate("GP-GOMEA\n(M$_\\phi$ N/A)",
             xy=(1, gp_gomea_agg_rmse), xycoords=("axes fraction", "data"),
             xytext=(-6, 4), textcoords="offset points",
             fontsize=8, color=GPGOMEA_COLOR, ha="right", va="bottom")

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
] + [plt.Line2D([0],[0], color=GPGOMEA_COLOR, linestyle="--",
                linewidth=1.5, label="GP-GOMEA (M$_\\phi$ N/A)")]
ax2.legend(handles=handles2, fontsize=8.5, loc="best")

plt.savefig(OUT_AGG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_AGG}")
