"""
plot_baselines_scatter_no_pysr.py

Same as plot_baselines_scatter.py but:
  - PySR excluded
  - Pareto-front points shown filled; dominated points shown hollow

RMSE plots  : Pareto front minimises both –M_phi and normalised RMSE
              (bottom-left corner = best interpretability AND best accuracy)
R² plots    : Pareto front maximises both M_phi and R²
              (top-right corner = best interpretability AND best accuracy)

Run from project root:
    python main/plot_baselines_scatter_no_pysr.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────────
_LOG  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
_FIGS = os.path.join(_LOG, "figs")
os.makedirs(_FIGS, exist_ok=True)

BASELINE_CSV = os.path.join(_LOG, "results_baselines.csv")
SLIM_CSV     = os.path.join(_LOG, "results_normalized_simplification.csv")

ALL_SLIM = ["SLIM+2SIG","SLIM*2SIG","SLIM+1SIG","SLIM*1SIG",
            "SLIM+ABS","SLIM*ABS","SLIM+NORM1","SLIM*NORM1","SLIM+NORM2","SLIM*NORM2"]

ROLE_STYLES = {
    "best_rmse": dict(label="Best RMSE",       color="#1f77b4", marker="o"),
    "best_mphi": dict(label="Best M$_\\phi$",  color="#2ca02c", marker="s"),
    "middle":    dict(label="Middle",           color="#ff7f0e", marker="^"),
}

BASELINE_STYLES = {
    "GPLearn":  dict(label="GPLearn",  color="#d62728", marker="D"),
    "Operon":   dict(label="Operon",   color="#9467bd", marker="P"),
    "GP-GOMEA": dict(label="GP-GOMEA", color="#17becf", marker="X"),
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


# ── Pareto utility ─────────────────────────────────────────────────────────────

def is_pareto(xs, ys, minimize_x=True, minimize_y=True):
    """Return boolean array: True where point is Pareto-non-dominated."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if not minimize_x:
        xs = -xs
    if not minimize_y:
        ys = -ys
    n = len(xs)
    front = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if xs[j] <= xs[i] and ys[j] <= ys[i] and (xs[j] < xs[i] or ys[j] < ys[i]):
                front[i] = False
                break
    return front


def scatter_with_pareto(ax, xvals, yvals, colors, markers, sizes,
                        pareto_mask, zorders, labels=None,
                        edgewidth=1.0, lw_pareto=1.2):
    """
    Draw each point filled if on Pareto front, hollow otherwise.
    xvals, yvals, colors, markers, sizes, pareto_mask, zorders : parallel arrays.
    """
    for k in range(len(xvals)):
        fc = colors[k] if pareto_mask[k] else "none"
        ec = colors[k]
        lw = lw_pareto if pareto_mask[k] else edgewidth
        ax.scatter(xvals[k], yvals[k],
                   facecolors=fc, edgecolors=ec,
                   marker=markers[k], s=sizes[k],
                   linewidths=lw, zorder=zorders[k])
        if labels is not None and labels[k]:
            ax.annotate(labels[k],
                        xy=(xvals[k], yvals[k]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=6.5, color=ec)


# ── load SLIM ──────────────────────────────────────────────────────────────────
slim_raw = pd.read_csv(SLIM_CSV)
slim_raw = slim_raw[slim_raw["algo"].isin(ALL_SLIM)].copy()
slim_raw["ell_after"]   = slim_raw[["ell_after",   "ell_before"  ]].min(axis=1)
slim_raw["m_phi_after"] = slim_raw[["m_phi_after", "m_phi_before"]].max(axis=1)

# ── load baselines (no PySR) ───────────────────────────────────────────────────
bl_raw = pd.read_csv(BASELINE_CSV)
bl_raw = bl_raw.dropna(subset=["test_rmse"])
gp_mask = bl_raw["algo"] == "GP-GOMEA"
bl_raw  = pd.concat([bl_raw[~gp_mask], bl_raw[gp_mask & bl_raw["m_phi_before"].notna()]],
                    ignore_index=True)
bl_raw["m_phi_after"] = bl_raw[["m_phi_after", "m_phi_before"]].max(axis=1)
bl_raw = bl_raw[bl_raw["algo"].isin(BASELINE_KEYS)].copy()

# ── combine & normalise (RMSE) ────────────────────────────────────────────────
cols = ["algo", "dataset", "seed", "test_rmse", "m_phi_after"]
combined = pd.concat([slim_raw[cols], bl_raw[cols]], ignore_index=True)
combined = combined[combined["dataset"].isin(DATASETS)]
ref = (combined[combined["algo"] == "GPLearn"]
       .groupby("dataset")["test_rmse"].median())
combined["rmse_norm"] = combined["test_rmse"] / combined["dataset"].map(ref)

med = (combined
       .groupby(["algo", "dataset"])
       .agg(m_phi=("m_phi_after", "median"), rmse_norm=("rmse_norm", "median"))
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


# ══════════════════════════════════════════════════════════════════════════════
# RMSE PANEL
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
axes = axes.flatten()

for i, ds in enumerate(DATASETS):
    ax    = axes[i]
    roles = roles_by_ds[ds]

    # collect all points for this panel
    pts_x, pts_y, pts_c, pts_m, pts_s, pts_z, pts_lbl = [], [], [], [], [], [], []

    for role, row in roles.items():
        st = ROLE_STYLES[role]
        pts_x.append(row["neg_mphi"]); pts_y.append(row["rmse_norm"])
        pts_c.append(st["color"]); pts_m.append(st["marker"])
        pts_s.append(110); pts_z.append(4); pts_lbl.append(row["algo"])

    for _, row in med[(med["algo"].isin(BASELINE_KEYS)) & (med["dataset"] == ds)].iterrows():
        st = BASELINE_STYLES[row["algo"]]
        pts_x.append(row["neg_mphi"]); pts_y.append(row["rmse_norm"])
        pts_c.append(st["color"]); pts_m.append(st["marker"])
        pts_s.append(90); pts_z.append(3); pts_lbl.append(st["label"])

    front = is_pareto(pts_x, pts_y, minimize_x=True, minimize_y=True)
    scatter_with_pareto(ax, pts_x, pts_y, pts_c, pts_m, pts_s,
                        front, pts_z, labels=pts_lbl)

    r = roles
    subtitle = (f"RMSE: {r['best_rmse']['algo']}  |  "
                f"Mφ: {r['best_mphi']['algo']}  |  "
                f"mid: {r['middle']['algo']}")
    ax.set_title(f"{DS_LABELS[ds]}\n{subtitle}", fontsize=8, fontweight="bold")
    ax.set_xlabel("−M$_\\phi$  (higher = more complex)", fontsize=8)
    ax.set_ylabel("RMSE / GPLearn median", fontsize=8)
    ax.tick_params(labelsize=7)

slim_handles = [
    plt.Line2D([0],[0], marker=ROLE_STYLES[r]["marker"],
               color=ROLE_STYLES[r]["color"], linestyle="None",
               markersize=8, markeredgecolor=ROLE_STYLES[r]["color"], markeredgewidth=1,
               markerfacecolor=ROLE_STYLES[r]["color"],
               label=f"SLIM – {ROLE_STYLES[r]['label']}")
    for r in ROLE_STYLES
]
bl_handles = [
    plt.Line2D([0],[0], marker=BASELINE_STYLES[a]["marker"],
               color=BASELINE_STYLES[a]["color"], linestyle="None",
               markersize=8, markeredgecolor=BASELINE_STYLES[a]["color"], markeredgewidth=1,
               markerfacecolor=BASELINE_STYLES[a]["color"],
               label=a)
    for a in BASELINE_KEYS
]
pareto_handle = plt.Line2D([0],[0], marker="o", color="k", linestyle="None",
                            markersize=8, markerfacecolor="#888888",
                            label="Filled = Pareto front")
hollow_handle = plt.Line2D([0],[0], marker="o", color="k", linestyle="None",
                            markersize=8, markerfacecolor="none",
                            label="Hollow = dominated")
fig.legend(handles=slim_handles + bl_handles + [pareto_handle, hollow_handle],
           loc="lower center", ncol=5,
           fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("−M$_\\phi$ vs Normalised Test RMSE — medians over 30 seeds\n"
             "Filled = Pareto-optimal  |  RMSE normalised by per-dataset GPLearn median",
             fontsize=10)
out = os.path.join(_FIGS, "baselines_scatter_panel_no_pysr.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# RMSE AGGREGATE
# ══════════════════════════════════════════════════════════════════════════════
agg_rows = []
for ds, roles in roles_by_ds.items():
    for role, row in roles.items():
        agg_rows.append({"role": role, "neg_mphi": row["neg_mphi"],
                         "rmse_norm": row["rmse_norm"]})
agg_slim = (pd.DataFrame(agg_rows)
              .groupby("role")
              .agg(neg_mphi=("neg_mphi", "median"), rmse_norm=("rmse_norm", "median"))
              .reset_index())
bl_agg = (med[med["algo"].isin(BASELINE_KEYS)]
          .groupby("algo")
          .agg(neg_mphi=("neg_mphi", "median"), rmse_norm=("rmse_norm", "median"))
          .reset_index())

pts_x, pts_y, pts_c, pts_m, pts_s, pts_z, pts_lbl = [], [], [], [], [], [], []
for _, row in agg_slim.iterrows():
    st = ROLE_STYLES[row["role"]]
    pts_x.append(row["neg_mphi"]); pts_y.append(row["rmse_norm"])
    pts_c.append(st["color"]); pts_m.append(st["marker"])
    pts_s.append(130); pts_z.append(4)
    pts_lbl.append(f"SLIM\n({st['label']})")
for _, row in bl_agg.iterrows():
    st = BASELINE_STYLES[row["algo"]]
    pts_x.append(row["neg_mphi"]); pts_y.append(row["rmse_norm"])
    pts_c.append(st["color"]); pts_m.append(st["marker"])
    pts_s.append(120); pts_z.append(3); pts_lbl.append(st["label"])

front = is_pareto(pts_x, pts_y, minimize_x=True, minimize_y=True)

fig2, ax2 = plt.subplots(figsize=(7, 5), constrained_layout=True)
scatter_with_pareto(ax2, pts_x, pts_y, pts_c, pts_m, pts_s,
                    front, pts_z, labels=pts_lbl, lw_pareto=1.5)
ax2.set_xlabel("−M$_\\phi$  (higher = more complex / less interpretable)", fontsize=10)
ax2.set_ylabel("RMSE / GPLearn median per dataset\n(< 1  =  better than GPLearn)", fontsize=9)
ax2.set_title("Interpretability vs Performance — medians across 6 datasets × 30 seeds\n"
              "Filled = Pareto-optimal  (lower-left corner dominates)", fontsize=10)
ax2.tick_params(labelsize=8)
h2 = (
    [plt.Line2D([0],[0], marker=ROLE_STYLES[r]["marker"],
                color=ROLE_STYLES[r]["color"], linestyle="None",
                markersize=9, markerfacecolor=ROLE_STYLES[r]["color"],
                label=f"SLIM – {ROLE_STYLES[r]['label']}")
     for r in ROLE_STYLES]
  + [plt.Line2D([0],[0], marker=BASELINE_STYLES[a]["marker"],
                color=BASELINE_STYLES[a]["color"], linestyle="None",
                markersize=9, markerfacecolor=BASELINE_STYLES[a]["color"],
                label=a)
     for a in BASELINE_KEYS]
  + [plt.Line2D([0],[0], marker="o", color="k", linestyle="None",
                markersize=8, markerfacecolor="#888888", label="Filled = Pareto front"),
     plt.Line2D([0],[0], marker="o", color="k", linestyle="None",
                markersize=8, markerfacecolor="none", label="Hollow = dominated")]
)
ax2.legend(handles=h2, fontsize=8.5, loc="best")
out2 = os.path.join(_FIGS, "baselines_scatter_agg_no_pysr.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# R² PANEL
# ══════════════════════════════════════════════════════════════════════════════
ROLE_STYLES_R2 = {
    "best_r2":   dict(label="Best R²",        color="#1f77b4", marker="o"),
    "best_mphi": dict(label="Best M$_\\phi$", color="#2ca02c", marker="s"),
    "middle":    dict(label="Middle",          color="#ff7f0e", marker="^"),
}

r2_cols = ["algo", "dataset", "seed", "test_r2", "m_phi_after"]
combined_r2 = pd.concat([slim_raw[r2_cols], bl_raw[r2_cols]], ignore_index=True)
combined_r2 = combined_r2[combined_r2["dataset"].isin(DATASETS)]
med_r2 = (combined_r2
          .groupby(["algo", "dataset"])
          .agg(m_phi=("m_phi_after", "median"), r2=("test_r2", "median"))
          .reset_index())
med_r2["neg_mphi"] = -med_r2["m_phi"]

def pick_roles_r2(ds):
    sub = med_r2[(med_r2["algo"].isin(ALL_SLIM)) & (med_r2["dataset"] == ds)].copy()
    sub["r2_rank"]   = sub["r2"].rank(ascending=False)
    sub["mphi_rank"] = sub["m_phi"].rank(ascending=False)
    sub["avg_rank"]  = (sub["r2_rank"] + sub["mphi_rank"]) / 2
    best_r2   = sub.loc[sub["r2_rank"].idxmin()]
    best_mphi = sub.loc[sub["mphi_rank"].idxmin()]
    rest      = sub[~sub["algo"].isin({best_r2["algo"], best_mphi["algo"]})]
    middle    = rest.loc[rest["avg_rank"].idxmin()]
    return {"best_r2": best_r2, "best_mphi": best_mphi, "middle": middle}

roles_r2_by_ds = {ds: pick_roles_r2(ds) for ds in DATASETS}

fig3, axes3 = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
axes3 = axes3.flatten()

for i, ds in enumerate(DATASETS):
    ax    = axes3[i]
    roles = roles_r2_by_ds[ds]

    pts_x, pts_y, pts_c, pts_m, pts_s, pts_z, pts_lbl = [], [], [], [], [], [], []
    for role, row in roles.items():
        st = ROLE_STYLES_R2[role]
        pts_x.append(row["m_phi"]); pts_y.append(row["r2"])
        pts_c.append(st["color"]); pts_m.append(st["marker"])
        pts_s.append(110); pts_z.append(4); pts_lbl.append(row["algo"])
    for _, row in med_r2[(med_r2["algo"].isin(BASELINE_KEYS)) & (med_r2["dataset"] == ds)].iterrows():
        st = BASELINE_STYLES[row["algo"]]
        pts_x.append(row["m_phi"]); pts_y.append(row["r2"])
        pts_c.append(st["color"]); pts_m.append(st["marker"])
        pts_s.append(90); pts_z.append(3); pts_lbl.append(st["label"])

    # Pareto: maximise both M_phi and R²
    front = is_pareto(pts_x, pts_y, minimize_x=False, minimize_y=False)
    scatter_with_pareto(ax, pts_x, pts_y, pts_c, pts_m, pts_s,
                        front, pts_z, labels=pts_lbl)

    r = roles
    subtitle = (f"R²: {r['best_r2']['algo']}  |  "
                f"Mφ: {r['best_mphi']['algo']}  |  "
                f"mid: {r['middle']['algo']}")
    ax.set_title(f"{DS_LABELS[ds]}\n{subtitle}", fontsize=8, fontweight="bold")
    ax.set_xlabel("M$_\\phi$  (higher = more interpretable)", fontsize=8)
    ax.set_ylabel("Test R²  (higher = better)", fontsize=8)
    ax.tick_params(labelsize=7)

sh3 = [plt.Line2D([0],[0], marker=ROLE_STYLES_R2[r]["marker"],
                   color=ROLE_STYLES_R2[r]["color"], linestyle="None",
                   markersize=8, markerfacecolor=ROLE_STYLES_R2[r]["color"],
                   label=f"SLIM – {ROLE_STYLES_R2[r]['label']}")
       for r in ROLE_STYLES_R2]
bh3 = [plt.Line2D([0],[0], marker=BASELINE_STYLES[a]["marker"],
                   color=BASELINE_STYLES[a]["color"], linestyle="None",
                   markersize=8, markerfacecolor=BASELINE_STYLES[a]["color"],
                   label=a)
       for a in BASELINE_KEYS]
ph3 = [plt.Line2D([0],[0], marker="o", color="k", linestyle="None",
                   markersize=8, markerfacecolor="#888888", label="Filled = Pareto front"),
       plt.Line2D([0],[0], marker="o", color="k", linestyle="None",
                   markersize=8, markerfacecolor="none", label="Hollow = dominated")]
fig3.legend(handles=sh3 + bh3 + ph3, loc="lower center", ncol=5,
            fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.06))
fig3.suptitle("M$_\\phi$ vs Test R² — medians over 30 seeds\n"
              "Filled = Pareto-optimal  (upper-right corner dominates)",
              fontsize=10)
out3 = os.path.join(_FIGS, "baselines_scatter_panel_r2_no_pysr.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")


# ══════════════════════════════════════════════════════════════════════════════
# R² AGGREGATE
# ══════════════════════════════════════════════════════════════════════════════
agg_r2_rows = []
for ds, roles in roles_r2_by_ds.items():
    for role, row in roles.items():
        agg_r2_rows.append({"role": role, "m_phi": row["m_phi"], "r2": row["r2"]})
agg_slim_r2 = (pd.DataFrame(agg_r2_rows)
               .groupby("role")
               .agg(m_phi=("m_phi", "median"), r2=("r2", "median"))
               .reset_index())
bl_agg_r2 = (med_r2[med_r2["algo"].isin(BASELINE_KEYS)]
             .groupby("algo")
             .agg(m_phi=("m_phi", "median"), r2=("r2", "median"))
             .reset_index())

pts_x, pts_y, pts_c, pts_m, pts_s, pts_z, pts_lbl = [], [], [], [], [], [], []
for _, row in agg_slim_r2.iterrows():
    st = ROLE_STYLES_R2[row["role"]]
    pts_x.append(row["m_phi"]); pts_y.append(row["r2"])
    pts_c.append(st["color"]); pts_m.append(st["marker"])
    pts_s.append(130); pts_z.append(4)
    pts_lbl.append(f"SLIM\n({st['label']})")
for _, row in bl_agg_r2.iterrows():
    st = BASELINE_STYLES[row["algo"]]
    pts_x.append(row["m_phi"]); pts_y.append(row["r2"])
    pts_c.append(st["color"]); pts_m.append(st["marker"])
    pts_s.append(120); pts_z.append(3); pts_lbl.append(st["label"])

front = is_pareto(pts_x, pts_y, minimize_x=False, minimize_y=False)

fig4, ax4 = plt.subplots(figsize=(7, 5), constrained_layout=True)
scatter_with_pareto(ax4, pts_x, pts_y, pts_c, pts_m, pts_s,
                    front, pts_z, labels=pts_lbl, lw_pareto=1.5)
ax4.set_xlabel("M$_\\phi$  (higher = more interpretable)", fontsize=10)
ax4.set_ylabel("Test R²  (higher = better)", fontsize=9)
ax4.set_title("M$_\\phi$ vs R² — medians across 6 datasets × 30 seeds\n"
              "Filled = Pareto-optimal  (upper-right corner dominates)", fontsize=10)
ax4.tick_params(labelsize=8)
h4 = (
    [plt.Line2D([0],[0], marker=ROLE_STYLES_R2[r]["marker"],
                color=ROLE_STYLES_R2[r]["color"], linestyle="None",
                markersize=9, markerfacecolor=ROLE_STYLES_R2[r]["color"],
                label=f"SLIM – {ROLE_STYLES_R2[r]['label']}")
     for r in ROLE_STYLES_R2]
  + [plt.Line2D([0],[0], marker=BASELINE_STYLES[a]["marker"],
                color=BASELINE_STYLES[a]["color"], linestyle="None",
                markersize=9, markerfacecolor=BASELINE_STYLES[a]["color"],
                label=a)
     for a in BASELINE_KEYS]
  + [plt.Line2D([0],[0], marker="o", color="k", linestyle="None",
                markersize=8, markerfacecolor="#888888", label="Filled = Pareto front"),
     plt.Line2D([0],[0], marker="o", color="k", linestyle="None",
                markersize=8, markerfacecolor="none", label="Hollow = dominated")]
)
ax4.legend(handles=h4, fontsize=8.5, loc="best")
out4 = os.path.join(_FIGS, "baselines_scatter_agg_r2_no_pysr.png")
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out4}")
