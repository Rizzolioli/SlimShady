"""
generate_decay_xo_figs.py
=========================
Analyse the cosine² p_xo decay experiment.

Compares:
  • decay_cos2  (results_decay_xo_22062026.csv)    – p_xo: 0.70 → 0.30
  • fixed p_xo=0.7 (results_op_stats_19062026.csv) – p_xo: 0.70 constant

Two figure sets per dataset:

  Fig-1  Fitness convergence: 3 variants × 2 metrics (train/test RMSE)
         decay_cos2 (solid orange) vs fixed-0.7 (dashed blue), median ± IQR

  Fig-2  Operator dynamics during decay: 3 variants, 2 columns
         col-0 = improvement rate (%), col-1 = n applications
         inflate=red, deflate=blue, xo=green
         secondary right axis: actual p_xo schedule (grey dashed)

Plus a summary CSV: final-gen (400) mean/std test_fit and nodes_count.
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── paths ─────────────────────────────────────────────────────────────────────

_HERE    = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR = os.path.join(_HERE, "..", "log")
_OUT     = os.path.join(_LOG_DIR, "latex_final", "decay_xo")

_DECAY_CSV    = os.path.join(_LOG_DIR, "results_decay_xo_22062026.csv")
_FIXED_CSV    = os.path.join(_LOG_DIR, "results_op_stats_19062026.csv")

# ── constants ─────────────────────────────────────────────────────────────────

_VARIANTS = ["SLIM+2SIG", "SLIM*ABS", "SLIM*1SIG"]
_DATASETS = ["toxicity", "concrete", "instanbul", "ppb",
             "resid_build_sale_price", "energy"]

_COLS9 = {0:'algo',1:'run_id',2:'dataset',3:'seed',4:'gen',
          5:'train_fit',6:'timing',7:'nodes',8:'test_fit',9:'nodes_count',
          10:'inflate_n',11:'inflate_improved',12:'deflate_n',13:'deflate_improved',
          14:'xo_n',15:'xo_improved',16:'log_level'}

_OPS       = ["inflate", "deflate", "xo"]
_OP_COLORS = {"inflate": "#e41a1c", "deflate": "#377eb8", "xo": "#4daf4a"}
_OP_LABELS = {"inflate": "Inflate", "deflate": "Deflate", "xo": "Head XO"}

_C_DECAY = "#e67e22"    # orange – decay_cos2
_C_FIXED = "#2980b9"    # blue   – fixed p_xo=0.7

N_ITER = 400

# ── schedule ──────────────────────────────────────────────────────────────────

def _cos2_pxo(t, n_iter=N_ITER, lo=0.3, hi=0.7):
    return lo + (hi - lo) * (0.5 * (1 + math.cos(math.pi * t / n_iter))) ** 2

_GEN_RANGE  = np.arange(0, N_ITER + 1)
_SCHEDULE   = np.array([_cos2_pxo(t) for t in _GEN_RANGE])

# ── loaders ───────────────────────────────────────────────────────────────────

def _load_log9(path):
    df = pd.read_csv(path, header=None).rename(columns=_COLS9)
    df["seed"]    = df["seed"].astype(int)
    df = df.drop_duplicates(subset=["algo","dataset","seed","gen"], keep="last")
    df["variant"] = df["algo"].str.extract(r'^(SLIM[+*]\w+)_pop')
    # operator rates
    for op in _OPS:
        df[f"{op}_rate"] = (
            df[f"{op}_improved"] / df[f"{op}_n"].replace(0, np.nan)
        ) * 100
    return df[df["variant"].isin(_VARIANTS)]


def _pivot_med_iqr(sub, col):
    piv = sub.pivot_table(index="gen", columns="seed", values=col)
    return piv.median(axis=1), piv.quantile(0.25, axis=1), piv.quantile(0.75, axis=1)

# ── save ──────────────────────────────────────────────────────────────────────

def _save(fig, name):
    path = os.path.join(_OUT, name)
    os.makedirs(_OUT, exist_ok=True)
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight")
    print(f"  saved: {name}.png")
    plt.close(fig)

# ── Fig-1: fitness convergence ────────────────────────────────────────────────

def fig_convergence(df_decay, df_fixed):
    metrics = [
        ("train_fit",    "Train RMSE"),
        ("test_fit",     "Test RMSE"),
        ("nodes_count",  "Model size (nodes)"),
    ]

    for dataset in _DATASETS:
        dec = df_decay[df_decay["dataset"] == dataset]
        fix = df_fixed[df_fixed["dataset"] == dataset]
        if dec.empty:
            print(f"  [SKIP] {dataset}")
            continue

        fig, axes = plt.subplots(
            len(_VARIANTS), len(metrics),
            figsize=(16, 3.4 * len(_VARIANTS)),
            sharex=True,
        )
        fig.suptitle(
            f"Convergence — {dataset.replace('_',' ')}\n"
            f"Cosine² decay (0.70->0.30) vs Fixed p_xo=0.70",
            fontsize=12, fontweight="bold", y=1.01,
        )
        for ci, (_, title) in enumerate(metrics):
            axes[0, ci].set_title(title, fontsize=10, pad=4)

        for ri, variant in enumerate(_VARIANTS):
            sd = dec[dec["variant"] == variant]
            sf = fix[fix["variant"] == variant]

            for ci, (col, _) in enumerate(metrics):
                ax = axes[ri, ci]

                for data, color, label, ls in [
                    (sd, _C_DECAY, "decay cos2", "-"),
                    (sf, _C_FIXED, "fixed 0.70", "--"),
                ]:
                    if data.empty:
                        continue
                    med, q25, q75 = _pivot_med_iqr(data, col)
                    ax.plot(med.index, med.values, color=color,
                            linewidth=1.6, linestyle=ls, label=label)
                    ax.fill_between(med.index, q25.values, q75.values,
                                    color=color, alpha=0.15)

                if ci == 0:
                    ax.set_ylabel(variant, fontsize=9, fontweight="bold")
                ax.grid(True, linewidth=0.4, alpha=0.5)
                ax.tick_params(labelsize=8)
                ax.set_ylim(bottom=0)

            if ri == len(_VARIANTS) - 1:
                for ci in range(len(metrics)):
                    axes[ri, ci].set_xlabel("Generation", fontsize=9)

        legend_handles = [
            Line2D([0],[0], color=_C_DECAY, lw=2, ls="-",  label="decay cos2 (0.70->0.30)"),
            Line2D([0],[0], color=_C_FIXED, lw=2, ls="--", label="fixed p_xo=0.70"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                   fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        _save(fig, f"convergence_{dataset}")


# ── Fig-2: operator dynamics + schedule overlay ───────────────────────────────

def fig_operator_dynamics(df_decay):
    for dataset in _DATASETS:
        dec = df_decay[df_decay["dataset"] == dataset]
        if dec.empty:
            print(f"  [SKIP] op {dataset}")
            continue

        fig, axes = plt.subplots(
            len(_VARIANTS), 2,
            figsize=(13, 3.6 * len(_VARIANTS)),
            sharex=True,
        )
        fig.suptitle(
            f"Operator dynamics under cosine² decay — {dataset.replace('_',' ')}",
            fontsize=12, fontweight="bold", y=1.01,
        )
        axes[0, 0].set_title("Improvement rate (%)", fontsize=10, pad=4)
        axes[0, 1].set_title("Applications per generation", fontsize=10, pad=4)

        for ri, variant in enumerate(_VARIANTS):
            sub = dec[dec["variant"] == variant]
            ax_rate  = axes[ri, 0]
            ax_count = axes[ri, 1]
            ax_rate.set_ylabel(variant, fontsize=9, fontweight="bold")

            for op in _OPS:
                color = _OP_COLORS[op]
                label = _OP_LABELS[op]

                med, q25, q75 = _pivot_med_iqr(sub, f"{op}_rate")
                ax_rate.plot(med.index, med.values, color=color,
                             linewidth=1.4, label=label)
                ax_rate.fill_between(med.index, q25.values, q75.values,
                                     color=color, alpha=0.15)

                med_n, q25_n, q75_n = _pivot_med_iqr(sub, f"{op}_n")
                ax_count.plot(med_n.index, med_n.values, color=color,
                              linewidth=1.4, label=label)
                ax_count.fill_between(med_n.index, q25_n.values, q75_n.values,
                                      color=color, alpha=0.15)

            # overlay schedule on right y-axis
            for ax in (ax_rate, ax_count):
                ax2 = ax.twinx()
                ax2.plot(_GEN_RANGE, _SCHEDULE, color="grey",
                         linewidth=1.2, linestyle=":", alpha=0.7)
                ax2.set_ylim(0, 1)
                ax2.set_ylabel("p_xo", fontsize=7, color="grey")
                ax2.tick_params(axis="y", labelsize=7, colors="grey")

            ax_rate.set_ylim(-2, 102)
            ax_rate.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax_rate.grid(True, linewidth=0.4, alpha=0.5)
            ax_rate.tick_params(labelsize=8)

            ax_count.set_ylim(bottom=0)
            ax_count.grid(True, linewidth=0.4, alpha=0.5)
            ax_count.tick_params(labelsize=8)

            if ri == len(_VARIANTS) - 1:
                for ax in (ax_rate, ax_count):
                    ax.set_xlabel("Generation", fontsize=9)

        op_handles = [Line2D([0],[0], color=_OP_COLORS[op], lw=2,
                             label=_OP_LABELS[op]) for op in _OPS]
        sched_handle = Line2D([0],[0], color="grey", lw=1.5, ls=":",
                              label="p_xo schedule")
        fig.legend(handles=op_handles + [sched_handle],
                   loc="lower center", ncol=4,
                   fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        _save(fig, f"op_dynamics_{dataset}")


# ── Fig-3: final-gen box plots (test RMSE) ───────────────────────────────────

def fig_final_boxplots(df_decay, df_fixed):
    final_dec = df_decay[df_decay["gen"] == N_ITER]
    final_fix = df_fixed[df_fixed["gen"] == N_ITER]

    row_metrics = [("test_fit", "Test RMSE"), ("nodes_count", "Model size (nodes)")]
    fig, axes = plt.subplots(
        len(row_metrics), len(_VARIANTS),
        figsize=(5 * len(_VARIANTS), 5 * len(row_metrics)),
    )
    fig.suptitle(
        "Final-generation comparison (gen 400)\nTest RMSE & Model size",
        fontsize=13, fontweight="bold",
    )

    for ci, variant in enumerate(_VARIANTS):
        axes[0, ci].set_title(variant, fontsize=10, fontweight="bold", pad=4)
        for ri, (col, ylabel) in enumerate(row_metrics):
            ax = axes[ri, ci]
            dec_v = final_dec[final_dec["variant"] == variant]
            fix_v = final_fix[final_fix["variant"] == variant]

            data_by_dataset = []
            labels = []
            colors = []
            for ds in _DATASETS:
                d = dec_v[dec_v["dataset"] == ds][col].dropna()
                f = fix_v[fix_v["dataset"] == ds][col].dropna()
                if not d.empty:
                    data_by_dataset.append(d.values)
                    labels.append(ds[:6])
                    colors.append(_C_DECAY)
                if not f.empty:
                    data_by_dataset.append(f.values)
                    labels.append("")
                    colors.append(_C_FIXED)

            bp = ax.boxplot(data_by_dataset, patch_artist=True,
                            widths=0.6, medianprops=dict(color="black", lw=1.5))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_xticklabels(labels, rotation=45, fontsize=7, ha="right")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
            ax.tick_params(labelsize=7)

    legend_handles = [
        Line2D([0],[0], color=_C_DECAY, lw=6, alpha=0.6, label="decay cos²"),
        Line2D([0],[0], color=_C_FIXED, lw=6, alpha=0.6, label="fixed p_xo=0.70"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "final_boxplots")


# ── Summary CSV ───────────────────────────────────────────────────────────────

def write_summary(df_decay, df_fixed):
    rows = []
    for df, label in [(df_decay, "decay_cos2"), (df_fixed, "fixed_0.7")]:
        final = df[df["gen"] == N_ITER]
        for variant in _VARIANTS:
            for dataset in _DATASETS:
                sub = final[(final["variant"] == variant) & (final["dataset"] == dataset)]
                if sub.empty:
                    continue
                rows.append({
                    "config":    label,
                    "variant":   variant,
                    "dataset":   dataset,
                    "test_mean": sub["test_fit"].mean(),
                    "test_std":  sub["test_fit"].std(),
                    "test_med":  sub["test_fit"].median(),
                    "nodes_mean":sub["nodes_count"].mean(),
                    "nodes_std": sub["nodes_count"].std(),
                })
    out = pd.DataFrame(rows)
    path = os.path.join(_OUT, "decay_xo_summary.csv")
    os.makedirs(_OUT, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.4f")
    print(f"  saved: decay_xo_summary.csv")

    # pivot for quick reading
    pivot = out.pivot_table(
        index=["variant","dataset"],
        columns="config",
        values=["test_med","nodes_mean"],
        aggfunc="first"
    )
    pivot["test_delta"] = pivot[("test_med","decay_cos2")] - pivot[("test_med","fixed_0.7")]
    pivot["nodes_delta"] = pivot[("nodes_mean","decay_cos2")] - pivot[("nodes_mean","fixed_0.7")]
    print("\n== Test RMSE median (decay - fixed): negative = decay wins ==")
    print(pivot["test_delta"].to_string())
    print("\n== Node count mean (decay - fixed): negative = decay more compact ==")
    print(pivot["nodes_delta"].to_string())

    return out


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading decay_cos2 log...")
    df_decay = _load_log9(_DECAY_CSV)
    print(f"  {len(df_decay)} rows")

    print("Loading fixed p_xo=0.7 log...")
    df_fixed = _load_log9(_FIXED_CSV)
    print(f"  {len(df_fixed)} rows")

    os.makedirs(_OUT, exist_ok=True)
    print(f"\nOutput -> {_OUT}\n")

    print("[1/4] Fitness convergence figures...")
    fig_convergence(df_decay, df_fixed)

    print("\n[2/4] Operator dynamics figures...")
    fig_operator_dynamics(df_decay)

    print("\n[3/4] Final-gen box plots...")
    fig_final_boxplots(df_decay, df_fixed)

    print("\n[4/4] Summary CSV + deltas...")
    write_summary(df_decay, df_fixed)

    print("\nDone.")
