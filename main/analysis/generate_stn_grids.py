"""
generate_stn_grids.py
=====================
Generate 18 STN comparison figures: 6 datasets × 3 (layout/model/view combos).

Each figure: 3 rows × 2 cols
  rows    = SLIM+2SIG, SLIM*ABS, SLIM*1SIG   (in that order)
  cols    = BASELINE  p_xo=0.0  |  TREATMENT  p_xo=0.7, hd=17

Output: main/log/latex/stns/
  {dataset}_stn_stress_clustering_tree.png / .tex
  {dataset}_stn_fitness_clustering_node.png / .tex
  {dataset}_stn_bivar_genotype_tree.png    / .tex

pkl files are expected in main/log/stns/{dataset}/ and must have been
produced by stn_prep.py + stn_build.py from results_prob_xo_12052026.csv.
Algo names in the pkls are sanitized (SLIM+2SIG_pxo0.0 → SLIM_2SIG_pxo0_0).
"""

import os
import sys
import re as _re
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import matplot2tikz as tikzplotlib

# ── PATH SETUP ────────────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR  = os.path.join(_HERE, "..", "log")
_STN_ROOT = os.path.join(_LOG_DIR, "stns")
_OUT_ROOT = os.path.join(_LOG_DIR, "latex", "stns")

sys.path.insert(0, _HERE)
from stn_plot import plot_stn, _get_layout, _NODE, FSIZE

# ── CONFIG ────────────────────────────────────────────────────────────────────

VARIANTS = ["SLIM+2SIG", "SLIM*ABS", "SLIM*1SIG"]
_SAFE_VARIANTS = [_re.sub(r'[^A-Za-z0-9_\-]', '_', v) for v in VARIANTS]
# → ["SLIM_2SIG", "SLIM_ABS", "SLIM_1SIG"]

DATASETS = ["toxicity", "concrete", "instanbul", "ppb",
            "resid_build_sale_price", "energy"]

# (column_header_label, pxo_tag_as_it_appears_in_sanitized_alg_name)
CONFIGS = [
    ("p_xo=0.0",        "pxo0_0"),
    ("p_xo=0.7, hd=17", "pxo0_7"),
]

# (output_stem_suffix, layout, model, node_size_attr, x_attr, y_attr)
FIGURE_JOBS = [
    ("stress_clustering_tree",  "stress",  "clustering", "TreeSize", "TreeSize", "Fitness"),
    ("fitness_clustering_node", "fitness", "clustering", "Count",    "TreeSize", "Fitness"),
    ("bivar_genotype_tree",     "bivar",   "genotype",   "TreeSize", "TreeSize", "Fitness"),
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _find_graph(infolder: str, safe_variant: str, config_tag: str, model: str):
    """
    Scan *infolder* for a pkl file whose stem contains *safe_variant*
    AND *config_tag* and whose model suffix matches *model*.

    stn_build.py names files {alg}_{model}_stn.pkl.
    Returns the loaded dict or None.
    """
    suffix = f"_{model}_stn.pkl"
    for fname in sorted(os.listdir(infolder)):
        if not fname.endswith(suffix):
            continue
        alg = fname[: -len(suffix)]
        if safe_variant in alg and config_tag in alg:
            with open(os.path.join(infolder, fname), "rb") as fh:
                return pickle.load(fh)
    return None


def _save_fig(fig, stem: str):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=150, bbox_inches="tight")
    tikzplotlib.save(stem + ".tex", figure=fig, strict=False)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(stem)}.png / .tex")


# ── FIGURE BUILDER ────────────────────────────────────────────────────────────

def make_stn_grid(benchmark: str,
                  layout: str, model: str,
                  size_attr: str, x_attr: str, y_attr: str,
                  stem_suffix: str):

    infolder = os.path.join(_STN_ROOT, benchmark)
    if not os.path.isdir(infolder):
        print(f"  [SKIP] {benchmark} — STN folder missing: {infolder}")
        return

    # ── Load graphs for every (variant, config) cell ─────────────────────────
    graphs: dict = {}   # (row, col) -> pkl dict
    for vi, (variant, safe_v) in enumerate(zip(VARIANTS, _SAFE_VARIANTS)):
        for ci, (col_label, cfg_tag) in enumerate(CONFIGS):
            d = _find_graph(infolder, safe_v, cfg_tag, model)
            if d is not None:
                graphs[(vi, ci)] = d
            else:
                print(f"    [MISSING] {variant} | {col_label} | {model}")

    if not graphs:
        print(f"  [SKIP] {benchmark} {stem_suffix} — no graphs found")
        return

    # ── Global scaling for node/edge sizes (consistent across all panels) ────
    all_tree, all_count, all_edge = [], [], []
    for d in graphs.values():
        G = d["G"]
        all_tree.extend(nx.get_node_attributes(G, "TreeSize").values())
        all_count.extend(nx.get_node_attributes(G, "Count").values())
        all_edge.extend(nx.get_edge_attributes(G, "Count").values() or [1])

    tree_range  = (min(all_tree),  max(all_tree))  if all_tree  else (0, 1)
    count_range = (min(all_count), max(all_count)) if all_count else (0, 1)
    edge_range  = (min(all_edge),  max(all_edge))  if all_edge  else (0, 1)
    size_range  = tree_range if size_attr == "TreeSize" else count_range

    # ── Per-row y-limits: the two side-by-side panels in each row share y ────
    # x is never shared — each panel auto-scales independently on x.
    row_y_limits = {}
    if layout in ("fitness", "bivar"):
        for vi in range(len(VARIANTS)):
            all_y_row = []
            for ci in range(len(CONFIGS)):
                if (vi, ci) in graphs:
                    G = graphs[(vi, ci)]["G"]
                    all_y_row.extend(nx.get_node_attributes(G, y_attr).values())
            if all_y_row:
                row_y_limits[vi] = (min(all_y_row), max(all_y_row))

    # ── Compute layouts once, reuse across the figure ─────────────────────────
    pos_cache = {
        k: _get_layout(d["G"], layout, x_attr=x_attr, y_attr=y_attr)
        for k, d in graphs.items()
    }

    # ── Build 3 × 2 figure ────────────────────────────────────────────────────
    # rect=[left, bottom, right, top] leaves room for row labels (left)
    # and the shared legend (bottom).
    fig, axes = plt.subplots(3, 2, figsize=(11, 15), squeeze=False)
    fig.suptitle(f"STN — {benchmark} | {layout}",
                 fontsize=13, fontweight="bold", y=0.99)

    # Column headers (annotated above the top-row panels)
    for ci, (col_label, _) in enumerate(CONFIGS):
        axes[0, ci].annotate(
            col_label,
            xy=(0.5, 1.06), xycoords="axes fraction",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
        )

    # Fill panels
    for vi in range(len(VARIANTS)):
        for ci in range(len(CONFIGS)):
            ax  = axes[vi, ci]
            key = (vi, ci)

            if key not in graphs:
                ax.text(0.5, 0.5, "no data",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                ax.axis("off")
                continue

            d = graphs[key]
            y_lim = row_y_limits.get(vi)   # shared per row; None for stress
            plot_stn(ax, d["G"], d["model"], "",
                     layout=layout, node_size_attr=size_attr,
                     size_range=size_range, edge_range=edge_range,
                     x_attr=x_attr, y_attr=y_attr,
                     x_limits=None,          # x never shared
                     y_limits=y_lim,         # y shared within row
                     fitness_limits=y_lim,   # also covers fitness layout
                     pos=pos_cache[key])
            ax.set_title("")    # clear the per-panel auto-title from plot_stn

    # Node-role legend at the bottom
    legend_handles = [
        mpatches.Patch(color=_NODE["Start"][0],  label="Start"),
        mpatches.Patch(color=_NODE["Medium"][0], label="Medium", alpha=0.45),
        mpatches.Patch(color=_NODE["End"][0],    label="End"),
        mpatches.Patch(color=_NODE["Best"][0],   label="Best"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               fontsize=FSIZE, framealpha=0.9, ncol=4,
               bbox_to_anchor=(0.55, 0.01))

    # Reserve left margin for row labels, bottom for legend
    fig.tight_layout(rect=[0.07, 0.04, 1.0, 0.97])

    # Row labels — added after tight_layout so panel positions are finalised
    for vi, variant in enumerate(VARIANTS):
        pos = axes[vi, 0].get_position()
        y_mid = (pos.y0 + pos.y1) / 2
        fig.text(0.01, y_mid, variant,
                 ha="left", va="center",
                 fontsize=10, fontweight="bold", rotation=90)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(_OUT_ROOT, exist_ok=True)
    stem = os.path.join(_OUT_ROOT, f"{benchmark}_stn_{stem_suffix}")
    _save_fig(fig, stem)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for benchmark in DATASETS:
        print(f"\n{'='*60}")
        print(f"  {benchmark}")
        print(f"{'='*60}")
        for stem_suffix, layout, model, size_attr, x_attr, y_attr in FIGURE_JOBS:
            print(f"\n  [{stem_suffix}]")
            make_stn_grid(benchmark, layout, model, size_attr, x_attr, y_attr, stem_suffix)

    print(f"\nDone. Figures written to {_OUT_ROOT}")
