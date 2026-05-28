"""
run_stn.py
End-to-end STN pipeline: prep → build → plot, for all benchmarks.

Steps
-----
1. stn_prep  : convert log=8 CSVs → per-algo STN input CSVs
2. stn_build : build genotype / hypercube / clustering STN graphs → .pkl
3. stn_plot  : render combined PNG figures for each layout

Edit the CONFIG block below before running.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

# ── CONFIG ────────────────────────────────────────────────────────────────────

LOG_CSV    = os.path.join(_HERE, "..", "log", "results_prob_xo_12052026.csv")
DATA_ROOT  = os.path.join(_HERE, "..", "log", "stn_data")
STN_ROOT   = os.path.join(_HERE, "..", "log", "stns")
PLOT_ROOT  = os.path.join(_HERE, "..", "log", "figs", "stns")

BENCHMARKS = ["toxicity", "concrete", "instanbul", "ppb",
              "resid_build_sale_price", "energy"]

NRUNS       = 5     # seeds 0..4 → Run 1..5
N_CLUSTERS  = 50

# Layout combinations to plot: (layout, node_size, x_attr, y_attr)
PLOT_CONFIGS = [
    ("stress",  "tree",  "TreeSize", "Fitness"),
    ("fitness", "tree",  "TreeSize", "Fitness"),
    ("bivar",   "tree",  "TreeSize", "Fitness"),
]

# ── IMPORTS ───────────────────────────────────────────────────────────────────

from stn_prep  import prep_stn_data
from stn_build import process_folder
from stn_plot  import combined_plot

# ── PIPELINE ─────────────────────────────────────────────────────────────────

def _banner(step, benchmark):
    print(f"\n{'='*60}")
    print(f"  [{step}]  {benchmark}")
    print(f"{'='*60}")


for benchmark in BENCHMARKS:

    # ── Step 1: Prep ─────────────────────────────────────────────────────────
    _banner("PREP", benchmark)
    prep_stn_data(LOG_CSV, benchmark, out_root=DATA_ROOT, nruns=NRUNS)

    # ── Step 2: Build ────────────────────────────────────────────────────────
    _banner("BUILD", benchmark)
    process_folder(benchmark,
                   data_root=DATA_ROOT,
                   stn_root=STN_ROOT,
                   nruns=NRUNS,
                   n_clusters=N_CLUSTERS,
                   build_clustering=True)

    # ── Step 3: Plot ─────────────────────────────────────────────────────────
    _banner("PLOT", benchmark)
    for layout, node_size, x_attr, y_attr in PLOT_CONFIGS:
        combined_plot(benchmark,
                      stn_root=STN_ROOT,
                      plot_root=PLOT_ROOT,
                      layout=layout,
                      node_size=node_size,
                      ncols=3,
                      x_attr=x_attr,
                      y_attr=y_attr)

print("\nAll benchmarks done.")
