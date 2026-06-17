"""
run_stn.py
End-to-end STN pipeline: prep → build → plot, for all benchmarks.

Steps
-----
1. stn_prep  : convert log=8 CSVs → per-algo STN input CSVs
2. stn_build : build genotype / hypercube / clustering STN graphs → .pkl
3. stn_plot  : render combined PNG figures for each layout

Prep and build run sequentially (they share the same log CSV and write to
shared folders). Plot runs in parallel across benchmarks — each benchmark's
figures are fully independent.

Edit the CONFIG block below before running.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Baseline (p_xo=0.0): from the prob_xo experiment (pop=100, n_iter=2000)
LOG_CSV_BASELINE  = os.path.join(_HERE, "..", "log", "results_prob_xo_12052026.csv")
# Treatment (pop=500, n_iter=400, p_xo=0.7): from the pop_xo experiment
LOG_CSV_TREATMENT = os.path.join(_HERE, "..", "log", "results_pop_xo_15052026.csv")

DATA_ROOT  = os.path.join(_HERE, "..", "log", "stn_data")
STN_ROOT   = os.path.join(_HERE, "..", "log", "stns")
PLOT_ROOT  = os.path.join(_HERE, "..", "log", "figs", "stns")

BENCHMARKS = ["toxicity", "concrete", "instanbul", "ppb",
              "resid_build_sale_price", "energy"]

NRUNS      = 5    # seeds 0..4 → Run 1..5
N_CLUSTERS = 50
N_WORKERS  = 2   # one worker per benchmark at most

# Layouts to plot (node_sizes 'tree' + 'node' produced for each)
LAYOUTS = ["stress", "fitness"]   # bivar handled separately (split by model)

# ── IMPORTS ───────────────────────────────────────────────────────────────────

from stn_prep  import prep_stn_data
from stn_build import process_folder
from stn_plot  import combined_plot

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _banner(step, benchmark):
    print(f"\n{'='*60}")
    print(f"  [{step}]  {benchmark}")
    print(f"{'='*60}", flush=True)


def _plot_benchmark(benchmark):
    """Plot all layouts for one benchmark (runs in a worker process)."""
    for layout in LAYOUTS:
        combined_plot(benchmark,
                      stn_root=STN_ROOT,
                      plot_root=PLOT_ROOT,
                      layout=layout,
                      node_sizes=('tree', 'node'),
                      ncols=3,
                      x_attr='TreeSize',
                      y_attr='Fitness')

    # bivar: genotype + hypercube → TreeSize vs Fitness
    combined_plot(benchmark,
                  stn_root=STN_ROOT,
                  plot_root=PLOT_ROOT,
                  layout='bivar',
                  node_sizes=('tree', 'node'),
                  ncols=2,
                  x_attr='TreeSize',
                  y_attr='Fitness',
                  models=('genotype', 'hypercube'))

    # bivar: clustering → ClusterID vs Fitness
    combined_plot(benchmark,
                  stn_root=STN_ROOT,
                  plot_root=PLOT_ROOT,
                  layout='bivar',
                  node_sizes=('tree', 'node'),
                  ncols=1,
                  x_attr='ClusterID',
                  y_attr='Fitness',
                  models=('clustering',))

    return benchmark

# ── STEP 1 + 2: Prep & Build (sequential — shared I/O) ───────────────────────

if __name__ == '__main__':
    for benchmark in BENCHMARKS:
        _banner("PREP", benchmark)
        prep_stn_data(LOG_CSV_BASELINE,  benchmark, out_root=DATA_ROOT, nruns=NRUNS)
        prep_stn_data(LOG_CSV_TREATMENT, benchmark, out_root=DATA_ROOT, nruns=NRUNS)

        _banner("BUILD", benchmark)
        process_folder(benchmark,
                       data_root=DATA_ROOT,
                       stn_root=STN_ROOT,
                       nruns=NRUNS,
                       n_clusters=N_CLUSTERS,
                       build_clustering=True)

    # ── STEP 3: Plot (parallel across benchmarks) ─────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  [PLOT]  launching {N_WORKERS} workers for {len(BENCHMARKS)} benchmarks")
    print(f"{'='*60}", flush=True)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_plot_benchmark, b): b for b in BENCHMARKS}
        for fut in as_completed(futures):
            b = futures[fut]
            try:
                fut.result()
                print(f"  [PLOT]  {b} done", flush=True)
            except Exception as exc:
                print(f"  [PLOT]  {b} FAILED: {exc}", flush=True)

    print("\nAll benchmarks done.")
