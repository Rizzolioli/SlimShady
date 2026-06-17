"""
run_stn_test_fitness.py
=======================
Full pipeline to produce STN grid figures using TEST fitness as the
node-fitness attribute (same graph topology as the train-fitness STNs —
the search trajectory is unchanged — but Fitness values on nodes reflect
generalisation performance instead of training performance).

Steps
-----
1. stn_prep  : extract elite-change trajectories from the log=8 sem_gen CSV,
               joining TEST fitness from the main log  →  stn_data_test/
2. stn_build : build genotype / hypercube / clustering graphs  →  stns_test/
3. plot      : 3-row x 2-col grid figures (SLIM+2SIG, SLIM*ABS, SLIM*1SIG
               vs p_xo=0.0 | p_xo=0.7)  →  latex/stns_test/

Outputs land in sibling folders of the train-fitness pipeline so both
versions coexist without overwriting each other:
  main/log/stn_data_test/   (intermediate CSVs)
  main/log/stns_test/       (pkl graphs)
  main/log/latex/stns_test/ (PNG + TikZ figures)
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from stn_prep  import prep_stn_data
from stn_build import process_folder
from generate_stn_grids import make_stn_grid, DATASETS, FIGURE_JOBS

# ── CONFIG ────────────────────────────────────────────────────────────────────

LOG_CSV_BASELINE  = os.path.join(_HERE, "..", "log", "results_prob_xo_12052026.csv")
LOG_CSV_TREATMENT = os.path.join(_HERE, "..", "log", "results_pop_xo_15052026.csv")
DATA_ROOT  = os.path.join(_HERE, "..", "log", "stn_data_test")
STN_ROOT   = os.path.join(_HERE, "..", "log", "stns_test")
OUT_ROOT   = os.path.join(_HERE, "..", "log", "latex", "stns_test")

NRUNS      = 5
N_CLUSTERS = 50
N_WORKERS  = 2

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _banner(step, benchmark):
    print(f"\n{'='*60}")
    print(f"  [{step}]  {benchmark}")
    print(f"{'='*60}", flush=True)


def _plot_benchmark(benchmark):
    for stem_suffix, layout, model, size_attr, x_attr, y_attr in FIGURE_JOBS:
        make_stn_grid(benchmark, layout, model, size_attr, x_attr, y_attr,
                      stem_suffix, stn_root=STN_ROOT, out_root=OUT_ROOT)
    return benchmark

# ── PIPELINE ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Step 1 + 2: prep and build (sequential — shared I/O) ─────────────────
    for benchmark in DATASETS:
        _banner("PREP (test fitness)", benchmark)
        prep_stn_data(LOG_CSV_BASELINE,  benchmark,
                      out_root=DATA_ROOT, nruns=NRUNS, fitness_col='test_fit')
        prep_stn_data(LOG_CSV_TREATMENT, benchmark,
                      out_root=DATA_ROOT, nruns=NRUNS, fitness_col='test_fit')

        _banner("BUILD", benchmark)
        process_folder(benchmark,
                       data_root=DATA_ROOT,
                       stn_root=STN_ROOT,
                       nruns=NRUNS,
                       n_clusters=N_CLUSTERS,
                       build_clustering=True)

    # ── Step 3: plot (parallel across benchmarks) ────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [PLOT]  {len(DATASETS)} benchmarks  x  {len(FIGURE_JOBS)} figure types")
    print(f"{'='*60}", flush=True)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_plot_benchmark, b): b for b in DATASETS}
        for fut in as_completed(futures):
            b = futures[fut]
            try:
                fut.result()
                print(f"  [PLOT]  {b} done", flush=True)
            except Exception as exc:
                print(f"  [PLOT]  {b} FAILED: {exc}", flush=True)

    print(f"\nAll done. Figures -> {OUT_ROOT}")
