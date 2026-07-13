"""
Quick local sanity check for the TabGPGO initial-population convex-hull
study (see chull_common.py for the method/math). Reduced synthetic prior
(20 datasets x 500 rows = 10,000 training rows) and a 2,000-row subsample for
the LP, so this runs on a laptop in well under a minute after the first
(cache-building) run. For the full-scale estimate, run chull_init_full.py
instead -- on a bigger/GPU machine, not here.

Usage:
    python tabgpgo_chull/chull_init_local.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main"))

from chull_common import run_study

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(THIS_DIR, "log", "local")

N_SYNTH_DATASETS = 20
LP_SAMPLE_SIZE = 2000
SEED = 0

if __name__ == "__main__":
    run_study(LOG_DIR, n_synth_datasets=N_SYNTH_DATASETS, lp_sample_size=LP_SAMPLE_SIZE, seed=SEED)
