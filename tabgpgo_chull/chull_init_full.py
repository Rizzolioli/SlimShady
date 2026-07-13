"""
Full-scale TabGPGO initial-population convex-hull study (see
chull_common.py for the method/math). Uses TabGPGOConfig's own default
synthetic prior (1000 datasets x 500 rows = 500,000 training rows) with NO
row-subsampling for the LP -- every training row is a constraint. This is
sized for a bigger/GPU machine, not a laptop: the LP alone has
pop_size + 2*n_train variables (~1,000,100 at these defaults) and
n_train + 1 equality constraints. Run chull_init_local.py first to sanity
check the method on a reduced scale.

Usage:
    python tabgpgo_chull/chull_init_full.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main"))

from chull_common import run_study

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(THIS_DIR, "log", "full")

N_SYNTH_DATASETS = None  # TabGPGOConfig's own default (1000 x 500 = 500,000 rows)
LP_SAMPLE_SIZE = None    # no subsampling -- solve against every training row
SEED = 0

if __name__ == "__main__":
    run_study(LOG_DIR, n_synth_datasets=N_SYNTH_DATASETS, lp_sample_size=LP_SAMPLE_SIZE, seed=SEED)
