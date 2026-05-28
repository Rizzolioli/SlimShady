"""
stn_prep.py
Convert SlimShady log=8 output files into STN-ready CSVs for stn_build.py.

log=8 writes two files:
  {log_path}.csv          — per-generation elite metrics (standard experiment log)
  {log_path}_sem_gen.csv  — elite genotype + semantics, written only when elite changes

_sem_gen.csv column layout  (no header):
  0: algo
  1: run_id
  2: dataset
  3: seed
  4: generation
  5: tree_repr       (genotype string)
  6: train_sem       (space-separated floats — the phenotype vector)
  7: test_sem        (space-separated floats, or "None")

Output: one CSV per unique algo in data/{benchmark}/{algo}.csv
  Row format (no header): Run, Iter, Fitness, genotype_string, sem_0, sem_1, ...

These files feed directly into stn_build.py.
"""

import os
import re
import numpy as np
import pandas as pd

# ── COLUMN MAPS ───────────────────────────────────────────────────────────────
_SEM_COLS = {0: 'algo', 1: 'run_id', 2: 'dataset', 3: 'seed',
             4: 'gen',  5: 'tree_repr', 6: 'train_sem', 7: 'test_sem'}
_LOG_COLS = {0: 'algo', 1: 'run_id', 2: 'dataset', 3: 'seed',
             4: 'gen',  5: 'train_fit', 6: 'timing',   7: 'nodes',
             8: 'test_fit', 9: 'nodes_count', 10: 'log_level'}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _parse_sem_col(series: pd.Series) -> np.ndarray:
    """
    Convert a column of space-separated float strings to a (n_rows × n_sem) array.
    Each entry looks like "1.23 4.56 7.89 ..."
    """
    split = series.str.split(expand=False)
    n_sem = len(split.iloc[0])
    mat = np.array(split.tolist(), dtype=float)   # (n, n_sem)
    return mat, n_sem


def _safe_algo_name(algo: str) -> str:
    """Sanitise algo string for use as a filename."""
    return re.sub(r'[^A-Za-z0-9_\-]', '_', algo)


# ── MAIN PREP FUNCTION ────────────────────────────────────────────────────────

def prep_stn_data(log_csv: str,
                  benchmark: str,
                  out_root:  str  = "data",
                  nruns:     int  = 10):
    """
    Prepare STN input CSVs for a single benchmark from a SlimShady log=8 run.

    Parameters
    ----------
    log_csv   : path to the main experiment log CSV
                (e.g. main/log/results_prob_xo_12052026.csv)
    benchmark : dataset name to extract (e.g. "istanbul")
    out_root  : output root folder; files are written to {out_root}/{benchmark}/
    nruns     : max seed index to include (seeds 0..nruns-1 → Run 1..nruns)
    """
    sem_csv = (log_csv[:-4] if log_csv.endswith('.csv') else log_csv) + '_sem_gen.csv'

    if not os.path.exists(sem_csv):
        raise FileNotFoundError(
            f"sem_gen file not found: {sem_csv}\n"
            f"Make sure you ran the experiment with log=8."
        )

    print(f"Loading {os.path.basename(log_csv)} …")
    df_main = pd.read_csv(log_csv, header=None).rename(columns=_LOG_COLS)
    print(f"Loading {os.path.basename(sem_csv)} …")
    df_sem  = pd.read_csv(sem_csv, header=None).rename(columns=_SEM_COLS)

    df_main['seed'] = df_main['seed'].astype(int)
    df_sem['seed']  = df_sem['seed'].astype(int)

    # Filter to requested benchmark
    df_main = df_main[df_main['dataset'] == benchmark].copy()
    df_sem  = df_sem[ df_sem['dataset']  == benchmark].copy()

    if df_sem.empty:
        print(f"  No sem_gen data found for dataset '{benchmark}' — nothing to write.")
        return

    # Only use runs with seed < nruns
    df_sem  = df_sem[df_sem['seed']  < nruns].copy()
    df_main = df_main[df_main['seed'] < nruns].copy()

    # ── Join fitness from main log ────────────────────────────────────────────
    # Main log has every generation; sem_gen only has elite-change generations.
    fit_lookup = (df_main
                  .drop_duplicates(subset=['algo', 'seed', 'gen'])
                  .set_index(['algo', 'seed', 'gen'])['train_fit'])

    def _get_fit(row):
        return fit_lookup.get((row['algo'], row['seed'], row['gen']), float('nan'))

    df_sem['Fitness'] = df_sem.apply(_get_fit, axis=1)
    n_missing = df_sem['Fitness'].isna().sum()
    if n_missing:
        print(f"  Warning: {n_missing} rows have no fitness match — they will be dropped.")
        df_sem = df_sem.dropna(subset=['Fitness'])

    # ── Expand train_sem string → float columns ───────────────────────────────
    sem_mat, n_sem = _parse_sem_col(df_sem['train_sem'])
    sem_col_names  = [f'sem_{i}' for i in range(n_sem)]
    sem_df = pd.DataFrame(sem_mat, columns=sem_col_names, index=df_sem.index)

    df_work = pd.concat([
        df_sem[['algo', 'seed', 'gen', 'tree_repr', 'Fitness']].reset_index(drop=True),
        sem_df.reset_index(drop=True),
    ], axis=1)

    df_work['Run'] = df_work['seed'] + 1   # 1-indexed runs for STN code

    # ── Write one CSV per algo ────────────────────────────────────────────────
    out_dir = os.path.join(out_root, benchmark)
    os.makedirs(out_dir, exist_ok=True)

    out_col_order = ['Run', 'gen', 'Fitness', 'tree_repr'] + sem_col_names

    for algo, grp in df_work.groupby('algo', sort=True):
        grp_sorted = (grp.sort_values(['seed', 'gen'])
                        [out_col_order])
        fname = _safe_algo_name(algo) + '.csv'
        fpath = os.path.join(out_dir, fname)
        grp_sorted.to_csv(fpath, index=False, header=False)

        n_runs_found = grp['seed'].nunique()
        print(f"  {algo}: {n_runs_found} runs, {len(grp_sorted)} rows → {fname}")

    print(f"\nDone. Data written to {out_dir}/")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os

    _HERE    = os.path.dirname(os.path.abspath(__file__))
    LOG_CSV  = os.path.join(_HERE, "..", "log", "results_prob_xo_12052026.csv")
    OUT_ROOT = os.path.join(_HERE, "..", "log", "stn_data")
    NRUNS    = 5   # seeds 0..4 → Run 1..5

    BENCHMARKS = ["toxicity", "concrete", "instanbul", "ppb",
                  "resid_build_sale_price", "energy"]

    for benchmark in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"  Preparing STN data: {benchmark}")
        print(f"{'='*60}")
        prep_stn_data(LOG_CSV, benchmark, out_root=OUT_ROOT, nruns=NRUNS)
