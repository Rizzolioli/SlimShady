"""
Held-out-split evaluation for the pop_size in {200, 100}/n_gens=5000
no-rotation sweep (main/main_tabgpgo_tabpfn_norotate_popsweep.py): TabGPGO
(fine-tuned AND zero-shot) vs. TabPFN V2/V3 (real-adapted AND zero-shot), on
the same 5x(80/20 split) protocol run_longrun_split_eval.py /
run_500pop_split_eval.py use.

TabPFN is NOT re-fit here. eval_tabpfn/eval_tabpfn_zeroshot only ever touch
the real dataset's own (X_train, X_test, y_train, y_test) for a given
(dataset, split) -- they never read the TabGPGO elite/config at all, so
their results are identical across experiments run on the same 6 datasets
with the same N_SPLITS/P_TEST/seed protocol. This module instead reuses the
TabPFN v2/v3 (real-adapted) and TabPFN v2/v3 (zero-shot) rows straight out
of pop500_split_eval_results.csv (produced by run_500pop_split_eval.py) and
only computes the NEW rows this sweep actually needs: tabgpgo_finetuned/
tabgpgo_zeroshot against the pop-sweep's own best-per-dataset elite (which
may come from a different (pop_size, ms) combo than pop500's winner).

Standard SLIM is NOT re-run here either -- see run_longrun_split_eval.py's
own module docstring for why (already-collected data, not tied to which
TabGPGO experiment it's compared against).

Usage:
    python tabpfn_baseline/run_popsweep_split_eval.py
"""
import csv
import os
import sys

import pandas as pd

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "main"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.data_loader import load_merged_data
from main_tabgpgo_funcset import CONSTANT_SETS, FUNCTION_SETS, function_constant_set
from run_longrun_split_eval import (_tag, eval_tabgpgo, fit_reducer,
                                    load_elite_tabpfn, load_reference_model,
                                    split_features)
from tabgpgo.config import TabGPGOConfig
from tabgpgo.tree_pool import make_terminals
from utils.utils import train_test_split

VAL_DATASETS = ["ppb", "toxicity", "resid_build_sale_price", "instanbul", "energy", "concrete"]
N_SPLITS = 5
P_TEST = 0.2

FUNCTION_SET_NAME = "full_extended"
CONSTANT_SET_NAME = "small_ints"

REUSED_TABPFN_ALGOS = ["tabpfn_v2", "tabpfn_v3", "tabpfn_v2_zeroshot", "tabpfn_v3_zeroshot"]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_norotate_popsweep_runs")
RESULTS_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_norotate_popsweep_results.csv")
POP500_RESULTS_CSV = os.path.join(THIS_DIR, "log", "pop500_split_eval_results.csv")
OUT_CSV = os.path.join(THIS_DIR, "log", "popsweep_split_eval_results.csv")
MEDIAN_CSV = os.path.join(THIS_DIR, "log", "popsweep_split_eval_median.csv")

BASE_COLS = ["algo", "run_id", "dataset", "seed", "generation", "elite_train_rmse", "time_s", "population_nodes"]
VAL_COLS = [f"val_{d}_{suf}" for d in VAL_DATASETS for suf in ("rmse_scaled", "rmse_raw", "r2")]
ELITE_COLS = ["elite_size", "elite_nodes", "elite_train_r2"]
RESULT_COLS = BASE_COLS + VAL_COLS + ELITE_COLS


def best_per_dataset():
    """Single results file spanning all 8 (pop_size, ms) combos -- last-
    logged-generation, best val_r2 per dataset, across every combo."""
    df = pd.read_csv(RESULTS_CSV, header=None, names=RESULT_COLS)
    last = df[df.generation == df.generation.max()]
    best = {}
    for dataset in VAL_DATASETS:
        row = last.loc[last[f"val_{dataset}_r2"].idxmax()]
        best[dataset] = {"algo": row["algo"], "run_id": row["run_id"]}
    return best


def load_reused_tabpfn_rows():
    df = pd.read_csv(POP500_RESULTS_CSV)
    reused = df[df["algo"].isin(REUSED_TABPFN_ALGOS)]
    return reused.to_dict("records")


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    best = best_per_dataset()
    print("popsweep best per dataset:", {k: v["algo"] for k, v in best.items()})

    reference_model, embed_dim = load_reference_model()
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]

    rows = load_reused_tabpfn_rows()
    print(f"reused {len(rows)} TabPFN v2/v3 (real-adapted + zero-shot) rows from "
         f"{os.path.basename(POP500_RESULTS_CSV)} -- not re-fit")

    with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
        TERMINALS = make_terminals(embed_dim)
        for dataset in VAL_DATASETS:
            X_raw, y_raw = load_merged_data(dataset, X_y=True)
            X_raw, y_raw = X_raw.float(), y_raw.float()

            run_dir = os.path.join(RUN_DIR_BASE, f"{best[dataset]['run_id']}_{_tag(best[dataset]['algo'])}_0")
            cfg, registry, ind = load_elite_tabpfn(run_dir)

            for split in range(N_SPLITS):
                X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, p_test=P_TEST, seed=split)
                reducer_meta = fit_reducer(X_train, y_train, cfg)
                Xtr100, ytr_z, Xte100, y_stats = split_features(X_train, y_train, X_test, y_test, cfg, reducer_meta)

                for algo, fine_tune in (("tabgpgo_finetuned", True), ("tabgpgo_zeroshot", False)):
                    tg_rmse, tg_r2 = eval_tabgpgo(ind, registry, reference_model, TERMINALS,
                                                  Xtr100, ytr_z, Xte100, y_test, y_stats, fine_tune)
                    rows.append({"dataset": dataset, "algo": algo, "split": split,
                                "rmse_raw": tg_rmse, "r2": tg_r2})
                    print(f"  [{dataset}/split{split}/{algo}] rmse={tg_rmse:.4f} r2={tg_r2:.4f}")
            print(f"[{dataset}] done")

    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "algo", "split", "rmse_raw", "r2"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT_CSV}")

    df = pd.DataFrame(rows)
    med = df.groupby(["dataset", "algo"])[["rmse_raw", "r2"]].median().reset_index()
    med.to_csv(MEDIAN_CSV, index=False)
    print(f"wrote medians -> {MEDIAN_CSV}")
    print(med.to_string(index=False))


if __name__ == "__main__":
    main()
