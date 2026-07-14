"""
TabGPGO fine-tuned vs. zero-shot: fresh-pool vs. no-fresh-pool, best-per-dataset.

Same 5x(80/20 split) protocol as run_split_eval.py, applied to the two SLIM*MIX
elites picked as "best per dataset" by zero-shot val R2 in
scratchpad/build_freshpool_data.py (one fresh-pool combo, one no-fresh-pool
(static pool) combo per dataset -- see FRESHPOOL_BEST / NOFRESH_BEST below,
copied from that script's printed comparison). This lets the fresh-pool vs.
no-fresh-pool dashboard comparison use the same fine-tuned/held-out protocol
TabPFN V2/V3 already use, instead of mixing zero-shot TabGPGO against
held-out TabPFN.

For each of the 6 real datasets, over 5 independent 80/20 splits
(utils.utils.train_test_split, seed = split index):
  - fresh-pool best elite: TabGPGOPredictor.fit(X_train, y_train,
    fine_tune=True) (per-block weights re-optimized on the split; tree
    structures/head frozen), then .predict(X_test).
  - no-fresh-pool (static pool) best elite: identical protocol.
Median RMSE (raw units) / R^2 across the 5 splits is reported, same feature
reducer (fit fresh per split, train-only) as run_split_eval.py.

Usage:
    python tabpfn_baseline/run_freshpool_finetune_eval.py
"""
import csv
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from datasets.data_loader import load_merged_data
from evaluators.fitness_functions import r2, rmse
from tabgpgo.inference import TabGPGOPredictor, load_run
from tabgpgo.preprocessing import reduce_features
from utils.utils import train_test_split

VAL_DATASETS = ["ppb", "toxicity", "resid_build_sale_price", "instanbul", "energy", "concrete"]
N_SPLITS = 5
P_TEST = 0.2
FT_STEPS = 200
FT_LR = 0.05

FRESHPOOL_RUN_ID = "7f4eed66-7ecc-11f1-bf6a-2c7ba0e6fa90"
FRESHPOOL_RUNS_DIR = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_freshpool_runs")
MAIN_RUN_ID = "4ca35ef8-7b63-11f1-8aa7-2c7ba0e6fa90"
MAIN_RUNS_DIR = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_runs")

# best (ms, patience) fresh-pool combo / best ms no-fresh-pool combo per
# dataset, picked by zero-shot val R2 at generation 200 -- see
# scratchpad/build_freshpool_data.py's "comparison" list.
FRESHPOOL_BEST = {
    "ppb": ("ms10", "pat10"), "toxicity": ("oms", "pat5"),
    "resid_build_sale_price": ("ms10", "pat10"), "instanbul": ("ms10", "pat50"),
    "energy": ("ms10", "pat10"), "concrete": ("oms", "pat50"),
}
NOFRESH_BEST = {
    "ppb": "oms", "toxicity": "ms1", "resid_build_sale_price": "oms",
    "instanbul": "ms10", "energy": "ms1", "concrete": "oms",
}

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(THIS_DIR, "log", "freshpool_finetune_results.csv")
MEDIAN_CSV = os.path.join(THIS_DIR, "log", "freshpool_finetune_median.csv")


def freshpool_run_dir(ms, pat):
    return os.path.join(FRESHPOOL_RUNS_DIR, f"{FRESHPOOL_RUN_ID}_SLIMxMIX_{ms}_{pat}_0")


def main_run_dir(ms):
    return os.path.join(MAIN_RUNS_DIR, f"{MAIN_RUN_ID}_SLIMxMIX_{ms}_0")


def fit_reducer(X_train_raw, y_train_raw, cfg):
    if X_train_raw.shape[1] > cfg.max_features:
        _, meta = reduce_features(X_train_raw, y_train_raw, cfg.reducer, cfg.max_features, cfg.val_seed)
        return meta
    return None


def eval_tabgpgo(elite, registry, model, cfg, X_train_raw, y_train_raw, X_test_raw, y_test_raw, reducer_meta):
    predictor = TabGPGOPredictor(elite, registry, model, cfg, reducer_meta=reducer_meta)
    predictor.fit(X_train_raw, y_train_raw, fine_tune=True, steps=FT_STEPS, lr=FT_LR)
    pred = predictor.predict(X_test_raw)
    return float(rmse(y_test_raw, pred)), float(r2(y_test_raw, pred))


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    rows = []
    for dataset in VAL_DATASETS:
        X_raw, y_raw = load_merged_data(dataset, X_y=True)
        X_raw, y_raw = X_raw.float(), y_raw.float()

        ms, pat = FRESHPOOL_BEST[dataset]
        fp_cfg, fp_model, fp_registry, fp_elite, *_ = load_run(freshpool_run_dir(ms, pat))
        nf_ms = NOFRESH_BEST[dataset]
        nf_cfg, nf_model, nf_registry, nf_elite, *_ = load_run(main_run_dir(nf_ms))

        for split in range(N_SPLITS):
            X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, p_test=P_TEST, seed=split)
            reducer_meta = fit_reducer(X_train, y_train, fp_cfg)

            for algo, (cfg, model_, registry, elite) in (
                ("freshpool", (fp_cfg, fp_model, fp_registry, fp_elite)),
                ("nofreshpool", (nf_cfg, nf_model, nf_registry, nf_elite)),
            ):
                rmse_, r2_ = eval_tabgpgo(elite, registry, model_, cfg,
                                         X_train, y_train, X_test, y_test, reducer_meta)
                rows.append({"dataset": dataset, "algo": algo, "split": split, "rmse_raw": rmse_, "r2": r2_})
                print(f"  [{dataset}/split{split}/{algo}] rmse={rmse_:.4f} r2={r2_:.4f}")
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
