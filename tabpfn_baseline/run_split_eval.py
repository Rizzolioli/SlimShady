"""
Held-out-split evaluation: TabGPGO (fine-tuned) vs. TabPFN V2/V3.

Supersedes run_tabpfn.py's full-unsplit protocol. For each of the 6 real
datasets, over 5 independent 80/20 splits (utils.utils.train_test_split,
seed = split index):
  - TabGPGO: TabGPGOPredictor.fit(X_train, y_train, fine_tune=True) (per-block
    weights re-optimized against the training split; tree structures/head
    stay frozen), then .predict(X_test).
  - TabPFN v2 / v3: .fit(X_train, y_train), .predict(X_test), fed the same
    PCA-100 representation TabGPGO itself uses (reduce_features + standardize
    -- fit on the training split only, applied to the test split, so neither
    the feature reducer nor any z-score statistic ever sees test rows).
Median RMSE (raw units) / R^2 across the 5 splits is what the dashboards
report -- a single lucky/unlucky split shouldn't drive the headline number.

Two TabGPGO "best per dataset" elites are evaluated, matching the two
published dashboards' own sweeps:
  - "main": the 36-algo sweep (main/log/tabgpgo_runs, pop=100/gens=200).
  - "hpt": the 31-combo p_inflate/n_elites sweep (main/log/tabgpgo_hpt_runs,
    pop=50/gens=200/ms=oms) -- filtered to tabgpgo_hpt_manifest.csv's valid,
    deduplicated run_ids (the raw results CSV also contains stale/partial
    runs from an earlier merge bug; the manifest is the source of truth for
    which run_ids are real, complete combos).
TabPFN doesn't depend on which TabGPGO sweep it's being compared against, so
it's computed once per (dataset, split) and reused for both dashboards' rows.

Usage:
    python tabpfn_baseline/run_split_eval.py
"""
import csv
import os
import sys

import numpy as np
import pandas as pd
import torch

# concrete (1030 rows) trips TabPFN's default CPU safeguard.
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from datasets.data_loader import load_merged_data
from evaluators.fitness_functions import r2, rmse
from tabgpgo.inference import TabGPGOPredictor, load_run
from tabgpgo.preprocessing import reduce_features, standardize_scale_pad, zscore
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from utils.utils import train_test_split

VAL_DATASETS = ["ppb", "toxicity", "resid_build_sale_price", "instanbul", "energy", "concrete"]
N_SPLITS = 5
P_TEST = 0.2
FT_STEPS = 200
FT_LR = 0.05

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_RESULTS_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_results.csv")
MAIN_RUNS_DIR = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_runs")
HPT_RESULTS_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_hpt_results.csv")
HPT_MANIFEST_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_hpt_manifest.csv")
HPT_RUNS_DIR = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_hpt_runs")
OUT_CSV = os.path.join(THIS_DIR, "log", "split_eval_results.csv")
MEDIAN_CSV = os.path.join(THIS_DIR, "log", "split_eval_median.csv")

COLS = (
    ["algo", "run_id", "dataset", "seed", "generation", "elite_train_rmse", "time_s", "population_nodes"]
    + [f"val_{d}_{suf}" for d in VAL_DATASETS for suf in ("rmse_scaled", "rmse_raw", "r2")]
    + ["elite_size", "elite_nodes", "elite_train_r2"]
)


def _tag(algo):
    return algo.replace("*", "x").replace("~", "t")


def _best_per_dataset(results_csv, group_col, valid_run_ids=None):
    """Last-logged-generation row per group_col, best val_r2 per dataset."""
    df = pd.read_csv(results_csv, header=None, names=COLS)
    if valid_run_ids is not None:
        df = df[df.run_id.isin(valid_run_ids)]
    last = df.sort_values("generation").groupby(group_col, as_index=False).tail(1)
    best = {}
    for dataset in VAL_DATASETS:
        row = last.loc[last[f"val_{dataset}_r2"].idxmax()]
        best[dataset] = {"algo": row["algo"], "run_id": row["run_id"]}
    return best


def best_main_per_dataset():
    # All 36 main-sweep algos share ONE run_id (a single multi-algo batch),
    # so "algo" -- not run_id -- is the unique key here.
    return _best_per_dataset(MAIN_RESULTS_CSV, group_col="algo")


def best_hpt_per_dataset():
    # HPT combos have unique run_ids but only 3 possible "algo" labels
    # (variant+"_oms"), so run_id is the correct key here -- filtered to the
    # manifest's validated combos, since the raw results CSV also contains
    # stale/partial run_ids left over from an earlier merge bug.
    manifest = pd.read_csv(HPT_MANIFEST_CSV)
    return _best_per_dataset(HPT_RESULTS_CSV, group_col="run_id", valid_run_ids=set(manifest.run_id))


def load_elite(runs_dir, run_id, algo):
    run_dir = os.path.join(runs_dir, f"{run_id}_{_tag(algo)}_0")
    return load_run(run_dir)


def fit_reducer(X_train_raw, y_train_raw, cfg):
    if X_train_raw.shape[1] > cfg.max_features:
        _, meta = reduce_features(X_train_raw, y_train_raw, cfg.reducer, cfg.max_features, cfg.val_seed)
        return meta
    return None


def reduce_with(X_raw, reducer_meta):
    if reducer_meta is None:
        return X_raw
    if reducer_meta["method"] == "rf":
        return X_raw[:, reducer_meta["columns"]]
    Xs, _, _ = zscore(X_raw, *reducer_meta["x_stats"])
    return torch.from_numpy(reducer_meta["pca"].transform(Xs.numpy())).float()


def tabpfn_split_features(X_train_raw, y_train_raw, X_test_raw, y_test_raw, cfg, reducer_meta):
    Xtr = reduce_with(X_train_raw, reducer_meta)
    Xte = reduce_with(X_test_raw, reducer_meta)
    Xtr100, ytr_z, meta = standardize_scale_pad(Xtr, y_train_raw, cfg.max_features)
    Xte100, _, _ = standardize_scale_pad(Xte, y_test_raw, cfg.max_features,
                                         x_stats=meta["x_stats"], y_stats=meta["y_stats"])
    return Xtr100, ytr_z, Xte100, meta["y_stats"]


def eval_tabpfn(version_enum, Xtr100, ytr_z, Xte100, y_test_raw, y_stats):
    model = TabPFNRegressor.create_default_for_version(version_enum)
    model.fit(Xtr100.numpy().astype(np.float32), ytr_z.numpy().astype(np.float32))
    pred_z = torch.as_tensor(np.asarray(model.predict(Xte100.numpy().astype(np.float32)))).float()
    mean, std = y_stats
    pred_raw = pred_z * std.squeeze() + mean.squeeze()
    return float(rmse(y_test_raw, pred_raw)), float(r2(y_test_raw, pred_raw))


def eval_tabgpgo(elite, registry, model, cfg, X_train_raw, y_train_raw, X_test_raw, y_test_raw, reducer_meta):
    predictor = TabGPGOPredictor(elite, registry, model, cfg, reducer_meta=reducer_meta)
    predictor.fit(X_train_raw, y_train_raw, fine_tune=True, steps=FT_STEPS, lr=FT_LR)
    pred = predictor.predict(X_test_raw)
    return float(rmse(y_test_raw, pred)), float(r2(y_test_raw, pred))


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    main_best = best_main_per_dataset()
    hpt_best = best_hpt_per_dataset()
    print("main-sweep best per dataset:", {k: v["algo"] for k, v in main_best.items()})
    print("hpt-sweep best per dataset: ", {k: v["algo"] for k, v in hpt_best.items()})

    rows = []
    for dataset in VAL_DATASETS:
        X_raw, y_raw = load_merged_data(dataset, X_y=True)
        X_raw, y_raw = X_raw.float(), y_raw.float()

        main_cfg, main_model, main_registry, main_elite, *_ = load_elite(
            MAIN_RUNS_DIR, main_best[dataset]["run_id"], main_best[dataset]["algo"])
        hpt_cfg, hpt_model, hpt_registry, hpt_elite, *_ = load_elite(
            HPT_RUNS_DIR, hpt_best[dataset]["run_id"], hpt_best[dataset]["algo"])

        for split in range(N_SPLITS):
            X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, p_test=P_TEST, seed=split)
            reducer_meta = fit_reducer(X_train, y_train, main_cfg)

            Xtr100, ytr_z, Xte100, y_stats = tabpfn_split_features(
                X_train, y_train, X_test, y_test, main_cfg, reducer_meta)
            for vname, venum in (("tabpfn_v2", ModelVersion.V2), ("tabpfn_v3", ModelVersion.V3)):
                pf_rmse, pf_r2 = eval_tabpfn(venum, Xtr100, ytr_z, Xte100, y_test, y_stats)
                rows.append({"dataset": dataset, "algo": vname, "split": split,
                            "rmse_raw": pf_rmse, "r2": pf_r2})
                print(f"  [{dataset}/split{split}/{vname}] rmse={pf_rmse:.4f} r2={pf_r2:.4f}")

            for dashboard, (cfg, model_, registry, elite) in (
                ("tabgpgo_main", (main_cfg, main_model, main_registry, main_elite)),
                ("tabgpgo_hpt", (hpt_cfg, hpt_model, hpt_registry, hpt_elite)),
            ):
                tg_rmse, tg_r2 = eval_tabgpgo(elite, registry, model_, cfg,
                                              X_train, y_train, X_test, y_test, reducer_meta)
                rows.append({"dataset": dataset, "algo": dashboard, "split": split,
                            "rmse_raw": tg_rmse, "r2": tg_r2})
                print(f"  [{dataset}/split{split}/{dashboard}] rmse={tg_rmse:.4f} r2={tg_r2:.4f}")
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
