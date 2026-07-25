"""
Held-out-split evaluation for the V3-pinned TabPFN embedding experiment
(main/main_tabgpgo_tabpfn_v3_pop200_ms001.py): TabGPGO (fine-tuned AND
zero-shot), evaluated with the SAME V3 reference embedding model the elite
was evolved with, on the same 5x(80/20 split) protocol every other
split-eval script in this directory uses.

Reference model: main/log/tabgpgo_tabpfn_v3_artifacts/tabpfn_v3_pool_bundle.pkl
is NOT present (the cache produced during the real evolve run is gone by the
time this script runs) -- but eval only ever needs the frozen FIRST
synthetic dataset's fitted TabPFNRegressor (see tabgpgo/tabpfn_encoder.py's
build_tabpfn_pool: reference_model is always that first dataset's model,
the rest of the ~1000-dataset pool is only used to build the training
context T_train/y_target, never touched again downstream). So instead of
re-generating the whole expensive pool, build_v3_reference_model below
reproduces just that one dataset (same seed=cfg.data_seed, i=0) and fits a
single V3-pinned TabPFNRegressor on it -- identical to what build_tabpfn_pool
would have handed back as reference_model, at a small fraction of the cost.

TabPFN v2/v3 (real-adapted AND zero-shot) columns are NOT re-fit here --
reused verbatim from tabpfn_baseline/log/pop500_split_eval_results.csv,
exactly like run_popsweep_split_eval.py's own REUSED_TABPFN_ALGOS: those
rows only ever touch the real dataset's own (X_train, X_test, y_train,
y_test) for a given (dataset, split), never the TabGPGO elite/config, so
they are identical across every experiment run on the same 6 datasets with
the same N_SPLITS/P_TEST/seed protocol.

Standard SLIM is NOT re-run here either -- see run_longrun_split_eval.py's
own module docstring for why (already-collected data, not tied to which
TabGPGO experiment it's compared against).

Usage:
    python tabpfn_baseline/run_v3_split_eval.py
"""
import csv
import os
import sys

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "main"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.data_loader import load_merged_data
from main_tabgpgo_funcset import CONSTANT_SETS, FUNCTION_SETS, function_constant_set
from run_longrun_split_eval import _tag, eval_tabgpgo, fit_reducer, load_elite_tabpfn, split_features
from tabgpgo.config import TabGPGOConfig
from tabgpgo.preprocessing import standardize_scale_pad
from tabgpgo.prior import generate_datasets
from tabgpgo.tree_pool import make_terminals
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from utils.utils import train_test_split

VAL_DATASETS = ["ppb", "toxicity", "resid_build_sale_price", "instanbul", "energy", "concrete"]
N_SPLITS = 5
P_TEST = 0.2

FUNCTION_SET_NAME = "full_extended"
CONSTANT_SET_NAME = "small_ints"

REUSED_TABPFN_ALGOS = ["tabpfn_v2", "tabpfn_v3", "tabpfn_v2_zeroshot", "tabpfn_v3_zeroshot"]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_v3_pop200ms001_runs")
V3_MANIFEST_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_v3_pop200ms001_manifest.csv")
POP500_RESULTS_CSV = os.path.join(THIS_DIR, "log", "pop500_split_eval_results.csv")
OUT_CSV = os.path.join(THIS_DIR, "log", "v3_split_eval_results.csv")
MEDIAN_CSV = os.path.join(THIS_DIR, "log", "v3_split_eval_median.csv")


def v3_combo():
    manifest = pd.read_csv(V3_MANIFEST_CSV)
    row = manifest.iloc[0]
    return {"algo": row["algo"], "run_id": row["run_id"]}


def build_v3_reference_model(cfg):
    """Reproduces build_tabpfn_pool's first-iteration reference_model
    (i=0, seed=cfg.data_seed, random_state=cfg.data_seed) without paying for
    the full ~1000-dataset pool -- see module docstring."""
    (X, y), = list(generate_datasets(1, cfg.n_rows, cfg.max_features, cfg.min_features,
                                     seed=cfg.data_seed, backend=cfg.prior_backend))
    X100, y_std, _ = standardize_scale_pad(X, y, cfg.max_features)
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3, n_estimators=1, random_state=cfg.data_seed)
    model.fit(X100.numpy().astype(np.float32), y_std.numpy().astype(np.float32))
    emb = model.get_embeddings(X100.numpy().astype(np.float32), data_source="train")
    embed_dim = emb.shape[-1]
    return model, embed_dim


def load_reused_tabpfn_rows():
    df = pd.read_csv(POP500_RESULTS_CSV)
    reused = df[df["algo"].isin(REUSED_TABPFN_ALGOS)]
    return reused.to_dict("records")


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    combo = v3_combo()
    print(f"V3 combo (all datasets): {combo['algo']}")

    cfg = TabGPGOConfig()
    reference_model, embed_dim = build_v3_reference_model(cfg)
    print(f"built V3 reference model (embed_dim={embed_dim})")
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]

    rows = load_reused_tabpfn_rows()
    print(f"reused {len(rows)} TabPFN v2/v3 (real-adapted + zero-shot) rows from "
         f"{os.path.basename(POP500_RESULTS_CSV)} -- not re-fit")

    run_dir = os.path.join(V3_RUN_DIR_BASE, f"{combo['run_id']}_{_tag(combo['algo'])}_0")
    run_cfg, registry, ind = load_elite_tabpfn(run_dir)

    with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
        TERMINALS = make_terminals(embed_dim)
        for dataset in VAL_DATASETS:
            X_raw, y_raw = load_merged_data(dataset, X_y=True)
            X_raw, y_raw = X_raw.float(), y_raw.float()

            for split in range(N_SPLITS):
                X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, p_test=P_TEST, seed=split)
                reducer_meta = fit_reducer(X_train, y_train, run_cfg)
                Xtr100, ytr_z, Xte100, y_stats = split_features(X_train, y_train, X_test, y_test, run_cfg, reducer_meta)

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
