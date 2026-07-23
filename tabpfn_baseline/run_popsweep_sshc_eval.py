"""
SSHC (Stochastic Search Hill Climbing) fine-tuning for the pop_size sweep
(200/100, n_gens=5000, main/main_tabgpgo_tabpfn_norotate_popsweep.py)'s
best-per-dataset elite -- same algorithm/parameters as
run_sshc_finetune_eval.py (see that module's docstring for the full
design), just pointed at the pop-sweep's own results/run directory via
run_popsweep_split_eval.best_per_dataset instead of the 2000-generation
longrun or pop_size=500 experiments'.

Usage:
    python tabpfn_baseline/run_popsweep_sshc_eval.py
"""
import csv
import os
import sys
import zlib

import pandas as pd
import torch

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "main"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.data_loader import load_merged_data
from main_tabgpgo_funcset import CONSTANT_SETS, FUNCTION_SETS, function_constant_set
from run_longrun_split_eval import _tag, fit_reducer, load_elite_tabpfn, load_reference_model, split_features
from run_popsweep_split_eval import RUN_DIR_BASE, best_per_dataset
from run_sshc_finetune_eval import POOL_SIZE, build_candidate_pool, eval_sshc
from tabgpgo.config import TabGPGOConfig
from tabgpgo.tabpfn_encoder import encode_val_tabpfn
from tabgpgo.tree_pool import make_terminals
from utils.utils import train_test_split

VAL_DATASETS = ["ppb", "toxicity", "resid_build_sale_price", "instanbul", "energy", "concrete"]
N_SPLITS = 5
P_TEST = 0.2

FUNCTION_SET_NAME = "full_extended"
CONSTANT_SET_NAME = "small_ints"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(THIS_DIR, "log", "popsweep_sshc_results.csv")
MEDIAN_CSV = os.path.join(THIS_DIR, "log", "popsweep_sshc_median.csv")


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    best = best_per_dataset()
    print("popsweep best per dataset:", {k: v["algo"] for k, v in best.items()})

    reference_model, embed_dim = load_reference_model()
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]
    cfg = TabGPGOConfig()

    variants = [("rt", "ms1"), ("rt", "oms")]

    rows = []
    with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
        TERMINALS = make_terminals(embed_dim)
        for dataset in VAL_DATASETS:
            X_raw, y_raw = load_merged_data(dataset, X_y=True)
            X_raw, y_raw = X_raw.float(), y_raw.float()

            run_dir = os.path.join(RUN_DIR_BASE, f"{best[dataset]['run_id']}_{_tag(best[dataset]['algo'])}_0")
            _, registry, ind = load_elite_tabpfn(run_dir)
            head_structure = registry[ind.head_idx]["structure"]
            elite_raw_blocks = [
                {"structure1": registry[b.idx1]["structure"],
                 "structure2": registry[b.idx2]["structure"] if b.idx2 is not None else None,
                 "ms": b.ms, "wrapper": b.wrapper}
                for b in ind.blocks
            ]

            for split in range(N_SPLITS):
                X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, p_test=P_TEST, seed=split)
                reducer_meta = fit_reducer(X_train, y_train, cfg)
                Xtr100, ytr_z, Xte100, y_stats = split_features(X_train, y_train, X_test, y_test, cfg, reducer_meta)
                T_train_embed = encode_val_tabpfn(reference_model, Xtr100)
                T_test_embed = encode_val_tabpfn(reference_model, Xte100)

                pool_seed = zlib.crc32(f"{dataset}_{split}".encode()) % (2 ** 31)
                candidate_structures = build_candidate_pool(POOL_SIZE, cfg, TERMINALS, seed=pool_seed)

                for candidate_mode, step_mode in variants:
                    algo = f"sshc_{candidate_mode}_{step_mode}"
                    rmse_, r2_ = eval_sshc(head_structure, elite_raw_blocks, candidate_structures,
                                           cfg, TERMINALS, T_train_embed, ytr_z, T_test_embed,
                                           y_test, y_stats, candidate_mode, step_mode, seed=split)
                    rows.append({"dataset": dataset, "algo": algo, "split": split,
                                "rmse_raw": rmse_, "r2": r2_})
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
