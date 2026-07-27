"""
Per-iteration convergence curve for the V3 SSHC fine-tuning
(run_v3_sshc_eval.py): logs the hill-climbing search's own train/test
RMSE and R2 after EVERY one of its SSHC_ITERS=50 iterations, instead of
only the final result -- so we can see whether the search has actually
plateaued by iteration 50 or is still improving (i.e. whether more
iterations would likely help), matching the "evolution curves" the main
TabGPGO dashboard already logs for the genetic-programming side.

Cheap by construction: sshc_search's per-iteration logging (see that
function's log_history option in run_sshc_finetune_eval.py) only adds one
more vectorized fold+score of the held-out test split per iteration -- no
new tree evaluation -- so this costs barely more than run_v3_sshc_eval.py
itself, just with history retained instead of discarded.

Usage:
    python tabpfn_baseline/run_v3_sshc_curve.py
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
from run_longrun_split_eval import _tag, fit_reducer, load_elite_tabpfn, split_features
from run_sshc_finetune_eval import POOL_SIZE, SSHC_ITERS, build_candidate_pool, eval_sshc
from run_v3_split_eval import V3_RUN_DIR_BASE, build_v3_reference_model, v3_combo
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
OUT_CSV = os.path.join(THIS_DIR, "log", "v3_sshc_curve_results.csv")
MEDIAN_CSV = os.path.join(THIS_DIR, "log", "v3_sshc_curve_median.csv")


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    combo = v3_combo()
    print(f"V3 combo (all datasets): {combo['algo']}")

    cfg = TabGPGOConfig()
    reference_model, embed_dim = build_v3_reference_model(cfg)
    print(f"built V3 reference model (embed_dim={embed_dim})")
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]

    variants = [("rt", "ms1"), ("rt", "oms")]

    run_dir = os.path.join(V3_RUN_DIR_BASE, f"{combo['run_id']}_{_tag(combo['algo'])}_0")
    _, registry, ind = load_elite_tabpfn(run_dir)
    head_structure = registry[ind.head_idx]["structure"]
    elite_raw_blocks = [
        {"structure1": registry[b.idx1]["structure"],
         "structure2": registry[b.idx2]["structure"] if b.idx2 is not None else None,
         "ms": b.ms, "wrapper": b.wrapper}
        for b in ind.blocks
    ]

    rows = []
    with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
        TERMINALS = make_terminals(embed_dim)
        for dataset in VAL_DATASETS:
            X_raw, y_raw = load_merged_data(dataset, X_y=True)
            X_raw, y_raw = X_raw.float(), y_raw.float()

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
                    _, _, history = eval_sshc(head_structure, elite_raw_blocks, candidate_structures,
                                              cfg, TERMINALS, T_train_embed, ytr_z, T_test_embed,
                                              y_test, y_stats, candidate_mode, step_mode, seed=split,
                                              log_history=True)
                    for it, entry in enumerate(history, start=1):
                        rows.append({"dataset": dataset, "algo": algo, "split": split, "iter": it,
                                    "train_rmse": entry["train_rmse"], "train_r2": entry["train_r2"],
                                    "test_rmse": entry["test_rmse"], "test_r2": entry["test_r2"]})
                    print(f"  [{dataset}/split{split}/{algo}] final train_r2={history[-1]['train_r2']:.4f} "
                         f"test_r2={history[-1]['test_r2']:.4f}")
            print(f"[{dataset}] done")

    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "algo", "split", "iter",
                                                "train_rmse", "train_r2", "test_rmse", "test_r2"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT_CSV}")

    df = pd.DataFrame(rows)
    med = df.groupby(["dataset", "algo", "iter"])[["train_rmse", "train_r2", "test_rmse", "test_r2"]] \
        .median().reset_index()
    med.to_csv(MEDIAN_CSV, index=False)
    print(f"wrote medians -> {MEDIAN_CSV}")
    print(f"iterations logged per (dataset,algo): {SSHC_ITERS}")


if __name__ == "__main__":
    main()
