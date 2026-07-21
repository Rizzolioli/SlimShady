"""
Held-out-split evaluation for the 2000-generation TabPFN-encoder longrun
experiment (main/main_tabgpgo_tabpfn_longrun.py): TabGPGO (fine-tuned AND
zero-shot) vs. TabPFN V2/V3, on the same 5x(80/20 split) protocol
run_split_eval.py uses for the older main/hpt sweeps.

Why this can't just reuse run_split_eval.py's load_elite/TabGPGOPredictor:
those assume the MLPAutoencoder encoder (tabgpgo.inference.load_run always
rebuilds a real MLPAutoencoder and loads encoder.pt into it). The longrun
elites were evolved with the TabPFN encoder instead (see
tabgpgo/tabpfn_encoder.py) -- their saved encoder.pt is an empty
nn.Identity() stand-in (see main_tabgpgo_tabpfn_longrun.py's _EncoderStub),
so load_run/TabGPGOPredictor would silently use the WRONG embedding space.
Instead this script:
  - parses config.json/elite.json directly (same fields load_run reads,
    minus the MLPAutoencoder reconstruction),
  - loads the cached TabPFN reference encoder from
    main/log/tabgpgo_tabpfn_artifacts/tabpfn_pool_bundle.pkl (the exact same
    frozen reference_model the longrun run used -- see that script's
    docstring: the bundle only depends on cfg's synthetic-prior fields,
    shared verbatim across every tabgpgo_tabpfn_* script), and
  - embeds each split's (X_train, X_test) via encode_val_tabpfn, then calls
    tabgpgo.inference.block_raw_terms / weighted_semantics / fine_tune_weights
    directly (the encoder-agnostic lower-level functions TabGPGOPredictor
    itself is built from) instead of TabGPGOPredictor's _encode().

Zero-shot TabGPGO ("not fine-tuned"): same held-out split, same TabPFN
embedding, but weights = each block's evolved `ms` (no re-optimization) and
the permissive zero-shot bound (BOUND=1e12) instead of fine_tune_weights'
bound=10.0 safeguard -- mirrors tabgpgo.inference.TabGPGOPredictor.predict's
own fine_tune-vs-not bound convention exactly.

Best-per-dataset elite selection: combines
main/log/tabgpgo_longrun_norotate_results.csv (29-col schema) and
tabgpgo_longrun_rotate10_results.csv (32-col schema) at generation 2000,
picks whichever of the 6 combos scored the highest val R2 for that dataset
-- same "last-logged-generation, best val_r2 per dataset" selection
run_split_eval.py's _best_per_dataset uses, just across these two files
instead of one.

Standard SLIM is NOT re-run here -- main/log/results_r2_generations.csv
already has 30-seed, 2000-generation, direct (no meta-learning) SLIM_GSGP
results on these exact 6 real datasets (the NORM1/NORM2 mutation study's own
baseline sweep, six classic variants: SLIM+/*ABS, SLIM+/*1SIG, SLIM+/*2SIG).
See scratchpad/build_longrun_data.py's standard-SLIM extraction, run
separately (no TabPFN/GP compute needed, it's already-collected data).

Usage:
    python tabpfn_baseline/run_longrun_split_eval.py
"""
import csv
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "main"))

from datasets.data_loader import load_merged_data
from evaluators.fitness_functions import r2, rmse
from main_tabgpgo_funcset import CONSTANT_SETS, FUNCTION_SETS, function_constant_set
from tabgpgo.config import TabGPGOConfig
from tabgpgo.inference import (Block, PoolIndividual, _from_jsonable,
                               block_raw_terms, fine_tune_weights,
                               weighted_semantics)
from tabgpgo.preprocessing import reduce_features, standardize_scale_pad, zscore
from tabgpgo.tabpfn_encoder import encode_val_tabpfn
from tabgpgo.tree_pool import BOUND, make_terminals
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from utils.utils import train_test_split

VAL_DATASETS = ["ppb", "toxicity", "resid_build_sale_price", "instanbul", "energy", "concrete"]
N_SPLITS = 5
P_TEST = 0.2
FT_STEPS = 200
FT_LR = 0.05

FUNCTION_SET_NAME = "full_extended"
CONSTANT_SET_NAME = "small_ints"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LONGRUN_RUN_DIR_BASE = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_longrun_runs")
NOROTATE_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_longrun_norotate_results.csv")
ROTATE_CSV = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_longrun_rotate10_results.csv")
TABPFN_ARTIFACTS_DIR = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_tabpfn_artifacts")
OUT_CSV = os.path.join(THIS_DIR, "log", "longrun_split_eval_results.csv")
MEDIAN_CSV = os.path.join(THIS_DIR, "log", "longrun_split_eval_median.csv")

BASE_COLS = ["algo", "run_id", "dataset", "seed", "generation", "elite_train_rmse", "time_s", "population_nodes"]
VAL_COLS = [f"val_{d}_{suf}" for d in VAL_DATASETS for suf in ("rmse_scaled", "rmse_raw", "r2")]
ELITE_COLS = ["elite_size", "elite_nodes", "elite_train_r2"]
NOROTATE_COLS = BASE_COLS + VAL_COLS + ELITE_COLS
ROTATE_COLS = NOROTATE_COLS + ["global_pool_rmse", "global_pool_r2", "active_dataset_idx"]


def _tag(algo):
    return algo.replace("*", "x").replace("~", "t")


def best_per_dataset():
    nr = pd.read_csv(NOROTATE_CSV, header=None, names=NOROTATE_COLS)
    rt = pd.read_csv(ROTATE_CSV, header=None, names=ROTATE_COLS)
    last = pd.concat([nr[nr.generation == nr.generation.max()],
                      rt[rt.generation == rt.generation.max()]], ignore_index=True)
    best = {}
    for dataset in VAL_DATASETS:
        row = last.loc[last[f"val_{dataset}_r2"].idxmax()]
        best[dataset] = {"algo": row["algo"], "run_id": row["run_id"]}
    return best


def load_elite_tabpfn(run_dir):
    """Like tabgpgo.inference.load_run, minus the MLPAutoencoder
    reconstruction (see module docstring) -- returns (cfg, ind)."""
    import json
    with open(os.path.join(run_dir, "config.json")) as fh:
        cfg_dict = json.load(fh)
    cfg_dict["variants"] = tuple(tuple(v) for v in cfg_dict.get("variants", ()))
    cfg_dict["val_datasets"] = tuple(cfg_dict.get("val_datasets", ()))
    cfg = TabGPGOConfig(**cfg_dict)
    with open(os.path.join(run_dir, "elite.json")) as fh:
        e = json.load(fh)
    registry = {e["head_idx"]: {"structure": _from_jsonable(e["head_structure"])}}
    for b, (s1, s2) in zip(e["blocks"], e["block_structures"]):
        registry[b[0]] = {"structure": _from_jsonable(s1)}
        if b[1] is not None:
            registry[b[1]] = {"structure": _from_jsonable(s2)}
    ind = PoolIndividual(e["head_idx"],
                         [Block(b[0], b[1], b[2], b[3], b[4]) for b in e["blocks"]],
                         fitness=e["fitness"], nodes_count=e["nodes_count"])
    return cfg, registry, ind


def load_reference_model():
    path = os.path.join(TABPFN_ARTIFACTS_DIR, "tabpfn_pool_bundle.pkl")
    with open(path, "rb") as fh:
        _, _, reference_model, embed_dim = pickle.load(fh)
    return reference_model, embed_dim


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


def split_features(X_train_raw, y_train_raw, X_test_raw, y_test_raw, cfg, reducer_meta):
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


def eval_tabgpgo(ind, registry, reference_model, TERMINALS, Xtr100, ytr_z, Xte100, y_test_raw, y_stats, fine_tune):
    T_train = encode_val_tabpfn(reference_model, Xtr100)
    T_test = encode_val_tabpfn(reference_model, Xte100)
    head_sem_tr, raw_terms_tr = block_raw_terms(ind, T_train, registry, TERMINALS)
    head_sem_te, raw_terms_te = block_raw_terms(ind, T_test, registry, TERMINALS)
    if fine_tune and ind.blocks:
        weights = fine_tune_weights(ind, head_sem_tr, raw_terms_tr, ytr_z, steps=FT_STEPS, lr=FT_LR, bound=10.0)
        bound = 10.0
    else:
        weights = torch.tensor([b.ms for b in ind.blocks], dtype=torch.float32)
        bound = BOUND
    pred_z = weighted_semantics(ind, head_sem_te, raw_terms_te, weights, bound=bound)
    mean, std = y_stats
    pred_raw = pred_z * std.squeeze() + mean.squeeze()
    return float(rmse(y_test_raw, pred_raw)), float(r2(y_test_raw, pred_raw))


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    best = best_per_dataset()
    print("longrun best per dataset:", {k: v["algo"] for k, v in best.items()})

    reference_model, embed_dim = load_reference_model()
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]

    rows = []
    with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
        TERMINALS = make_terminals(embed_dim)
        for dataset in VAL_DATASETS:
            X_raw, y_raw = load_merged_data(dataset, X_y=True)
            X_raw, y_raw = X_raw.float(), y_raw.float()

            run_dir = os.path.join(LONGRUN_RUN_DIR_BASE, f"{best[dataset]['run_id']}_{_tag(best[dataset]['algo'])}_0")
            cfg, registry, ind = load_elite_tabpfn(run_dir)

            for split in range(N_SPLITS):
                X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, p_test=P_TEST, seed=split)
                reducer_meta = fit_reducer(X_train, y_train, cfg)
                Xtr100, ytr_z, Xte100, y_stats = split_features(X_train, y_train, X_test, y_test, cfg, reducer_meta)

                for vname, venum in (("tabpfn_v2", ModelVersion.V2), ("tabpfn_v3", ModelVersion.V3)):
                    pf_rmse, pf_r2 = eval_tabpfn(venum, Xtr100, ytr_z, Xte100, y_test, y_stats)
                    rows.append({"dataset": dataset, "algo": vname, "split": split,
                                "rmse_raw": pf_rmse, "r2": pf_r2})
                    print(f"  [{dataset}/split{split}/{vname}] rmse={pf_rmse:.4f} r2={pf_r2:.4f}")

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
