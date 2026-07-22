"""
SSHC (Stochastic Search Hill Climbing) fine-tuning for TabGPGO's best-per-
dataset longrun elite -- an ALTERNATIVE to the existing Adam-based
"tabgpgo_finetuned"/"tabgpgo_zeroshot" columns (run_longrun_split_eval.py),
evaluated on the exact same 5x(80/20 split) held-out protocol so results
drop directly into the same comparison table.

Algorithm (per spec): starting from the frozen elite, repeat n_iters times:
  1. Build a NEIGHBORHOOD of candidate individuals around the current one:
     - one DEFLATE neighbor per existing block (remove that block),
     - one INFLATE neighbor per candidate tree in a precomputed pool (add
       that tree as one new block; every SSHC-proposed block uses
       wrapper="abs" -- a single fixed wrapper, simpler and faster than
       replicating the original elite's own per-block "mix" wrapper draw,
       and orthogonal to the RT/OMT/MS/OMS choices below).
  2. Score every neighbor's REAL training-split RMSE (never the test split)
     and move to the best neighbor -- but ONLY if it is a strict improvement
     over the current individual (standard hill-climbing: sideways/worse
     moves are never taken). If nothing improves, the current individual is
     kept unchanged for this iteration and the next one is tried.

Speed (the whole point of this design): candidate trees are generated ONCE
per (dataset, split) -- not once per iteration -- and their raw semantics are
evaluated ONCE against that split's real TabPFN-embedded train/test tokens,
exactly the "evaluate a priori, then index" pattern this session already
used for TensorSLIM's own static pool and the rotate-dataset caching work
(Block/PoolIndividual/wrapper_output from tabgpgo/evolution.py). Every one
of the n_iters search steps then costs only vectorized tensor arithmetic
over the small real-dataset row count -- no tree evaluation happens inside
the search loop itself. The current individual's own aggregate is
maintained incrementally block-by-block (never refolded from scratch),
matching FreshPoolSLIM's own O(1)-update philosophy.

Two independent knobs (per the "SSHC + OMT and OMS, otherwise RT and MS
default 1" spec) control how each newly-accepted block is proposed:
  - candidate_mode: "rt" (Random Tree -- draw from the precomputed pool;
    default; fully compatible with the "precompute upfront" speedup above)
    or "omt" (Optimal Mutation Tree -- see sshc_search's docstring for why
    this is a SIMPLIFIED, real-data-adapted stand-in for FreshPoolSLIM's own
    nested-GSGP OMT, not a byte-for-byte reuse of it, and why it is
    inherently NOT precompute-friendly -- much slower, implemented but not
    run by this script's __main__).
  - step_mode: "ms1" (literal ms=1.0 for every newly accepted block,
    default) or "oms" (closed-form-optimal step, TensorSLIM._optimal_ms's
    own math -- cheap and fully vectorizable across the whole candidate
    pool at once, since it only needs each candidate's already-precomputed
    raw semantics plus the current aggregate/residual, no new tree
    evaluation -- so "rt"+"oms" is exactly as fast as "rt"+"ms1").

This script's __main__ runs candidate_mode="rt" x step_mode in {"ms1",
"oms"} (both fast); candidate_mode="omt" is implemented (see sshc_search)
but not invoked here given its materially higher per-iteration cost.

Usage:
    python tabpfn_baseline/run_sshc_finetune_eval.py
"""
import csv
import os
import random
import sys
import zlib

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "main"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.data_loader import load_merged_data
from evaluators.fitness_functions import r2, rmse
from main_tabgpgo_funcset import CONSTANT_SETS, FUNCTION_SETS, function_constant_set
from run_longrun_split_eval import (best_per_dataset, fit_reducer,
                                    load_elite_tabpfn, load_reference_model,
                                    split_features)
from tabgpgo.config import TabGPGOConfig
from tabgpgo.evolution import wrapper_output
from tabgpgo.evolution_freshpool import _raw_semantics
from tabgpgo.tabpfn_encoder import encode_val_tabpfn
from tabgpgo.tree_pool import BOUND, generate_ramped_structures, make_terminals
from utils.utils import protected_div, train_test_split

VAL_DATASETS = ["ppb", "toxicity", "resid_build_sale_price", "instanbul", "energy", "concrete"]
N_SPLITS = 5
P_TEST = 0.2

SSHC_ITERS = 200          # hill-climbing accept/reject steps -- matches the
                          # existing Adam fine-tuning's FT_STEPS, so runtime
                          # stays comparable to what's already been run.
POOL_SIZE = 300           # precomputed random-tree candidate pool size per
                          # (dataset, split) -- real-dataset train splits are
                          # a few hundred rows, so this is ample diversity.
OMS_BOUND = 1.0           # matches TabGPGOConfig.oms_bound
OMS_EPS = 1e-4            # matches TabGPGOConfig.oms_eps
OMT_LITE_M = 20           # fresh random trees tried per iteration when
                          # candidate_mode="omt" (see sshc_search docstring)

FUNCTION_SET_NAME = "full_extended"
CONSTANT_SET_NAME = "small_ints"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(THIS_DIR, "log", "sshc_finetune_results.csv")
MEDIAN_CSV = os.path.join(THIS_DIR, "log", "sshc_finetune_median.csv")


def build_candidate_pool(K, cfg, TERMINALS, seed):
    """Generates K random tree structures ONCE (see module docstring) --
    generate_ramped_structures draws from the global random/np.random
    modules, so this seeds them explicitly first for reproducibility."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    return [r["structure"] for r in generate_ramped_structures(K, cfg.init_depth, cfg.p_c, TERMINALS)]


def build_registry(head_structure, elite_raw_blocks, candidate_structures):
    """Combined registry: index 0 = head; then each elite block's own
    structure(s) (structure2 only if sig2); then the candidate pool trees.
    Returns (registry: list[structure], elite_block_specs: list of
    {idx1, idx2, ms, wrapper} referencing registry indices, candidate_start_idx).
    """
    registry = [head_structure]
    elite_block_specs = []
    for b in elite_raw_blocks:
        idx1 = len(registry)
        registry.append(b["structure1"])
        idx2 = None
        if b["structure2"] is not None:
            idx2 = len(registry)
            registry.append(b["structure2"])
        elite_block_specs.append({"idx1": idx1, "idx2": idx2, "ms": b["ms"], "wrapper": b["wrapper"]})
    candidate_start_idx = len(registry)
    registry.extend(candidate_structures)
    return registry, elite_block_specs, candidate_start_idx


def evaluate_registry(registry, T_embed, TERMINALS):
    """Evaluate every registry entry's raw semantics on T_embed ONCE ->
    (n_registry, n_rows) -- the precomputed pool the search then only
    indexes into."""
    return torch.stack([_raw_semantics(s, T_embed, TERMINALS) for s in registry])


def _fold(pool, blocks):
    """Full fold of a block list (operator fixed to "mul", matching
    SLIM*MIX) from raw registry semantics -- used only for the initial
    aggregate before the search loop and for final held-out scoring; the
    search loop itself maintains its aggregate incrementally instead."""
    agg = pool[0]
    for b in blocks:
        tr1 = pool[b["idx1"]]
        tr2 = pool[b["idx2"]] if b["idx2"] is not None else None
        delta = b["ms"] * wrapper_output(b["wrapper"], tr1, tr2)
        agg = agg * (1 + delta)
    return torch.clamp(agg, -BOUND, BOUND)


def _oms_for_candidates(cand_sR, current_agg, y_train_z):
    """Vectorized closed-form-optimal step size (TensorSLIM._optimal_ms's
    own math, see tabgpgo/evolution_freshpool.py) for EVERY candidate at
    once -- only needs each candidate's already-precomputed raw semantics
    plus the current aggregate/residual, no new tree evaluation, so this is
    just as cheap as the literal ms=1.0 path."""
    residual = protected_div(y_train_z, current_agg) - 1
    numer = (cand_sR * residual.unsqueeze(0)).sum(dim=1)
    denom = (cand_sR * cand_sR).sum(dim=1)
    ms = torch.where(denom > 1e-12, numer / denom, torch.zeros_like(numer))
    ms = torch.clamp(ms, -OMS_BOUND, OMS_BOUND)
    return torch.where(ms.abs() < OMS_EPS, torch.zeros_like(ms), ms)


def sshc_search(elite_block_specs, candidate_start_idx, n_candidates, pool_train, y_train_z,
                n_iters=SSHC_ITERS, candidate_mode="rt", step_mode="ms1",
                cfg=None, TERMINALS=None, T_train_embed=None, seed=0):
    """Runs the hill-climbing search described in the module docstring.
    Returns (final_block_specs, final_train_rmse, appended_structures).

    Every block in the returned final_block_specs (and in current_blocks at
    every point during the loop) has an idx1/idx2 that resolves against
    `pool_train` AS GROWN so far by this function -- candidate_mode="omt"
    grows pool_train in place (torch.cat) the moment a freshly-generated
    tree is accepted, exactly like an "rt" acceptance references a fixed
    index into the precomputed pool, so every subsequent iteration's
    deflate/inflate arithmetic can index it normally. appended_structures
    lists those freshly-generated structures in acceptance order, so a
    caller can extend a PARALLEL pool (e.g. the held-out test split's own
    pool, see eval_sshc) with the same structures in the same order and
    keep indices aligned between the two.

    candidate_mode="omt": a SIMPLIFIED, real-data-adapted stand-in for
    FreshPoolSLIM's own nested-GSGP Optimal Mutation Tree, not a reuse of
    that exact machinery (which is deeply tied to a FreshPoolSLIM instance's
    own reservoir/threading state and searches over many GENERATIONS of a
    nested population, not a handful of single trees). Here, at every
    iteration, OMT_LITE_M fresh random trees are generated and evaluated on
    the spot (not precomputed -- the whole point of OMT is adapting to the
    CURRENT residual, which changes every iteration), and the best of that
    fresh batch (by real training RMSE, after picking its own ms via the
    same OMS math if step_mode="oms") competes against the deflate
    neighbors and the current individual exactly like an "rt" inflate
    neighbor would. This is materially slower than "rt" (new tree
    evaluations every iteration, not just a pool lookup) and is not invoked
    by this module's own __main__.
    """
    device = pool_train.device

    current_blocks = list(elite_block_specs)
    current_agg = _fold(pool_train, current_blocks)
    current_rmse = float(rmse(y_train_z, current_agg))

    cand_idx = list(range(candidate_start_idx, candidate_start_idx + n_candidates))
    cand_sem = pool_train[cand_idx]                     # (K, n_train)
    cand_sR_abs = wrapper_output("abs", cand_sem)        # (K, n_train)

    appended_structures = []

    for _ in range(n_iters):
        best_rmse, best_blocks, best_agg, best_appended = current_rmse, None, None, None

        # -- deflate neighbors: remove one existing block at a time --
        for i, b in enumerate(current_blocks):
            tr1 = pool_train[b["idx1"]]
            tr2 = pool_train[b["idx2"]] if b["idx2"] is not None else None
            delta_i = b["ms"] * wrapper_output(b["wrapper"], tr1, tr2)
            agg_wo_i = torch.clamp(current_agg / (1 + delta_i), -BOUND, BOUND)
            r = float(rmse(y_train_z, agg_wo_i))
            if r < best_rmse - 1e-9:
                best_rmse = r
                best_blocks = current_blocks[:i] + current_blocks[i + 1:]
                best_agg = agg_wo_i
                best_appended = None

        # -- inflate neighbors: add one new block, vectorized over the pool --
        if step_mode == "oms":
            ms_used = _oms_for_candidates(cand_sR_abs, current_agg, y_train_z)
        else:
            ms_used = torch.ones(n_candidates, device=device)

        cand_agg = torch.clamp(current_agg.unsqueeze(0) * (1 + ms_used.unsqueeze(1) * cand_sR_abs),
                               -BOUND, BOUND)             # (K, n_train)
        cand_rmse = rmse(y_train_z, cand_agg)             # (K,)
        j = int(torch.argmin(cand_rmse))
        if float(cand_rmse[j]) < best_rmse - 1e-9:
            best_rmse = float(cand_rmse[j])
            best_blocks = current_blocks + [{"idx1": cand_idx[j], "idx2": None,
                                             "ms": float(ms_used[j]), "wrapper": "abs"}]
            best_agg = cand_agg[j]
            best_appended = None

        # -- OMT-lite inflate neighbor (only when requested; see docstring) --
        if candidate_mode == "omt":
            fresh = [r["structure"] for r in generate_ramped_structures(
                OMT_LITE_M, cfg.init_depth, cfg.p_c, TERMINALS)]
            fresh_sem = torch.stack([_raw_semantics(s, T_train_embed, TERMINALS) for s in fresh])
            fresh_sR = wrapper_output("abs", fresh_sem)
            if step_mode == "oms":
                ms_fresh = _oms_for_candidates(fresh_sR, current_agg, y_train_z)
            else:
                ms_fresh = torch.ones(OMT_LITE_M, device=device)
            fresh_agg = torch.clamp(current_agg.unsqueeze(0) * (1 + ms_fresh.unsqueeze(1) * fresh_sR),
                                    -BOUND, BOUND)
            fresh_rmse = rmse(y_train_z, fresh_agg)
            k = int(torch.argmin(fresh_rmse))
            if float(fresh_rmse[k]) < best_rmse - 1e-9:
                best_rmse = float(fresh_rmse[k])
                # grow pool_train NOW so the new block's idx1 is resolvable
                # by every later iteration's deflate/inflate arithmetic --
                # see docstring.
                new_idx = pool_train.shape[0]
                pool_train = torch.cat([pool_train, fresh_sem[k:k + 1]], dim=0)
                best_blocks = current_blocks + [{"idx1": new_idx, "idx2": None,
                                                 "ms": float(ms_fresh[k]), "wrapper": "abs"}]
                best_agg = fresh_agg[k]
                best_appended = fresh[k]

        if best_blocks is not None:
            current_blocks, current_agg, current_rmse = best_blocks, best_agg, best_rmse
            if best_appended is not None:
                appended_structures.append(best_appended)

    return current_blocks, current_rmse, appended_structures


def eval_sshc(head_structure, elite_raw_blocks, candidate_structures, cfg, TERMINALS,
             T_train_embed, ytr_z, T_test_embed, y_test_raw, y_stats,
             candidate_mode, step_mode, seed):
    registry, elite_block_specs, candidate_start_idx = build_registry(
        head_structure, elite_raw_blocks, candidate_structures)
    pool_train = evaluate_registry(registry, T_train_embed, TERMINALS)
    pool_test = evaluate_registry(registry, T_test_embed, TERMINALS)

    final_blocks, _, appended_structures = sshc_search(
        elite_block_specs, candidate_start_idx, len(candidate_structures),
        pool_train, ytr_z, n_iters=SSHC_ITERS, candidate_mode=candidate_mode,
        step_mode=step_mode, cfg=cfg, TERMINALS=TERMINALS,
        T_train_embed=T_train_embed, seed=seed)

    # OMT-lite may have grown pool_train past the original registry (see
    # sshc_search) -- extend pool_test with the SAME freshly-generated
    # structures, in the SAME order, so its indices stay aligned with
    # final_blocks' idx1/idx2 references.
    if appended_structures:
        extra_sem_test = torch.stack([_raw_semantics(s, T_test_embed, TERMINALS) for s in appended_structures])
        pool_test = torch.cat([pool_test, extra_sem_test], dim=0)

    pred_z = _fold(pool_test, final_blocks)
    mean, std = y_stats
    pred_raw = pred_z * std.squeeze() + mean.squeeze()
    return float(rmse(y_test_raw, pred_raw)), float(r2(y_test_raw, pred_raw))


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    best = best_per_dataset()
    print("longrun best per dataset:", {k: v["algo"] for k, v in best.items()})

    reference_model, embed_dim = load_reference_model()
    p_c, constants = CONSTANT_SETS[CONSTANT_SET_NAME]
    cfg = TabGPGOConfig()

    from run_longrun_split_eval import LONGRUN_RUN_DIR_BASE, _tag

    variants = [("rt", "ms1"), ("rt", "oms")]

    rows = []
    with function_constant_set(FUNCTION_SETS[FUNCTION_SET_NAME], constants):
        TERMINALS = make_terminals(embed_dim)
        for dataset in VAL_DATASETS:
            X_raw, y_raw = load_merged_data(dataset, X_y=True)
            X_raw, y_raw = X_raw.float(), y_raw.float()

            run_dir = os.path.join(LONGRUN_RUN_DIR_BASE,
                                   f"{best[dataset]['run_id']}_{_tag(best[dataset]['algo'])}_0")
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

                # zlib.crc32 (not the builtin hash(), which is randomized
                # per-process for strings) so the candidate pool is
                # reproducible across runs/processes.
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
