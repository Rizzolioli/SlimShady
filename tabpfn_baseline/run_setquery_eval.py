"""
Standalone test of the set-query autoencoder (tabgpgo/setquery_encoder.py):
train ONLY on the synthetic prior, evaluate ZERO-SHOT on the 6 real
validation datasets, with NO TabGPGO/SLIM-GSGP evolution involved anywhere
-- purely an encoder-quality experiment, per this session's request.

Protocol:
  - Train: cfg.n_synth_datasets synthetic datasets (default from
    TabGPGOConfig), one at a time, each with its own randomly sampled row
    count (see tabgpgo/setquery_encoder.py's ROW_RANGE) so every row_pos
    index the real datasets might use during eval actually gets trained.
  - Eval: for each real dataset, the SAME 5x(80/20 split) protocol every
    other eval script in this directory uses. Per split: fit the PCA/RF
    reducer (only if the dataset has >100 raw features) and the z-score
    stats on the TRAIN rows only (reduce_with/split_features, reused from
    run_longrun_split_eval.py), then encode ALL rows -- train split AND
    test split together, features only, y never touched -- in one forward
    pass, and read off the target-query's prediction at the TEST rows'
    positions only. This is fully zero-shot: the model is never fit or
    adapted to the real dataset in any way, it only ever sees real X.
  - rmse/r2 computed against the real (raw-unit) y_test, inverting the
    z-score with the TRAIN split's own y_stats (same convention as every
    other eval_* function in this directory).

Usage:
    python tabpfn_baseline/run_setquery_eval.py
"""
import csv
import os
import sys

import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "main"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.data_loader import load_merged_data
from evaluators.fitness_functions import r2, rmse
from run_longrun_split_eval import fit_reducer, split_features
from tabgpgo.config import TabGPGOConfig
from tabgpgo.setquery_encoder import encode_and_predict, train_setquery_ae
from utils.utils import train_test_split

VAL_DATASETS = ["ppb", "toxicity", "resid_build_sale_price", "instanbul", "energy", "concrete"]
N_SPLITS = 5
P_TEST = 0.2

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(THIS_DIR, "log", "setquery_eval_results.csv")
MEDIAN_CSV = os.path.join(THIS_DIR, "log", "setquery_eval_median.csv")
RECON_CSV = os.path.join(THIS_DIR, "log", "setquery_recon_r2.csv")

# A controlled overfitting test this session (train_setquery_ae's own
# earlier one-step-per-dataset version, on a tiny fixed/repeated pool) showed
# both losses sitting at their trivial baseline (MSE~=1.0, "predict the
# mean") after only ~1 gradient step per dataset with no repeats -- but
# clearly learning given more steps on the same data. N_DATASETS x N_EPOCHS
# = 10,000 total gradient steps here, vs. the 500 steps (0 repeats) the
# first real run got.
N_DATASETS = 500
N_EPOCHS = 20
HIDDEN = 128
N_LATENTS = 32
TARGET_WEIGHT = 1.0
LR = 1e-3


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    cfg = TabGPGOConfig()

    print(f"training set-query autoencoder on {N_DATASETS} synthetic datasets x {N_EPOCHS} epochs "
         f"(hidden={HIDDEN}, n_latents={N_LATENTS}, target_weight={TARGET_WEIGHT})...")
    model = train_setquery_ae(cfg, n_datasets=N_DATASETS, n_epochs=N_EPOCHS, target_weight=TARGET_WEIGHT,
                              lr=LR, hidden=HIDDEN, n_latents=N_LATENTS)

    ckpt_path = os.path.join(THIS_DIR, "log", "setquery_ae.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint -> {ckpt_path}")

    rows, recon_rows = [], []
    for dataset in VAL_DATASETS:
        X_raw, y_raw = load_merged_data(dataset, X_y=True)
        X_raw, y_raw = X_raw.float(), y_raw.float()

        for split in range(N_SPLITS):
            X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, p_test=P_TEST, seed=split)
            reducer_meta = fit_reducer(X_train, y_train, cfg)
            Xtr100, ytr_z, Xte100, y_stats = split_features(X_train, y_train, X_test, y_test, cfg, reducer_meta)

            X_all100 = torch.cat([Xtr100, Xte100], dim=0)
            n_train = Xtr100.shape[0]
            X_hat, y_hat_z = encode_and_predict(model, X_all100)

            y_hat_test_z = y_hat_z[n_train:]
            mean, std = y_stats
            y_hat_test_raw = y_hat_test_z * std.squeeze() + mean.squeeze()
            sq_rmse, sq_r2 = float(rmse(y_test, y_hat_test_raw)), float(r2(y_test, y_hat_test_raw))
            rows.append({"dataset": dataset, "algo": "setquery_zeroshot", "split": split,
                        "rmse_raw": sq_rmse, "r2": sq_r2})
            print(f"  [{dataset}/split{split}/setquery_zeroshot] rmse={sq_rmse:.4f} r2={sq_r2:.4f}")

            recon_test_r2 = float(r2(Xte100.reshape(-1), X_hat[n_train:].reshape(-1)))
            recon_rows.append({"dataset": dataset, "split": split, "recon_r2_test": recon_test_r2})
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

    recon_df = pd.DataFrame(recon_rows)
    recon_df.to_csv(RECON_CSV, index=False)
    print(f"\nwrote feature-reconstruction R2 (diagnostic only) -> {RECON_CSV}")
    print(recon_df.groupby("dataset")["recon_r2_test"].median().to_string())


if __name__ == "__main__":
    main()
