"""
TabPFN baseline: fit + predict TabPFN V2 (oldest regression-capable release)
and V3 (latest, released May 2026) on each of TabGPGO's 6 real validation
datasets, for a bottom-of-dashboard comparison against TabGPGO's own
best-per-dataset results.

Protocol (per explicit user decisions -- see the plan this script was built
from):
  - Full unsplit dataset: fit and predict on the same data, matching
    TabGPGO's own val_r2 protocol exactly (tabgpgo/evolution.py::
    elite_val_metrics is diagnostic tracking on the whole unsplit set, never
    a held-out test split -- see tabgpgo/preprocessing.py::
    load_validation_sets' docstring). TabPFN is in-context/inference-only
    (no gradient-based fitting on the target dataset), so this doesn't carry
    the usual train/test leakage concern a from-scratch-trained model would.
  - Same input representation TabGPGO itself uses: `load_validation_sets`
    (reused unmodified) already produces each dataset's PCA-to-100-feature,
    standardized/padded X and z-scored y -- both models see the identical
    input space.

Note on versions: TabPFN v1 (2022) was classification-only and never had a
regressor; the current unified `tabpfn` package's ModelVersion enum only
exposes V2/V2.5/V2.6/V3 (confirmed empirically -- no V1 at all). V2 is used
here as the earliest regression-capable stand-in for "the old one", clearly
labeled as such, not mislabeled as v1.

TabPFN V3 requires a one-time interactive license acceptance
(https://ux.priorlabs.ai) before its weights can download -- this only needs
doing once per machine (cached locally afterward), and can't be automated
here since it requires a human to log in.

Usage:
    python tabpfn_baseline/run_tabpfn.py
"""
import csv
import os
import sys
import time

import numpy as np

# concrete has 1030 rows, just over TabPFN's default CPU safeguard (1000) --
# override rather than let that combo come back "unsupported" every run.
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tabgpgo.config import TabGPGOConfig
from tabgpgo.preprocessing import load_validation_sets
from evaluators.fitness_functions import r2, rmse
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(THIS_DIR, "log")
LOG_PATH = os.path.join(LOG_DIR, "tabpfn_results.csv")

# "v1" is deliberately not here -- TabPFN v1 never supported regression (see
# module docstring). V2 stands in as the earliest regression-capable release.
VERSIONS = [("v2", ModelVersion.V2), ("v3", ModelVersion.V3)]


def fit_predict_one(version_enum, X, y):
    model = TabPFNRegressor.create_default_for_version(version_enum)
    model.fit(X, y)
    return model.predict(X)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    cfg = TabGPGOConfig()
    val_sets = load_validation_sets(cfg)

    rows = []
    for dataset, d in val_sets.items():
        X = d["X"].numpy().astype(np.float32)
        y = d["y"].numpy().astype(np.float32)
        y_mean, y_std = d["meta"]["y_stats"]
        y_mean, y_std = float(y_mean.squeeze()), float(y_std.squeeze())

        for version_name, version_enum in VERSIONS:
            print(f"[{dataset} / {version_name}] fitting...")
            t0 = time.time()
            status = "ok"
            rmse_scaled = rmse_raw = r2_val = None
            try:
                pred = fit_predict_one(version_enum, X, y)
                pred_t = d["y"].new_tensor(pred)
                rmse_scaled = float(rmse(d["y"], pred_t))
                r2_val = float(r2(d["y"], pred_t))
                rmse_raw = float(rmse(d["y"] * y_std + y_mean, pred_t * y_std + y_mean))
            except Exception as exc:
                status = f"unsupported: {exc}"
            elapsed = time.time() - t0
            print(f"  status={status} rmse_scaled={rmse_scaled} r2={r2_val} "
                  f"({elapsed:.1f}s)")
            rows.append({
                "dataset": dataset, "tabpfn_version": version_name,
                "rmse_scaled": rmse_scaled, "rmse_raw": rmse_raw, "r2": r2_val,
                "status": status,
            })

    with open(LOG_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"{'dataset':<24}{'version':<8}{'rmse_scaled':>12}{'r2':>10}{'status':>14}")
    for row in rows:
        rmse_str = f"{row['rmse_scaled']:.4f}" if row["rmse_scaled"] is not None else "n/a"
        r2_str = f"{row['r2']:.4f}" if row["r2"] is not None else "n/a"
        print(f"{row['dataset']:<24}{row['tabpfn_version']:<8}{rmse_str:>12}{r2_str:>10}"
              f"{row['status']:>14}")
    print(f"\nlogged -> {LOG_PATH}")


if __name__ == "__main__":
    main()
