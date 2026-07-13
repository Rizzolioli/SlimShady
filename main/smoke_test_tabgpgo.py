"""
End-to-end CPU smoke test for the TabGPGO pipeline.

Runs the whole pipeline (both prior backends' generation, preprocessing,
autoencoder, tree pool, all SLIM variants x ms_hi sweep run concurrently,
inference equivalence, artifact-cache reuse, persistence round-trip,
standalone prediction) at tiny sizes with hard assertions. Must finish in
minutes on a CPU-only machine.

Usage:  python main/smoke_test_tabgpgo.py
"""
import os
import sys
import shutil
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from datasets.data_loader import load_merged_data
from evaluators.fitness_functions import rmse
from main_tabgpgo import evaluate_autoencoder, evolve, prepare
from tabgpgo.config import TabGPGOConfig
from tabgpgo.inference import TabGPGOPredictor, load_run
from tabgpgo.prior import generate_datasets
from utils.utils import train_test_split

tmp_dir = tempfile.mkdtemp(prefix="tabgpgo_smoke_")
cfg = TabGPGOConfig(
    device="cpu",
    prior_backend="simple_scm",
    n_synth_datasets=20, n_rows=100, min_features=2, max_features=100,
    val_datasets=("instanbul", "energy", "concrete", "ppb"),  # incl. one >100-feat
    reducer="pca",
    ae_epochs=5, ae_batch=512,
    pool_size=200,
    pop_size=20, n_gens=5, n_runs=1,
    log_path=os.path.join(tmp_dir, "results.csv"),
    run_dir_base=os.path.join(tmp_dir, "runs"),
    artifacts_dir=os.path.join(tmp_dir, "artifacts"),
)

# --- prior backends generate valid, reproducible data ------------------------
for backend in ("simple_scm", "tabpfn_v1"):
    for X, y in generate_datasets(3, 50, 30, 2, seed=1, backend=backend):
        assert X.dtype == torch.float32 and X.shape[0] == 50
        assert torch.isfinite(X).all() and torch.isfinite(y).all()
        assert y.std() > 1e-8
print("prior backends OK")

# --- a-priori AE check, decoupled from the evolution run -----------------------
ctx = prepare(cfg, verbose=1)
ae_stats = evaluate_autoencoder(cfg, ctx, verbose=1)
assert set(ae_stats) == {"synthetic(train)", *cfg.val_datasets}
for name, s in ae_stats.items():
    assert s["mse"] >= 0 and s["mse"] == s["mse"], (name, s)  # finite, non-negative
    assert s["r2"] == s["r2"], (name, s)  # finite (can be negative if the AE fails to generalize)
print("a-priori AE reconstruction check OK")

# --- full pipeline: variants x ms_hi sweep, concurrent (reuses prepared artifacts) --
elites, run_id = evolve(cfg, ctx, verbose=1)

assert ctx["T_train"].shape == (20 * 100, cfg.latent_dim)
assert ctx["T_train"].device.type == cfg.get_device().type
assert ctx["pool_train"].shape == (cfg.pool_size, 20 * 100)
# ppb (628 feats) must be reduced column-wise to exactly max_features comps
assert ctx["val_sets"]["ppb"]["meta"]["k"] == cfg.max_features
assert len(elites) == len(cfg.variants) * len(cfg.ms_hi_values) * cfg.n_runs
for (algo, seed), elite in elites.items():
    assert torch.isfinite(torch.tensor(elite.fitness)), f"{algo}: bad fitness"
print("pipeline shapes/fitness OK")

# --- artifact cache: second prepare() must reuse, not recompute ----------------
for fname in ("synthetic_pool.pt", "val_sets.pkl", "autoencoder.pt",
              "T_train.pt", "T_val.pt", "registry.pkl"):
    assert os.path.exists(os.path.join(cfg.artifacts_dir, fname)), fname
t0 = time.time()
ctx2 = prepare(cfg, verbose=False)
assert torch.equal(ctx2["T_train"], ctx["T_train"]), "cache changed tokens"
print(f"artifact cache reuse OK ({time.time() - t0:.1f}s)")

# --- CSV: one row per generation (incl. gen 0) per run -------------------------
with open(cfg.log_path) as fh:
    n_rows_csv = sum(1 for _ in fh)
expected = len(cfg.variants) * len(cfg.ms_hi_values) * cfg.n_runs * (cfg.n_gens + 1)
assert n_rows_csv == expected, f"CSV rows {n_rows_csv} != {expected}"
print("CSV logging OK")

# --- persistence round-trip + standalone predictor -----------------------------
# inference must work WITHOUT registry.pkl (elite.json is self-contained)
run_dirs = sorted(os.listdir(cfg.run_dir_base))
assert len(run_dirs) == len(cfg.variants) * len(cfg.ms_hi_values) * cfg.n_runs
run_dir = os.path.join(cfg.run_dir_base, run_dirs[0])
os.remove(os.path.join(run_dir, "registry.pkl"))
loaded_cfg, model, registry, elite, wrapper, operator = load_run(run_dir)

X_raw, y_raw = load_merged_data("energy", X_y=True)
X_raw, y_raw = X_raw.float(), y_raw.float()

predictor = TabGPGOPredictor(elite, registry, model, loaded_cfg)
preds_scaled = predictor.predict(X_raw)                # zero-shot, no fit() call
predictor.fit(X_raw, y_raw)                             # derive y_stats only (fine_tune=False)
preds_raw = predictor.predict(X_raw)

assert preds_scaled.shape == (X_raw.shape[0],) and torch.isfinite(preds_scaled).all()
assert preds_raw.shape == (X_raw.shape[0],) and torch.isfinite(preds_raw).all()
mean, std = predictor.y_stats
mean, std = mean.squeeze(), std.squeeze()
y_scaled = (y_raw - mean) / std
# raw-unit inversion must round-trip exactly against the scaled prediction
assert torch.allclose(preds_raw, preds_scaled * std + mean, atol=1e-4)
print(f"standalone predict on energy: rmse (scaled) = "
      f"{float(rmse(y_scaled, preds_scaled)):.4f}  "
      f"rmse (raw units) = {float(rmse(y_raw, preds_raw)):.4f}")
print("persistence + inference OK (registry.pkl not needed)")

# --- fine-tuning: per-block weights adapt to a labeled context split -----------
# pick a run whose elite actually has blocks, so the optimization path (not
# just its head-only no-op branch) is genuinely exercised.
ft_elite = None
for d in run_dirs:
    c, m, r, e, _, _ = load_run(os.path.join(cfg.run_dir_base, d))
    if e.blocks:
        ft_cfg, ft_model, ft_registry, ft_elite = c, m, r, e
        break
assert ft_elite is not None, "no run in this smoke sweep produced a blocked elite"

X_ctx, X_hold, y_ctx, y_hold = train_test_split(X_raw, y_raw, p_test=0.5, seed=0)
ft_predictor = TabGPGOPredictor(ft_elite, ft_registry, ft_model, ft_cfg)
orig_weights = ft_predictor.weights.clone()
ft_predictor.fit(X_ctx, y_ctx, fine_tune=True, steps=50)
preds_hold = ft_predictor.predict(X_hold)
assert preds_hold.shape == (X_hold.shape[0],) and torch.isfinite(preds_hold).all()
assert not torch.allclose(ft_predictor.weights, orig_weights), \
    "fine_tune=True did not move any block weights"
print(f"fine-tuned predict on energy held-out split: rmse (raw units) = "
      f"{float(rmse(y_hold, preds_hold)):.4f}  "
      f"(weights moved from {orig_weights.tolist()} to {ft_predictor.weights.tolist()})")
print("fine-tuning OK")

shutil.rmtree(tmp_dir, ignore_errors=True)
print("\nSMOKE TEST PASSED")
