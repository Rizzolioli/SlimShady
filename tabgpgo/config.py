"""
Single source of truth for all TabGPGO pipeline hyperparameters.

Every stage (synthetic prior, preprocessing, autoencoder, tree pool,
evolution, inference) reads its knobs from a TabGPGOConfig instance, so the
smoke test and the full experiment differ only in the overrides they pass.
"""
import os
from dataclasses import dataclass, field

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A variant is (wrapper, operator). wrapper selects the mutation function
# used to build each new block: "abs" | "sig1" | "sig2" (fixed for the whole
# run) or "mix" (chosen uniformly at random, independently, for every new
# block). operator selects how blocks aggregate onto the head: "sum" | "mul"
# (fixed) or "mix" (chosen uniformly at random per block, sum vs mul). The
# *MIX variants keep the wrapper (or operator) that was actually drawn on
# each Block, so heterogeneous individuals remain fully reconstructable.
WRAPPERS = ("abs", "sig1", "sig2")
OPERATORS = ("sum", "mul")

VARIANTS = [
    ("sig2", "sum"),   # SLIM+2SIG
    ("sig2", "mul"),   # SLIM*2SIG
    ("sig1", "sum"),   # SLIM+1SIG
    ("sig1", "mul"),   # SLIM*1SIG
    ("abs",  "sum"),   # SLIM+ABS
    ("abs",  "mul"),   # SLIM*ABS
    ("mix",  "sum"),   # SLIM+MIX: wrapper uniform per block, sum-only aggregation
    ("mix",  "mul"),   # SLIM*MIX: wrapper uniform per block, mul-only aggregation
    ("mix",  "mix"),   # SLIM~MIX: wrapper AND aggregation both uniform per block
]

ALGO_NAMES = {
    ("sig2", "sum"): "SLIM+2SIG",
    ("sig2", "mul"): "SLIM*2SIG",
    ("sig1", "sum"): "SLIM+1SIG",
    ("sig1", "mul"): "SLIM*1SIG",
    ("abs",  "sum"): "SLIM+ABS",
    ("abs",  "mul"): "SLIM*ABS",
    ("mix",  "sum"): "SLIM+MIX",
    ("mix",  "mul"): "SLIM*MIX",
    ("mix",  "mix"): "SLIM~MIX",
}


@dataclass
class TabGPGOConfig:
    # --- device -------------------------------------------------------------
    device: str = "auto"                # "auto" | "cuda" | "cpu"

    # --- synthetic prior (Phase 1) -------------------------------------------
    prior_backend: str = "tabpfn_v1"    # "tabpfn_v1" | "simple_scm"
    n_synth_datasets: int = 1000
    n_rows: int = 500                   # rows per synthetic dataset
    max_features: int = 100
    min_features: int = 2
    data_seed: int = 42

    # --- real validation datasets (merged files, used unsplit) -----------------
    val_datasets: tuple = ("ppb", "toxicity", "resid_build_sale_price",
                           "instanbul", "energy", "concrete")
    reducer: str = "pca"                # "pca" | "rf" (for datasets with >100 features)
    val_seed: int = 1                   # random_state for the reducer

    # --- autoencoder (Phase 2) ------------------------------------------------
    ae_hidden: int = 256
    latent_dim: int = 512
    ae_epochs: int = 50
    ae_batch: int = 4096
    ae_lr: float = 1e-3

    # --- tree pool (Phase 3) ---------------------------------------------------
    pool_size: int = 5000
    pool_dtype: str = "float32"         # "float32" | "float16"
    init_depth: int = 6
    p_c: float = 0.0                    # probability of constants in random trees

    # --- evolution (Phase 4) ----------------------------------------------------
    pop_size: int = 200
    n_gens: int = 2000
    p_inflate: float = 0.5
    ms_lo: float = 0.0
    ms_hi: float = 1.0                  # default/fallback, overridden per run by ms_hi_values
    ms_hi_values: tuple = (1.0, 10.0, 100.0)   # mutation-step upper-bound sweep
    tournament_size: int = 2
    n_elites: int = 1
    n_runs: int = 1                     # seeds per variant
    variants: tuple = tuple(VARIANTS)
    max_workers: int = 4                # concurrent (variant, ms_hi, seed) runs in evolve()

    # --- paths / artifact caching -------------------------------------------------
    log_path: str = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_results.csv")
    run_dir_base: str = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_runs")
    # Phase 1-3 artifacts (synthetic data, encoder, latent tokens, tree
    # registry) are cached here and reused across invocations instead of
    # being recomputed. Delete the directory or set reuse_artifacts=False to
    # start fresh. Pool semantics (pool_size x n_rows, up to ~10 GB) are NOT
    # cached — they are recomputed from registry + tokens at startup.
    artifacts_dir: str = os.path.join(REPO_ROOT, "main", "log", "tabgpgo_artifacts")
    reuse_artifacts: bool = True

    def get_device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def get_pool_dtype(self):
        return {"float32": torch.float32, "float16": torch.float16}[self.pool_dtype]
