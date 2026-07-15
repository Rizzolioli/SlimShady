"""
Phase 2 (TabPFN-encoder variant): frozen TabPFN embeddings in place of the
custom MLPAutoencoder (see tabgpgo/autoencoder.py).

TabPFN never "trains" in the usual sense -- it's a frozen, pretrained
in-context transformer; `.fit()` only stores a training context for the
frozen network to attend over at inference time, it never updates a single
weight. That means, unlike the MLP autoencoder (one global function
X -> latent, trained once on the pooled synthetic X and applied identically
everywhere afterward), TabPFN's embeddings are tied to whatever context you
fit it on -- there is no way to embed the whole ~500k-row pooled tensor in
one shot the way encode() does.

So the pool is built per SYNTHETIC DATASET instead of per pooled tensor:
each of cfg.n_synth_datasets datasets (cfg.n_rows rows each) gets its own
fresh TabPFNRegressor fit on its own (X, y), then embedded via
get_embeddings(X, data_source="train") -- mirroring exactly how the pool is
already built dataset-by-dataset in
tabgpgo.preprocessing.build_synthetic_pool, just swapping in TabPFN's
embedding as this dataset's own tokens instead of concatenating raw
(standardized/padded) features directly.

n_estimators=1 (not TabPFNRegressor's default of 8) is deliberate: measured
empirically, embedding 500 rows costs ~7.9s at n_estimators=1 vs ~177s at
the default 8 -- a ~22x difference dominated by ensemble members that would
be pure surplus cost here (we need embeddings AT ALL, not a well-averaged
ensemble of them).

Real validation data must NOT be fit on its own (X, y) -- that would use
the real target as TabPFN's in-context conditioning, leaking target
information into the embedding itself before evolution even starts, unlike
every other zero-shot dashboard this session (whose encoder never sees real
y at all). Instead, the FIRST synthetic dataset's already-fitted
TabPFNRegressor is kept as a frozen "reference encoder" and reused, via
get_embeddings(X, data_source="test"), to embed every real dataset --
mirroring the MLP autoencoder's own "train once on synthetic, freeze, apply
unchanged to real" shape.
"""
import torch

from tabpfn import TabPFNRegressor

from .preprocessing import standardize_scale_pad
from .prior import generate_datasets


def _embed(model, X, data_source):
    """(n_estimators, n_rows, embed_dim) -> (n_rows, embed_dim), squeezing
    out the estimator axis (n_estimators=1 throughout this module, so axis
    0 always has size 1)."""
    emb = model.get_embeddings(X.numpy(), data_source=data_source)
    return torch.from_numpy(emb[0]).float()


def build_tabpfn_pool(cfg, n_estimators=1, verbose=True):
    """Phase 1-2 (TabPFN-encoder variant): builds (T_train, y_target,
    reference_model, embed_dim) in place of build_synthetic_pool +
    train_autoencoder + encode.

    T_train is the concatenation of every synthetic dataset's own frozen
    TabPFN train-mode embeddings ((n_synth_datasets*n_rows, embed_dim));
    y_target is the matching concatenation of standardized targets, same
    convention as build_synthetic_pool. reference_model is the FIRST
    dataset's fitted TabPFNRegressor, kept around to embed real validation
    data later (see encode_val_tabpfn) -- it is never refit on real data.
    """
    T_chunks, y_chunks = [], []
    reference_model = None
    embed_dim = None
    for i, (X, y) in enumerate(generate_datasets(
            cfg.n_synth_datasets, cfg.n_rows, cfg.max_features, cfg.min_features,
            seed=cfg.data_seed, backend=cfg.prior_backend)):
        X100, y_std, _ = standardize_scale_pad(X, y, cfg.max_features)
        model = TabPFNRegressor(n_estimators=n_estimators, random_state=cfg.data_seed + i)
        model.fit(X100.numpy(), y_std.numpy())
        emb = _embed(model, X100, "train")
        if embed_dim is None:
            embed_dim = emb.shape[1]
        T_chunks.append(emb)
        y_chunks.append(y_std)
        if reference_model is None:
            reference_model = model
        if verbose and (i + 1) % 50 == 0:
            print(f"  [tabpfn pool] {i + 1}/{cfg.n_synth_datasets} synthetic datasets embedded")
    T_train = torch.cat(T_chunks)
    y_target = torch.cat(y_chunks)
    return T_train, y_target, reference_model, embed_dim


def encode_val_tabpfn(reference_model, X):
    """Embed real validation data via the frozen reference_model's
    test-mode pass -- never refit on X's own target, so this stays a
    genuine zero-shot embedding, exactly like the MLP autoencoder's own
    (X -> latent) function applied unchanged to unseen data."""
    return _embed(reference_model, X, "test")
