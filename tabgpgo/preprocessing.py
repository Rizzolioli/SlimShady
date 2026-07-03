"""
Phase 1: standardization to a fixed 100-feature input space.

Every dataset (synthetic or real) ends up as an (n, max_features) float32
matrix: per-feature z-score, magnitude scaling by max_features/k, zero-padding
to max_features columns. Real datasets with more than max_features columns are
first reduced to exactly max_features via Random-Forest importance or PCA.
"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

from datasets.data_loader import load_merged_data


def zscore(t, mean=None, std=None, eps=1e-8):
    """Column-wise z-score; constant columns are centered only."""
    if mean is None:
        mean = t.mean(dim=0, keepdim=True)
        std = t.std(dim=0, keepdim=True)
    return (t - mean) / torch.where(std < eps, torch.ones_like(std), std), mean, std


def standardize_scale_pad(X, y, max_features=100, x_stats=None, y_stats=None):
    """z-score X and y, scale features by max_features/k, zero-pad to max_features.

    Returns (X_padded, y_std, meta). Pass the returned meta's stats back in to
    apply a fitted transform to another split of the same dataset.
    """
    k = X.shape[1]
    if k > max_features:
        raise ValueError(f"dataset has {k} > {max_features} features; reduce first")
    X, x_mean, x_std = zscore(X, *(x_stats or (None, None)))
    y, y_mean, y_std = zscore(y.reshape(-1, 1), *(y_stats or (None, None)))
    y = y.flatten()
    X = X * (max_features / k)
    if k < max_features:
        X = torch.cat([X, torch.zeros(X.shape[0], max_features - k)], dim=1)
    meta = {"k": k, "x_stats": (x_mean, x_std), "y_stats": (y_mean, y_std)}
    return X.float(), y.float(), meta


def reduce_features(X, y, method, n=100, seed=0):
    """Reduce >n-feature data to exactly n columns (column-wise only).

    'rf':  top-n features by RandomForestRegressor importance (column subset).
    'pca': first n principal components of the z-scored features.
    Returns (X_n, reducer_meta); X_n is float32.
    """
    if method == "rf":
        rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        rf.fit(X.numpy(), y.numpy())
        top = np.argsort(rf.feature_importances_)[::-1][:n].copy()
        top.sort()  # keep original column order
        return X[:, top].clone(), {"method": "rf", "columns": top}
    elif method == "pca":
        Xs, mean, std = zscore(X)
        if Xs.shape[0] < n:
            raise ValueError(f"PCA to {n} components needs >= {n} rows, "
                             f"got {Xs.shape[0]}; use the 'rf' reducer")
        pca = PCA(n_components=n, random_state=seed)
        Xn = torch.from_numpy(pca.fit_transform(Xs.numpy())).float()
        return Xn, {"method": "pca", "pca": pca, "x_stats": (mean, std)}
    raise ValueError(f"unknown reducer: {method}")


def build_synthetic_pool(cfg):
    """Generate, standardize and concatenate all synthetic datasets.

    Returns (X_train (M, max_features), y_target (M,)) on cfg's device.
    """
    from .prior import generate_datasets
    xs, ys = [], []
    for X, y in generate_datasets(cfg.n_synth_datasets, cfg.n_rows,
                                  cfg.max_features, cfg.min_features,
                                  seed=cfg.data_seed, backend=cfg.prior_backend):
        X100, y_std, _ = standardize_scale_pad(X, y, cfg.max_features)
        xs.append(X100)
        ys.append(y_std)
    device = cfg.get_device()
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


def load_validation_sets(cfg):
    """Load the real validation datasets, standardized to max_features columns.

    Returns dict name -> {"X", "y", "meta"}. The full merged dataset is used
    unsplit: these sets never influence fitness (tracking only), so there is
    no train/test split and reducers/statistics are fitted on all rows.
    Tensors stay on CPU here (encoded and moved to device later).
    """
    out = {}
    for name in cfg.val_datasets:
        X, y = load_merged_data(name, X_y=True)
        X, y = X.float(), y.float()
        reducer_meta = None
        if X.shape[1] > cfg.max_features:
            X, reducer_meta = reduce_features(X, y, cfg.reducer,
                                              cfg.max_features, cfg.val_seed)
        X100, y_s, meta = standardize_scale_pad(X, y, cfg.max_features)
        meta["reducer"] = reducer_meta
        out[name] = {"X": X100, "y": y_s, "meta": meta}
    return out
