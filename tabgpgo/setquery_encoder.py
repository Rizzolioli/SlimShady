"""
Standalone experiment: a set-query autoencoder for the TabGPGO synthetic
prior, entirely independent of TabGPGO/SLIM-GSGP (no evolution, no symbolic
search) -- see tabpfn_baseline/run_setquery_eval.py for the training +
zero-shot-eval driver.

Design (derived over this session's conversation -- kept here for anyone
picking this back up):

  - The encoder NEVER sees the target column, at train OR eval time,
    matching how the frozen encoder is actually used downstream elsewhere
    in this repo (encoder always maps X only; y only ever enters as a
    training/fitting SIGNAL, never as model input -- see
    tabgpgo/autoencoder.py's own module docstring for the analogous
    ae_aux_weight convention already established there).

  - A naive "reconstruct D and D^T" objective, where D packs the target in
    as an extra column and a decoder is built from independently-computed
    per-row and per-column tokens (D[i,j] = dec(R_i, C_j)), is VACUOUS:
    D^T[j,i] is the exact same computation, so requiring both adds no new
    loss term, it's the same sum relabeled. The constraint only bites if
    the shared code has NO row- or column-indexed slots at all.

  - So instead: the encoder pools X (n rows x max_features) into a small,
    FIXED set of M latent vectors z via Perceiver-style cross-attention --
    no row- or column-specific slot survives into z, only a globally pooled
    summary. A single shared decoder is then queried with LEARNED,
    data-independent positional embeddings -- row_pos(i) + col_pos(j) -- to
    reconstruct entry X[i, j], for every (i, j) pair. This covers the whole
    matrix AND its transpose in one computation (they're the same entries),
    which is the correct resolution once the vacuity issue above is fixed:
    there's no meaningful way to keep "reconstruct D" and "reconstruct D^T"
    as two separate loss terms without reintroducing the same vacuity.

  - The target is treated as one more column slot, decoder-side only:
    row_pos(i) + target_marker -> predicted y_i. target_marker is a single
    extra learned vector, added ONLY on the decoder side -- the encoder's
    forward pass (encode()) never receives it or anything derived from y.

  - Row positions are also arbitrary indices within a single forward pass
    (row order is exchangeable, "row i" has no persistent identity across
    datasets) -- exactly like position embeddings in a masked-autoencoder
    decoder. Since real validation datasets range up to ~1000 rows while
    the synthetic prior's own n_rows is fixed at 500 (see
    TabGPGOConfig.n_rows), training samples a fresh random row count per
    synthetic dataset (ROW_RANGE below) so every row_pos index actually
    gets trained, not just the ones synthetic datasets would otherwise use.
"""
import random

import torch
from torch import nn

MAX_ROWS = 1536      # >= largest val dataset (concrete, 1030 rows) with slack
ROW_RANGE = (50, 1280)  # per-synthetic-dataset row count sampled from this range at train time


class SetQueryAutoencoder(nn.Module):
    def __init__(self, n_features=100, hidden=128, n_latents=32, n_pool_rounds=2, n_heads=4):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden

        self.row_proj = nn.Linear(n_features, hidden)
        self.latents = nn.Parameter(torch.randn(n_latents, hidden) * 0.02)
        self.pool_attn = nn.ModuleList(
            nn.MultiheadAttention(hidden, n_heads, batch_first=True) for _ in range(n_pool_rounds))
        self.pool_norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_pool_rounds))

        self.row_pos = nn.Embedding(MAX_ROWS, hidden)
        self.col_pos = nn.Embedding(n_features, hidden)
        self.target_marker = nn.Parameter(torch.randn(hidden) * 0.02)

        self.query_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.query_norm = nn.LayerNorm(hidden)
        self.out_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, X):
        """X: (n, n_features), features only -- never touches y. Returns the
        pooled bottleneck z: (n_latents, hidden), with no row/column-specific
        identity retained (see module docstring)."""
        tok = self.row_proj(X).unsqueeze(0)          # (1, n, hidden)
        z = self.latents.unsqueeze(0)                 # (1, n_latents, hidden)
        for attn, norm in zip(self.pool_attn, self.pool_norm):
            attn_out, _ = attn(z, tok, tok)
            z = norm(z + attn_out)
        return z.squeeze(0)                            # (n_latents, hidden)

    def _decode_queries(self, z, queries):
        """queries: (n_q, hidden) -> (n_q,) scalar predictions, each cross-
        attending into z independently of any other query."""
        attn_out, _ = self.query_attn(queries.unsqueeze(0), z.unsqueeze(0), z.unsqueeze(0))
        ctx = self.query_norm(queries + attn_out.squeeze(0))
        return self.out_head(ctx).squeeze(-1)

    def reconstruct_X(self, z, n_rows):
        """Full entrywise reconstruction of X (n_rows, n_features) -- the
        row axis and column axis (X^T) are reconstructed by the identical
        formula, since an entry and its transposed counterpart are the same
        query result (see module docstring)."""
        device = z.device
        ri, ci = torch.meshgrid(torch.arange(n_rows, device=device),
                                torch.arange(self.n_features, device=device), indexing="ij")
        queries = self.row_pos(ri.reshape(-1)) + self.col_pos(ci.reshape(-1))
        return self._decode_queries(z, queries).view(n_rows, self.n_features)

    def predict_target(self, z, n_rows):
        """Target-query prediction for every row, decoder-only: the target
        marker never enters encode(), only this query."""
        rows = torch.arange(n_rows, device=z.device)
        queries = self.row_pos(rows) + self.target_marker.unsqueeze(0)
        return self._decode_queries(z, queries)

    def forward(self, X):
        z = self.encode(X)
        n_rows = X.shape[0]
        return self.reconstruct_X(z, n_rows), self.predict_target(z, n_rows)


def train_setquery_ae(cfg, n_datasets=None, n_epochs=20, target_weight=1.0, lr=1e-3,
                      hidden=128, n_latents=32, verbose=True, seed=None, log_every=50):
    """Train on the synthetic prior: pre-generate `n_datasets` synthetic
    datasets ONCE (row/column pooling is inherently per-dataset, so each is
    still its own gradient step, not batched), then train `n_epochs` passes
    over that fixed, shuffled pool -- mirroring how tabgpgo/autoencoder.py's
    train_autoencoder trains many epochs over one pooled synthetic set,
    rather than one gradient step per dataset (confirmed via a controlled
    overfitting test this session: with only ~1 step/dataset and no repeats,
    both losses sit at their trivial baseline -- MSE~=1.0, i.e. "predict the
    mean" -- even though the same architecture visibly learns, given enough
    steps, on a small repeated set). Each dataset gets its own randomly
    sampled row count (ROW_RANGE) so every row_pos index gets trained --
    real validation datasets are larger than the synthetic prior's fixed
    n_rows (see module docstring)."""
    from .prior import generate_datasets
    from .preprocessing import standardize_scale_pad

    device = cfg.get_device()
    model = SetQueryAutoencoder(cfg.max_features, hidden, n_latents).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = n_datasets or cfg.n_synth_datasets
    seed = cfg.data_seed if seed is None else seed
    row_rng = random.Random(seed)

    if verbose:
        print(f"  generating {n} synthetic datasets (once, cached for {n_epochs} epochs)...")
    pool = []
    for i in range(n):
        n_rows_i = row_rng.randint(*ROW_RANGE)
        (X, y), = list(generate_datasets(1, n_rows_i, cfg.max_features, cfg.min_features,
                                         seed=seed + i, backend=cfg.prior_backend))
        X100, y_std, _ = standardize_scale_pad(X, y, cfg.max_features)
        pool.append((X100.to(device), y_std.to(device)))

    epoch_rng = random.Random(seed + 1)
    for epoch in range(n_epochs):
        order = list(range(n))
        epoch_rng.shuffle(order)
        total_recon, total_tgt, count = 0.0, 0.0, 0
        for idx in order:
            X100, y_std = pool[idx]
            opt.zero_grad()
            X_hat, y_hat = model(X100)
            recon_loss = nn.functional.mse_loss(X_hat, X100)
            target_loss = nn.functional.mse_loss(y_hat, y_std)
            loss = recon_loss + target_weight * target_loss
            loss.backward()
            opt.step()

            total_recon += recon_loss.item()
            total_tgt += target_loss.item()
            count += 1
        if verbose:
            print(f"  epoch {epoch + 1}/{n_epochs}  recon MSE {total_recon / count:.5f}  "
                 f"target MSE {total_tgt / count:.5f}")

    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def encode_and_predict(model, X100):
    """X100: (n, max_features), features only. Returns (X_hat, y_hat_std),
    both entirely zero-shot -- no fitting/adaptation, y never touched."""
    device = next(model.parameters()).device
    X100 = X100.to(device)
    z = model.encode(X100)
    n_rows = X100.shape[0]
    return model.reconstruct_X(z, n_rows).cpu(), model.predict_target(z, n_rows).cpu()
