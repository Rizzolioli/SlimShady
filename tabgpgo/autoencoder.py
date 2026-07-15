"""
Phase 2: the a-priori MLP autoencoder.

Trained unsupervised (MSE) on the pooled synthetic X; afterwards the decoder
is discarded and the frozen encoder maps any 100-feature input to the static
512-dim latent tokens the symbolic evolution operates on.

Optional target-aware auxiliary head (see train_autoencoder's y_target/
aux_weight): every downstream sweep this session (function sets, ms, linear
scaling, latent sizes) hit the same near-zero zero-shot ceiling regardless
of what else changed -- nothing in the encoder's own training objective
ever pushes it to keep target-relevant structure, only to reconstruct X.
The auxiliary head adds aux_weight * MSE(pred_head(encoder(x)), y) to the
training loss, jointly with reconstruction, so the encoder is now also
pushed to preserve whatever latent structure predicts y. Defaults (
y_target=None / aux_weight=0.0) reproduce the original unsupervised-only
training exactly, so every existing experiment's behavior is unaffected
unless a caller explicitly opts in.
"""
import torch
from torch import nn


class MLPAutoencoder(nn.Module):
    def __init__(self, n_features=100, hidden=256, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_features))
        # Auxiliary target-prediction head -- a separate submodule from
        # encoder/decoder, so save_run/load_run's encoder.state_dict()-only
        # persistence is completely unaffected by its presence.
        self.pred_head = nn.Linear(latent_dim, 1)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_train, cfg, y_target=None, verbose=True):
    """Train on pooled synthetic X (already on device); return frozen model.

    If y_target is given AND cfg.ae_aux_weight > 0, also trains the
    auxiliary pred_head to predict y_target from the latent code, jointly
    with reconstruction (see module docstring). y_target is otherwise
    accepted but ignored, so callers that don't care about the auxiliary
    head can pass it unconditionally without checking cfg.ae_aux_weight
    themselves (main_tabgpgo.py's prepare() does exactly this).
    """
    device = cfg.get_device()
    model = MLPAutoencoder(cfg.max_features, cfg.ae_hidden, cfg.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ae_lr)
    loss_fn = nn.MSELoss()
    use_aux = y_target is not None and cfg.ae_aux_weight > 0
    n = X_train.shape[0]
    for epoch in range(cfg.ae_epochs):
        perm = torch.randperm(n, device=X_train.device)
        total, total_recon, total_aux, batches = 0.0, 0.0, 0.0, 0
        for start in range(0, n, cfg.ae_batch):
            idx = perm[start:start + cfg.ae_batch]
            batch = X_train[idx]
            opt.zero_grad()
            z = model.encoder(batch)
            recon_loss = loss_fn(model.decoder(z), batch)
            if use_aux:
                aux_loss = loss_fn(model.pred_head(z).squeeze(-1), y_target[idx])
                loss = recon_loss + cfg.ae_aux_weight * aux_loss
            else:
                aux_loss = torch.zeros((), device=device)
                loss = recon_loss
            loss.backward()
            opt.step()
            total += loss.item()
            total_recon += recon_loss.item()
            total_aux += aux_loss.item()
            batches += 1
        if verbose:
            if use_aux:
                print(f"  AE epoch {epoch + 1}/{cfg.ae_epochs}  "
                      f"recon MSE {total_recon / batches:.5f}  aux MSE {total_aux / batches:.5f}")
            else:
                print(f"  AE epoch {epoch + 1}/{cfg.ae_epochs}  train MSE {total / batches:.5f}")
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def reconstruction_stats(model, X):
    """Full-model (encoder+decoder) reconstruction MSE and R^2 (fraction of
    X's overall variance explained), so datasets preprocessed at different
    scales (zero-padded low-feature-count vs PCA/RF-reduced) are comparable.
    """
    recon = model(X)
    mse = nn.functional.mse_loss(recon, X).item()
    var = X.var(unbiased=False).item()
    r2 = 1.0 - mse / var if var > 0 else float("nan")
    return mse, r2


@torch.no_grad()
def encode(model, X, batch=8192):
    """Pass X through the frozen encoder in batches; returns latent tokens."""
    return torch.cat([model.encoder(X[i:i + batch]) for i in range(0, X.shape[0], batch)])
