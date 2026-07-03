"""
Phase 2: the a-priori MLP autoencoder.

Trained unsupervised (MSE) on the pooled synthetic X; afterwards the decoder
is discarded and the frozen encoder maps any 100-feature input to the static
512-dim latent tokens the symbolic evolution operates on.
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

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_train, cfg, verbose=True):
    """Train on pooled synthetic X (already on device); return frozen model."""
    device = cfg.get_device()
    model = MLPAutoencoder(cfg.max_features, cfg.ae_hidden, cfg.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ae_lr)
    loss_fn = nn.MSELoss()
    n = X_train.shape[0]
    for epoch in range(cfg.ae_epochs):
        perm = torch.randperm(n, device=X_train.device)
        total, batches = 0.0, 0
        for start in range(0, n, cfg.ae_batch):
            batch = X_train[perm[start:start + cfg.ae_batch]]
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        if verbose:
            print(f"  AE epoch {epoch + 1}/{cfg.ae_epochs}  train MSE {total / batches:.5f}")
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def reconstruction_mse(model, X):
    return nn.functional.mse_loss(model(X), X).item()


@torch.no_grad()
def encode(model, X, batch=8192):
    """Pass X through the frozen encoder in batches; returns latent tokens."""
    return torch.cat([model.encoder(X[i:i + batch]) for i in range(0, X.shape[0], batch)])
