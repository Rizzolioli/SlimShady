"""
Vendored TabPFN v1 synthetic-data prior (regression variant).

Adapted from https://github.com/automl/TabPFN, tag v1.0.0 (MIT license,
Copyright 2022 The University of Freiburg — Noah Hollmann, Samuel Mueller,
Katharina Eggensperger, Frank Hutter):
  - tabpfn/priors/mlp.py                (the causal-MLP / BNN generator)
  - tabpfn/model_configs.get_diff_causal (hyperparameter distributions)
  - tabpfn/priors/differentiable_prior   (meta-distribution samplers)

Adaptations for TabGPGO:
  - the continuous scalar output is kept as the regression target y
    (the flexible_categorical classification binning step is skipped);
  - the config/dataloader machinery is replaced by `sample_prior_config()`
    which draws each dataset's hyperparameters from the same distributions
    (meta_choice weights simplified to uniform choices);
  - one MLP per dataset, batch dimension removed.
"""
import math
import random

import numpy as np
import torch
from torch import nn
from scipy import stats


# ---------------------------------------------------------------------------
# Samplers (tabpfn/priors/utils.py lines 103-105)
# ---------------------------------------------------------------------------

def _trunc_norm(mu, sigma):
    return stats.truncnorm((0 - mu) / sigma, (1000000 - mu) / sigma,
                           loc=mu, scale=sigma).rvs(1)[0]


def _meta_gamma(max_alpha, max_scale, lower_bound):
    # differentiable_prior.py meta_gamma: alpha=exp(U(0, log max_alpha)),
    # scale=U(0, max_scale); sample gamma(alpha, scale/alpha), round, offset.
    alpha = math.exp(random.uniform(0.0, math.log(max_alpha)))
    scale = random.uniform(0.0, max_scale)
    return lower_bound + round(np.random.gamma(alpha, scale / alpha))


def _meta_trunc_norm_log_scaled(min_mean, max_mean, lower_bound,
                                min_std=0.01, max_std=1.0):
    log_mean = random.uniform(math.log(min_mean), math.log(max_mean))
    log_std = random.uniform(math.log(min_std), math.log(max_std))
    return lower_bound + _trunc_norm(math.exp(log_mean),
                                     math.exp(log_mean) * math.exp(log_std))


def _meta_beta(scale, mn, mx):
    b, k = random.uniform(mn, mx), random.uniform(mn, mx)
    return scale * np.random.beta(b, k)


def sample_prior_config():
    """Draw one dataset's hyperparameters (model_configs.get_diff_causal)."""
    return {
        "num_layers": _meta_gamma(max_alpha=2, max_scale=3, lower_bound=2),
        "prior_mlp_hidden_dim": _meta_gamma(max_alpha=3, max_scale=100, lower_bound=4),
        "prior_mlp_dropout_prob": _meta_beta(scale=0.6, mn=0.1, mx=5.0),
        "noise_std": _meta_trunc_norm_log_scaled(0.0001, 0.3, lower_bound=0.0),
        "init_std": _meta_trunc_norm_log_scaled(0.01, 10.0, lower_bound=0.0),
        "num_causes": _meta_gamma(max_alpha=3, max_scale=7, lower_bound=2),
        "is_causal": random.choice([True, False]),
        "pre_sample_weights": random.choice([True, False]),
        "y_is_effect": random.choice([True, False]),
        "sampling": random.choice(["normal", "mixed"]),
        "prior_mlp_activations": random.choice([nn.Tanh, nn.Identity, nn.ReLU]),
        "block_wise_dropout": random.choice([True, False]),
        "sort_features": random.choice([True, False]),
        "in_clique": random.choice([True, False]),
        # fixed in get_general_config()
        "pre_sample_causes": True,
        "prior_mlp_scale_weights_sqrt": True,
        "random_feature_rotation": True,
    }


# ---------------------------------------------------------------------------
# Generator MLP (tabpfn/priors/mlp.py, batch dimension removed)
# ---------------------------------------------------------------------------

class _GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.normal(torch.zeros_like(x), self.std)


class _PriorMLP(nn.Module):
    def __init__(self, n_rows, num_features, hp):
        super().__init__()
        self.n_rows = n_rows
        self.num_features = num_features
        for key, val in hp.items():
            setattr(self, key, val)

        with torch.no_grad():
            if self.is_causal:
                self.prior_mlp_hidden_dim = max(self.prior_mlp_hidden_dim,
                                                1 + 2 * num_features)
            else:
                self.num_causes = num_features

            if self.pre_sample_causes:
                means = np.random.normal(0, 1, self.num_causes)
                stds = np.abs(np.random.normal(0, 1, self.num_causes) * means)
                self.causes_mean = torch.tensor(means).unsqueeze(0).tile((n_rows, 1)).float()
                self.causes_std = torch.tensor(stds).unsqueeze(0).tile((n_rows, 1)).float()

            def generate_module(out_dim):
                noise = (_GaussianNoise(torch.abs(torch.normal(
                            torch.zeros(1, out_dim), float(self.noise_std))))
                         if self.pre_sample_weights
                         else _GaussianNoise(float(self.noise_std)))
                return nn.Sequential(self.prior_mlp_activations(),
                                     nn.Linear(self.prior_mlp_hidden_dim, out_dim),
                                     noise)

            layers = [nn.Linear(self.num_causes, self.prior_mlp_hidden_dim)]
            layers += [generate_module(self.prior_mlp_hidden_dim)
                       for _ in range(self.num_layers - 1)]
            if not self.is_causal:
                layers += [generate_module(1)]
            self.layers = nn.Sequential(*layers)

            for i, (name, p) in enumerate(self.layers.named_parameters()):
                if self.block_wise_dropout:
                    if len(p.shape) == 2:
                        nn.init.zeros_(p)
                        n_blocks = random.randint(1, math.ceil(math.sqrt(min(p.shape))))
                        w, h = p.shape[0] // n_blocks, p.shape[1] // n_blocks
                        keep_prob = (n_blocks * w * h) / p.numel()
                        for b in range(n_blocks):
                            nn.init.normal_(
                                p[w * b: w * (b + 1), h * b: h * (b + 1)],
                                std=self.init_std / keep_prob ** (
                                    1 / 2 if self.prior_mlp_scale_weights_sqrt else 1))
                else:
                    if len(p.shape) == 2:
                        dropout_prob = self.prior_mlp_dropout_prob if i > 0 else 0.0
                        dropout_prob = min(dropout_prob, 0.99)
                        nn.init.normal_(p, std=self.init_std / (
                            1. - dropout_prob ** (
                                1 / 2 if self.prior_mlp_scale_weights_sqrt else 1)))
                        p *= torch.bernoulli(torch.zeros_like(p) + 1. - dropout_prob)

    @torch.no_grad()
    def forward(self):
        n_rows, num_features = self.n_rows, self.num_features

        def sample_normal():
            if self.pre_sample_causes:
                return torch.normal(self.causes_mean, self.causes_std.abs()).float()
            return torch.normal(0., 1., (n_rows, self.num_causes)).float()

        if self.sampling == "normal":
            causes = sample_normal()
        elif self.sampling == "mixed":
            zipf_p, multi_p, normal_p = (random.random() * 0.66,
                                         random.random() * 0.66,
                                         random.random() * 0.66)

            def sample_cause(n):
                if random.random() > normal_p:
                    if self.pre_sample_causes:
                        return torch.normal(self.causes_mean[:, n],
                                            self.causes_std[:, n].abs()).float()
                    return torch.normal(0., 1., (n_rows,)).float()
                elif random.random() > multi_p:
                    x = torch.multinomial(torch.rand(random.randint(2, 10)),
                                          n_rows, replacement=True).float()
                    return (x - torch.mean(x)) / torch.std(x)
                else:
                    x = torch.minimum(
                        torch.tensor(np.random.zipf(2.0 + random.random() * 2,
                                                    size=(n_rows,))).float(),
                        torch.tensor(10.0))
                    return x - torch.mean(x)

            causes = torch.stack([sample_cause(n) for n in range(self.num_causes)], -1)
        elif self.sampling == "uniform":
            causes = torch.rand((n_rows, self.num_causes))
        else:
            raise ValueError(f"invalid sampling: {self.sampling}")

        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        outputs = outputs[2:]

        if self.is_causal:
            outputs_flat = torch.cat(outputs, -1)
            if self.in_clique:
                start = random.randint(0, outputs_flat.shape[-1] - 1 - num_features)
                random_perm = start + torch.randperm(1 + num_features)
            else:
                random_perm = torch.randperm(outputs_flat.shape[-1] - 1)

            random_idx_y = (list(range(-1, 0)) if self.y_is_effect
                            else random_perm[0:1])
            random_idx = random_perm[1:1 + num_features]
            if self.sort_features:
                random_idx, _ = torch.sort(random_idx)
            y = outputs_flat[:, random_idx_y].squeeze(-1)
            x = outputs_flat[:, random_idx]
        else:
            y = outputs[-1].squeeze(-1)
            x = causes

        if self.random_feature_rotation:
            x = x[..., (torch.arange(x.shape[-1])
                        + random.randrange(x.shape[-1])) % x.shape[-1]]

        return x, y


def generate_dataset(n_rows, n_features, max_tries=10):
    """Generate one synthetic regression dataset (X: (n_rows, k), y: (n_rows,)).

    Uses the global `random`/`numpy`/`torch` RNG state — callers seed per
    dataset. Degenerate draws (NaN/Inf or constant target) are resampled with
    a fresh hyperparameter config.
    """
    for _ in range(max_tries):
        hp = sample_prior_config()
        x, y = _PriorMLP(n_rows, n_features, hp)()
        finite = torch.isfinite(x).all() and torch.isfinite(y).all()
        if finite and y.std() > 1e-8:
            return x.float(), y.float()
    raise RuntimeError("tabpfn_v1 prior: could not draw a non-degenerate dataset "
                       f"in {max_tries} tries")
