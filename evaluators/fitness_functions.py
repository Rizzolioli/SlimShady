import torch
import torch.nn.functional as F
import math

# def rmse(y_true, y_pred):
    # return torch.sqrt_(F.mse_loss(y_true, y_pred))
def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean(torch.pow(torch.sub(y_true, y_pred), 2), len(y_pred.shape)-1))
def mse(y_true, y_pred):
    return torch.mean(torch.pow(torch.sub(y_true, y_pred), 2), len(y_pred.shape)-1)


# -------- Weibull time-concordance index (C-index) helpers -------- #

def _safe_positive(x: torch.Tensor, min_value: float = 1e-6) -> torch.Tensor:
    # ensure strictly positive parameters
    return torch.clamp(x, min=min_value)


def weibull_expected_time(params: torch.Tensor) -> torch.Tensor:
    """
    Compute expected time E[T] for a Weibull(k, lambda) distribution for each sample.

    params: tensor of shape [2, N] or [N, 2]: first row/col -> scale (lambda), second -> shape (k)
    returns: tensor [N]
    """
    if params.dim() != 2 or 2 not in params.shape:
        raise ValueError("params must be 2xN or Nx2 tensor for Weibull parameters")

    if params.shape[0] == 2:
        scale_raw, shape_raw = params[0], params[1]
    else:
        scale_raw, shape_raw = params[:, 0], params[:, 1]

    # map raw outputs to strictly positive domain
    scale = _safe_positive(torch.exp(scale_raw))
    shape = _safe_positive(torch.exp(shape_raw))

    # E[T] = lambda * Gamma(1 + 1/k)
    gamma_arg = 1.0 + 1.0 / shape
    # use torch.lgamma for numerical stability: Gamma(x) = exp(lgamma(x))
    expected_time = scale * torch.exp(torch.lgamma(gamma_arg))
    return expected_time


def time_concordance_index(y_true: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    """
    Harrell's C-index for continuous outcomes without censoring.
    y_true: [N]
    preds: [N] predicted scores (higher means larger predicted time)
    returns: scalar c-index in [0,1]
    """
    y_true = y_true.reshape(-1)
    preds = preds.reshape(-1)
    n = y_true.shape[0]
    if n < 2:
        return torch.tensor(1.0, dtype=y_true.dtype, device=y_true.device)

    # pairwise comparisons i<j
    idx_i = torch.arange(n, device=y_true.device).unsqueeze(1)
    idx_j = torch.arange(n, device=y_true.device).unsqueeze(0)
    mask = idx_i < idx_j

    yi = y_true.unsqueeze(1).expand(n, n)[mask]
    yj = y_true.unsqueeze(0).expand(n, n)[mask]
    pi = preds.unsqueeze(1).expand(n, n)[mask]
    pj = preds.unsqueeze(0).expand(n, n)[mask]

    # comparable if yi != yj
    comparable = (yi != yj)
    if comparable.any():
        yi = yi[comparable]
        yj = yj[comparable]
        pi = pi[comparable]
        pj = pj[comparable]
    else:
        return torch.tensor(0.5, dtype=y_true.dtype, device=y_true.device)

    concordant = ((yi < yj) & (pi < pj)) | ((yi > yj) & (pi > pj))
    ties = (pi == pj)
    c = (concordant.float().sum() + 0.5 * ties.float().sum()) / yi.numel()
    return c


def weibull_cindex_loss(y_true: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Loss = 1 - C-index using Weibull-predicted expected times as scores.
    params: raw outputs shaped [2, N] or [N, 2].
    """
    scores = weibull_expected_time(params)
    c = time_concordance_index(y_true, scores)
    return 1.0 - c

def mae(y_true, y_pred):
    return torch.mean(torch.abs(torch.sub(y_true, y_pred)), len(y_pred.shape)-1)

def mae_int(y_true, y_pred):
    return torch.mean(torch.abs(torch.sub(y_true, torch.round(y_pred))), len(y_pred.shape)-1)

def signed_errors(y_true, y_pred):
    return torch.sub(y_true, y_pred)


def inverse_r2(y_true, y_pred):
    # Calculate mean of true values along the appropriate dimension
    y_mean = torch.mean(y_true, dim=len(y_pred.shape) - 1, keepdim=True)

    # Sum of squared residuals
    ss_res = torch.sum(torch.pow(torch.sub(y_true, y_pred), 2), dim=len(y_pred.shape) - 1)

    # Total sum of squares
    ss_tot = torch.sum(torch.pow(torch.sub(y_true, y_mean), 2), dim=len(y_pred.shape) - 1)

    # R² score
    r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to avoid division by zero

    # Return 1 - R² for minimization
    # Clamp to handle negative R² values (worse than baseline)
    # When R² is negative, 1 - R² becomes > 1, which properly penalizes bad predictions
    return 1 - r2