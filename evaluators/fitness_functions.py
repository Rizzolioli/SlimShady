import torch
import torch.nn.functional as F

# def rmse(y_true, y_pred):
    # return torch.sqrt_(F.mse_loss(y_true, y_pred))
def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean(torch.pow(torch.sub(y_true, y_pred), 2), len(y_pred.shape)-1))
def mse(y_true, y_pred):
    return torch.mean(torch.pow(torch.sub(y_true, y_pred), 2), len(y_pred.shape)-1)

def r2(y_true, y_pred):
    dim = len(y_true.shape) - 1
    ss_res = torch.sum(torch.pow(torch.sub(y_true, y_pred), 2), dim)
    ss_tot = torch.sum(torch.pow(torch.sub(y_true, torch.mean(y_true, dim, keepdim=True)), 2), dim)
    return 1 - ss_res / ss_tot

def mae(y_true, y_pred):
    return torch.mean(torch.abs(torch.sub(y_true, y_pred)), len(y_pred.shape)-1)

def mae_int(y_true, y_pred):
    return torch.mean(torch.abs(torch.sub(y_true, torch.round(y_pred))), len(y_pred.shape)-1)

def signed_errors(y_true, y_pred):
    return torch.sub(y_true, y_pred)

def sign(y_true, y_pred):
    sign_y_true = torch.greater_equal(y_true, 0)
    sign_y_pred = torch.greater_equal(y_pred, 0)
    return torch.sum(torch.ne(sign_y_true, sign_y_pred))

def sign_rmse(y_true, y_pred):
    return torch.add(rmse(y_true, y_pred), torch.mul(torch.div(rmse(y_true, y_pred), y_true.size()[0]), sign(y_true,y_pred)))

def linear_scaling(y_true, y_pred):
    """Keijzer/Bianco linear scaling: closed-form (a, b) minimizing
    sum((y_true - (a + b*y_pred))^2). Applying a+b*y_pred is guaranteed to
    never increase RMSE vs. y_pred alone (b=0 recovers the constant predictor
    when y_pred has ~zero variance)."""
    t_mean, p_mean = torch.mean(y_true), torch.mean(y_pred)
    p_centered = y_pred - p_mean
    denom = torch.sum(p_centered * p_centered)
    b = torch.sum((y_true - t_mean) * p_centered) / denom if denom > 1e-12 else torch.zeros_like(denom)
    a = t_mean - b * p_mean
    return a, b
