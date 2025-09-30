import torch
import torch.nn.functional as F

# def rmse(y_true, y_pred):
    # return torch.sqrt_(F.mse_loss(y_true, y_pred))
def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean(torch.pow(torch.sub(y_true, y_pred), 2), len(y_pred.shape)-1))
def mse(y_true, y_pred):
    return torch.mean(torch.pow(torch.sub(y_true, y_pred), 2), len(y_pred.shape)-1)

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