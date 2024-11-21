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

def sign(y_true, y_pred):
    sign_y_true = torch.greater_equal(y_true, 0)
    sign_y_pred = torch.greater_equal(y_pred, 0)
    return torch.sum(torch.eq(sign_y_true, sign_y_pred))

def sign_rmse(y_true, y_pred):
    return torch.add(rmse(y_true, y_pred), torch.mul(torch.div(rmse(y_true, y_pred), y_true.size()[0]), sign(y_true,y_pred)))
