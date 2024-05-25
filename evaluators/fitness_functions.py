import torch
import torch.nn.functional as F


# def rmse(y_true, y_pred):
# return torch.sqrt_(F.mse_loss(y_true, y_pred))
def rmse(y_true, y_pred):
    return torch.sqrt(
        torch.mean(
            torch.pow(
                torch.sub(
                    y_true, y_pred), 2), len(
                y_pred.shape) - 1))


def mse(y_true, y_pred):
    return torch.mean(
        torch.pow(
            torch.sub(
                y_true, y_pred), 2), len(
            y_pred.shape) - 1)


def mae(y_true, y_pred):
    return torch.mean(
        torch.abs(
            torch.sub(
                y_true, y_pred)), len(
            y_pred.shape) - 1)


def mae_int(y_true, y_pred):
    return torch.mean(
        torch.abs(
            torch.sub(
                y_true, torch.round(y_pred))), len(
            y_pred.shape) - 1)


def signed_errors(y_true, y_pred):
    return torch.sub(y_true, y_pred)
