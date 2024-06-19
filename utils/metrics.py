import numpy as np
import torch


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.astype(np.float32)
    mask /= np.mean((mask))
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(preds - labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)


# def MAE(preds, labels, null_val=np.nan):
#     preds = torch.Tensor(preds)
#     labels = torch.Tensor(labels)
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels!=null_val)
#     mask = mask.float()
#     mask /=  torch.mean((mask))
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs(preds-labels)
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def MSE(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.astype(np.float32)
    mask /= np.mean((mask))
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)


def RMSE(pred, true, null_value=np.nan):
    return np.sqrt(MSE(pred, true, null_value))


def MAPE(preds, labels, null_val: float = np.nan):
    """Masked mean absolute percentage error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value.
                                    In the mape metric, null_val is set to 0.0 by all default.
                                    We keep this parameter for consistency, but we do not allow it to be changed.
                                    Zeros in labels will lead to inf in mape. Therefore, null_val is set to 0.0 by default.

    Returns:
        torch.Tensor: masked mean absolute percentage error
    """
    # we do not allow null_val to be changed
    null_val = 0.0
    # delete small values to avoid abnormal results
    # TODO: support multiple null values
    labels = np.where(np.abs(labels) < 1e-4, np.zeros_like(labels), labels)
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        eps = 5e-5
        mask = ~np.isclose(labels, np.full_like(labels, fill_value=null_val), atol=eps, rtol=0.)
    mask = mask.astype(np.float32)
    mask /= np.mean((mask))
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(np.abs(preds - labels) / labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)


def MSPE(preds, labels, null_val: float = np.nan):
    """Masked mean square percentage error.

        Args:
            preds (torch.Tensor): predicted values
            labels (torch.Tensor): labels
            null_val (float, optional): null value.
                                        In the mape metric, null_val is set to 0.0 by all default.
                                        We keep this parameter for consistency, but we do not allow it to be changed.
                                        Zeros in labels will lead to inf in mape. Therefore, null_val is set to 0.0 by default.

        Returns:
            torch.Tensor: masked mean absolute percentage error
        """
    # we do not allow null_val to be changed
    null_val = 0.0
    # delete small values to avoid abnormal results
    # TODO: support multiple null values
    labels = np.where(np.abs(labels) < 1e-4, np.zeros_like(labels), labels)
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        eps = 5e-5
        mask = ~np.isclose(labels, np.full_like(labels, fill_value=null_val), atol=eps, rtol=0.)
    mask = mask.astype(np.float32)
    mask /= np.mean((mask))
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.square((preds - labels) / labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)


def metric(pred, true):
    mae = MAE(pred, true, 0.0)
    mse = MSE(pred, true, 0.0)
    rmse = RMSE(pred, true, 0.0)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
