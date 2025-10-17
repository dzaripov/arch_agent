import numpy as np
import torch


def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    """Mean Absolute Error"""
    return torch.mean(torch.abs(pred - true))

def MSE(pred, true):
    """Mean Squared Error"""
    return torch.mean((pred - true)**2)

def RMSE(pred, true):
    """Root Mean Squared Error"""
    return torch.sqrt(MSE(pred, true))

def MAPE(pred, true):
    """Mean Absolute Percentage Error"""
    # Add a small epsilon to the denominator to avoid division by zero.
    epsilon = 1e-8
    return torch.mean(torch.abs((pred - true) / (true + epsilon)))

def MSPE(pred, true):
    """Mean Squared Percentage Error"""
    # Add a small epsilon to the denominator to avoid division by zero.
    epsilon = 1e-8
    return torch.mean(torch.square((pred - true) / (true + epsilon)))

def metric(pred, true):
    """
    Calculates all metrics and returns them as standard Python floats.
    
    Args:
        pred (torch.Tensor): The predicted values.
        true (torch.Tensor): The true values.
    
    Returns:
        tuple: A tuple containing mae, mse, rmse, mape, mspe as float values.
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae.item(), mse.item(), rmse.item(), mape.item(), mspe.item()