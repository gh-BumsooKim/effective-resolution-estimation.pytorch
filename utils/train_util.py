"""Small training helpers: MAPE loss and staircase LR decay."""

import torch


def mape_loss(pred, target, eps=1e-8):
    """Mean absolute percentage error.

    ``mean(|target - pred| / |target|)``.  The paper uses MAPE because it
    weighs the error relative to the (small) regression target.
    """
    return (torch.abs(target - pred) / torch.clamp(torch.abs(target), min=eps)).mean()


def reset_LR(optimizer, lr_decay):
    """Multiply every param-group learning rate by ``lr_decay`` (staircase)."""
    for group in optimizer.param_groups:
        group['lr'] = group['lr'] * lr_decay
