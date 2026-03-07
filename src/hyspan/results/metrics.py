"""Detector evaluation metrics implemented in pure PyTorch (no sklearn needed)."""
from __future__ import annotations

from typing import Tuple

import torch


def roc_curve(
    scores: torch.Tensor,
    gt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the ROC curve.

    Args:
        scores: (H, W) or (N,) float detection scores (higher = more likely target).
        gt:     (H, W) or (N,) binary int/float labels (1 = target, 0 = background).

    Returns:
        fpr: (T,) false-positive rate at each threshold.
        tpr: (T,) true-positive rate at each threshold.
    """
    scores = scores.flatten().float()
    labels = gt.flatten().float()

    n_pos = labels.sum().item()
    n_neg = (1.0 - labels).sum().item()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("gt must contain both positive and negative samples.")

    sorted_idx = torch.argsort(scores, descending=True)
    labels_sorted = labels[sorted_idx]

    tp = torch.cumsum(labels_sorted, dim=0)
    fp = torch.cumsum(1.0 - labels_sorted, dim=0)

    tpr = torch.cat([torch.zeros(1, device=scores.device), tp / n_pos])
    fpr = torch.cat([torch.zeros(1, device=scores.device), fp / n_neg])

    return fpr, tpr


def roc_auc(scores: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Area under the ROC curve (trapezoidal rule).

    Args:
        scores: (H, W) or (N,) detection scores.
        gt:     (H, W) or (N,) binary labels.

    Returns:
        AUC as a float in [0, 1].
    """
    fpr, tpr = roc_curve(scores, gt)
    return torch.trapz(tpr, fpr).item()
