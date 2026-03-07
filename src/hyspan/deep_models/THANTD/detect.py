"""Convenience wrapper for THANTD inference on a full HSI."""
from __future__ import annotations

import torch
from .model import THANTD


def thantd_detect(
    model:          THANTD,
    image:          torch.Tensor,
    prior_spectrum: torch.Tensor,
    batch_size:     int  = 4096,
    normalise:      bool = True,
) -> torch.Tensor:
    """
    Run THANTD detection on a full hyperspectral image.

    Args:
        model:          trained THANTD
        image:          (H, W, D) float tensor
        prior_spectrum: (D,) target spectrum
        batch_size:     pixels per GPU batch (tune for memory)
        normalise:      min-max normalise output to [0, 1]
    Returns:
        scores: (H, W) detection map
    """
    scores = model.detect(image, prior_spectrum, batch_size=batch_size)
    if normalise:
        lo, hi = scores.min(), scores.max()
        if hi > lo:
            scores = (scores - lo) / (hi - lo)
    return scores
