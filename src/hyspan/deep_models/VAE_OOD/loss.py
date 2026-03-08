"""
ELBO loss for SpectralVAE.

L_ELBO = L_rec + β · KL
  L_rec = ||x - x̂||²          (MSE reconstruction)
  KL    = -½ Σ(1 + log σ² - μ² - σ²)   (diagonal Gaussian vs N(0,I))

The β parameter (β-VAE, Higgins et al. 2017) controls the
reconstruction–regularisation trade-off.  β < 1 prioritises sharp
reconstruction (useful when the anomaly score is L_rec); β > 1
encourages a more regular, disentangled latent space.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ELBOLoss(nn.Module):
    """
    Args:
        beta: weight on the KL term (β-VAE formulation).
              Paper uses β=1; values 0.1–0.5 often give sharper
              reconstruction scores for OOD detection.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        x:       torch.Tensor,   # (N, D) input
        x_hat:   torch.Tensor,   # (N, D) reconstruction
        mu:      torch.Tensor,   # (N, L) posterior mean
        log_var: torch.Tensor,   # (N, L) posterior log-variance
    ):
        """
        Returns:
            loss:    scalar ELBO loss
            metrics: dict with detached float values for logging
        """
        # Reconstruction: sum over bands, mean over samples
        l_rec = ((x - x_hat) ** 2).sum(dim=-1).mean()

        # KL divergence: -½ Σ(1 + log σ² - μ² - σ²)
        l_kl  = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()

        loss  = l_rec + self.beta * l_kl

        metrics = dict(
            loss    = loss.item(),
            l_rec   = l_rec.item(),
            l_kl    = l_kl.item(),
        )
        return loss, metrics
