"""
SpectralVAE — real-valued VAE for hyperspectral OOD detection.

Trained exclusively on background (H0) pixel spectra.
At inference, the reconstruction error  ||x - x̂||²  serves as the
anomaly/detection score: target pixels are out-of-distribution and
reconstruct with larger error than background pixels.

Adapted from:
  Rouzoumka et al., "Out-of-Distribution Radar Detection with Complex VAEs:
  Theory, Whitening, and ANMF Fusion", IEEE TSP 2025.

  Original paper uses complex-valued convolutions for 1-D Doppler profiles.
  Here we adapt to real-valued MLP for D-band spectral vectors, matching
  the HSI domain (no phase information, spectral correlation structure).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List


class SpectralVAE(nn.Module):
    """
    MLP-based VAE for hyperspectral pixel spectra.

    Args:
        n_bands:     spectral dimension D (input / output)
        latent_dim:  latent space dimension q
        hidden_dims: list of hidden layer widths for encoder
                     (decoder mirrors in reverse)
        dropout:     dropout probability applied after each hidden ReLU
    """

    def __init__(
        self,
        n_bands:     int,
        latent_dim:  int        = 32,
        hidden_dims: List[int]  = None,
        dropout:     float      = 0.1,
    ):
        super().__init__()
        self.n_bands    = n_bands
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # ── Encoder ────────────────────────────────────────────────────────
        enc_layers: list = []
        in_dim = n_bands
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        # Variational heads: mean and log-variance
        self.fc_mu      = nn.Linear(in_dim, latent_dim)
        self.fc_log_var = nn.Linear(in_dim, latent_dim)

        # ── Decoder ────────────────────────────────────────────────────────
        dec_layers: list = []
        dec_dims = list(reversed(hidden_dims))
        in_dim = latent_dim
        for h in dec_dims:
            dec_layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h
        dec_layers += [
            nn.Linear(in_dim, n_bands),
            nn.Sigmoid(),   # spectra normalised to [0, 1]
        ]
        self.decoder = nn.Sequential(*dec_layers)

    # ── Forward passes ─────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor):
        """x: (N, D) → (mu, log_var) each (N, latent_dim)."""
        h       = self.encoder(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick (Kingma & Welling 2014)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, latent_dim) → x_hat: (N, D)."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        x: (N, D)
        Returns:
            x_hat:   (N, D) reconstructed spectrum
            mu:      (N, latent_dim)
            log_var: (N, latent_dim)
        """
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        x_hat       = self.decode(z)
        return x_hat, mu, log_var

    @torch.no_grad()
    def reconstruction_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample anomaly score (sum of squared reconstruction errors).

        x: (N, D)  →  scores: (N,)   [higher = more anomalous]
        """
        self.eval()
        mu, _   = self.encode(x)          # use mean (no randomness at test time)
        x_hat   = self.decode(mu)
        return ((x - x_hat) ** 2).sum(dim=-1)
