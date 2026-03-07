"""
Dataset and sample construction for THANTD training  (Sec. III-A / Eq. 2).

Sample construction:
  anchor   = S_prior (the known target spectrum)
  negative = S_negative (a random background pixel)
  positive = S_positive = μ·S_negative + (1-μ)·S_prior,  μ ~ U(0, mu_max=0.1)

Background pixels can come from:
  - ground-truth mask  (gt == 0)
  - coarse CEM-based pseudo-background (bottom fraction of CEM scores)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    """
    Yields (anchor, positive, negative) triplets on the fly.

    Args:
        bg_pixels:        (N, D) float tensor of background pixels.
        prior_spectrum:   (D,)   known target spectrum.
        n_triplets:       virtual epoch length (number of __getitem__ calls).
        mu_max:           upper bound of R2TM mixing coefficient (paper: 0.1).
        anchor_noise_std: optional Gaussian noise std on the anchor for
                          regularisation (0 = no noise, as in the paper).
    """

    def __init__(
        self,
        bg_pixels:        torch.Tensor,
        prior_spectrum:   torch.Tensor,
        n_triplets:       int   = 50_000,
        mu_max:           float = 0.1,
        anchor_noise_std: float = 0.0,
    ):
        super().__init__()
        self.bg        = bg_pixels.float()
        self.prior     = prior_spectrum.float()
        self.n         = n_triplets
        self.mu_max    = mu_max
        self.noise_std = anchor_noise_std
        self._n_bg     = self.bg.shape[0]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, _idx: int):
        # Random background pixel
        idx = torch.randint(0, self._n_bg, ()).item()
        neg = self.bg[idx]                                      # (D,)

        # Positive via simplified R2TM (Eq. 2)
        mu  = torch.empty(1).uniform_(0.0, self.mu_max).item()
        pos = mu * neg + (1.0 - mu) * self.prior               # (D,)

        # Anchor = prior target spectrum (+ optional tiny noise)
        anc = self.prior.clone()
        if self.noise_std > 0:
            anc = anc + torch.randn_like(anc) * self.noise_std

        return anc, pos, neg


def build_dataset_from_image(
    image:          torch.Tensor,
    gt:             torch.Tensor | None,
    prior_spectrum: torch.Tensor,
    n_triplets:     int   = 50_000,
    mu_max:         float = 0.1,
    coarse_frac:    float = 0.5,
) -> TripletDataset:
    """
    Convenience factory: build a TripletDataset from a labelled (or unlabelled) HSI.

    If gt is provided, background = pixels where gt == 0.
    If gt is None, a coarse CEM pass selects the bottom `coarse_frac` fraction of
    pixels as pseudo-background.

    Args:
        image:          (H, W, D) tensor
        gt:             (H, W)    binary tensor (1=target, 0=background), or None
        prior_spectrum: (D,)      target spectrum
        n_triplets:     virtual epoch length
        mu_max:         R2TM mixing upper bound
        coarse_frac:    fraction of CEM-lowest pixels used when gt=None
    """
    H, W, D = image.shape
    pixels   = image.reshape(-1, D)

    if gt is not None:
        bg_mask   = (gt.reshape(-1) == 0)
        bg_pixels = pixels[bg_mask]
    else:
        # Coarse CEM: solve R·c = t then score every pixel
        t   = prior_spectrum.to(pixels.device)
        R   = (pixels.T @ pixels) / pixels.shape[0]
        R   = R + 1e-6 * torch.eye(D, device=pixels.device, dtype=pixels.dtype)
        c   = torch.linalg.solve(R, t.unsqueeze(-1)).squeeze(-1)
        c   = c / (t @ c + 1e-12)
        cem = (pixels @ c.unsqueeze(-1)).squeeze(-1)
        n_bg      = max(1, int(coarse_frac * pixels.shape[0]))
        bg_pixels = pixels[torch.argsort(cem)[:n_bg]]

    return TripletDataset(
        bg_pixels.cpu(),
        prior_spectrum.cpu(),
        n_triplets=n_triplets,
        mu_max=mu_max,
    )
