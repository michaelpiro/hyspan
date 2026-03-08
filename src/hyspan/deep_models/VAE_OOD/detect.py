"""
Detection interface for SpectralVAE and p-value fusion with CEM.

Three detection modes (matching Sec. IV-C/D of the paper):

1. VAE-only: score = ||x - x̂||²  (reconstruction error)
2. CEM-only: standard Constrained Energy Minimisation
3. Fused:    weighted log-p combination of VAE and CEM scores
             (paper Eq. 18: S* = -(w·log p_CEM + (1-w)·log p_VAE))
"""
from __future__ import annotations

import torch
from .model import SpectralVAE


# ── VAE detection ─────────────────────────────────────────────────────────

@torch.no_grad()
def vae_ood_detect(
    model:      SpectralVAE,
    image:      torch.Tensor,    # (H, W, D)
    batch_size: int  = 4096,
    normalise:  bool = True,
    device:     str  = "cuda",
) -> torch.Tensor:
    """
    Compute per-pixel anomaly score (reconstruction error) for a full HSI.

    Returns:
        scores: (H, W) tensor, higher = more anomalous / more likely target
    """
    H, W, D = image.shape
    model.eval()
    model.to(device)

    pixels  = image.reshape(-1, D)
    chunks  = [pixels[i:i + batch_size] for i in range(0, pixels.shape[0], batch_size)]
    scores  = []

    for chunk in chunks:
        chunk   = chunk.to(device)
        mu, _   = model.encode(chunk)          # use mean at test time (no noise)
        x_hat   = model.decode(mu)
        rec_err = ((chunk - x_hat) ** 2).sum(dim=-1)
        scores.append(rec_err.cpu())

    scores = torch.cat(scores).reshape(H, W)

    if normalise:
        lo, hi = scores.min(), scores.max()
        if hi > lo:
            scores = (scores - lo) / (hi - lo)

    return scores


# ── CEM detection ─────────────────────────────────────────────────────────

@torch.no_grad()
def cem_detect(
    image:     torch.Tensor,    # (H, W, D)
    prior:     torch.Tensor,    # (D,)
    normalise: bool = True,
) -> torch.Tensor:
    """
    Constrained Energy Minimisation (CEM) detector.

    Score = x^T · c  where  c = R^{-1} t / (t^T R^{-1} t),
    R = global sample covariance, t = prior spectrum.

    Returns:
        scores: (H, W)
    """
    H, W, D = image.shape
    pixels  = image.reshape(-1, D).float()
    t       = prior.float().to(pixels.device)

    R = pixels.T @ pixels / pixels.shape[0]
    R = R + 1e-6 * torch.eye(D, device=pixels.device)
    c = torch.linalg.solve(R, t.unsqueeze(-1)).squeeze(-1)
    c = c / (t @ c + 1e-12)

    scores = (pixels @ c).reshape(H, W)

    if normalise:
        lo, hi = scores.min(), scores.max()
        if hi > lo:
            scores = (scores - lo) / (hi - lo)

    return scores


# ── Empirical CDF (for p-value computation) ───────────────────────────────

def _empirical_right_tail_p(
    scores:    torch.Tensor,   # test scores (any shape, flattened internally)
    bg_scores: torch.Tensor,   # background (H0) scores used to build ECDF
) -> torch.Tensor:
    """
    Compute right-tail empirical p-values:
        p(s) = P(S >= s | H0) = 1 - ECDF_{H0}(s)

    A high anomaly score → small p-value (anomalous under H0).
    """
    flat_test = scores.reshape(-1)
    flat_bg   = bg_scores.reshape(-1).sort().values     # sorted H0 scores

    N   = flat_bg.shape[0]
    # For each test score, count how many H0 scores are BELOW it (= ECDF value)
    # searchsorted returns index of first element >= s, so rank = index
    ranks  = torch.searchsorted(flat_bg, flat_test.contiguous())  # (M,)
    ecdf   = ranks.float() / N                                    # in [0, 1]
    p_vals = (1.0 - ecdf).clamp(min=1e-12)                       # right-tail p
    return p_vals.reshape(scores.shape)


# ── Log-p fusion (paper Eq. 18) ───────────────────────────────────────────

def fuse_vae_cem(
    vae_scores:    torch.Tensor,   # (H, W) VAE detection scores
    cem_scores:    torch.Tensor,   # (H, W) CEM detection scores
    bg_vae_scores: torch.Tensor,   # (N,)   VAE scores on background pixels
    bg_cem_scores: torch.Tensor,   # (N,)   CEM scores on background pixels
    w:             float = 0.5,    # weight on CEM branch (1-w on VAE)
    normalise:     bool  = True,
) -> torch.Tensor:
    """
    Weighted log-p fusion (paper Eq. 18):
        S* = -(w · log p_CEM + (1-w) · log p_VAE)

    Both p-values are right-tail: high score → small p → large -log p.
    The fused score S* is large when either (or both) detectors flag the pixel.

    Args:
        vae_scores:    (H, W) — raw (unnormalised) VAE reconstruction errors
        cem_scores:    (H, W) — raw (unnormalised) CEM scores
        bg_vae_scores: (N,)   — VAE scores on confirmed background pixels (H0 calibration)
        bg_cem_scores: (N,)   — CEM scores on background pixels
        w:             weight in [0, 1] for CEM branch

    Returns:
        fused: (H, W) detection map (higher = target more likely)
    """
    p_vae = _empirical_right_tail_p(vae_scores, bg_vae_scores)
    p_cem = _empirical_right_tail_p(cem_scores, bg_cem_scores)

    fused = -(w * p_cem.log() + (1.0 - w) * p_vae.log())

    if normalise:
        lo, hi = fused.min(), fused.max()
        if hi > lo:
            fused = (fused - lo) / (hi - lo)

    return fused
