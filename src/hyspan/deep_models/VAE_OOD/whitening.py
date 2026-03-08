"""
Local spatial whitening for HSI — HSI analog of the radar local covariance
whitening in Rouzoumka et al. (2025), Sec. IV-B.

Two whitening strategies are provided:

1. SCM whitening  (`local_whiten_image`)
   For each pixel (h, w), gather the K = k² spatial neighbours, compute
   their sample covariance R̂, regularise, and whiten:
       x^w = R̂_reg^{-1/2} · x
   Rank condition: K > D required for full-rank SCM. Apply PCA first when D > K.

2. GMM whitening  (`gmm_whiten_image`)
   Fit a GMM globally on background pixels.  Each pixel's k×k neighbourhood
   votes for a GMM component (majority of hard assignments).  The winning
   component's Cholesky factor is used to whiten the center pixel:
       x^w = L_k^{-1} (x − μ_k)
   Advantage: the per-component covariance is estimated from all pixels
   in that cluster (not just a k×k patch), so it is full-rank even when
   D >> k². No PCA pre-reduction required.

Regularisation (analogous to Eq. 13 in the radar paper):
    R̂_reg = R̂ + ε_ridge · (tr(R̂) / D) · I_D
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _pca_reduce(pixels: torch.Tensor, n_components: int):
    """
    Simple PCA: (N, D) → (N, k), also returns (mean, components).
    """
    mean   = pixels.mean(dim=0)                          # (D,)
    X      = pixels - mean
    cov    = X.T @ X / (X.shape[0] - 1)                 # (D, D)
    _, V   = torch.linalg.eigh(cov)                      # ascending eigenvalues
    comps  = V[:, -n_components:].flip(-1)               # (D, k) top-k
    return (X @ comps), mean, comps                      # (N, k), (D,), (D, k)


@torch.no_grad()
def local_whiten_image(
    image:        torch.Tensor,   # (H, W, D)
    kernel_size:  int   = 7,
    ridge_frac:   float = 1e-2,   # ε_ridge: fraction of mean eigenvalue added to diagonal
    n_components: int | None = None,  # if set, apply PCA first: D → n_components
    device:       str   = "cpu",
) -> torch.Tensor:
    """
    Spatially-local whitening: for each pixel, use k×k neighborhood to
    estimate the local covariance and whiten the pixel spectrum.

    Args:
        image:        (H, W, D) float tensor, spectra in [0, 1]
        kernel_size:  spatial neighbourhood side length (odd)
        ridge_frac:   regularisation — adds ridge_frac * (tr(R)/D) * I to SCM
        n_components: if given, reduce to this many PCA dimensions before
                      whitening.  Required when D > kernel_size².
        device:       computation device

    Returns:
        whitened image (H, W, D') where D' = n_components (or D if None)
    """
    image = image.to(device).float()
    H, W, D = image.shape
    k     = kernel_size
    pad   = k // 2

    # ── Optional PCA reduction ────────────────────────────────────────────
    pca_mean  = None
    pca_comps = None
    if n_components is not None and D > n_components:
        pixels = image.reshape(-1, D)
        proj, pca_mean, pca_comps = _pca_reduce(pixels, n_components)
        image = proj.reshape(H, W, n_components)
        D     = n_components

    # ── Extract k×k patches for covariance estimation ────────────────────
    # image: (H, W, D) → permute → (1, D, H, W) → unfold → patches
    img_t = image.permute(2, 0, 1).unsqueeze(0)         # (1, D, H, W)
    # unfold: (1, D, H, W) → (1, D*k*k, H*W)  via unfold on H then W
    patches = F.unfold(img_t, kernel_size=k, padding=pad)   # (1, D*k*k, H*W)
    patches = patches.squeeze(0).T                           # (H*W, D*k*k)
    patches = patches.reshape(H * W, k * k, D)              # (N, K, D)

    K = k * k  # number of neighbors per pixel

    # ── Per-pixel local whitening ─────────────────────────────────────────
    # Vectorised: compute SCM for every pixel simultaneously
    # SCM: (N, D, D) — expensive for large D; works well after PCA reduction
    X   = patches                                            # (N, K, D)
    mu  = X.mean(dim=1, keepdim=True)                       # (N, 1, D)
    Xc  = X - mu                                            # (N, K, D)
    # R = Xc^T @ Xc / (K-1): batch matmul  (N, D, K) @ (N, K, D) = (N, D, D)
    R   = torch.bmm(Xc.transpose(1, 2), Xc) / max(1, K - 1)  # (N, D, D)

    # Regularise: R_reg = R + ridge_frac * (tr(R)/D) * I
    tr_R  = R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)        # (N,)
    ridge = ridge_frac * tr_R / D                           # (N,)
    I     = torch.eye(D, device=device).unsqueeze(0)        # (1, D, D)
    R_reg = R + ridge.view(-1, 1, 1) * I                    # (N, D, D)

    # R_reg^{-1/2} via eigendecomposition: R = V Λ V^T → R^{-1/2} = V Λ^{-1/2} V^T
    eigvals, eigvecs = torch.linalg.eigh(R_reg)             # ascending; (N,D), (N,D,D)
    eigvals          = eigvals.clamp(min=1e-8)
    inv_sqrt_lambda  = eigvals.rsqrt()                       # (N, D)
    # R^{-1/2} = eigvecs @ diag(inv_sqrt_lambda) @ eigvecs^T
    R_inv_sqrt = eigvecs * inv_sqrt_lambda.unsqueeze(-2)     # (N, D, D)  (broadcast diag)
    R_inv_sqrt = R_inv_sqrt @ eigvecs.transpose(-2, -1)      # (N, D, D)

    # Whiten each pixel: x^w = R^{-1/2} @ x
    x_flat    = image.reshape(H * W, D, 1)                  # (N, D, 1)
    x_whitened = torch.bmm(R_inv_sqrt, x_flat).squeeze(-1)  # (N, D)

    return x_whitened.cpu().reshape(H, W, D)


# ── GMM-based local whitening ─────────────────────────────────────────────

@torch.no_grad()
def gmm_whiten_image(
    image:        torch.Tensor,        # (H, W, D)
    n_components: int   = 10,
    kernel_size:  int   = 7,
    gmm=None,                          # pre-fitted TorchGMM (skips fitting if provided)
    fit_pixels:   torch.Tensor | None = None,  # pixels to fit GMM on; defaults to all pixels
    device:       str   = "cpu",
):
    """
    GMM-based local whitening for HSI.

    Algorithm:
      1. Fit a TorchGMM on `fit_pixels` (background spectra, or all pixels).
      2. Hard-assign every pixel in the image to its most likely component.
      3. For each pixel, find the majority component in its k×k neighbourhood
         (spatial voting — same idea as ClassicDetectorGMM).
      4. Whiten using that component's lower-Cholesky factor L_k:
             x^w = L_k^{-1} (x − μ_k)
         so that background pixels ≈ N(0, I) in the whitened space.

    Advantage over SCM whitening:
      - Component covariance is estimated from ALL pixels in that cluster,
        not just a k×k patch → full-rank even when D >> kernel².
      - Naturally adapts to multi-modal backgrounds (different terrain types).

    Args:
        image:        (H, W, D) float tensor, spectra in [0, 1]
        n_components: number of GMM components (ignored if `gmm` provided)
        kernel_size:  spatial neighbourhood for majority-vote assignment
        gmm:          pre-fitted TorchGMM; if None, one is fitted internally
        fit_pixels:   (N, D) tensor to fit the GMM on; defaults to all image pixels
        device:       computation device

    Returns:
        whitened:  (H, W, D) whitened image tensor (on CPU)
        gmm:       the fitted TorchGMM (re-use across calls to avoid re-fitting)
    """
    from ..GMM.TorchGmm import TorchGMM

    image = image.float()
    H, W, D = image.shape
    k   = kernel_size
    pad = k // 2

    # ── 1. Fit GMM ────────────────────────────────────────────────────────
    if gmm is None:
        pixels_for_fit = (
            fit_pixels.float() if fit_pixels is not None
            else image.reshape(-1, D)
        )
        gmm = TorchGMM(
            n_components=n_components,
            covariance_type="full",
            device=device,
            random_state=0,
        )
        print(f"  Fitting GMM ({n_components} components) on {pixels_for_fit.shape[0]:,} pixels …")
        gmm.fit(pixels_for_fit.to(device))
        print("  GMM fitted.")

    # ── 2. Hard-assign every image pixel to a component ───────────────────
    pixels_all = image.reshape(-1, D)
    labels_np  = gmm.predict(pixels_all)                      # (H*W,) numpy int64
    labels     = torch.tensor(labels_np.tolist(), dtype=torch.long)  # (H*W,)

    # ── 3. Spatial majority-vote in k×k neighbourhood ─────────────────────
    labels_4d = labels.reshape(1, 1, H, W).float()            # (1, 1, H, W)
    # unfold to (1, k², H*W) then reshape to (H*W, k²)
    neigh = F.unfold(labels_4d, kernel_size=k, padding=pad)   # (1, k², H*W)
    neigh = neigh.squeeze(0).T.long()                          # (H*W, k²)

    # torch.mode returns the most frequent value (smallest in case of ties)
    majority, _ = torch.mode(neigh, dim=1)                    # (H*W,)

    # ── 4. Per-component whitening ─────────────────────────────────────────
    # L_k: lower Cholesky of Σ_k  (from the GMM internal cache)
    means_t = gmm._means.to(device).float()    # (K, D)
    chol_t  = gmm._cov_chol.to(device).float() # (K, D, D)  lower triangular

    pixels_dev  = pixels_all.to(device)        # (N, D)
    majority_dev = majority.to(device)          # (N,)
    x_whitened  = torch.zeros_like(pixels_dev) # (N, D)

    K = gmm.n_components
    for k_idx in range(K):
        mask = (majority_dev == k_idx)
        if not mask.any():
            continue
        px_k  = pixels_dev[mask]              # (N_k, D)
        mu_k  = means_t[k_idx]               # (D,)
        L_k   = chol_t[k_idx]               # (D, D) lower triangular

        # Centre then whiten: solve L_k y = (x - μ_k)^T  →  y = L_k^{-1}(x-μ)^T
        diff = (px_k - mu_k).T             # (D, N_k)
        y    = torch.linalg.solve_triangular(L_k, diff, upper=False)  # (D, N_k)
        x_whitened[mask] = y.T

    return x_whitened.cpu().reshape(H, W, D), gmm
