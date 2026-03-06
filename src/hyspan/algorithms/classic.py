from __future__ import annotations

import numpy as np
import torch


# ---------- Matched Filter ----------


def matched_filter_scores(image: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    simple matched filter implementation
    Args:
        image: (H,W,D)
        target: (D,)
    Returns:
        scores: (H*W,) real-valued
    """
    H, W, D = image.shape
    x = image.reshape(-1, D)
    t = target.reshape(D, -1)
    scores = x @ t  # (H*W,D) @ (D,1) -> (H*W,1)
    return scores


def glrt(centered_image: torch.Tensor, centered_targets: torch.Tensor, inv_cov, phi1: int, phi2: int):
    N, D = centered_image.shape
    x = centered_image.view(-1, D, 1)

    if centered_targets.dim() == 1:
        centered_targets = centered_targets.unsqueeze(0)
    # if centered_targets.dim() == 2:

    P = centered_targets.shape[-1]
    target = centered_targets.view(-1, D, P)

    # compute the detector scores
    a = inv_cov @ x  # (N,D,1)
    den = phi1 + (phi2 * x.transpose(-1, -2) @ a)  # (N,1,1)

    b = target.transpose(-1, -2) @ a  # (N,1,1)

    middle = target.transpose(-1, -2) @ inv_cov @ target  # (N,1,1)
    if P == 1:
        scores = (b ** 2) / (den * middle)
    else:
        scores = (b.transpose(-1, -2) @ torch.linalg.inv(middle) @ b) / den  # (N,1,1)
    return scores


def get_orthogonal_subspace(B: torch.Tensor, lstsqrs: bool = False):
    """
    Compute the orthogonal subspace projection matrix for the background subspace B.
    Args:
        B: (D, Mb) background subspace matrix
    Returns:
        P_orth_B: (D,D) orthogonal projection matrix
    """
    D, Mb = B.shape

    # compute the orthogonal projection matrix for the background subspace
    BtB = B.t() @ B  # (Mb,Mb)

    # x is the result of (BtB)^-1 @ B^T
    if lstsqrs:
        x = torch.linalg.lstsq(BtB, B.t()).solution  # (Mb,D)
    else:
        x = torch.linalg.solve(BtB, B.t())  # (Mb,D)
    P_B = B @ x  # (D,D)
    P_B_orth = torch.eye(D, device=B.device, dtype=B.dtype) - P_B  # (D,D)

    return P_B_orth


def OSP_detector(image: torch.Tensor, background_matrix: torch.Tensor, target_matrix: torch.Tensor):
    H, W, D = image.shape
    N = H * W
    x = image.view(-1, D, 1)  # (N,D,1) N standing vectors

    # compute the projection matrix for the background
    B = background_matrix  # (D, Mb)
    T = target_matrix  # (D, Mt)

    # compute the orthogonal background subspace avoiding using inverse
    P_B_orth = get_orthogonal_subspace(B, lstsqrs=True)  # (D,D)

    # compute the orthogonal target and background subspace avoiding using inverse
    S_B = torch.cat([B, T], dim=1)  # (D, Mb + Mt)
    P_SB_orth = get_orthogonal_subspace(S_B, lstsqrs=True)  # (D,D)

    # compute the projection matrix for the background subspace
    num = x.transpose(-1, -2) @ P_B_orth @ x  # (N,1,1)
    den = x.transpose(-1, -2) @ P_SB_orth @ x  # (N,1,1)

    scores = (num / den).squeeze()
    return scores

# def kellys_detector(image: torch.Tensor, target: torch.Tensor, gmm: GMM):
#     H, W, D = image.shape
#     x = image.view(-1, D)
#     # comps_pixel = get_top_gaussian(image, gmm).view(-1).long()
#     comps = get_top_gaussian_using_neighbors_only(image, gmm, n=5).view(-1).long()
#
#     N = H * W
#     mu_s = gmm.means  # (K,D)
#     # move the samples by the mean of the selected component
#     x_centered = x - mu_s[comps]  # (N,D)
#     x_centered = x_centered.unsqueeze(-1)  # (N,D,1)
#
#     # compute the inverse cov for each component
#     inv_chol_k = torch.cholesky_inverse(gmm.scale_tril())
#     inv_k = inv_chol_k[comps]  # (N,D,D)
#
#     # compute the detector scores
#     t = target.view(D, 1).unsqueeze(0)  # (1,D,1)
#     a = inv_k @ x_centered  # (N,D,1)
#     den = N + x_centered.transpose(-1, -2) @ a  # (N,1,1)
#
#     b = target.transpose(-1, -2) @ a  # (N,1,1)
#     middle = target.transpose(-1, -2) @ inv_k @ t  # (N,1,1)
#     scores = (b.transpose(-1, -2) @ torch.linalg.inv(middle) @ b) / den  # (N,1,1)
#     return scores.squeeze()
