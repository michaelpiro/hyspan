from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from ..classic import glrt
from ..utils import ts_generation


class ClassicDetectorSCM:
    # TODO: add assertions for input shapes and types
    def __init__(self, target_signature: torch.Tensor = None, kernel_size: int = 7, remove_central: bool = True,
                 remove_neighbors_size: int = 1, eps: float = 1e-8):
        self.target_signature = target_signature
        self.kernel_size = kernel_size
        self.remove_central = remove_central
        self.remove_neighbors_size = remove_neighbors_size
        self.eps = eps

    @torch.no_grad()
    def set_target(self, target_signature: torch.Tensor):
        self.target_signature = target_signature

    @torch.no_grad()
    def set_target_from_image(self, image: torch.Tensor, gt, method: int = 0):
        self.target_signature = ts_generation(image, gt, method)

    @torch.no_grad()
    def CEM_SCM(self, image: torch.Tensor, kernel_size=None, remove_central=None, remove_neighbors_size=None, eps=None,
                all_image=False):
        kernel_size = kernel_size if kernel_size is not None else self.kernel_size
        remove_central = remove_central if remove_central is not None else self.remove_central
        remove_neighbors_size = remove_neighbors_size if remove_neighbors_size is not None else self.remove_neighbors_size
        eps = eps if eps is not None else self.eps

        return CEM_scm_algorithm(image, self.target_signature, kernel_size=kernel_size, remove_central=remove_central,
                                 remove_neighbors_size=remove_neighbors_size, eps=eps, all_image=all_image)

    @torch.no_grad()
    def AMF_SCM(self, image: torch.Tensor, kernel_size=None, remove_central=None, remove_neighbors_size=None, eps=None):

        kernel_size = kernel_size if kernel_size is not None else self.kernel_size
        remove_central = remove_central if remove_central is not None else self.remove_central
        remove_neighbors_size = remove_neighbors_size if remove_neighbors_size is not None else self.remove_neighbors_size
        eps = eps if eps is not None else self.eps

        return AMF_SCM_algorithm(image, self.target_signature, kernel_size=kernel_size, remove_central=remove_central,
                                 remove_neighbors_size=remove_neighbors_size, eps=eps)

    @torch.no_grad()
    def ACE_SCM(self, image: torch.Tensor, kernel_size=None, remove_central=None, remove_neighbors_size=None, eps=None):
        kernel_size = kernel_size if kernel_size is not None else self.kernel_size
        remove_central = remove_central if remove_central is not None else self.remove_central
        remove_neighbors_size = remove_neighbors_size if remove_neighbors_size is not None else self.remove_neighbors_size
        eps = eps if eps is not None else self.eps

        return ACE_SCM_algorithm(image, self.target_signature, kernel_size=kernel_size, remove_central=remove_central,
                                 remove_neighbors_size=remove_neighbors_size, eps=eps)

    @torch.no_grad()
    def Kellys_SCM(self, image: torch.Tensor, kernel_size=None, remove_central=None, remove_neighbors_size=None, eps=None):
        kernel_size = kernel_size if kernel_size is not None else self.kernel_size
        remove_central = remove_central if remove_central is not None else self.remove_central
        remove_neighbors_size = remove_neighbors_size if remove_neighbors_size is not None else self.remove_neighbors_size
        eps = eps if eps is not None else self.eps

        return kellys_SCM_algorithm(image, self.target_signature, kernel_size=kernel_size, remove_central=remove_central,
                                    remove_neighbors_size=remove_neighbors_size, eps=eps)

    # todo: add a method to compute the SCM score using explicit loops (for debugging and comparison)
    # todo: add simple MF



# TODO: ADD A METHOD TO RUN ALL DETECTORS FOR EACH PIXEL AT A TIME USING GENERATOR
#  FOR THE PATCHES OR SIMILAR FOR MEMEORY EFFICIENCY

@torch.no_grad()
def get_local_mean_inv_cov(image: torch.Tensor, kernel_size: int = 7, eps: float = 1e-8, remove_central_pxl=True,
                           neighborhood_size=2) -> tuple[Tensor, Tensor]:
    """
    Estimate a local mean and inverse covariance for each pixel from its k×k neighborhood.

    Args:
        image:       (H, W, D) float tensor
        kernel_size: odd window size (e.g. 3/5/7)
        eps:         Tikhonov regularization added to covariance diagonals

    Returns:
        local_means:   (H*W, D)
        local_inv_cov: (H*W, D, D)
    """
    H, W, D = image.shape
    pad = kernel_size // 2

    # Extract patches as (HW, k*k, D)
    patches = F.unfold(image.permute(2, 0, 1).unsqueeze(0), kernel_size=kernel_size, padding=pad)
    if not remove_central_pxl:
        patches = patches.view(D, kernel_size * kernel_size, H * W).permute(2, 1, 0)  # (HW, k*k, D)
    else:
        # remove the central pixel and it's 2D kernel_size from the patch
        patches = patches.view(D, kernel_size * kernel_size, H * W).permute(2, 1, 0)  # (HW, k*k, D)
        center_idx = kernel_size // 2
        mask = torch.ones(kernel_size, kernel_size, dtype=torch.bool, device=image.device)
        start_idx = center_idx - neighborhood_size
        end_idx = center_idx + neighborhood_size + 1
        mask[start_idx:end_idx, start_idx:end_idx] = False
        mask = mask.view(-1)  # (k*k,)
        patches = patches[:, mask, :]  # (HW, k*k - num_removed, D)

    # Local mean and centered samples
    mean = patches.mean(dim=1)  # (HW, D)
    centered = patches - mean.unsqueeze(1)  # (HW, k*k, D)

    # Empirical covariance per pixel: (HW, D, D)
    n_samples = patches.shape[1]  # actual count after any central-pixel removal
    cov = torch.bmm(centered.transpose(1, 2), centered) / max(1, n_samples - 1)
    # Regularize and invert via Cholesky for stability
    eye = torch.eye(D, device=image.device, dtype=image.dtype).unsqueeze(0).expand_as(cov)
    cov = (cov + cov.transpose(1, 2)) * 0.5 + eps * eye

    # make cov diagonal
    # todo: remove this line to have full covariance
    # cov = torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))

    L = torch.linalg.cholesky(cov)  # (HW, D, D)
    inv_cov = torch.cholesky_inverse(L)  # (HW, D, D)

    return mean, inv_cov


@torch.no_grad()
def get_scm_score_loop(image: torch.Tensor, target: torch.Tensor, kernel_size: int = 7, neighberhood_size=1,
                       remove_central_pxl=True, eps: float = 1e-8) -> torch.Tensor:
    """
    Baseline implementation of SCM score using explicit loops (slower).
    Args:
        image: (H,W,D)
        target: (D,)
        kernel_size:
        neighborhood size: size of central area to remove
        remove_central_pxl: whether to remove central pixel and its kernel_size
        eps: numerical stability
    Returns:
        scores: (H*W,)
    """
    H, W, D = image.shape
    pad = kernel_size // 2
    scores = torch.zeros(H * W, device=image.device, dtype=image.dtype)

    t = target.view(D, 1)

    padded_image = F.pad(image.permute(2, 0, 1), (pad, pad, pad, pad), mode='reflect').permute(1, 2, 0)

    for i in range(H):
        for j in range(W):
            patch = padded_image[i:i + kernel_size, j:j + kernel_size, :].reshape(-1, D)
            if remove_central_pxl:
                center_idx = kernel_size // 2
                mask = torch.ones(kernel_size, kernel_size, dtype=torch.bool, device=image.device)
                start_idx = center_idx - neighberhood_size
                end_idx = center_idx + neighberhood_size + 1
                mask[start_idx:end_idx, start_idx:end_idx] = False
                mask = mask.view(-1)
                patch = patch[mask, :]

            mu = patch.mean(dim=0, keepdim=True)  # (1,D)
            centered = patch - mu
            cov = centered.t() @ centered / max(1, patch.shape[0] - 1)  # (D,D)
            cov = (cov + cov.t()) * 0.5 + eps * torch.eye(D, device=image.device, dtype=image.dtype)
            L = torch.linalg.cholesky(cov)
            inv_cov = torch.cholesky_inverse(L)
            dx = image[i, j, :].view(1, D) - mu
            num = (t.t() @ inv_cov @ dx.t()).squeeze()
            den = (t.t() @ inv_cov @ t).squeeze()
            scores[i * W + j] = num / den
    return scores

@torch.no_grad()
def CEM_scm_algorithm(image: torch.Tensor, target: torch.Tensor, kernel_size: int = 7, remove_central=True,
                      remove_neighbors_size=1, eps: float = 1e-8, all_image=False):
    H, W, D = image.shape
    x = image.reshape(-1, D)
    # local stats from neighborhood
    # compute the R matrix using all the pixels in the image
    if all_image:
        R = (torch.matmul(image.view(-1, D).t(), image.view(-1, D))) / (H * W)
        R += eps * torch.eye(D, device=image.device, dtype=image.dtype)
        z = torch.linalg.solve(R, target.view(D, 1))  # (D,1)
        den = target.view(1, D) @ z
        c = z / den  # (D,1)
        scores = c.transpose(-1, -2) @ x.unsqueeze(-1)  # (N,1,1)
        return scores.squeeze()

    local_mean, local_inv_cov = get_local_mean_inv_cov(image,
                                                       kernel_size=kernel_size,
                                                       eps=eps,
                                                       remove_central_pxl=remove_central,
                                                       neighborhood_size=remove_neighbors_size
                                                       )  # (HW,D), (HW,D,D)
    mu_s = local_mean  # (N,D)
    # cov = torch.linalg.inv(local_inv_cov)  # (N,D,D)
    cov = torch.cholesky_inverse(torch.linalg.cholesky(local_inv_cov))

    R_x = cov + (mu_s.unsqueeze(1) @ mu_s.unsqueeze(2))  # (N,D,D)
    z = torch.linalg.solve(R_x, target.view(D, 1))  # (N,D,1)
    den = target.view(1, D).unsqueeze(1) @ z  # (N,1,1)
    c = z / den  # (N,D,1)
    scores = c.transpose(-1, -2) @ x.unsqueeze(-1)  # (N,1,1)
    return scores.squeeze()

@torch.no_grad()
def glrt_scm_algorithm(image: torch.Tensor, target: torch.Tensor, phi1, phi2, kernel_size: int = 7, remove_central=True,
                       remove_neighbor_size=1, eps: float = 1e-8):
    H, W, D = image.shape
    x = image.reshape(-1, D)

    # local stats from neighborhood
    local_mean, local_inv_cov = get_local_mean_inv_cov(image,
                                                       kernel_size=kernel_size,
                                                       eps=eps,
                                                       remove_central_pxl=remove_central,
                                                       neighborhood_size=remove_neighbor_size
                                                       )  # (HW,D), (HW,D,D)

    # center the samples
    x_centered = x - local_mean  # (N,D)

    # center the target
    centered_target = target - local_mean  # (N,D)

    # compute the detector scores
    centered_target = centered_target.view(-1, D, 1)  # (N,D,1)

    scores = glrt(x_centered, centered_target, local_inv_cov, phi1=phi1, phi2=phi2)

    return scores.squeeze()

@torch.no_grad()
def AMF_SCM_algorithm(image: torch.Tensor, target: torch.Tensor, kernel_size: int = 7, remove_central=True,
                      remove_neighbors_size=0, eps: float = 1e-8):
    return glrt_scm_algorithm(image, target, phi1=1, phi2=0, kernel_size=kernel_size, remove_central=remove_central,
                              remove_neighbor_size=remove_neighbors_size, eps=eps)

@torch.no_grad()
def ACE_SCM_algorithm(image: torch.Tensor, target: torch.Tensor, kernel_size: int = 7, remove_central=True,
                      remove_neighbors_size=0, eps: float = 1e-8):
    return glrt_scm_algorithm(image, target, phi1=0, phi2=1, kernel_size=kernel_size, remove_central=remove_central,
                              remove_neighbor_size=remove_neighbors_size, eps=eps)

@torch.no_grad()
def kellys_SCM_algorithm(image: torch.Tensor, target: torch.Tensor, kernel_size: int = 7, remove_central=True,
                         remove_neighbors_size=0, eps: float = 1e-8):
    # phi1 = N = number of secondary (background) training samples in the window
    if remove_central:
        s = 2 * remove_neighbors_size + 1
        n_secondary = kernel_size ** 2 - s * s
    else:
        n_secondary = kernel_size ** 2
    n_secondary = max(1, n_secondary)
    return glrt_scm_algorithm(image, target, phi1=n_secondary, phi2=1, kernel_size=kernel_size,
                              remove_central=remove_central, remove_neighbor_size=remove_neighbors_size, eps=eps)


# TODO: ADD TESTS TO THE MODULE