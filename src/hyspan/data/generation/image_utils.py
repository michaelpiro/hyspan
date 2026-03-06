from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from skimage.util import view_as_windows

import matplotlib.pyplot as plt

from .utils import get_gaussian_kernel


def generate_spatial_comp_map(
        H,
        W,
        ratios,
        smoothness: float = 5.0,
        segment_size: int = 25,
        device: str = "cpu",
        peak=10000.0
) -> torch.Tensor:
    """
    Returns a (H, W) map of mixture component indices in {0..K-1}.

    Spatially smooth "region labels" with bias according to mixture weights.
    """
    K = ratios.numel()
    if ratios is None:
        ratios = torch.full((K,), 1.0 / K, device=device)

    noise = torch.randn((1, K, H, W), device=device)
    # noise += ratios.view(1, K, 1, 1)

    # pick random pixels to boost their values according to ratios
    num_pixels = H * W
    class_pixels = ratios * num_pixels
    segment_size = (segment_size // 2) * 2 + 1  # make it odd
    kernel = get_gaussian_kernel(kernel_size=segment_size, smoothness=10.0).to(device) * peak
    for k in range(K):
        num_random_pixels = int(class_pixels[k].item())
        num_segments = min(max(1, num_random_pixels // (segment_size * segment_size)), 20000)
        # print(f"Class {k}: boosting {num_segments} segments of size {segment_size}x{segment_size}")
        for s in range(num_segments):
            start_idx = torch.randint(0, num_pixels - segment_size * segment_size, (1,), device=device).item()
            row_start = min(start_idx // W, H - segment_size)
            col_start = min(start_idx % W, W - segment_size)
            fac = kernel
            noise[0, k, row_start:row_start + segment_size, col_start:col_start + segment_size] += fac

    # Gaussian kernel for spatial smoothing
    kernel_size = max(1, 2 * int(2 * smoothness) + 1)
    padding = kernel_size // 2

    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - padding
    gauss_1d = torch.exp(-coords ** 2 / (2 * smoothness ** 2 + 1e-12))
    gauss_1d /= gauss_1d.sum()

    kernel2d = gauss_1d[:, None] * gauss_1d[None, :]  # (k,k)
    kernel2d = kernel2d.expand(K, 1, -1, -1)  # (K,1,k,k)

    smoothed = F.conv2d(noise, kernel2d, padding=padding, groups=K)
    labels = smoothed.argmax(dim=1).squeeze(0)  # (H, W), long
    return labels


def generate_lmm_weights(K, comp_map, kernel_tensor):
    one_hot = F.one_hot(comp_map, num_classes=K).float()  # (H, W, K)
    one_hot = one_hot.permute(2, 0, 1).unsqueeze(0)  # (1, K, H, W)

    kernel2d = kernel_tensor.expand(K, 1, -1, -1)  # (K,1,k,k)
    # pad the 1 hot vector
    p = kernel_tensor.shape[-1] // 2
    one_hot = F.pad(one_hot, (p, p, p, p), mode='replicate')

    smoothed = F.conv2d(one_hot, kernel2d, padding=0, groups=K)  # (1, K, H, W)
    smoothed = smoothed.squeeze(0)  # (K, H, W)

    # 4) Normalize across components to get convex weights
    smoothed = torch.clamp(smoothed, min=1e-12)
    smoothed = smoothed / smoothed.sum(dim=0, keepdim=True)  # (K, H, W)

    mix_weights = smoothed
    return mix_weights





