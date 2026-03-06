from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from skimage.util import view_as_windows
from torch import Tensor


def patch_cosin_similarity(patches, centers):
    centers = centers[:, None, :]
    patches_norm = np.sqrt(np.sum(patches ** 2, axis=-1, keepdims=True))
    centers_norm = np.sqrt(np.sum(centers ** 2, axis=-1, keepdims=True))
    x_y_multi = np.sum(np.multiply(patches, centers), axis=-1, keepdims=True)
    return x_y_multi / (patches_norm * centers_norm + 1e-8)

def cosin_similarity(x, y):
    norm_x = np.sqrt(np.sum(x ** 2, axis=-1))
    norm_y = np.sqrt(np.sum(y ** 2, axis=-1))
    x_y = np.sum(np.multiply(x, y), axis=-1)
    similarity = np.clip(x_y / (norm_x * norm_y), -1, 1)
    return np.arccos(similarity)


def ts_generation(data, gt, type=0):
    '''
    using different methods to select the target spectrum
    :param type:
    0-Averaged spectrum
    1-Averaged spectrum after spatial debiasing (L2 norm)
    2-Averaged spectrum after spectral debiasing (spectral angle)
    3-Spatial median spectrum (median of L2 norm)
    4-Spectral median spectrum (median of spectral angles)
    5-The target pixel with the closest Euclidean distance to all the other target pixels
    6-The target pixel with the closest cosine distance to all the target pixels
    7-The target pixel closest to the averaged spectrum (L2 norm)
    8-The target pixel closest to the averaged spectrum (spectral angle)
    :return:
    '''
    ind = np.where(gt == 1)
    ts = data[ind]
    avg_target_spectrum = np.mean(ts, axis=0)
    avg_target_spectrum = np.expand_dims(avg_target_spectrum, axis=-1)
    if type == 0:
        return avg_target_spectrum
    elif type == 1:
        spatial_distance = np.sqrt(np.sum((ts - avg_target_spectrum.T) ** 2, axis=-1))
        arg_distance = np.argsort(spatial_distance)
        saved_num = int(ts.shape[0] * 0.8)
        saved_spectrums = ts[arg_distance[:saved_num]]
        removed_deviation_target_spectrum = np.mean(saved_spectrums, axis=0)
        removed_deviation_target_spectrum = np.expand_dims(removed_deviation_target_spectrum, axis=-1)
        return removed_deviation_target_spectrum
    elif type == 2:
        spatial_distance = cosin_similarity(ts, avg_target_spectrum.T)
        arg_distance = np.argsort(spatial_distance)
        saved_num = int(ts.shape[0] * 0.8)
        saved_spectrums = ts[arg_distance[:saved_num]]
        removed_deviation_target_spectrum = np.mean(saved_spectrums, axis=0)
        removed_deviation_target_spectrum = np.expand_dims(removed_deviation_target_spectrum, axis=-1)
        return removed_deviation_target_spectrum
    elif type == 3:
        dist_list = np.zeros([ts.shape[0]])
        for i in range(ts.shape[0]):
            dist_list[i] = np.mean(np.sqrt(np.sum(np.square(ts - ts[i]), axis=-1)))
        arg_distance = np.argsort(dist_list)
        mid = ts.shape[0] // 2
        mid_target_spectrum = ts[arg_distance[mid]]
        mid_target_spectrum = np.expand_dims(mid_target_spectrum, axis=-1)
        return mid_target_spectrum
    elif type == 4:
        dist_list = np.zeros([ts.shape[0]])
        for i in range(ts.shape[0]):
            dist_list[i] = np.mean(cosin_similarity(ts, ts[i]))
        arg_distance = np.argsort(dist_list)
        mid = ts.shape[0] // 2
        mid_target_spectrum = ts[arg_distance[mid]]
        mid_target_spectrum = np.expand_dims(mid_target_spectrum, axis=-1)
        return mid_target_spectrum
    elif type == 5:
        min_distance = 10000
        opd_i = 0
        for i in range(ts.shape[0]):
            dist = np.mean(np.sqrt(np.sum(np.square(ts - ts[i]), axis=-1)))
            # print(dist)
            if dist < min_distance:
                min_distance = dist
                opd_i = i
        target_spectrum = ts[opd_i]
        target_spectrum = np.expand_dims(target_spectrum, axis=-1)
        return target_spectrum
    elif type == 6:
        min_distance = 10000
        opd_i = 0
        for i in range(ts.shape[0]):
            dist = np.mean(cosin_similarity(ts, ts[i]))
            # print(dist)
            if dist < min_distance:
                min_distance = dist
                opd_i = i
        target_spectrum = ts[opd_i]
        target_spectrum = np.expand_dims(target_spectrum, axis=-1)
        return target_spectrum

    elif type == 7:
        distance = np.sqrt(np.sum((ts - avg_target_spectrum.T) ** 2, axis=-1))
        arg_distance = np.argsort(distance)
        avg_L2_target_spectrum = ts[arg_distance[0]]
        avg_L2_target_spectrum = np.expand_dims(avg_L2_target_spectrum, axis=-1)
        return avg_L2_target_spectrum
    elif type == 8:
        distance = cosin_similarity(ts, avg_target_spectrum.T)
        # print(distance)
        arg_distance = np.argsort(distance)
        avg_cosin_target_spectrum = ts[arg_distance[0]]
        avg_cosin_target_spectrum = np.expand_dims(avg_cosin_target_spectrum, axis=-1)
        return avg_cosin_target_spectrum
    else:
        return avg_target_spectrum


def encode_patches(patch, center):
    encoded_weight = patch_cosin_similarity(patch, center)  # shape (num_patches, K*K , 1)
    encoded_weight = np.exp(encoded_weight) / np.sum(np.exp(encoded_weight), axis=1, keepdims=True)
    # encoded_weight = encoded_weight[:, None]
    encoded_vector = np.sum(encoded_weight * patch, axis=1)
    # encoded_vector = encoded_vector[None, :]
    return encoded_vector


def extrcat_patches(image, kernel_size):
    H, W, D = image.shape
    pad = kernel_size // 2
    image_padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='symmetric')
    # x = image_padded[np.newaxis, ...]  # add batch dimension

    # patches = view_as_windows(x, (1,kernel_size, kernel_size , D), step=(1,1, 1, 1))
    patches = view_as_windows(image_padded, (kernel_size, kernel_size, D), step=(1, 1, 1))
    # patches = patches[0, :, :, 0, :, :, :]  # remove batch and channel dimension
    patches = patches.reshape(H * W, kernel_size * kernel_size, D)
    return patches


def spectral_encoding(image: torch.Tensor, kernel_size):
    patches = extrcat_patches(image.numpy(), kernel_size=kernel_size)
    centers = patches[:, kernel_size * kernel_size // 2, :]
    encoded_patches = encode_patches(patches, centers)  # shape (H*W, D)
    return torch.from_numpy(encoded_patches.reshape(image.shape))


def add_targets_to_image(image: torch.Tensor, target_percent: float, target: torch.Tensor,
                         is_additive: bool = True, boost=0.5, allow_scale=False) -> tuple[Tensor, Tensor]:
    """
    Adds target vector to a percentage of image pixels (element-wise addition),
    returns indices of modified pixels (2D indices).
    """
    H, W, D = image.size()
    N = H * W
    k = max(1, int(N * (target_percent / 100.0)))
    flat_idx = torch.randperm(N)[:k]
    row_idx = flat_idx // W
    col_idx = flat_idx % W

    new_image = torch.zeros_like(image)
    new_image.copy_(image)

    random_target_weights = torch.sqrt(torch.rand(k, 1, device=image.device)) + boost
    # clip to [0.4, 1.0] using torch clamp
    random_target_weights = torch.clamp(random_target_weights, boost, 1.0)

    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).to(image.device)

    # easy case
    if is_additive:
        new_image[row_idx, col_idx, :] += target * random_target_weights

    # harder case, slightly suppressive
    else:
        random_image_weights = 1.0 - random_target_weights
        new_image[row_idx, col_idx, :] = (image[row_idx, col_idx,
                                          :] * random_image_weights) + target * random_target_weights

    if allow_scale:
        # randomly scale the target by a factor between 0.5 and 1.5
        k = max(1, int(N * (4.0 / 100.0)))
        flat_idx = torch.randperm(N)[:k]
        row_idx = flat_idx // W
        col_idx = flat_idx % W
        scale_factors = torch.clamp(torch.empty(k, 1, device=image.device).uniform_(0.8, 1.2), 0.8, 1.0)
        new_image[row_idx, col_idx, :] *= scale_factors

    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[row_idx, col_idx] = True
    return new_image, mask


# ---------- Image generators ----------

import torch
import torch.nn.functional as F


def get_gaussian_kernel(kernel_size=3, smoothness=0.8):
    padding = kernel_size // 2

    coords = torch.arange(kernel_size, dtype=torch.float32) - padding
    gauss_1d = torch.exp(-coords ** 2 / (2 * smoothness ** 2 + 1e-12))
    gauss_1d /= gauss_1d.sum()

    kernel2d = gauss_1d[:, None] * gauss_1d[None, :]  # (k,k)
    return kernel2d
