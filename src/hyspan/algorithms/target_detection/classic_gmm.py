from __future__ import annotations

import torch
import numpy as np
from scipy.ndimage import convolve
from torch.nn import functional as F

from src.hyspan.algorithms.classic import glrt
from src.hyspan.deep_models import GMM


class ClassicDetectorGMM:
    def __init__(self, target_signature: torch.Tensor, gmm: GMM, neighbors: int = 9, remove_pxl_undr_tst=True,
                 remove_neighbors=2):
        # TODO: add assertions for input shapes and types
        self.target_signature = target_signature
        self.gmm = gmm
        self.kernel_size = neighbors
        self.remove_neighbors = remove_neighbors
        self.remove_pxl_undr_tst = remove_pxl_undr_tst

    @torch.no_grad()
    def get_patches_votes(self, image: torch.Tensor, kernel_size=None, neighborhood_size=None,
                          remove_central_pixel=None,
                          memory_efficient: bool = False) -> torch.Tensor:

        kernel_size = kernel_size if kernel_size is not None else self.kernel_size
        neighborhood_size = neighborhood_size if neighborhood_size is not None else self.remove_neighbors
        remove_central_pixel = remove_central_pixel if remove_central_pixel is not None else self.remove_pxl_undr_tst

        np_image = image.cpu().numpy()
        orig_shape = np_image.shape
        if np_image.ndim == 2:
            samples = np_image  # (W,D) or (H,D)

        elif np_image.ndim == 3:
            H, W, D = np_image.shape
            samples = np_image.reshape(-1, D)  # (H*W, D)
        else:
            raise ValueError(f"Unsupported image shape: {np_image.shape}")
        comps = self.gmm.predict(samples)
        comp_map = comps.reshape(orig_shape[:-1])  # (H,W) or (W,)

        if memory_efficient:
            patches_votes = majority_vote_loop(comp_map, kernel_size=kernel_size, neighborhood_size=neighborhood_size,
                                               remove_central_pixel=remove_central_pixel)
        else:
            patches_votes = majority_vote_filter(comp_map, kernel_size=kernel_size, neighborhood_size=neighborhood_size,
                                                 remove_central_pixel=remove_central_pixel)

        return torch.from_numpy(patches_votes).long().to(image.device)

    @torch.no_grad()
    def CEM(self, image: torch.Tensor, kernel_size=None, remove_central_pixel=None, neighborhood_size=None,
            memory_efficient=False) -> torch.Tensor:

        kernel_size = kernel_size if kernel_size is not None else self.kernel_size
        neighborhood_size = neighborhood_size if neighborhood_size is not None else self.remove_neighbors
        remove_central_pixel = remove_central_pixel if remove_central_pixel is not None else self.remove_pxl_undr_tst

        H, W, D = image.shape
        x = image.reshape(-1, D)
        # comps_pixel = get_top_gaussian(image, gmm).view(-1).long()
        comps = self.get_patches_votes(image, kernel_size=kernel_size, remove_central_pixel=remove_central_pixel,
                                       neighborhood_size=neighborhood_size,
                                       memory_efficient=memory_efficient).view(-1).long()

        mu_s = self.gmm.means_torch  # (K,D)
        cov = self.gmm.covariances_torch  # (K,D,D)
        R_x = cov + (mu_s.unsqueeze(1) @ mu_s.unsqueeze(2))  # (K,D,D)
        z = torch.linalg.solve(R_x, self.target_signature.view(D, 1))  # (K,D,1)

        den = self.target_signature.view(1, D).unsqueeze(1) @ z  # (K,1,1)
        c_k = z / den  # (K,D,1)
        c = c_k[comps]  # (N,D,1)
        scores = c.transpose(-1, -2) @ x.unsqueeze(-1)  # (N,1,1)
        return scores.squeeze()

    @torch.no_grad()
    def glrt_neighbors_algorithm(self, image: torch.Tensor, phi1, phi2, kernel_size=None, remove_central_pixel=None,
                                 neighborhood_size=None, memory_efficient=False) -> torch.Tensor:

        kernel_size = kernel_size if kernel_size is not None else self.kernel_size
        neighborhood_size = neighborhood_size if neighborhood_size is not None else self.remove_neighbors
        remove_central_pixel = remove_central_pixel if remove_central_pixel is not None else self.remove_pxl_undr_tst
        H, W, D = image.shape
        x = image.reshape(-1, D)
        # comps_pixel = get_top_gaussian(image, gmm).view(-1).long()
        comps = self.get_patches_votes(image, kernel_size=kernel_size, remove_central_pixel=remove_central_pixel,
                                       neighborhood_size=neighborhood_size,
                                       memory_efficient=memory_efficient).view(-1).long()

        mu_s = self.gmm.means_torch
        # move the samples by the mean of the selected component
        x_centered = x - mu_s[comps]  # (N,D)

        # # compute the inverse cov for each component
        # inv_chol_k = torch.cholesky_inverse(gmm.scale_tril())
        # inv_k = inv_chol_k[comps]  # (N,D,D)

        inv_chol_k = self.gmm.inv_cholesky_covariances_torch
        inv_cov_k = inv_chol_k @ inv_chol_k.transpose(-1, -2)  # (K,D,D)
        inv_k = inv_cov_k[comps]  # (N,D,D)

        # center the target by the mean of the selected component
        centered_target = self.target_signature - mu_s[comps]  # (N,D)

        # compute the detector scores
        centered_target = centered_target.view(-1, D, 1)

        scores = glrt(x_centered, centered_target, inv_k, phi1=phi1, phi2=phi2)

        return scores.squeeze()

    @torch.no_grad()
    def AMF(self, image: torch.Tensor, kernel_size=None, remove_central_pixel=None,
            neighborhood_size=None, memory_efficient=False):
        return self.glrt_neighbors_algorithm(image, phi1=1, phi2=0, kernel_size=kernel_size,
                                             remove_central_pixel=remove_central_pixel,
                                             neighborhood_size=neighborhood_size, memory_efficient=memory_efficient)

    @torch.no_grad()
    def ACE(self, image: torch.Tensor, kernel_size=None, remove_central_pixel=None,
            neighborhood_size=None, memory_efficient=False):
        return self.glrt_neighbors_algorithm(image, phi1=0, phi2=1, kernel_size=kernel_size,
                                             remove_central_pixel=remove_central_pixel,
                                             neighborhood_size=neighborhood_size, memory_efficient=memory_efficient)

    @torch.no_grad()
    def kellys(self, image: torch.Tensor, kernel_size=None, remove_central_pixel=None,
               neighborhood_size=None, memory_efficient=False):
        H, W, D = image.shape
        N = H * W
        return self.glrt_neighbors_algorithm(image, phi1=N, phi2=1, kernel_size=kernel_size,
                                             remove_central_pixel=remove_central_pixel,
                                             neighborhood_size=neighborhood_size, memory_efficient=memory_efficient)

@torch.no_grad()
def majority_vote_filter(image: np.ndarray, kernel_size=11, neighborhood_size=2,
                         remove_central_pixel=True) -> np.ndarray:
    """
    Applies a fast majority-vote filter to a 2D class map using convolution.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D (class map).")
    # 1. Determine the number of classes (assuming values [0, n])
    n = int(np.max(image)) + 1

    # 2. Build the exclusion mask (1s where we count, 0s where we ignore)
    mask = np.ones((kernel_size, kernel_size), dtype=np.int32)
    c = kernel_size // 2  # The center index

    # Exclude the inner patch
    if neighborhood_size > 0:
        # Calculate bounds to keep it as centered as possible
        start = c - (neighborhood_size // 2)
        end = start + neighborhood_size
        mask[start:end, start:end] = 0

    # Exclude the exact central pixel
    if remove_central_pixel:
        mask[c, c] = 0

    # 3. Initialize an array to hold the "votes" for each class.
    # Shape is (n, Height, Width). Index 0 is ignored since classes start at 0.
    counts = np.zeros((n, image.shape[0], image.shape[1]), dtype=np.int32)

    # 4. Count frequencies via Convolution
    # A convolution of a binary map with our mask effectively counts the
    # occurrences of that class in the valid neighborhood for every pixel at once.
    for k in range(n):
        binary_map = (image == k).astype(np.int32)

        # scipy.ndimage.convolve handles the mirroring padding automatically
        # 'reflect' mode mirrors across the edge exactly as requested (d c b | a b c d | c b a)
        counts[k] = convolve(binary_map, mask, mode='reflect')

    # 5. Get the majority class for each pixel
    # argmax along the class axis (axis=0) returns the index (class) with the highest vote
    filtered_image = np.argmax(counts, axis=0)

    return filtered_image

@torch.no_grad()
def majority_vote_loop(image: np.ndarray, kernel_size=11, neighborhood_size=2, remove_central_pixel=True) -> np.ndarray:
    H, W = image.shape
    pad_size = kernel_size // 2

    # 1. Pad the image (mirroring/'reflect' mode)
    padded_img = np.pad(image, pad_width=pad_size, mode='reflect')

    # 2. Build the boolean exclusion mask (True = keep, False = ignore)
    mask = np.ones((kernel_size, kernel_size), dtype=bool)
    c = kernel_size // 2

    # Exclude the inner patch
    if neighborhood_size > 0:
        start = c - (neighborhood_size // 2)
        end = start + neighborhood_size
        mask[start:end, start:end] = False

    # Exclude the exact central pixel
    if remove_central_pixel:
        mask[c, c] = False

    # 3. Initialize the output array
    filtered_image = np.zeros_like(image)

    # 4. Iterate over every single pixel
    for i in range(H):
        for j in range(W):
            # Extract the local kernel_size x kernel_size patch
            patch = padded_img[i:i + kernel_size, j:j + kernel_size]

            # Filter the patch using our boolean mask (flattens into a 1D array of valid pixels)
            valid_pixels = patch[mask]

            # Find the most frequent class
            # np.bincount is extremely fast for arrays of small positive integers
            filtered_image[i, j] = np.bincount(valid_pixels).argmax()

    return filtered_image

########################################################################################################################
########################################################################################################################

############### following functions should not be used, only for reference, old code from the old project ##############

########################################################################################################################
########################################################################################################################
@torch.no_grad()
def get_top_gaussian_using_neighbors_only(image: torch.Tensor, gmm, n: int = 5, remove_neighbors=2) -> torch.Tensor:
    """
    For each pixel in the image, find the most likely Gaussian component using neighborhood information only.
    Steps:
      1. Compute per-pixel most likely component (using the GMM) once.
      2. Build an n x n neighborhood over the component map.
      3. For each pixel, take a majority vote over the components in its neighborhood,
         EXCLUDING the center pixel.

    Args:
        image: Tensor of shape (H, W, D)
        gmm:   Object with method log_prob_per_comp(x) -> (N, K), where x is (N, D)
        n:     Neighborhood size (must be odd)

    Returns:
        comp_map: LongTensor of shape (H, W) with the chosen component index per pixel.
    """
    assert n % 2 == 1, "Neighborhood size n must be odd."
    H, W, D = image.shape
    device = image.device

    # 1) Per-pixel GMM classification (only once)
    # Flatten to (H*W, D)
    pixels = image.reshape(-1, D)

    # log_p_per_comp: (H*W, K)
    log_p_per_comp = gmm.log_prob_per_comp(pixels)

    # comps_per_pixel: (H*W,)
    comps_per_pixel = torch.argmax(log_p_per_comp, dim=1)

    # Reshape to (1, 1, H, W) for unfolding; keep as float for F.unfold
    comp_map_img = comps_per_pixel.view(1, 1, H, W).float()

    # 2) Reflect padding on the component map
    pad = n // 2
    comp_map_padded = F.pad(comp_map_img, (pad, pad, pad, pad), mode="reflect")  # (1, 1, H+2p, W+2p)

    # 3) Extract n x n neighborhoods of component indices
    # patches: (1, n*n, H*W)
    patches = F.unfold(comp_map_padded, kernel_size=n)

    # -> (H*W, n*n)
    patches = patches.squeeze(0).transpose(0, 1)

    # 4) Exclude center pixel and it's nearest neighbors from each neighborhood
    center_idx = (n * n) // 2
    if n * n > 1:
        # idx_before = torch.arange(center_idx, device=device)  # [0, ..., center_idx-1]
        # idx_after = torch.arange(center_idx + 1, n * n, device=device)  # [center_idx+1, ..., n*n-1]
        # idx = torch.cat([idx_before, idx_after], dim=0)  # length = n*n - 1
        # exclude nearest neighbors as well
        mask = torch.ones(n, n, dtype=torch.bool, device=device)
        start_idx = center_idx // n - remove_neighbors
        end_idx = center_idx // n + remove_neighbors + 1
        mask[start_idx:end_idx, start_idx:end_idx] = False
        mask = mask.view(-1)  # (n*n,)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)  # indices to keep

        # neighbors: (H*W, n*n - 1)
        neighbors = patches[:, idx].long()
    else:
        # Degenerate n=1 case; no neighbors to exclude
        neighbors = patches.long()

    # 5) Majority vote (mode) over neighbors for each pixel
    # top_comp: (H*W,)
    top_comp, _ = torch.mode(neighbors, dim=1)

    # 6) Reshape back to (H, W)
    comp_map = top_comp.view(H, W).long()

    return comp_map


@torch.no_grad()
def CEM_neighbors_algorithm(image: torch.Tensor, target: torch.Tensor, gmm: GMM, neighbors: int = 5,
                            remove_neighbors=2):
    H, W, D = image.shape
    x = image.reshape(-1, D)
    # comps_pixel = get_top_gaussian(image, gmm).view(-1).long()
    comps = get_top_gaussian_using_neighbors_only(image, gmm, n=neighbors, remove_neighbors=remove_neighbors).view(
        -1).long()
    # comps = comp_map.view(-1).long()

    mu_s = gmm.means  # (K,D)
    cov = gmm.covariances()  # (K,D,D)
    R_x = cov + (mu_s.unsqueeze(1) @ mu_s.unsqueeze(2))  # (K,D,D)
    z = torch.linalg.solve(R_x, target.view(D, 1))  # (K,D,1)

    den = target.view(1, D).unsqueeze(1) @ z  # (K,1,1)
    c_k = z / den  # (K,D,1)
    c = c_k[comps]  # (N,D,1)
    scores = c.transpose(-1, -2) @ x.unsqueeze(-1)  # (N,1,1)
    return scores.squeeze()


@torch.no_grad()
def glrt_neighbors_algorithm(image: torch.Tensor, target: torch.Tensor, gmm: GMM, phi1, phi2, neighbors: int = 5,
                             remove_neighbors=2):
    H, W, D = image.shape
    x = image.reshape(-1, D)
    # comps_pixel = get_top_gaussian(image, gmm).view(-1).long()
    comps = get_top_gaussian_using_neighbors_only(image, gmm, n=neighbors, remove_neighbors=remove_neighbors).view(
        -1).long()

    mu_s = gmm.means  # (K,D)
    # move the samples by the mean of the selected component
    x_centered = x - mu_s[comps]  # (N,D)

    # compute the inverse cov for each component
    inv_chol_k = torch.cholesky_inverse(gmm.scale_tril())
    inv_k = inv_chol_k[comps]  # (N,D,D)

    # center the target by the mean of the selected component
    centered_target = target - mu_s[comps]  # (N,D)

    # compute the detector scores
    centered_target = centered_target.view(-1, D, 1)

    scores = glrt(x_centered, centered_target, inv_k, phi1=phi1, phi2=phi2)

    return scores.squeeze()


@torch.no_grad()
def AMF_neighbors_algorithm(image: torch.Tensor, target: torch.Tensor, gmm: GMM, neighbors: int = 5,
                            remove_neighbors=2):
    return glrt_neighbors_algorithm(image, target, gmm, phi1=1, phi2=0, neighbors=neighbors,
                                    remove_neighbors=remove_neighbors)


@torch.no_grad()
def ACE_neighbors_algorithm(image: torch.Tensor, target: torch.Tensor, gmm: GMM, neighbors: int = 5,
                            remove_neighbors=2):
    return glrt_neighbors_algorithm(image, target, gmm, phi1=0, phi2=1, neighbors=neighbors,
                                    remove_neighbors=remove_neighbors)


@torch.no_grad()
def kellys_neighbors_algorithm(image: torch.Tensor, target: torch.Tensor, gmm: GMM, neighbors: int = 5,
                               remove_neighbors=2):
    H, W, D = image.shape
    N = H * W
    return glrt_neighbors_algorithm(image, target, gmm, phi1=N, phi2=1, neighbors=neighbors,
                                    remove_neighbors=remove_neighbors)

########################################################################################################################
########################################################################################################################

########################################### END OLD FUNCTIONS ########################################################

########################################################################################################################
########################################################################################################################

# TODO: ADD TESTS TO THE MODULE