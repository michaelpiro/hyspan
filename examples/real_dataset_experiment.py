"""
Real-dataset experiment
=======================

Runs all 6 detectors (CEM/AMF/ACE × SCM/GMM) on a real HSI benchmark
dataset, with optional PCA dimensionality reduction to make SCM detectors
tractable on high-dimensional data.

Supported datasets (all in datasets/real/):
    Sandiego       — 100×100×189,  binary gt,     uint16 radiance
    abu-airport-2  — 100×100×205,  binary gt,     uint16 radiance
    pavia-u        — 610×340×103,  multiclass gt, float32 (pick one class as target)

Usage:
    python examples/real_dataset_experiment.py                       # Sandiego + PCA→30
    python examples/real_dataset_experiment.py --dataset abu-airport-2
    python examples/real_dataset_experiment.py --dataset pavia-u --target-class 6
    python examples/real_dataset_experiment.py --pca 0              # disable PCA

The script normalises each image to [0, 1] per-band before detection.
PCA is fitted on background pixels only and applied to both image and
target spectrum (pure-torch SVD, no sklearn required).
"""
from __future__ import annotations

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "experiments" / "real_results"
PLOTS_DIR = ROOT / "experiments" / "real_plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(name: str, target_class: int = 1):
    """
    Load a .mat dataset.

    Returns
    -------
    image    : (H, W, D) float32 torch tensor, normalised to [0, 1] per-band
    binary_gt: (H, W)    float32 torch tensor, 1 = target, 0 = background
    """
    path = ROOT / "datasets" / "real" / f"{name}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = loadmat(str(path))
    data_np = raw["data"].astype(np.float32)  # (H, W, D)
    map_np = raw["map"]

    # per-band normalisation to [0, 1]
    H, W, D = data_np.shape
    flat = data_np.reshape(-1, D)
    band_min = flat.min(axis=0)
    band_max = flat.max(axis=0)
    data_np = (data_np - band_min) / np.maximum(band_max - band_min, 1e-8)

    # binarise gt
    unique = np.unique(map_np)
    if len(unique) == 2:
        binary_gt_np = (map_np != 0).astype(np.float32)
    else:
        print(f"    Multiclass gt — labels: {unique.tolist()}  →  using class {target_class} as target")
        binary_gt_np = (map_np == target_class).astype(np.float32)

    # safe numpy→torch (no C bridge)
    image = torch.tensor(data_np.tolist(), dtype=torch.float32)
    binary_gt = torch.tensor(binary_gt_np.tolist(), dtype=torch.float32)

    n_tgt = int(binary_gt.sum().item())
    print(f"    Image : {tuple(image.shape)}  D={D}")
    print(f"    Target: {n_tgt} pixels ({n_tgt / (H * W) * 100:.2f}% of image)")
    return image, binary_gt


# ─────────────────────────────────────────────────────────────────────────────
# PCA (pure torch — no sklearn / numpy bridge)
# ─────────────────────────────────────────────────────────────────────────────

class TorchPCA:
    """
    Simple PCA fitted on the background pixels of an HSI.

    Uses torch.linalg.eigh on the (D × D) covariance matrix — efficient
    when D < N (standard case for HSI).
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self._mean: torch.Tensor | None = None  # (D,)
        self._components: torch.Tensor | None = None  # (D, n_components)
        self._explained_var: torch.Tensor | None = None

    def fit(self, pixels: torch.Tensor) -> "TorchPCA":
        """
        Fit PCA to (N, D) background pixels.
        """
        N, D = pixels.shape
        self._mean = pixels.mean(0)  # (D,)
        centered = pixels - self._mean  # (N, D)

        # Covariance matrix (D, D)
        cov = centered.T @ centered / (N - 1)

        # Eigendecomposition — eigh returns eigenvalues in ascending order
        eigvals, eigvecs = torch.linalg.eigh(cov)  # (D,), (D, D)

        # Take the top-k eigenvectors (largest eigenvalues)
        k = min(self.n_components, D)
        self._components = eigvecs[:, -k:].flip(-1).contiguous()  # (D, k)
        self._explained_var = eigvals[-k:].flip(-1)

        total_var = eigvals.sum().item()
        expl_var = self._explained_var.sum().item()
        print(f"    PCA: {k} components explain "
              f"{expl_var / total_var * 100:.1f}% of variance  (D {D} → {k})")
        return self

    def transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Project image (H, W, D) or pixels (N, D) onto PCA components.

        Returns the same shape with the last dimension replaced by n_components.
        """
        shape = image.shape
        pixels = image.reshape(-1, shape[-1])
        projected = (pixels - self._mean) @ self._components  # (N, k)
        return projected.reshape(*shape[:-1], self._components.shape[1])

    def transform_vector(self, v: torch.Tensor) -> torch.Tensor:
        """Project a single (D,) vector."""
        return (v - self._mean) @ self._components  # (k,)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
        dataset_name: str,
        target_class: int,
        n_gmm_components: int,
        n_pca: int,
        scm_k: int,

):
    from hyspan import (
        CEM_SCM, AMF_SCM, ACE_SCM,
        CEM_GMM, AMF_GMM, ACE_GMM,
        Experiment, ResultsStore,
    )

    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1] Loading dataset ...")
    image, binary_gt = load_dataset(dataset_name, target_class=target_class)
    D_orig = image.shape[-1]

    # ── PCA ───────────────────────────────────────────────────────────────────
    if n_pca > 0 and n_pca < D_orig:
        print(f"\n[2] Fitting PCA on background pixels (n_components={n_pca}) ...")
        bg_pixels = image[binary_gt == 0].reshape(-1, D_orig)  # (N_bg, D)
        pca = TorchPCA(n_components=n_pca)
        pca.fit(bg_pixels)

        image_pca = pca.transform(image)  # (H, W, n_pca)
        tgt_pixels = image[binary_gt == 1]  # (N_tgt, D)
        target_sig = pca.transform_vector(tgt_pixels.mean(0))  # (n_pca,)
        D = n_pca
        step_offset = 1
    else:
        print(f"\n[2] Skipping PCA (D={D_orig})")
        image_pca = image
        target_sig = image[binary_gt == 1].mean(0)
        D = D_orig
        step_offset = 0

    print(f"    Working dimension: D={D}")
    print(f"    Target signature shape: {tuple(target_sig.shape)}")

    # SCM kernel: need N_eff = k² - 9 > D

    # scm_k = int(np.ceil(np.sqrt(D + 10)))
    if scm_k % 2 == 0:
        scm_k += 1
    scm_k = max(9, min(scm_k, 35))
    print(f"    SCM kernel_size={scm_k}  (D={D})")

    # ── Experiment ────────────────────────────────────────────────────────────
    print(f"\n[{2 + step_offset}] Running detectors (GMM K={n_gmm_components}) ...")
    exp_name = f"{dataset_name}_pca{n_pca}_K{n_gmm_components}"
    exp = Experiment(image_pca, binary_gt, name=exp_name)
    (exp
     .add(CEM_SCM(kernel_size=scm_k))
     .add(AMF_SCM(kernel_size=scm_k))
     .add(ACE_SCM(kernel_size=scm_k))
     .add(CEM_GMM(n_components=n_gmm_components))
     .add(AMF_GMM(n_components=n_gmm_components))
     .add(ACE_GMM(n_components=n_gmm_components))
     )

    result = exp.run(target_signature=target_sig, verbose=True)

    print(f"\n{'─' * 45}")
    print(result.summary())
    print(f"{'─' * 45}")
    print(f"Best: {result.best().name}  AUC={result.best().auc:.4f}")

    # ── Store ─────────────────────────────────────────────────────────────────
    print(f"\n[{3 + step_offset}] Saving results ...")
    store = ResultsStore(str(RESULTS_DIR))
    store.save(result)
    print(f"    Saved '{exp_name}'")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n[{4 + step_offset}] Generating plots → {PLOTS_DIR}")
    from hyspan.plots import plot_roc_curves, plot_detection_maps, plot_auc_bar

    plot_roc_curves(
        result,
        title=f"ROC — {dataset_name}  (PCA→{n_pca}, GMM K={n_gmm_components})",
        save_path=str(PLOTS_DIR / f"{exp_name}_roc.png"),
        show=False,
    )
    plot_detection_maps(
        result,
        cols=4,
        save_path=str(PLOTS_DIR / f"{exp_name}_maps.png"),
        show=False,
    )
    plot_auc_bar(
        result,
        save_path=str(PLOTS_DIR / f"{exp_name}_auc_bar.png"),
        show=False,
    )
    print("    Saved: _roc.png, _maps.png, _auc_bar.png")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Sandiego")
    p.add_argument("--target-class", type=int, default=1)
    p.add_argument("--gmm-components", type=int, default=10)
    p.add_argument("--pca", type=int, default=30,
                   help="Number of PCA components (0 = disable, default: 30)")
    p.add_argument("--scm-kernel", type=int, default=15,
                   help="SCM kernel size (default: 15)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        dataset_name=args.dataset,
        target_class=args.target_class,
        n_gmm_components=args.gmm_components,
        n_pca=args.pca,
        scm_k=args.scm_kernel,
    )
