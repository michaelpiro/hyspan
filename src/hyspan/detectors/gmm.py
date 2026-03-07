"""
GMM-based detectors.

Background is modelled as a Gaussian Mixture Model fitted to background
pixels (gt == 0).  Call ``detector.fit(image, gt)`` before ``detect()``,
or let ``Experiment.run()`` handle it automatically.
"""
from __future__ import annotations

import torch
from typing import Optional, Union

from .base import BaseDetector
from ..algorithms.target_detection.classic_gmm import ClassicDetectorGMM
from ..deep_models.GMM.TorchGmm import TorchGMM


class _BaseGMMDetector(BaseDetector):
    """
    Shared fit logic for all GMM-based detectors.

    Parameters
    ----------
    n_components : int
        Number of GMM components for the background model.
    kernel_size : int
        Neighbourhood size used for the majority-vote component assignment.
    remove_central : bool
        Exclude the pixel under test and its neighbourhood from local stats.
    remove_neighbors : int
        Radius of the excluded central region.
    device : str
        PyTorch device for GMM fitting ('cpu', 'cuda', 'mps').
    gmm_kwargs : dict
        Extra keyword arguments forwarded to ``TorchGMM``.
    """

    def __init__(
        self,
        n_components: int = 10,
        kernel_size: int = 9,
        remove_central: bool = True,
        remove_neighbors: int = 2,
        device: str = "cpu",
        **gmm_kwargs,
    ):
        self.n_components = n_components
        self.kernel_size = kernel_size
        self.remove_central = remove_central
        self.remove_neighbors = remove_neighbors
        self.device = device
        self._gmm_kwargs = gmm_kwargs
        self._inner: Optional[ClassicDetectorGMM] = None

    def fit(
        self,
        image: torch.Tensor,
        gt: Optional[torch.Tensor] = None,
    ) -> "_BaseGMMDetector":
        H, W, D = image.shape
        pixels = image.reshape(-1, D).to(self.device)

        if gt is not None:
            bkg_mask = gt.reshape(-1) == 0
            pixels = pixels[bkg_mask]

        gmm = TorchGMM(
            n_components=self.n_components,
            covariance_type="full",
            reg_covar=1e-6,
            tol=1e-3,
            max_iter=100,
            n_init=1,
            device=self.device,
            **self._gmm_kwargs,
        )
        gmm.fit(pixels)

        self._inner = ClassicDetectorGMM(
            target_signature=None,
            gmm=gmm,
            neighbors=self.kernel_size,
            remove_pxl_undr_tst=self.remove_central,
            remove_neighbors=self.remove_neighbors,
        )
        return self

    @classmethod
    def from_fitted_gmm(
        cls,
        gmm: TorchGMM,
        kernel_size: int = 9,
        remove_central: bool = True,
        remove_neighbors: int = 2,
    ) -> "_BaseGMMDetector":
        """Build a detector from an already-fitted TorchGMM (avoids re-fitting)."""
        inst = cls.__new__(cls)
        inst.n_components = gmm.n_components
        inst.kernel_size = kernel_size
        inst.remove_central = remove_central
        inst.remove_neighbors = remove_neighbors
        inst.device = str(gmm.device)
        inst._gmm_kwargs = {}
        inst._inner = ClassicDetectorGMM(
            target_signature=None,
            gmm=gmm,
            neighbors=kernel_size,
            remove_pxl_undr_tst=remove_central,
            remove_neighbors=remove_neighbors,
        )
        return inst

    def _check_fitted(self):
        if self._inner is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling detect(). "
                "Call .fit(image, gt) or use Experiment.run() which does this automatically."
            )

    def _run_inner(self, method_name: str, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        H, W = image.shape[:2]
        self._inner.set_target(target)
        scores = getattr(self._inner, method_name)(image)
        return scores.view(H, W)


class LRT_GMM(BaseDetector):
    """
    Likelihood Ratio Test detector using two fitted GMMs.

    Score for pixel x = log p(x | target GMM) - log p(x | background GMM).

    This is the Neyman-Pearson optimal detector under the GMM approximations:
    no threshold tuning or covariance estimation at test time — just evaluate
    both densities and take the log-ratio.

    Parameters
    ----------
    n_bg_components : int
        Number of GMM components for the background model.
    n_target_components : int
        Number of GMM components for the target model.
        Use 1 for a single Gaussian (sufficient when targets are homogeneous).
    device : str
        PyTorch device for GMM fitting ('cpu', 'cuda', 'mps').
    gmm_kwargs : dict
        Extra keyword arguments forwarded to both TorchGMM instances.

    Notes
    -----
    ``detect()`` ignores the ``target`` argument required by the BaseDetector
    interface — the target distribution is encoded in the fitted target GMM.
    ``fit(image, gt)`` is mandatory; gt must contain at least one target pixel
    (gt == 1).
    """

    def __init__(
        self,
        n_bg_components: int = 10,
        n_target_components: int = 3,
        device: str = "cpu",
        **gmm_kwargs,
    ):
        self.n_bg_components = n_bg_components
        self.n_target_components = n_target_components
        self.device = device
        self._gmm_kwargs = gmm_kwargs
        self._bg_gmm: Optional[TorchGMM] = None
        self._tgt_gmm: Optional[TorchGMM] = None

    def fit(
        self,
        image: torch.Tensor,
        gt: Optional[torch.Tensor] = None,
    ) -> "LRT_GMM":
        if gt is None:
            raise ValueError(
                "LRT_GMM.fit() requires gt with target pixels (gt == 1). "
                "Pass a binary ground-truth map or use a background-only detector."
            )
        H, W, D = image.shape
        pixels = image.reshape(-1, D).to(self.device)
        flat_gt = gt.reshape(-1)

        bg_pixels  = pixels[flat_gt == 0]
        tgt_pixels = pixels[flat_gt == 1]

        if tgt_pixels.shape[0] == 0:
            raise ValueError("No target pixels (gt == 1) found — cannot fit target GMM.")

        def _make_gmm(n_components: int) -> TorchGMM:
            return TorchGMM(
                n_components=n_components,
                covariance_type="full",
                reg_covar=1e-6,
                tol=1e-3,
                max_iter=100,
                n_init=1,
                device=self.device,
                **self._gmm_kwargs,
            )

        self._bg_gmm = _make_gmm(self.n_bg_components)
        self._bg_gmm.fit(bg_pixels)

        self._tgt_gmm = _make_gmm(self.n_target_components)
        self._tgt_gmm.fit(tgt_pixels)

        return self

    def detect(
        self,
        image: torch.Tensor,
        target: torch.Tensor,   # unused — kept for BaseDetector compatibility
    ) -> torch.Tensor:
        if self._bg_gmm is None or self._tgt_gmm is None:
            raise RuntimeError(
                "LRT_GMM must be fitted before calling detect(). "
                "Call .fit(image, gt) or use Experiment.run()."
            )
        H, W, D = image.shape
        pixels = image.reshape(-1, D)

        log_p_bg  = self._bg_gmm.log_prob_torch(pixels)   # (N,)
        log_p_tgt = self._tgt_gmm.log_prob_torch(pixels)  # (N,)

        return (log_p_tgt - log_p_bg).view(H, W)


class CEM_GMM(_BaseGMMDetector):
    """Constrained Energy Minimisation with GMM background."""

    def detect(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._run_inner("CEM", image, target)


class AMF_GMM(_BaseGMMDetector):
    """Adaptive Matched Filter with GMM background (GLRT φ₁=1, φ₂=0)."""

    def detect(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._run_inner("AMF", image, target)


class ACE_GMM(_BaseGMMDetector):
    """Adaptive Cosine/Coherence Estimator with GMM background (GLRT φ₁=0, φ₂=1)."""

    def detect(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._run_inner("ACE", image, target)


class Kellys_GMM(_BaseGMMDetector):
    """Kelly's detector with GMM background (GLRT φ₁=N, φ₂=1)."""

    def detect(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._run_inner("kellys", image, target)
