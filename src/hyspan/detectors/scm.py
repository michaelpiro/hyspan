"""
SCM-based detectors (Sample Covariance Matrix).

Background statistics are estimated locally per-pixel from a k×k
neighbourhood — no fitting step required.
"""
from __future__ import annotations

import torch
from typing import Optional

from .base import BaseDetector
from ..algorithms.target_detection.classic_scm import (
    CEM_scm_algorithm,
    AMF_SCM_algorithm,
    ACE_SCM_algorithm,
    kellys_SCM_algorithm,
)


class _BaseSCM(BaseDetector):
    """Shared constructor for all SCM detectors."""

    def __init__(
        self,
        kernel_size: int = 7,
        remove_central: bool = True,
        remove_neighbors_size: int = 1,
        eps: float = 1e-8,
    ):
        self.kernel_size = kernel_size
        self.remove_central = remove_central
        self.remove_neighbors_size = remove_neighbors_size
        self.eps = eps

    def _run(self, fn, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        H, W = image.shape[:2]
        scores = fn(
            image,
            target,
            kernel_size=self.kernel_size,
            remove_central=self.remove_central,
            remove_neighbors_size=self.remove_neighbors_size,
            eps=self.eps,
        )
        return scores.view(H, W)


class CEM_SCM(_BaseSCM):
    """Constrained Energy Minimisation with local SCM background."""

    def detect(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._run(CEM_scm_algorithm, image, target)


class AMF_SCM(_BaseSCM):
    """Adaptive Matched Filter with local SCM background (GLRT φ₁=1, φ₂=0)."""

    def detect(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._run(AMF_SCM_algorithm, image, target)


class ACE_SCM(_BaseSCM):
    """Adaptive Cosine/Coherence Estimator with local SCM (GLRT φ₁=0, φ₂=1)."""

    def detect(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._run(ACE_SCM_algorithm, image, target)


class Kellys_SCM(_BaseSCM):
    """Kelly's detector with local SCM background (GLRT φ₁=N, φ₂=1)."""

    def detect(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._run(kellys_SCM_algorithm, image, target)
