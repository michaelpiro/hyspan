from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseDetector(ABC):
    """
    Minimal interface every detector must implement.

    Lifecycle
    ---------
    1. ``detector.fit(image, gt)``   — fit a background model (no-op by default).
    2. ``scores = detector.detect(image, target)``  — run detection.

    Both steps happen automatically inside ``Experiment.run()``.
    """

    def fit(
        self,
        image: torch.Tensor,
        gt: Optional[torch.Tensor] = None,
    ) -> "BaseDetector":
        """
        Fit a background model from the image.

        Args:
            image: (H, W, D) float tensor.
            gt:    (H, W) binary int tensor (1 = target, 0 = background).
                   Detectors that model only background should mask out gt==1.

        Default implementation is a no-op (e.g. SCM detectors estimate
        background statistics locally per-pixel at detect-time).
        """
        return self

    @abstractmethod
    def detect(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-pixel detection scores.

        Args:
            image:  (H, W, D) float tensor.
            target: (D,)      target spectrum.

        Returns:
            scores: (H, W) float tensor — higher value = more likely target.
        """
        ...

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({params})"
