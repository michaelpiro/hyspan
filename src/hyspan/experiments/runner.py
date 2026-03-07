"""Experiment runner — orchestrates detection, evaluation, and result collection."""
from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch

from ..detectors.base import BaseDetector
from ..results.metrics import roc_auc
from ..results.store import DetectorResult, ExperimentResult
from ..algorithms.utils import ts_generation


class Experiment:
    """
    Runs one or more detectors on a single (image, ground-truth) pair and
    collects evaluation results.

    Parameters
    ----------
    image : (H, W, D) tensor or ndarray
        Hyperspectral image.
    gt : (H, W) tensor or ndarray
        Binary ground truth: 1 = target pixel, 0 = background.
    name : str
        Identifier used when saving to a ``ResultsStore``.

    Example
    -------
    ::
        exp = Experiment(image, gt, name="sandiego")
        exp.add(ACE_SCM()).add(ACE_GMM(n_components=10))
        result = exp.run()
        print(result.summary())
    """

    def __init__(
        self,
        image: Union[torch.Tensor, np.ndarray],
        gt: Union[torch.Tensor, np.ndarray],
        name: str = "experiment",
    ):
        self.image = self._to_tensor(image, torch.float32)
        self.gt = self._to_tensor(gt, torch.float32)
        self.name = name
        self._detectors: Dict[str, BaseDetector] = {}

    def add(
        self,
        detector: BaseDetector,
        name: Optional[str] = None,
    ) -> "Experiment":
        """
        Register a detector.

        Args:
            detector: any ``BaseDetector`` subclass.
            name:     display name; defaults to the class name.
                      Append a suffix to avoid collisions, e.g. ``name="ACE_GMM_k10"``.

        Returns self for chaining: ``exp.add(A).add(B).add(C)``.
        """
        key = name or detector.__class__.__name__
        if key in self._detectors:
            # auto-suffix to avoid silent overwrite
            idx = sum(1 for k in self._detectors if k.startswith(key))
            key = f"{key}_{idx}"
        self._detectors[key] = detector
        return self

    def run(
        self,
        target_signature: Optional[Union[torch.Tensor, np.ndarray]] = None,
        ts_method: int = 0,
        verbose: bool = True,
    ) -> ExperimentResult:
        """
        Fit each detector and compute detection scores + AUC.

        Args:
            target_signature : (D,) tensor/ndarray. If None, estimated from gt
                               using ``ts_generation(image, gt, ts_method)``.
            ts_method        : int 0-8 — method index for ``ts_generation``.
                               Ignored when ``target_signature`` is provided.
            verbose          : print progress per detector.

        Returns:
            ExperimentResult with scores and AUC for every registered detector.
        """
        target = self._resolve_target(target_signature, ts_method)
        image = self.image
        gt = self.gt

        detector_results: Dict[str, DetectorResult] = {}

        for det_name, detector in self._detectors.items():
            if verbose:
                print(f"  [{det_name}] fitting...", end=" ", flush=True)

            detector.fit(image, gt)

            if verbose:
                print("detecting...", end=" ", flush=True)

            scores = detector.detect(image, target)  # (H, W)

            auc = roc_auc(scores, gt)

            if verbose:
                print(f"AUC = {auc:.4f}")

            detector_results[det_name] = DetectorResult(
                name=det_name, scores=scores.cpu(), auc=auc
            )

        return ExperimentResult(
            name=self.name,
            gt=gt.cpu(),
            target=target.cpu(),
            detector_results=detector_results,
            metadata={"ts_method": ts_method, "n_detectors": len(self._detectors)},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_target(
        self,
        target_signature: Optional[Union[torch.Tensor, np.ndarray]],
        ts_method: int,
    ) -> torch.Tensor:
        if target_signature is not None:
            t = self._to_tensor(target_signature, torch.float32)
            return t.squeeze()

        # ts_generation expects numpy arrays
        image_np = self.image.numpy()
        gt_np = self.gt.numpy().astype(int)
        ts_np = ts_generation(image_np, gt_np, ts_method)  # returns (D, 1)
        return torch.from_numpy(ts_np).float().squeeze()

    @staticmethod
    def _to_tensor(x: Union[torch.Tensor, np.ndarray], dtype) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            # Avoid numpy C bridge (broken on NumPy 2 + old torch)
            return torch.tensor(x.tolist(), dtype=dtype)
        return x.to(dtype)
