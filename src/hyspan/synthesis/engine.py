"""
High-level synthesis API.

Two entry points:

* ``SynthesisEngine.from_image(image, n_components)``
      Fit a GMM to the image pixels (optionally excluding target pixels),
      build a DataGeneratorModel.

* ``SynthesisEngine.from_labeled_image(image, gt)``
      Build the mixture directly from per-class statistics (mean + cov),
      no GMM fitting needed — faster and exact.

Both return a ``SynthesisEngine`` whose ``.generate()`` / ``.generate_lmm()``
methods produce synthetic HSI cubes.
"""
from __future__ import annotations

from typing import Optional, Union, Sequence

import numpy as np
import torch

from ..data.generation.data_generator import DataGeneratorModel
from ..deep_models.GMM.TorchGmm import TorchGMM


class SynthesisEngine:
    """
    Generates synthetic hyperspectral images from a fitted background model.

    Do not construct directly — use one of the class methods below.

    Parameters
    ----------
    model : DataGeneratorModel
        The fitted mixture model that drives synthesis.
    """

    def __init__(self, model: DataGeneratorModel):
        self.model = model

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_image(
        cls,
        image: Union[np.ndarray, torch.Tensor],
        n_components: int = 10,
        gt: Optional[Union[np.ndarray, torch.Tensor]] = None,
        exclude_target: bool = True,
        device: str = "cpu",
        verbose: bool = False,
    ) -> "SynthesisEngine":
        """
        Fit a GMM to image pixels, then build a synthesis engine.

        Args:
            image         : (H, W, D) hyperspectral image (numpy or torch).
            n_components  : number of GMM components.
            gt            : (H, W) binary ground truth; used to exclude target
                            pixels from fitting when ``exclude_target=True``.
            exclude_target: if True and ``gt`` is provided, only background
                            pixels (gt == 0) are used for GMM fitting.
            device        : torch device for TorchGMM (e.g. 'cuda').
            verbose       : print fitting progress.
        """
        # Accept numpy or torch; keep as torch to avoid numpy C bridge
        if isinstance(image, torch.Tensor):
            img_t = image.float()
        else:
            img_t = torch.tensor(image.tolist(), dtype=torch.float32)

        H, W, D = img_t.shape
        pixels = img_t.reshape(-1, D)  # (H*W, D) torch tensor

        if gt is not None and exclude_target:
            if isinstance(gt, torch.Tensor):
                gt_t = gt.reshape(-1).float()
            else:
                gt_t = torch.tensor(gt.reshape(-1).tolist(), dtype=torch.float32)
            pixels = pixels[gt_t == 0]

        gmm = TorchGMM(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-6,
            tol=1e-3,
            max_iter=100,
            n_init=1,
            device=device,
            verbose=int(verbose),
        )
        gmm.fit(pixels.to(device))

        from ..data.generation.distribution_estimation import build_data_model_from_gmm
        model = build_data_model_from_gmm(gmm)
        return cls(model)

    @classmethod
    def from_labeled_image(
        cls,
        image: Union[np.ndarray, torch.Tensor],
        gt: Union[np.ndarray, torch.Tensor],
        target_label: int = 1,
        comp_type: str = "gaussian",
    ) -> "SynthesisEngine":
        """
        Build a synthesis engine from per-class statistics (no GMM fitting).

        Each unique label in ``gt`` (except ``target_label``) becomes one
        mixture component, parameterised by its empirical mean and covariance.

        Args:
            image        : (H, W, D) hyperspectral image.
            gt           : (H, W) integer label map.
            target_label : label value to exclude (the target class).
            comp_type    : component distribution — 'gaussian', 'student_t', or 'uniform'.
        """
        from ..data.generation.distribution_estimation import get_classes_stats, build_data_model_from_stats

        image_np = image if isinstance(image, np.ndarray) else np.array(image.tolist())
        gt_np = gt if isinstance(gt, np.ndarray) else np.array(gt.tolist(), dtype=np.int32)

        stats = get_classes_stats(image_np, gt_np)
        # Remove target class
        stats.pop(target_label, None)

        ratios = np.array([s[0] for s in stats.values()])
        ratios /= ratios.sum()
        means = np.array([s[1] for s in stats.values()])
        covs = np.array([s[2] for s in stats.values()])

        model = build_data_model_from_stats(
            means=means,
            covs=covs,
            comp_types=comp_type,
            weights=ratios.astype(float),
        )
        return cls(model)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        height: int,
        width: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a synthetic unmixed background image.

        Each pixel is drawn from exactly one mixture component.

        Returns:
            (H, W, D) float tensor.
        """
        return self.model.sample_unmixed_image(height, width, seed=seed)

    def generate_lmm(
        self,
        height: int,
        width: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a synthetic Linear Mixing Model (LMM) image.

        Each pixel is a spatially-smoothed convex combination of component samples.

        Returns:
            (H, W, D) float tensor.
        """
        return self.model.sample_lmm_image(height, width, seed=seed)
