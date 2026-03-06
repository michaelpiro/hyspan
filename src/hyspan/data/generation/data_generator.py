from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any

import numpy as np
import torch
from torch import Tensor
import json

from torch.distributions import MultivariateNormal as TorchMVN
from torch.distributions import Uniform as TorchUniform
from torch.distributions import Independent

from scipy.stats import multivariate_t

from typing import Union
# from image_utils import generate_spatial_comp_map, generate_lmm_weights
from .image_utils import generate_spatial_comp_map, generate_lmm_weights
from .utils import get_gaussian_kernel
import sys


# ============
# Base classes
# ============

class BaseComponent(ABC):
    """
    Abstract base class for a single mixture component.

    All components must implement:
      - mean: torch.Tensor[D]
      - event_dim: int
      - sample(n): torch.Tensor[n, D]
      - log_prob(x): torch.Tensor[N]
    """

    def __init__(self, device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype

    @property
    @abstractmethod
    def mean(self) -> torch.Tensor:
        """Mean vector of the component, shape (D,)."""
        ...

    @property
    @abstractmethod
    def event_dim(self) -> int:
        """Dimensionality of the component (D)."""
        ...

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """Draw n samples, shape (n, D)."""
        ...

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log probability of x under this component.

        Args:
            x: torch.Tensor of shape (N, D)

        Returns:
            log_p: torch.Tensor of shape (N,)
        """
        ...


# ===========================
# Concrete component classes
# ===========================

class GaussianComponent(BaseComponent):
    """
    Multivariate Gaussian component with mean and full covariance.
    """

    def __init__(
            self,
            mean: Union[torch.Tensor, np.ndarray, List[float]],
            cov: Union[torch.Tensor, np.ndarray],
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device=device, dtype=dtype)
        mean = torch.as_tensor(mean, dtype=dtype, device=self.device)
        cov = torch.as_tensor(cov, dtype=dtype, device=self.device)

        if cov.shape != (mean.shape[0], mean.shape[0]):
            raise ValueError(f"covariance shape {cov.shape} is not (D,D) for D={mean.shape[0]}")

        self._dist = TorchMVN(loc=mean, covariance_matrix=cov)

    @property
    def mean(self) -> torch.Tensor:
        return self._dist.mean

    @property
    def event_dim(self) -> int:
        return self._dist.event_shape[0]

    def sample(self, n: int) -> torch.Tensor:
        # (n, D)
        return self._dist.sample((n,))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, self.dtype)
        return self._dist.log_prob(x)


class StudentTComponent(BaseComponent):
    """
    Multivariate Student-t component with mean and full covariance.

    This implementation wraps scipy.stats.multivariate_t for sampling
    and log-density evaluation (CPU-based, numpy <-> torch).
    """

    def __init__(
            self,
            mean: Union[torch.Tensor, np.ndarray, List[float]],
            cov: Union[torch.Tensor, np.ndarray],
            df: float,
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device=device, dtype=dtype)

        mean_np = np.asarray(mean, dtype=np.float64)
        cov_np = np.asarray(cov, dtype=np.float64)

        if cov_np.shape != (mean_np.shape[0], mean_np.shape[0]):
            raise ValueError(f"covariance shape {cov_np.shape} is not (D,D) for D={mean_np.shape[0]}")

        self._mean_np = mean_np
        self._cov_np = cov_np
        self._df = df
        self._dist = multivariate_t(loc=self._mean_np, shape=self._cov_np, df=self._df)

    @property
    def mean(self) -> torch.Tensor:
        return torch.as_tensor(self._mean_np, dtype=self.dtype, device=self.device)

    @property
    def event_dim(self) -> int:
        return self._mean_np.shape[0]

    def sample(self, n: int) -> torch.Tensor:
        # scipy returns (D,) for n=1 and (n,D) otherwise; enforce (n,D)
        samples_np = self._dist.rvs(size=n)
        samples_np = np.atleast_2d(samples_np)
        return torch.as_tensor(samples_np, dtype=self.dtype, device=self.device)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D)
        x_np = x.detach().cpu().numpy()
        logp_np = self._dist.logpdf(x_np)  # (N,)
        return torch.as_tensor(logp_np, dtype=self.dtype, device=self.device)


class UniformComponent(BaseComponent):
    """
    Multivariate uniform distribution centered around a mean vector.

    The support is:
        [mean_i - half_range_i, mean_i + half_range_i]  for each dimension i.
    """

    def __init__(
            self,
            mean: Union[torch.Tensor, np.ndarray, List[float]],
            half_range: Union[float, np.ndarray, torch.Tensor, List[float]],
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device=device, dtype=dtype)

        mean = torch.as_tensor(mean, dtype=dtype, device=self.device)
        half_range = torch.as_tensor(half_range, dtype=dtype, device=self.device)

        # Broadcast scalar half_range if needed
        if half_range.ndim == 0:
            half_range = half_range.expand_as(mean)
        elif half_range.shape != mean.shape:
            raise ValueError(
                f"half_range must be scalar or have shape {mean.shape}, "
                f"got {half_range.shape}"
            )

        low = mean - half_range
        high = mean + half_range

        if (high <= low).any():
            raise ValueError("All high values must be strictly greater than low values")

        base = TorchUniform(low, high, validate_args=False)
        self._dist = Independent(base, 1)

    @property
    def mean(self) -> torch.Tensor:
        # = (low + high) / 2
        return self._dist.base_dist.mean

    @property
    def event_dim(self) -> int:
        return self._dist.base_dist.low.shape[0]

    def sample(self, n: int) -> torch.Tensor:
        return self._dist.sample((n,))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, self.dtype)
        return self._dist.log_prob(x)


# =====================
# Mixture model wrapper
# =====================

class DataGeneratorModel:
    """
    Generic finite mixture model over a list of components.

    components: list of BaseComponent
    weights: torch.Tensor of shape (K,), mixture weights (will be normalized)
    """
# TODO: GO OVER THE SEEDING AND REMOVE ANY UNNECESSARY SEEDING (e.g. in sample_spatial_comp_map and sample_lmm_weights) AND MAKE SURE THE SEEDING IS CONSISTENT AND WORKS AS EXPECTED
    def __init__(
            self,
            components: List[BaseComponent],
            weights: Union[torch.Tensor, np.ndarray, List[float]],
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.float32,
    ):
        if len(components) == 0:
            raise ValueError("Must provide at least one component")

        self.device = torch.device(device)
        self.dtype = dtype
        self.components = components
        self.K = len(components)

        # Check all components have same dimension
        dims = {c.event_dim for c in components}
        if len(dims) != 1:
            raise ValueError(f"All components must have same event_dim, got {dims}")
        self.dim = dims.pop()

        # Set and normalize weights
        weights = torch.as_tensor(weights, dtype=dtype, device=self.device)
        if weights.ndim != 1 or weights.shape[0] != self.K:
            raise ValueError(f"weights must be 1D with length K={self.K}, got shape {weights.shape}")

        weights = weights / weights.sum()
        self.weights = weights

        # Precompute component means
        self.means = torch.stack(
            [comp.mean.to(self.device, self.dtype) for comp in self.components],
            dim=0,
        )  # (K, D)

    def sample(
            self,
            n: int,
            comps: Optional[torch.Tensor] = None,
            return_labels: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample from the mixture model.

        Args:
            n: number of samples
            comps: optional pre-specified component indices, shape (n,)
            return_labels: if True, also return the component indices

        Returns:
            x: (n, D) mixture samples
            labels (optional): (n,) component indices
        """
        if comps is None:
            # Draw component assignments from categorical(weights)
            comps = torch.multinomial(self.weights, num_samples=n, replacement=True)
        else:
            comps = comps.to(self.device)

        x = torch.zeros((n, self.dim), dtype=self.dtype, device=self.device)

        for k in range(self.K):
            idx = (comps == k).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            samples_k = self.components[k].sample(idx.numel())
            samples_k = samples_k.to(self.device, self.dtype)
            x[idx] = samples_k

        if return_labels:
            return x, comps
        return x

    def sample_mixing(
            self,
            mix_weights: torch.Tensor,
            return_weights: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample from the *mixing model* version of this mixture.

        Each output sample is a convex combination of samples from all components:
            x_i = sum_k w_{k,i} * x_{k,i},
        where x_{k,i} ~ component k independently.

        Parameters
        ----------
        mix_weights : torch.Tensor
            Mixing weights specifying how much each component contributes
            to each sample.

            Supported shapes:
              - (K, N)
              - (K, H, W)

            where:
              K = number of components = self.K.
              N = number of samples.
              H, W = spatial dimensions (e.g. image pixels, N = H*W).

            For each sample i, mix_weights[:, i] (or mix_weights[:, h, w])
            are the raw (non-negative) mixing weights per component.
            They will be clamped to >= 0 and renormalized to sum to 1.

        return_weights : bool, default=False
            If True, also return the *normalized* mixing weights used
            internally (flattened if (K, H, W) was provided).

        Returns
        -------
        x : torch.Tensor of shape (N, D)
            Mixed samples, where N = number of positions (N or H*W).

        weights (optional) : torch.Tensor of shape (N, K)
            The normalized mixing weights used for each sample, flattened
            along spatial dims if needed.
        """
        mix_weights = mix_weights.to(self.device, self.dtype)

        if mix_weights.ndim == 2:
            # (K, N)
            K_w, N = mix_weights.shape
            if K_w != self.K:
                raise ValueError(
                    f"mix_weights first dim must be K={self.K}, got {K_w}"
                )
            flat_weights = mix_weights.transpose(0, 1)  # (N, K)
        elif mix_weights.ndim == 3:
            # (K, H, W)
            K_w, H, W = mix_weights.shape
            if K_w != self.K:
                raise ValueError(
                    f"mix_weights first dim must be K={self.K}, got {K_w}"
                )
            N = H * W
            flat_weights = mix_weights.view(self.K, N).transpose(0, 1)  # (N, K)
        else:
            raise ValueError(
                f"mix_weights must have shape (K, N) or (K, H, W), got {mix_weights.shape}"
            )

        # Ensure convex combination: clamp >= 0 and normalize along K
        flat_weights = torch.clamp(flat_weights, min=0.0)  # (N, K)
        weight_sums = flat_weights.sum(dim=1, keepdim=True) + 1e-12
        flat_weights = flat_weights / weight_sums  # (N, K)

        # For each component k, draw N samples: s_k ~ p_k, shape (N, D)
        samples_per_comp = []
        for k in range(self.K):
            s_k = self.components[k].sample(N)  # (N, D)
            s_k = s_k.to(self.device, self.dtype)
            samples_per_comp.append(s_k)

        # Stack to (K, N, D) then reshape to (N, K, D)
        samples = torch.stack(samples_per_comp, dim=0)  # (K, N, D)
        samples = samples.permute(1, 0, 2)  # (N, K, D)

        # Apply weights: (N, K, 1) * (N, K, D) -> sum over K -> (N, D)
        w = flat_weights.unsqueeze(-1)  # (N, K, 1)
        x = (w * samples).sum(dim=1)  # (N, D)

        # reshape back to (H, W, D) if needed
        H, W = mix_weights.shape[1], mix_weights.shape[2]
        x = x.view(H, W, self.dim)
        if return_weights:
            return x, flat_weights
        return x

    def compose_mixing_image(self, mix_weights: torch.Tensor, seed: int = None) -> torch.Tensor:
        """Compose a mixing image from the provided mixing weights."""
        assert mix_weights.ndim == 3 and mix_weights.shape[0] == self.K, \
            f"mix_weights must have shape (K, H, W), got {mix_weights.shape}"

        height, width = mix_weights.size(1), mix_weights.size(2)
        mix_image = torch.zeros(height, width, self.dim, dtype=torch.float32)
        # set the seed for reproducibility if provided
        self.set_seeds(seed)

        for k in range(self.K):
            # sample a comp map full of curr component in order to sample from the component distribution only
            comp_k_map = torch.full((height * width,), k)
            comp_k_samples = self.sample(height * width, comps=comp_k_map).view(height, width, self.dim)
            mix_image += mix_weights[k, :, :].unsqueeze(-1) * comp_k_samples
        return mix_image

    @staticmethod
    def set_seeds(seed: Optional[int]):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Optional: for full determinism on CUDA
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def sample_lmm_image(self, height: int = 128, width: int = 128, comp_map: torch.Tensor = None,
                         mix_weights: torch.Tensor = None, seed: int = None, return_weights=False,
                         return_comp_map=False) -> torch.Tensor:
        """Sample a mixing image (LMM) using the provided component map and mixing weights."""
        if comp_map is not None:
            if comp_map.shape != (height, width):
                print(f"Warning: comp_map shape {comp_map.shape} does not match (height, width) = ({height}, {width})"
                      f"generating image with comp_map shape", file=sys.stderr)
                height, width = comp_map.shape
        else:
            # self.set_seeds(seed)
            comp_map = self.sample_spatial_comp_map(height, width, ratios=self.weights, device=self.device, seed=seed)

        if mix_weights is not None:
            assert mix_weights.shape == (self.K, height, width), \
                f"mix_weights shape {mix_weights.shape} does not match (K, height, width) = ({self.K}, {height}, {width})"
        else:
            mix_weights = self.sample_lmm_weights(self.K, comp_map)
        mixed_image = self.compose_mixing_image(mix_weights, seed=seed)
        return_tuple = (mixed_image,)
        if return_weights:
            return_tuple += (mix_weights,)
        if return_comp_map:
            return_tuple += (comp_map,)

        return return_tuple if len(return_tuple) > 1 else return_tuple[0]

    def sample_unmixed_image(self, height: int = 128, width: int = 128, comp_map: torch.Tensor = None, seed: int = None,
                             return_comp_map=False) -> tuple[Any, Tensor] | Any:
        """Sample an unmixed image using the provided component map."""
        if comp_map is not None:
            if comp_map.shape != (height, width):
                print(f"Warning: comp_map shape {comp_map.shape} does not match (height, width) = ({height}, {width})"
                      f"generating image with comp_map shape", file=sys.stderr)
                height, width = comp_map.shape
        else:
            # self.set_seeds(seed)
            comp_map = self.sample_spatial_comp_map(height, width, ratios=self.weights, device=self.device,seed=seed)
        self.set_seeds(seed)
        image = self.sample(height * width, comps=comp_map.view(-1)).view(height, width, self.dim)

        if return_comp_map:
            return image, comp_map
        return image


    def log_prob_per_component(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log probability of each point under each component.

        Args:
            x: (N, D)

        Returns:
            log_p: (N, K)
        """
        x = x.to(self.device, self.dtype)
        N = x.shape[0]
        log_p = torch.empty((N, self.K), dtype=self.dtype, device=self.device)
        for k, comp in enumerate(self.components):
            log_p[:, k] = comp.log_prob(x)
        return log_p

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log probability of each point under the mixture.

        Args:
            x: (N, D)

        Returns:
            log_p: (N,)
        """
        log_p_per_comp = self.log_prob_per_component(x)  # (N, K)
        log_weights = torch.log(self.weights + 1e-12)  # (K,)
        return torch.logsumexp(log_p_per_comp + log_weights.unsqueeze(0), dim=1)

    @classmethod
    def sample_spatial_comp_map(cls, H, W, ratios, smoothness: float = 5.0, segment_size: int = 25, seed: int = None,
                                device: str = "cpu",
                                peak=10000.0) -> torch.Tensor:
        """Generate a spatial map of component assignments with smooth regions biased by the provided ratios."""
        if seed is not None:
            cls.set_seeds(seed)
        return generate_spatial_comp_map(H, W, ratios=ratios, smoothness=smoothness, segment_size=segment_size,
                                         device=device, peak=peak)

    @classmethod
    def sample_lmm_weights(cls, K, comp_map, kernel_tensor=None) -> torch.Tensor:
        """Generate mixing weights for the LMM based on a component map and a smoothing kernel."""
        if kernel_tensor is None:
            kernel_tensor = get_gaussian_kernel().to(comp_map.device)
        return generate_lmm_weights(K, comp_map, kernel_tensor)

    def save(self, filepath: str):
        """
        Instance method to save the model's configuration and parameters to disk.
        """
        components_state = []
        for comp in self.components:
            # Extract raw parameters based on component type
            if isinstance(comp, GaussianComponent):
                comp_state = {
                    'type': 'Gaussian',
                    'mean': comp.mean.cpu(),
                    'cov': comp._dist.covariance_matrix.cpu()
                }
            elif isinstance(comp, StudentTComponent):
                comp_state = {
                    'type': 'StudentT',
                    'mean': comp._mean_np,
                    'cov': comp._cov_np,
                    'df': comp._df
                }
            elif isinstance(comp, UniformComponent):
                # Reconstruct half_range from the Uniform base distribution bounds
                low = comp._dist.base_dist.low
                high = comp._dist.base_dist.high
                half_range = (high - low) / 2.0
                comp_state = {
                    'type': 'Uniform',
                    'mean': comp.mean.cpu(),
                    'half_range': half_range.cpu()
                }
            else:
                raise ValueError(f"Unknown component type: {type(comp)}")

            components_state.append(comp_state)

        # Build the overarching state dictionary
        state = {
            'weights': self.weights.cpu(),
            'components': components_state,
            'dtype': self.dtype,
            # We don't save device, as the user should dictate device on load
        }

        torch.save(state, filepath)
        # adds a json file with the same name containing the model configuration (e.g. component types and parameters)
        # for easier inspection without loading the full model
        json_path = filepath.rsplit('.', 1)[0] + '.json'

        with open(json_path, 'w') as f:
            json.dump({
                'weights': self.weights.cpu().tolist(),
                'components': components_state,
            }, f, indent=4)



    @classmethod
    def load(cls, filepath: str, device: Union[str, torch.device] = "cpu"):
        """
        Class method to instantiate a new DataGeneratorModel from a saved file.
        """
        # Load the dictionary (map_location ensures safe loading across CPU/GPU)
        state = torch.load(filepath, map_location="cpu")

        dtype = state['dtype']
        weights = state['weights']

        components = []
        for c_state in state['components']:
            c_type = c_state['type']

            # Reconstruct the specific component
            if c_type == 'Gaussian':
                comp = GaussianComponent(
                    mean=c_state['mean'], cov=c_state['cov'],
                    device=device, dtype=dtype
                )
            elif c_type == 'StudentT':
                comp = StudentTComponent(
                    mean=c_state['mean'], cov=c_state['cov'], df=c_state['df'],
                    device=device, dtype=dtype
                )
            elif c_type == 'Uniform':
                comp = UniformComponent(
                    mean=c_state['mean'], half_range=c_state['half_range'],
                    device=device, dtype=dtype
                )
            else:
                raise ValueError(f"Corrupted file: Unknown component type {c_type}")

            components.append(comp)

        # Use 'cls' to create and return the new instance
        return cls(components=components, weights=weights, device=device, dtype=dtype)




def _test_seed_reproducibility():
    seed=0
    """test for checking that seeding does indeed produces the same image across runs"""
    model = generate_dummy_gen()

    img1 = model.sample_lmm_image(height=64, width=64, seed=seed)
    img2 = model.sample_lmm_image(height=64, width=64, seed=seed)
    assert torch.allclose(img1, img2), "Images generated with the same seed should be identical"
    print("Test passed: Images are identical across runs with the same seed.")


def generate_dummy_gen():
    device = "cpu"
    dtype = torch.float32
    K = 3
    D = 5
    components = [
        GaussianComponent(mean=torch.zeros(D), cov=torch.eye(D), device=device, dtype=dtype),
        GaussianComponent(mean=torch.ones(D) * 5, cov=torch.eye(D) * 2, device=device, dtype=dtype),
        GaussianComponent(mean=torch.ones(D) * -5, cov=torch.eye(D) * 0.5, device=device, dtype=dtype),
    ]
    weights = [0.5, 0.3, 0.2]
    model = DataGeneratorModel(components=components, weights=weights, device=device, dtype=dtype)
    return model


if __name__ == "__main__":
    # example for how to generate a mixing image with the DataGeneratorModel
    model = generate_dummy_gen()
    img1,weights,comp_map = model.sample_lmm_image(height=64, width=64, seed=0, return_weights=True, return_comp_map=True)

    weights2 = model.sample_lmm_weights(model.K, comp_map)
    img2 = model.compose_mixing_image(weights2, seed=0)
    # test for reproducibility
    assert torch.allclose(weights, weights2), "Mixing weights generated with the same seed should be identical"
    print("Test passed: Mixing weights are identical across runs with the same seed.")
    assert torch.allclose(img1, img2), "Images generated with the same seed should be identical"
    print("Test passed: Images are identical across runs with the same seed.")

    comp_map3 = model.sample_spatial_comp_map(64, 64, ratios=model.weights, device=model.device, seed=0)
    weights3 = model.sample_lmm_weights(model.K, comp_map3)
    img3 = model.sample_lmm_image(height=64, width=64,mix_weights=weights3,comp_map=comp_map3, seed=0)
    assert torch.allclose(img1, img3), "Images generated with the same seed should be identical"
    print("Test passed: Images are identical across runs with the same seed.")




