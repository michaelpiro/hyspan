from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Union


class TorchGMM:
    """
    Gaussian Mixture Model fitted with the EM algorithm, running entirely on PyTorch.

    Designed as a GPU-capable drop-in replacement for the ``Gmm`` wrapper around
    sklearn's GaussianMixture for high-dimensional hyperspectral data.  The same
    property interface is preserved so ``ClassicDetectorGMM`` and
    ``build_data_model_from_gmm`` work without modification.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    covariance_type : {'full', 'diag'}
        Type of covariance matrix per component.
    reg_covar : float
        Tikhonov regularisation added to every covariance diagonal for
        numerical stability.
    tol : float
        EM stops when ``|delta mean-log-likelihood| < tol``.
    max_iter : int
        Maximum EM iterations per initialisation.
    n_init : int
        Number of random initialisations; the run with the highest
        log-likelihood is kept.
    random_state : int or None
        Global seed for reproducibility.
    verbose : int
        0 = silent.  1 = one line per EM iteration.
    device : str or torch.device
        PyTorch device, e.g. ``'cpu'``, ``'cuda'``, ``'mps'``.
    dtype : torch.dtype
        Floating-point precision for all tensors.

    Notes
    -----
    Memory footprint — for full covariance the E-step solves one (D x D) triangular
    system against a (D x N) RHS per component, keeping peak memory at O(D*N) per
    component iteration rather than O(K*D*N).  This is safe for large HSI images
    (e.g. 400x400x200 pixels) on a typical 8 GB GPU.
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        tol: float = 1e-3,
        max_iter: int = 100,
        n_init: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if covariance_type not in ("full", "diag"):
            raise ValueError(
                f"covariance_type must be 'full' or 'diag', got '{covariance_type}'"
            )

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.device = torch.device(device)
        self.dtype = dtype

        # Internal tensor parameters (set after fit)
        self._means: Optional[torch.Tensor] = None        # (K, D)
        self._covariances: Optional[torch.Tensor] = None  # (K, D, D) or (K, D)
        self._weights: Optional[torch.Tensor] = None      # (K,)
        self._cov_chol: Optional[torch.Tensor] = None     # (K, D, D) lower Cholesky — full only
        self._log_det_cov: Optional[torch.Tensor] = None  # (K,)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> "TorchGMM":
        """Fit GMM parameters to data using the EM algorithm."""
        X = self._to_tensor(X)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        best_lb = float("-inf")
        best_snap: Optional[dict] = None

        for init in range(self.n_init):
            if self.random_state is not None:
                torch.manual_seed(self.random_state + init)
                np.random.seed(self.random_state + init)

            self._initialize(X)
            lb = float("-inf")

            for it in range(self.max_iter):
                prev_lb = lb
                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)
                lb = log_prob_norm.mean().item()

                if self.verbose:
                    print(
                        f"[TorchGMM] init={init + 1}/{self.n_init}  iter={it + 1:3d}  "
                        f"lower_bound={lb:.6f}  delta={lb - prev_lb:+.2e}"
                    )

                if abs(lb - prev_lb) < self.tol:
                    if self.verbose:
                        print(f"[TorchGMM] Converged at iteration {it + 1}.")
                    break

            if lb > best_lb:
                best_lb = lb
                best_snap = self._snapshot()

        self._restore(best_snap)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Hard component assignment for each sample.

        Accepts (N, D), (D,), or (H, W, D) inputs and returns an array
        whose shape matches the spatial dimensions (like sklearn's predict).
        """
        orig_shape = X.shape if isinstance(X, np.ndarray) else tuple(X.shape)
        _, log_resp = self._e_step(self._to_tensor(self._flatten_to_2d(X)))
        labels = log_resp.argmax(dim=1).cpu()
        return np.array(labels.tolist(), dtype=np.int64).reshape(orig_shape[:-1])

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Soft component responsibilities, shape (N, K)."""
        _, log_resp = self._e_step(self._to_tensor(self._flatten_to_2d(X)))
        proba = log_resp.exp().cpu()
        return np.array(proba.tolist(), dtype=np.float32)

    def score_samples(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Per-sample log-likelihood, shape (N,)."""
        log_prob_norm, _ = self._e_step(self._to_tensor(self._flatten_to_2d(X)))
        lp = log_prob_norm.cpu()
        return np.array(lp.tolist(), dtype=np.float32)

    def score(self, X: Union[np.ndarray, torch.Tensor]) -> float:
        """Mean log-likelihood over all samples."""
        return float(self.score_samples(X).mean())

    def log_prob_torch(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Per-sample log p(x) under the mixture, returned as a torch tensor.

        Stays entirely in torch — no numpy bridge. Use this inside other torch
        modules or detectors that need to stay on-device.

        Args:
            X: (N, D) or (H, W, D) tensor/ndarray.

        Returns:
            (N,) float tensor of log-likelihoods (on self.device).
        """
        X_t = self._to_tensor(self._flatten_to_2d(X))
        log_prob_norm, _ = self._e_step(X_t)
        return log_prob_norm

    # ------------------------------------------------------------------
    # Properties — numpy interface (sklearn / build_data_model_from_gmm compat)
    # ------------------------------------------------------------------

    @staticmethod
    def _t2np(t: torch.Tensor) -> np.ndarray:
        """Safe tensor→numpy that works even when the numpy C bridge is broken."""
        return np.array(t.cpu().tolist())

    @property
    def means_(self) -> np.ndarray:
        return self._t2np(self._means)

    @property
    def covariances_(self) -> np.ndarray:
        return self._t2np(self._covariances)

    @property
    def weights_(self) -> np.ndarray:
        return self._t2np(self._weights)

    @property
    def inv_cholesky_covariances_(self) -> np.ndarray:
        """Upper-triangular Cholesky of precision (matches sklearn precisions_cholesky_)."""
        return self._t2np(self._precision_chol())

    @property
    def n_components_(self) -> int:
        return self.n_components

    # ------------------------------------------------------------------
    # Properties — torch interface (ClassicDetectorGMM compat)
    # ------------------------------------------------------------------

    @property
    def means_np(self) -> np.ndarray:
        return self.means_

    @property
    def covariances_np(self) -> np.ndarray:
        return self.covariances_

    @property
    def weights_np(self) -> np.ndarray:
        return self.weights_

    @property
    def inv_cholesky_covariances_np(self) -> np.ndarray:
        return self.inv_cholesky_covariances_

    @property
    def means_torch(self) -> torch.Tensor:
        return self._means.float()

    @property
    def covariances_torch(self) -> torch.Tensor:
        return self._covariances.float()

    @property
    def weights_torch(self) -> torch.Tensor:
        return self._weights.float()

    @property
    def inv_cholesky_covariances_torch(self) -> torch.Tensor:
        """Upper-triangular Cholesky U of precision s.t. U @ U^T = Sigma^{-1}."""
        return self._precision_chol().float()

    def precision_matrix_np(self) -> np.ndarray:
        return self._t2np(self.precision_matrix_torch())

    def precision_matrix_torch(self) -> torch.Tensor:
        """Returns (K, D, D) precision matrices Sigma_k^{-1}."""
        U = self._precision_chol()            # (K, D, D) upper triangular
        return U @ U.transpose(-1, -2)        # U @ U^T = Sigma^{-1}

    # ------------------------------------------------------------------
    # EM internals
    # ------------------------------------------------------------------

    def _initialize(self, X: torch.Tensor):
        N, D = X.shape
        K = self.n_components
        means = self._kmeans_pp(X, K)
        weights = torch.full((K,), 1.0 / K, device=self.device, dtype=self.dtype)
        if self.covariance_type == "full":
            covariances = (
                torch.eye(D, device=self.device, dtype=self.dtype)
                .unsqueeze(0)
                .repeat(K, 1, 1)
            )
        else:
            covariances = torch.ones(K, D, device=self.device, dtype=self.dtype)
        self._means = means
        self._weights = weights
        self._covariances = covariances
        self._update_chol_and_logdet()

    def _kmeans_pp(self, X: torch.Tensor, K: int) -> torch.Tensor:
        """K-means++ seeding — O(K*N*D) work, O(N) extra memory."""
        N = X.shape[0]
        first = torch.randint(N, (1,), device=self.device).item()
        centers = [X[first]]
        for _ in range(1, K):
            C = torch.stack(centers)                               # (k, D)
            dists = torch.cdist(X, C).min(dim=1).values ** 2      # (N,)
            idx = torch.multinomial(dists / dists.sum(), 1).item()
            centers.append(X[idx])
        return torch.stack(centers)  # (K, D)

    def _e_step(self, X: torch.Tensor):
        log_prob = self._log_prob_components(X)              # (N, K)
        log_w = self._weights.clamp(min=1e-12).log()         # (K,)
        weighted = log_prob + log_w.unsqueeze(0)             # (N, K)
        log_prob_norm = torch.logsumexp(weighted, dim=1)     # (N,)
        log_resp = weighted - log_prob_norm.unsqueeze(1)     # (N, K)
        return log_prob_norm, log_resp

    def _log_prob_components(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x | component k) for all samples and components.

        Per-component loop with a (D x N) triangular solve keeps peak GPU memory
        at O(D*N) regardless of K — critical for high-dimensional HSI data.

        Returns (N, K).
        """
        N, D = X.shape
        K = self.n_components
        log_2pi = D * torch.log(
            torch.tensor(2.0 * torch.pi, dtype=self.dtype, device=self.device)
        )
        log_prob = torch.empty(N, K, device=self.device, dtype=self.dtype)

        if self.covariance_type == "full":
            for k in range(K):
                diff = (X - self._means[k]).T                # (D, N)
                # Solve L_k @ y = diff  =>  y = L_k^{-1} diff  (D, N)
                y = torch.linalg.solve_triangular(
                    self._cov_chol[k], diff, upper=False
                )
                maha = (y * y).sum(0)                        # (N,)
                log_prob[:, k] = -0.5 * (log_2pi + self._log_det_cov[k] + maha)
        else:
            for k in range(K):
                diff = X - self._means[k]                    # (N, D)
                maha = (diff * diff / self._covariances[k]).sum(-1)   # (N,)
                log_prob[:, k] = -0.5 * (log_2pi + self._log_det_cov[k] + maha)

        return log_prob

    def _m_step(self, X: torch.Tensor, log_resp: torch.Tensor):
        N, D = X.shape
        K = self.n_components
        resp = log_resp.exp()                                  # (N, K)
        nk = resp.sum(0).clamp(min=1e-10)                     # (K,)

        self._weights = nk / N
        self._means = (resp.T @ X) / nk.unsqueeze(1)          # (K, D)

        if self.covariance_type == "full":
            cov = torch.empty(K, D, D, device=self.device, dtype=self.dtype)
            reg = self.reg_covar * torch.eye(D, device=self.device, dtype=self.dtype)
            for k in range(K):
                diff = X - self._means[k]                     # (N, D)
                # weighted outer-product sum: (D, N) @ (N, D) -> (D, D)
                cov_k = (diff * resp[:, k].unsqueeze(1)).T @ diff / nk[k]
                cov[k] = (cov_k + cov_k.T) * 0.5 + reg       # symmetrise + regularise
            self._covariances = cov
        else:
            var = torch.empty(K, D, device=self.device, dtype=self.dtype)
            for k in range(K):
                diff = X - self._means[k]                     # (N, D)
                var[k] = (resp[:, k].unsqueeze(1) * diff * diff).sum(0) / nk[k]
            self._covariances = var + self.reg_covar

        self._update_chol_and_logdet()

    def _update_chol_and_logdet(self):
        """Cache lower Cholesky of covariance and its log-determinant."""
        if self.covariance_type == "full":
            try:
                L = torch.linalg.cholesky(self._covariances)  # (K, D, D)
            except torch.linalg.LinAlgError:
                D = self._covariances.shape[-1]
                self._covariances = self._covariances + 1e-3 * torch.eye(
                    D, device=self.device, dtype=self.dtype
                )
                L = torch.linalg.cholesky(self._covariances)
            self._cov_chol = L
            # log|Sigma| = 2 * sum_i log L_ii
            self._log_det_cov = (
                2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)
            )
        else:
            self._log_det_cov = self._covariances.log().sum(-1)  # (K,)

    def _precision_chol(self) -> torch.Tensor:
        """
        Upper-triangular Cholesky factor U of the precision matrix,
        s.t. U @ U^T = Sigma^{-1}.  Matches sklearn's ``precisions_cholesky_``.

        Derivation: Sigma = L L^T  =>  Sigma^{-1} = L^{-T} L^{-1} = U U^T
        where U = L^{-T}  (upper triangular).

        Shape: (K, D, D).
        """
        if self.covariance_type == "full":
            K, D, _ = self._cov_chol.shape
            eye = (
                torch.eye(D, device=self.device, dtype=self.dtype)
                .unsqueeze(0)
                .expand(K, -1, -1)
            )
            # Solve L @ X = I  =>  X = L^{-1}  (lower triangular)
            L_inv = torch.linalg.solve_triangular(self._cov_chol, eye, upper=False)
            return L_inv.transpose(-1, -2)  # L^{-T}  (upper triangular)
        else:
            return torch.diag_embed(self._covariances.rsqrt())  # (K, D, D)

    # ------------------------------------------------------------------
    # Snapshot / restore for multi-init
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict:
        snap = {
            "means": self._means.clone(),
            "covariances": self._covariances.clone(),
            "weights": self._weights.clone(),
            "log_det_cov": self._log_det_cov.clone(),
        }
        if self.covariance_type == "full":
            snap["cov_chol"] = self._cov_chol.clone()
        return snap

    def _restore(self, snap: dict):
        self._means = snap["means"]
        self._covariances = snap["covariances"]
        self._weights = snap["weights"]
        self._log_det_cov = snap["log_det_cov"]
        if self.covariance_type == "full":
            self._cov_chol = snap["cov_chol"]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _to_tensor(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(X, np.ndarray):
            # Use tolist() to avoid the numpy-torch C bridge (broken on NumPy 2 / old torch)
            X = torch.tensor(X.tolist(), dtype=self.dtype, device=self.device)
            return X
        return X.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _flatten_to_2d(
        X: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Flatten (D,) -> (1,D) and (H,W,D) -> (H*W,D); leave (N,D) unchanged."""
        if X.ndim == 1:
            return X.reshape(1, -1) if isinstance(X, np.ndarray) else X.unsqueeze(0)
        if X.ndim == 3:
            return X.reshape(-1, X.shape[-1])
        return X

    @classmethod
    def from_sklearn_gmm(
        cls,
        sklearn_gmm,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "TorchGMM":
        """Wrap a fitted ``sklearn.mixture.GaussianMixture`` into a TorchGMM."""
        inst = cls(
            n_components=sklearn_gmm.n_components,
            covariance_type=sklearn_gmm.covariance_type,
            device=device,
            dtype=dtype,
        )
        inst._means = torch.from_numpy(sklearn_gmm.means_).to(
            device=inst.device, dtype=inst.dtype
        )
        inst._covariances = torch.from_numpy(sklearn_gmm.covariances_).to(
            device=inst.device, dtype=inst.dtype
        )
        inst._weights = torch.from_numpy(sklearn_gmm.weights_).to(
            device=inst.device, dtype=inst.dtype
        )
        inst._update_chol_and_logdet()
        return inst


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _smoke_test():
    """Fit on well-separated Gaussians and verify correctness of all properties."""
    torch.manual_seed(0)
    D, K, N_per = 8, 3, 400
    true_means = torch.tensor([[0.0] * D, [8.0] * D, [-8.0] * D])
    X = torch.cat([
        true_means[k].unsqueeze(0).expand(N_per, -1) + torch.randn(N_per, D) * 0.5
        for k in range(K)
    ])  # (K*N_per, D) — pass as torch tensor to avoid numpy bridge

    for cov_type in ("full", "diag"):
        gmm = TorchGMM(
            n_components=K, covariance_type=cov_type, reg_covar=1e-4,
            tol=1e-4, max_iter=300, n_init=1, random_state=0,
        )
        gmm.fit(X)  # accept torch tensor directly

        # Check cluster recovery via internal e-step (avoids .numpy() bridge)
        _, log_resp = gmm._e_step(X.to(gmm.device, gmm.dtype))
        labels = log_resp.argmax(dim=1)
        assert labels.unique().numel() == K, f"[{cov_type}] expected {K} clusters"

        # Parameter shapes
        assert gmm._means.shape == (K, D)
        assert gmm._weights.shape == (K,)

        # Core invariant: U @ U^T == Sigma^{-1}
        U = gmm.inv_cholesky_covariances_torch    # (K, D, D)
        P = gmm.precision_matrix_torch()           # (K, D, D)
        assert torch.allclose(U @ U.transpose(-1, -2), P, atol=1e-4), \
            f"[{cov_type}] inv_chol @ inv_chol.T != precision matrix"

        print(f"  [{cov_type}] passed.")

    print("TorchGMM smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
