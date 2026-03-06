import torch
from sklearn.mixture import GaussianMixture
import numpy as np

class Gmm:
    # TODO: ADD SAMPLING FUNCTION
    def __init__(self, n_components=1, covariance_type='full'):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None

        self.inv_cholesky_covariances_ = None # cholesky decomposition of the inverse of the covariance matrices
        self.gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type,reg_covar=1e-6,
                                   tol=1e-3, max_iter=100, random_state=0,n_init=1,verbose=0)

    def fit(self, X):
        self.gmm.fit(X)
        self.means_ = self.gmm.means_
        self.covariances_ = self.gmm.covariances_
        self.weights_ = self.gmm.weights_
        self.inv_cholesky_covariances_ = self.gmm.precisions_cholesky_


    def predict(self, X:np.ndarray) -> np.ndarray:
        """Predicts the component labels for each sample in X."""
        shape_orig = X.shape
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim == 3:
            H, W, D = X.shape
            X = X.reshape(-1, D)

        pred = self.gmm.predict(X)

        return pred.reshape(shape_orig[:-1])

    def score_samples(self, X):
        """Returns the log-likelihood of each sample in X under the model."""
        return self.gmm.score_samples(X)

    @property
    def means_np(self):
        return self.means_

    @property
    def covariances_np(self):
        return self.covariances_

    @property
    def weights_np(self):
        return self.weights_

    @property
    def inv_cholesky_covariances_np(self):
        return self.inv_cholesky_covariances_


    @property
    def n_components_(self):
        return self.n_components

    @property
    def means_torch(self):
        return torch.from_numpy(self.means_).float()

    @property
    def covariances_torch(self):
        return torch.from_numpy(self.covariances_).float()

    @property
    def weights_torch(self):
        return torch.from_numpy(self.weights_).float()

    @property
    def inv_cholesky_covariances_torch(self):
        return torch.from_numpy(self.inv_cholesky_covariances_).float()


    def precision_matrix_np(self) -> np.ndarray:
        """Returns the precision matrix (inverse of covariance) for each component."""
        if self.covariance_type == 'full':
            # uses the cholesky decomposition of the inverse of the covariance matrices to compute the precision matrices
            inv_covariances = np.zeros_like(self.covariances_)
            for k in range(self.n_components):
                inv_covariances[k] = self.inv_cholesky_covariances_[k].T @ self.inv_cholesky_covariances_[k]
            return inv_covariances
        elif self.covariance_type == 'diag':
            return 1.0 / self.covariances_
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

    def precision_matrix_torch(self) -> torch.Tensor:
        """Returns the precision matrix (inverse of covariance) for each component as a torch tensor."""
        precision_np = self.precision_matrix_np()
        return torch.from_numpy(precision_np).float()


    @classmethod
    def from_sklearn_gmm(cls, sklearn_gmm:GaussianMixture):
        """Creates a Gmm instance from a fitted sklearn GaussianMixture model."""
        gmm = cls(n_components=sklearn_gmm.n_components, covariance_type=sklearn_gmm.covariance_type)
        gmm.means_ = sklearn_gmm.means_
        gmm.covariances_ = sklearn_gmm.covariances_
        gmm.weights_ = sklearn_gmm.weights_
        gmm.inv_cholesky_covariances_ = sklearn_gmm.precisions_cholesky_
        return gmm
