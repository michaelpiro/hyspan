from typing import Union, Sequence, Dict, Any, Optional
import numpy as np
import torch

from .data_generator import BaseComponent, GaussianComponent, StudentTComponent, \
    UniformComponent, DataGeneratorModel
GAUSSIAN_COMP = "gaussian"
COV_TYPE = "full"
DF_DEFAULT = 25


def create_component(
        dist_name: str,
        *,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        **params: Any,
) -> BaseComponent:
    """
    Factory for a single mixture component.

    Parameters
    ----------
    dist_name : str
        Name of the distribution type. Supported aliases:
          - Gaussian: "gaussian", "normal", "mvnormal", "multivariate_normal"
          - Student-t: "student_t", "student-t", "t", "multivariate_student_t", "mvt"
          - Uniform: "uniform", "box", "hyperrectangle"
    device : torch.device or str, optional
        Device for torch tensors.
    dtype : torch.dtype, optional
        Torch dtype.
    **params : dict
        Distribution-specific parameters:

        Gaussian:
            mean: (D,)
            cov: (D, D)

        Student-t:
            mean: (D,)
            cov: (D, D)
            df: float

        Uniform:
            low: (D,)
            high: (D,)

    Returns
    -------
    BaseComponent
        An instance of the appropriate concrete component class.

    Raises
    ------
    ValueError
        If the distribution name is unknown or required params are missing.
    """
    name = dist_name.lower()

    def _check_required(required_keys):
        missing = [k for k in required_keys if k not in params]
        if missing:
            raise ValueError(
                f"Missing required parameters for '{dist_name}': {missing}. "
                f"Got keys: {list(params.keys())}"
            )

    if name in ("gaussian", "normal", "mvnormal", "multivariate_normal"):
        _check_required(["mean", "cov"])
        return GaussianComponent(
            mean=params["mean"],
            cov=params["cov"],
            device=device,
            dtype=dtype,
        )

    elif name in ("student_t", "student-t", "t", "multivariate_student_t", "mvt"):
        _check_required(["mean", "cov", "df"])
        return StudentTComponent(
            mean=params["mean"],
            cov=params["cov"],
            df=params["df"],
            device=device,
            dtype=dtype,
        )

    elif name in ("uniform", "box", "hyperrectangle"):
        _check_required(["low", "high"])
        return UniformComponent(
            mean=params["mean"],
            half_range=params["half_range"],
            device=device,
            dtype=dtype,
        )

    else:
        raise ValueError(
            f"Unknown distribution name '{dist_name}'. "
            "Supported types: gaussian, student_t, uniform."
        )


def create_data_generator_model(
        specs: Sequence[Dict[str, Any]],
        weights: Union[torch.Tensor, Sequence[float]],
        *,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
) -> DataGeneratorModel:
    """
    Factory for DataGeneratorModel from a list of component specs.

    Parameters
    ----------
    specs : sequence of dict
        Each dict must have a key "type" and the corresponding
        component parameters, for example:
            {"type": "gaussian", "mean": ..., "cov": ...}
            {"type": "student_t", "mean": ..., "cov": ..., "df": ...}
            {"type": "uniform", "mean": ..., "half_range": ...}
    weights : array-like of shape (K,)
        Mixture weights (will be normalized).
    device : torch.device or str, optional
        Device for all tensors.
    dtype : torch.dtype, optional
        Dtype for tensors.

    Returns
    -------
    DataGeneratorModel
    """
    components: list[BaseComponent] = []

    for spec in specs:
        if "type" not in spec:
            raise ValueError(f"Each spec must contain a 'type' key. Got: {spec}")
        spec = dict(spec)  # shallow copy
        dist_type = spec.pop("type")
        comp = create_component(
            dist_type,
            device=device,
            dtype=dtype,
            **spec,
        )
        components.append(comp)

    return DataGeneratorModel(
        components=components,
        weights=weights,
        device=device,
        dtype=dtype,
    )


def build_data_model_from_stats(
        means: Union[np.ndarray, torch.Tensor],
        covs: Union[np.ndarray, torch.Tensor],
        comp_types: Optional[Union[str, Sequence[Optional[str]]]] = None,
        *,
        df_default: float = 10.0,
        weights: Optional[Union[Sequence[float], torch.Tensor]] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
) -> DataGeneratorModel:
    """
    Build a DataGeneratorModel from per-component statistics.

    Default behavior:
      - Unspecified or 'gaussian' -> GaussianComponent(mean, cov)
      - 'student_t'                -> StudentTComponent(mean, cov, df_default)
      - 'uniform'                  -> UniformComponent(mean, half_range)
        with half_range = sqrt(diag(cov))  (i.e., mean ± std).
    """
    device = torch.device(device)

    # ---- convert means and covs to torch ----
    # Use torch.tensor() (not as_tensor/from_numpy) to avoid the numpy C bridge
    def _to_t(x):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)
        return torch.tensor(np.asarray(x).tolist(), dtype=dtype, device=device)

    means_t = _to_t(means)
    covs_t = _to_t(covs)

    if means_t.ndim != 2:
        raise ValueError(f"means must have shape (K, D), got {means_t.shape}")
    if covs_t.ndim != 3:
        raise ValueError(f"covs must have shape (K, D, D), got {covs_t.shape}")

    K, D = means_t.shape
    if covs_t.shape != (K, D, D):
        raise ValueError(f"covs must have shape (K, D, D), got {covs_t.shape}")

    # ---- setup weights ----
    if weights is None:
        weights_t = torch.full((K,), 1.0 / K, dtype=dtype, device=device)
    else:
        weights_t = _to_t(weights)
        if weights_t.shape != (K,):
            raise ValueError(f"weights must have shape (K,), got {weights_t.shape}")
        weights_t = weights_t / weights_t.sum()

    # ---- interpret comp_types ----
    if comp_types is None:
        types_list = ["gaussian"] * K
    elif isinstance(comp_types, str):
        types_list = [comp_types] * K
    else:
        if len(comp_types) != K:
            raise ValueError(f"comp_types must have length K={K}, got {len(comp_types)}")
        types_list = [
            ("gaussian" if (t is None or t == "") else str(t))
            for t in comp_types
        ]

    # ---- build components ----
    components: list[BaseComponent] = []

    for k in range(K):
        t = types_list[k].lower()

        if t in ("gaussian", "normal", "mvnormal", "multivariate_normal"):
            comp = GaussianComponent(
                mean=means_t[k],
                cov=covs_t[k],
                device=device,
                dtype=dtype,
            )

        elif t in ("student_t", "student-t", "t", "multivariate_student_t", "mvt"):
            comp = StudentTComponent(
                mean=means_t[k],
                cov=covs_t[k],
                df=df_default,
                device=device,
                dtype=dtype,
            )

        elif t in ("uniform", "box", "hyperrectangle"):
            # Option A: derive half_range from covariance diag -> mean ± std
            var_diag = torch.diagonal(covs_t[k], dim1=-2, dim2=-1)  # (D,)
            std = torch.sqrt(torch.clamp(var_diag, min=1e-12))  # (D,)
            half_range = std

            comp = UniformComponent(
                mean=means_t[k],
                half_range=half_range,
                device=device,
                dtype=dtype,
            )

        else:
            raise ValueError(
                f"Unknown component type '{t}' for component {k}. "
                "Supported: gaussian, student_t, uniform."
            )

        components.append(comp)

    model = DataGeneratorModel(
        components=components,
        weights=weights_t,
        device=device,
        dtype=dtype,
    )
    return model


def get_classes_stats(image: np.ndarray, labels: np.ndarray) -> Dict[int, tuple[float, np.ndarray, np.ndarray]]:
    """
    Compute mean and covariance for each class in the hyperspectral image.

    Args:
        image:  (H, W, D) or (n,D) hyperspectral image
        labels: (H, W) integer labels for each pixel

    Returns:
        class_stats: dict mapping class label to (ratio, mean, covariance) tuple
    """
    if image.ndim == 3:
        H, W, D = image.shape
        image_reshaped = image.reshape(-1, D)
        labels_reshaped = labels.reshape(-1)
    else:
        image_reshaped = image
        labels_reshaped = labels
        D = image.shape[1]

    class_stats = {}
    unique_labels, counts = np.unique(labels_reshaped, return_counts=True)
    ratios = counts / (labels_reshaped.shape[0])
    means = []
    for label, ratio in zip(unique_labels, ratios):
        class_pixels = image_reshaped[labels_reshaped == label].reshape(-1, D)
        mean = np.mean(class_pixels, axis=0)
        covariance = np.cov(class_pixels, rowvar=False)
        class_stats[label] = (ratio, mean, covariance)
        means.append(mean)
    return class_stats


def create_generator_from_stats(comp_types, stats):
    ratios = np.array([x[0] for x in stats.values()])
    ratios = ratios / np.sum(ratios)
    means = np.array([x[1] for x in stats.values()])
    covs = np.array([x[2] for x in stats.values()])
    if isinstance(comp_types, list) and len(comp_types) != len(stats):
        raise ValueError(
            f"Length of comp_types list must match number of classes (excluding target). Got {len(comp_types)} vs {len(stats)}")
    data_generator = build_data_model_from_stats(means=means, covs=covs, comp_types=comp_types, weights=ratios,
                                                 df_default=DF_DEFAULT)
    return data_generator, ratios, means, covs

#TODO: ADD THE USAGE OF THE NEW TORCHGMM CLASS
def estimate_dist_gmm(image: np.ndarray, num_components: int, cov_type: str = COV_TYPE, n_init: int = 1,
                      max_iter: int = 100, tol=1e-3):
    from sklearn.mixture import GaussianMixture  # lazy import — sklearn may not be available
    bkg_samples = image.reshape(-1, image.shape[-1])
    gmm = GaussianMixture(n_components=num_components, covariance_type=cov_type, random_state=0, n_init=n_init,
                          tol=tol,
                          verbose=0, verbose_interval=5,
                          max_iter=max_iter)
    gmm.fit(bkg_samples)
    return gmm


def build_data_model_from_gmm(gmm,
                              comp_dist: Optional[Union[str, Sequence[Optional[str]]]] = GAUSSIAN_COMP,
                              df_default: int = DF_DEFAULT) -> DataGeneratorModel:
    means = gmm.means_
    covs = gmm.covariances_
    if covs.ndim < 3:
        covs = np.array([np.diag(covs[i]) for i in range(covs.shape[0])])
    weights = gmm.weights_
    num_components = gmm.n_components
    comp_types = [comp_dist for _ in range(num_components)]
    data_generator = build_data_model_from_stats(means=means, covs=covs, comp_types=comp_types, weights=weights,
                                                 df_default=df_default)
    return data_generator


def get_generator_via_gmm(image: np.ndarray, num_components: int, gt: np.ndarray = None,
                          classes_to_exclude: Optional[Union[int, Sequence[int]]] = None,
                          comp_dist: Optional[Union[str, Sequence[Optional[str]]]] = GAUSSIAN_COMP,
                          cov_type: str = COV_TYPE,
                          df_default: int = DF_DEFAULT, n_init: int = 1, max_iter: int = 100,
                          tol=1e-3) -> DataGeneratorModel:
    if gt is not None and classes_to_exclude is not None:
        mask = np.ones_like(gt, dtype=bool)
        for cls in classes_to_exclude:
            mask &= (gt != cls)
        image = image[mask].reshape(-1, image.shape[-1])

    gmm = estimate_dist_gmm(image=image, num_components=num_components, cov_type=cov_type, n_init=n_init,
                            max_iter=max_iter, tol=tol)

    data_generator = build_data_model_from_gmm(gmm=gmm, comp_dist=comp_dist, df_default=df_default)
    return data_generator


def multiclass_data_estimation_using_labels(image: np.ndarray, gt: np.ndarray, target_labels: Union[int, Sequence[int]],
                                            bkg_comp_types: Optional[Union[str, Sequence[Optional[str]]]] = "gaussian",
                                            target_comp_types: Optional[
                                                Union[str, Sequence[Optional[str]]]] = "gaussian"):
    # data = loadmat(file_path)
    # gt = data['map']
    # image = data['data']

    classes = np.unique(gt)
    if isinstance(target_labels, int):
        target_labels = [target_labels]
    classes_stats = get_classes_stats(image, gt)

    bkg_mask = np.ones_like(gt, dtype=bool)

    target_stats = {}
    target_samples = np.empty((0, image.shape[-1]))
    # target_stats = stats.get(target_label, None)
    for target_label in target_labels:
        if target_label < 0:
            continue
        elif target_label not in classes_stats:
            print(
                f"Warning: Target label {target_label} not found in ground truth labels. Skipping target distribution estimation.")
            continue

        bkg_mask &= (gt != target_label)
        target_stats[target_label] = classes_stats[target_label]
        target_samples = np.vstack((target_samples, image[gt == target_label].reshape(-1, image.shape[-1])))
        classes_stats.pop(target_label)

    bkg_data_generator, _, __, ___ = create_generator_from_stats(bkg_comp_types, classes_stats)
    if target_stats == {}:
        # raise ValueError(f"Target label {target_label} not found in ground truth labels.")
        print(
            f"Warning: Target label {target_labels} not found in ground truth labels. Skipping target distribution estimation.")
        return bkg_data_generator, None

    target_data_generator, _, __, ___ = create_generator_from_stats(target_comp_types, target_stats)

    return bkg_data_generator, target_data_generator


def multiclass_data_estimation_using_gmm(image: np.ndarray, gt: np.ndarray, target_labels: Union[int, Sequence[int]],
                                         num_bkg_components: int, num_target_components: int,
                                         bkg_comp_dist: Optional[Union[str, Sequence[Optional[str]]]] = GAUSSIAN_COMP,
                                         target_comp_dist: Optional[
                                             Union[str, Sequence[Optional[str]]]] = GAUSSIAN_COMP,
                                         cov_type: str = COV_TYPE, df_default: int = DF_DEFAULT, n_init: int = 1,
                                         max_iter: int = 100, tol=1e-3):
    # data = loadmat(file_path)
    # gt = data['map']
    # image = data['data']

    classes = np.unique(gt)
    if isinstance(target_labels, int):
        target_labels = [target_labels]
    bkg_classes = [c for c in classes if c not in target_labels]
    bkg_generator = get_generator_via_gmm(image=image, num_components=num_bkg_components, gt=gt,
                                          classes_to_exclude=target_labels,
                                          comp_dist=bkg_comp_dist, cov_type=cov_type, df_default=df_default,
                                          n_init=n_init,
                                          max_iter=max_iter, tol=tol)

    if len(target_labels) == 1 and target_labels[0] not in classes:
        print(
            f"Warning: Target label {target_labels[0]} not found in ground truth labels. Skipping target distribution estimation.")
        return bkg_generator, None

    if len(target_labels) == 0:
        print("No target labels specified. Skipping target distribution estimation.")
        return bkg_generator, None

    target_generator = get_generator_via_gmm(image=image, num_components=num_target_components, gt=gt,
                                             classes_to_exclude=bkg_classes,
                                             comp_dist=target_comp_dist, cov_type=cov_type, df_default=df_default,
                                             n_init=n_init,
                                             max_iter=max_iter, tol=tol)
    return bkg_generator, target_generator


def binary_data_estimation_using_gmm(image: np.ndarray, gt: np.ndarray,
                                     num_bkg_components: int, num_target_components: int,
                                     bkg_comp_dist: Optional[Union[str, Sequence[Optional[str]]]] = GAUSSIAN_COMP,
                                     target_comp_dist: Optional[Union[str, Sequence[Optional[str]]]] = GAUSSIAN_COMP,
                                     cov_type: str = COV_TYPE, df_default: int = DF_DEFAULT, n_init: int = 1,
                                     max_iter: int = 100, tol=1e-3) -> tuple[
    DataGeneratorModel, Optional[DataGeneratorModel]]:
    bkg_generator, target_generator = multiclass_data_estimation_using_gmm(
        image=image,
        gt=gt,
        target_labels=[1],
        num_bkg_components=num_bkg_components,
        num_target_components=num_target_components,
        bkg_comp_dist=bkg_comp_dist,
        target_comp_dist=target_comp_dist,
        cov_type=cov_type,
        df_default=df_default,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol
    )
    return bkg_generator, target_generator
