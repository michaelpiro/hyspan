from typing import Union, Sequence, Dict, Any, Optional
import numpy as np
import torch
from scipy.io import savemat, loadmat
from .distribution_estimation import binary_data_estimation_using_gmm, multiclass_data_estimation_using_gmm, \
    multiclass_data_estimation_using_labels
from .data_generator import DataGeneratorModel
from .utils import add_targets_to_image, ts_generation
import os

DATA_SUFFIX = "synthetic_dataset.mat"
BKG_GEN_SUFFIX = "bkg_generator.pt"
TARGET_GEN_SUFFIX = "target_generator.pt"
GAUSSIAN_COMP = "gaussian"
COV_TYPE = "full"
DF_DEFAULT = 25


def save_synthetic_dataset(save_dir: str, bkg_generator: DataGeneratorModel, target_generator: DataGeneratorModel,
                           lmm_img_targets: np.ndarray, lmm_img_no_target: np.ndarray, unmixed_img: np.ndarray,
                           comp_map: np.ndarray, target_mask: np.ndarray, target_spectrum: np.ndarray = None):
    # check if the save_dir exists, if not create it

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_path = os.path.join(save_dir, DATA_SUFFIX)
    savemat(
        data_path,
        {
            'data': lmm_img_targets,
            'original_data': lmm_img_no_target,
            'non_mixed': unmixed_img,
            'map': target_mask,
            'comp_map': comp_map,
            'target_spectrum': target_spectrum
        }
    )
    bkg_path = os.path.join(save_dir, BKG_GEN_SUFFIX)
    bkg_generator.save(bkg_path)
    target_path = os.path.join(save_dir, TARGET_GEN_SUFFIX)
    target_generator.save(target_path)


def target_spec_from_generator(target_generator: DataGeneratorModel, type=0, num_samples: int = 100):
    target_samples = target_generator.sample(num_samples, return_labels=False).numpy()  # (num_samples, D)
    target_spec = ts_generation(target_samples, gt=np.ones(num_samples), type=type).squeeze()  # (D,)
    return target_spec


def generate_synthetic_hsi_image_from_binary(file_path: str, save_dir: str,
                                             h: int, w: int, seed: int = None,  # image dimensions
                                             num_bkg_components: int = 10, num_target_components: int = 1,
                                             bkg_dist_type: Optional[
                                                 Union[str, Sequence[Optional[str]]]] = GAUSSIAN_COMP,
                                             target_dist_type: Optional[
                                                 Union[str, Sequence[Optional[str]]]] = GAUSSIAN_COMP,
                                             cov_type: str = COV_TYPE,
                                             df_default: int = DF_DEFAULT,
                                             n_init: int = 1,
                                             max_iter: int = 100,
                                             tol: float = 1e-3,
                                             target_percent: float = 10,
                                             is_additive: bool = True,
                                             allow_scale=False,
                                             target_type=0,
                                             ):
    data = loadmat(file_path)

    image = data['data']  # (H, W, D)
    gt = data['map']  # (H, W)
    bkg_data_gen, target_data_gen = binary_data_estimation_using_gmm(image, gt,
                                                                     num_bkg_components, num_target_components,
                                                                     bkg_dist_type, target_dist_type,
                                                                     cov_type, df_default, n_init, max_iter, tol
                                                                     )
    create_image_from_gens(save_dir, h, w, bkg_data_gen, target_data_gen, target_percent, target_type, is_additive,
                           allow_scale, seed)


def create_image_from_gens(save_dir, h, w, bkg_data_gen, target_data_gen, target_percent, target_type, is_additive,
                           allow_scale, seed):
    lmm_img, comp_map = bkg_data_gen.sample_lmm_image(height=h, width=w, seed=seed, return_weights=False,
                                                      return_comp_map=True)
    non_mixing_img = bkg_data_gen.sample_unmixed_image(height=h, width=w, seed=seed, comp_map=comp_map)
    target_spec = target_spec_from_generator(target_data_gen, type=target_type, num_samples=100)
    lmm_img_targets, target_mask = add_targets_to_image(lmm_img, target_percent=target_percent, target=target_spec,
                                                        is_additive=is_additive, allow_scale=allow_scale)
    save_synthetic_dataset(save_dir, bkg_data_gen, target_data_gen, lmm_img_targets.numpy(), lmm_img, non_mixing_img,
                           comp_map, target_mask.numpy(), target_spec)


def generate_synthetic_image_from_multiclass_gmm(file_path: str, save_dir: str,
                                                 h: int, w: int, seed: int = None,  # image dimensions
                                                 target_labels=6,
                                                 num_bkg_components: int = 15, num_target_components: int = 1,
                                                 bkg_dist_type: Optional[
                                                     Union[str, Sequence[Optional[str]]]] = GAUSSIAN_COMP,
                                                 target_dist_type: Optional[
                                                     Union[str, Sequence[Optional[str]]]] = GAUSSIAN_COMP,
                                                 cov_type: str = COV_TYPE,
                                                 df_default: int = DF_DEFAULT,
                                                 n_init: int = 1,
                                                 max_iter: int = 100,
                                                 tol: float = 1e-3,
                                                 target_percent: float = 10,
                                                 is_additive: bool = True,
                                                 allow_scale=False,
                                                 target_type=0,
                                                 ):
    if target_labels is None:
        target_labels = [1]
    data = loadmat(file_path)

    image = data['data']  # (H, W, D)
    gt = data['map']  # (H, W)
    bkg_data_gen, target_data_gen = multiclass_data_estimation_using_gmm(image, gt,
                                                                         target_labels=[target_labels] if isinstance(
                                                                             target_labels, int) else
                                                                         target_labels,
                                                                         num_bkg_components=num_bkg_components,
                                                                         num_target_components=num_target_components,
                                                                         bkg_comp_dist=bkg_dist_type,
                                                                         target_comp_dist=target_dist_type,
                                                                         cov_type=cov_type, df_default=df_default,
                                                                         n_init=n_init, max_iter=max_iter, tol=tol)

    create_image_from_gens(save_dir, h, w, bkg_data_gen, target_data_gen, target_percent, target_type, is_additive,
                           allow_scale, seed)


def generate_synthetic_image_from_multiclass_labels(file_path: str, save_dir: str,
                                                    h: int, w: int, seed: int = None,  # image dimensions
                                                    target_labels=6,
                                                    bkg_dist_type: Optional[
                                                        Union[str, Sequence[Optional[str]]]] = GAUSSIAN_COMP,
                                                    target_dist_type: Optional[
                                                        Union[str, Sequence[Optional[str]]]] = GAUSSIAN_COMP,
                                                    target_percent: float = 10,
                                                    is_additive: bool = True,
                                                    allow_scale=False,
                                                    target_type=0,
                                                    ):
    if target_labels is None:
        target_labels = [1]
    data = loadmat(file_path)

    image = data['data']  # (H, W, D)
    gt = data['map']  # (H, W)
    bkg_data_gen, target_data_gen = multiclass_data_estimation_using_labels(image, gt,
                                                                            target_labels=[target_labels] if isinstance(
                                                                                target_labels, int) else
                                                                            target_labels,
                                                                            bkg_comp_types=bkg_dist_type,
                                                                            target_comp_types=target_dist_type)

    create_image_from_gens(save_dir, h, w, bkg_data_gen, target_data_gen, target_percent, target_type, is_additive,
                           allow_scale, seed)


def generate_synthetic_image_from_dir(path_to_gen_dir, save_dir, h, w, target_percent=0.01, is_additive=True,
                                      allow_scale=False,
                                      target_type=0, seed=None):
    bkg_data_gen = DataGeneratorModel.load(os.path.join(path_to_gen_dir, BKG_GEN_SUFFIX))
    target_data_gen = DataGeneratorModel.load(os.path.join(path_to_gen_dir, TARGET_GEN_SUFFIX))
    create_image_from_gens(save_dir, h, w, bkg_data_gen, target_data_gen, target_percent, target_type, is_additive,
                           allow_scale, seed)

# TODO: ADD A CLASS OF IMAGE GENERATOR FOR SIMPLER INTERFACE


if __name__ == '__main__':
    # usage example
    dataset_name = "abu-airport-2.mat"
    dataset_path = os.path.join("/datasets", dataset_name)
    save_directory = os.path.join("", dataset_name)
    generate_synthetic_hsi_image_from_binary(dataset_path, save_directory,
                                             h=128, w=128, seed=42,
                                             num_bkg_components=10, num_target_components=1,
                                             bkg_dist_type=GAUSSIAN_COMP, target_dist_type=GAUSSIAN_COMP,
                                             cov_type=COV_TYPE, df_default=DF_DEFAULT, n_init=1, max_iter=100,
                                             tol=1e-3, target_percent=10, is_additive=True, target_type=0,
                                             allow_scale=False)
