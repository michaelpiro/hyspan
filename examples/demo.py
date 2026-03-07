"""
hyspan demo
===========

Tests the full library pipeline on a small synthetic demo image:

  1. Build & save a (64, 64, 8) demo HSI with planted targets  → datasets/real/demo.mat
  2. Synthesis test A: SynthesisEngine.from_image  (TorchGMM-based)
  3. Synthesis test B: SynthesisEngine.from_labeled_image (multiclass / stats-based)
  4. Experiment: run SCM + GMM detectors on the demo image
  5. Store results
  6. Plot ROC curves, detection maps, AUC bar chart

Run from the repo root:
    python examples/demo.py
or (after pip install -e .):
    python -m examples.demo
"""
from __future__ import annotations

import sys
import numpy as np
import torch
from pathlib import Path
from scipy.io import savemat, loadmat
from torch.distributions import MultivariateNormal

# ── make sure the package is importable when run as a plain script ──────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── output directories ───────────────────────────────────────────────────────
DATASET_PATH = ROOT / "datasets" / "real" / "demo.mat"
RESULTS_DIR  = ROOT / "experiments" / "demo_results"
PLOTS_DIR    = ROOT / "experiments" / "demo_plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build & save demo dataset  (pure torch — no numpy bridge needed)
# ─────────────────────────────────────────────────────────────────────────────

def build_demo_dataset(seed: int = SEED):
    """
    Create a (64, 64, 8) HSI with 3 Gaussian background classes + 1 target.
    Everything built in torch to avoid the numpy-torch C bridge.

    Returns:
        image        : (H, W, D) float32 torch tensor
        binary_gt    : (H, W) int32 torch tensor  — 1=target, 0=background
        multiclass_gt: (H, W) int32 torch tensor  — 0/1/2=background classes, 3=target
    """
    torch.manual_seed(seed)
    H, W, D = 64, 64, 8
    half = H // 2

    # ── spectral signatures ──────────────────────────────────────────────────
    bg0_mean = torch.linspace(0.1, 0.4, D)
    bg1_mean = torch.linspace(0.4, 0.7, D)
    bg2_mean = torch.tensor([0.6, 0.5, 0.4, 0.3, 0.4, 0.5, 0.6, 0.5])
    tgt_mean = torch.tensor([0.9, 0.1, 0.8, 0.1, 0.9, 0.1, 0.8, 0.1])

    cov_bg  = torch.eye(D) * 0.02
    cov_tgt = torch.eye(D) * 0.01

    dist_bg0 = MultivariateNormal(bg0_mean, cov_bg)
    dist_bg1 = MultivariateNormal(bg1_mean, cov_bg)
    dist_bg2 = MultivariateNormal(bg2_mean, cov_bg)
    dist_tgt = MultivariateNormal(tgt_mean, cov_tgt)

    image = torch.zeros(H, W, D)
    multiclass_gt = torch.zeros(H, W, dtype=torch.int32)

    # top-left quadrant: bg class 0
    n_tl = half * half
    image[:half, :half, :] = dist_bg0.sample((n_tl,)).view(half, half, D)
    multiclass_gt[:half, :half] = 0

    # top-right quadrant: bg class 1
    n_tr = half * (W - half)
    image[:half, half:, :] = dist_bg1.sample((n_tr,)).view(half, W - half, D)
    multiclass_gt[:half, half:] = 1

    # bottom half: bg class 2
    n_bot = (H - half) * W
    image[half:, :, :] = dist_bg2.sample((n_bot,)).view(H - half, W, D)
    multiclass_gt[half:, :] = 2

    # ── plant target pixels: cross pattern near centre ───────────────────────
    cy, cx = H // 2, W // 2
    target_mask = torch.zeros(H, W, dtype=torch.bool)
    target_mask[cy - 1 : cy + 2, cx - 6 : cx + 7] = True   # horizontal bar
    target_mask[cy - 6 : cy + 7, cx - 1 : cx + 2] = True   # vertical bar

    n_tgt = target_mask.sum().item()
    image[target_mask] = dist_tgt.sample((n_tgt,))
    multiclass_gt[target_mask] = 3

    image = image.clamp(0.0, 1.0)
    binary_gt = (multiclass_gt == 3).to(torch.int32)

    return image, binary_gt, multiclass_gt


def save_demo_mat(image, binary_gt, multiclass_gt, path: Path):
    """Save torch tensors to .mat via safe tolist() conversion."""
    image_np    = np.array(image.tolist(),         dtype=np.float32)
    gt_np       = np.array(binary_gt.tolist(),     dtype=np.int32)
    mc_gt_np    = np.array(multiclass_gt.tolist(), dtype=np.int32)

    savemat(str(path), {
        "data":            image_np,
        "map":             gt_np,
        "map_multiclass":  mc_gt_np,
    })
    print(f"[1] Saved demo dataset → {path}")
    print(f"    image shape : {image.shape}  dtype={image.dtype}")
    print(f"    binary gt   : {binary_gt.sum().item()} target pixels")
    print(f"    multiclass gt labels: {multiclass_gt.unique().tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Synthesis test A — from_image (TorchGMM)
# ─────────────────────────────────────────────────────────────────────────────

def test_synthesis_gmm(image, binary_gt):
    from hyspan import SynthesisEngine

    print("\n[2] Synthesis A: SynthesisEngine.from_image (TorchGMM, K=4)")
    engine = SynthesisEngine.from_image(
        image=image,          # torch tensor — no numpy bridge needed
        n_components=4,
        gt=binary_gt,
        exclude_target=True,
        device="cpu",
        verbose=True,
    )

    synth = engine.generate(64, 64, seed=SEED)
    print(f"    Unmixed synthetic image: {tuple(synth.shape)}  "
          f"val range=[{synth.min():.3f}, {synth.max():.3f}]")

    synth_lmm = engine.generate_lmm(64, 64, seed=SEED)
    print(f"    LMM synthetic image   : {tuple(synth_lmm.shape)}  "
          f"val range=[{synth_lmm.min():.3f}, {synth_lmm.max():.3f}]")

    print("    [OK] Synthesis A passed")
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Synthesis test B — from_labeled_image (multiclass stats)
# ─────────────────────────────────────────────────────────────────────────────

def test_synthesis_multiclass(image, multiclass_gt):
    from hyspan import SynthesisEngine

    print("\n[3] Synthesis B: SynthesisEngine.from_labeled_image (multiclass)")
    # Convert to numpy via tolist for the stats-based path (uses numpy internals)
    image_np    = np.array(image.tolist(),         dtype=np.float32)
    mc_gt_np    = np.array(multiclass_gt.tolist(), dtype=np.int32)

    engine = SynthesisEngine.from_labeled_image(
        image=image_np,
        gt=mc_gt_np,
        target_label=3,
        comp_type="gaussian",
    )

    synth = engine.generate(64, 64, seed=SEED)
    print(f"    Unmixed synthetic image: {tuple(synth.shape)}  "
          f"val range=[{synth.min():.3f}, {synth.max():.3f}]")

    synth_lmm = engine.generate_lmm(64, 64, seed=SEED)
    print(f"    LMM synthetic image   : {tuple(synth_lmm.shape)}  "
          f"val range=[{synth_lmm.min():.3f}, {synth_lmm.max():.3f}]")

    print("    [OK] Synthesis B passed")
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# 4 + 5.  Experiment: SCM + GMM detectors, store results
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(image, binary_gt):
    from hyspan import (
        ACE_SCM, AMF_SCM, CEM_SCM,
        ACE_GMM, AMF_GMM, CEM_GMM,
        Experiment,
        ResultsStore,
    )

    print("\n[4] Running experiment on demo image ...")

    # Target signature: mean spectrum of planted target pixels (pure torch)
    tgt_pixels = image[binary_gt == 1]           # (N_tgt, D)
    target_sig = tgt_pixels.mean(dim=0).float()  # (D,)
    print(f"    Target signature from {tgt_pixels.shape[0]} target pixels  "
          f"(shape={tuple(target_sig.shape)})")

    exp = Experiment(image.float(), binary_gt.float(), name="demo")
    (exp
     .add(CEM_SCM())
     .add(AMF_SCM())
     .add(ACE_SCM())
     .add(CEM_GMM(n_components=4))
     .add(AMF_GMM(n_components=4))
     .add(ACE_GMM(n_components=4))
    )

    result = exp.run(target_signature=target_sig, verbose=True)

    print(f"\n    --- AUC Summary ---\n{result.summary()}")
    print(f"\n    Best detector: {result.best().name} (AUC={result.best().auc:.4f})")

    print(f"\n[5] Saving results to {RESULTS_DIR} ...")
    store = ResultsStore(str(RESULTS_DIR))
    store.save(result)
    print(f"    Saved. Listing store: {store.list()}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plots
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(result):
    from hyspan.plots import plot_roc_curves, plot_detection_maps, plot_auc_bar

    print(f"\n[6] Generating plots → {PLOTS_DIR}")

    plot_roc_curves(
        result,
        title="ROC curves — demo image",
        save_path=str(PLOTS_DIR / "roc_curves.png"),
        show=False,
    )
    print("    roc_curves.png saved")

    plot_detection_maps(
        result,
        cols=4,
        save_path=str(PLOTS_DIR / "detection_maps.png"),
        show=False,
    )
    print("    detection_maps.png saved")

    plot_auc_bar(
        result,
        save_path=str(PLOTS_DIR / "auc_bar.png"),
        show=False,
    )
    print("    auc_bar.png saved")

    print("    [OK] All plots saved")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  hyspan demo")
    print("=" * 60)

    # 1. Build dataset entirely in torch
    image, binary_gt, multiclass_gt = build_demo_dataset()
    save_demo_mat(image, binary_gt, multiclass_gt, DATASET_PATH)

    # 2. Synthesis A: GMM (torch-native path)
    test_synthesis_gmm(image, binary_gt)

    # 3. Synthesis B: multiclass stats (numpy path, safe)
    test_synthesis_multiclass(image, multiclass_gt)

    # 4+5. Experiment + store
    result = run_experiment(image, binary_gt)

    # 6. Plots
    make_plots(result)

    print("\n" + "=" * 60)
    print("  Demo complete.  Figures in experiments/demo_plots/")
    print("=" * 60)
