# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`hyspan` is a Python library for hyperspectral image (HSI) analysis. Core capabilities: classical statistical target detection (CEM, AMF, ACE, Kelly's), GMM-based variants, synthetic HSI generation, and experiment orchestration with result storage.

## Installation

```bash
pip install -e .
```

`src/` layout — package root is `src/hyspan/`. Python 3.8+ required. Key dependencies: `torch`, `numpy`, `scipy`, `scikit-learn`, `einops`.

**Dev environment note:** The development machine has NumPy 2.x but torch 2.2.2 was built against NumPy 1.x. `torch.Tensor.numpy()` and `torch.from_numpy()` may fail. All new code should avoid the numpy bridge where possible, or test with pure torch paths.

## Development Commands

No formal test suite yet. To run module-level demos:

```bash
# Run as installed package (required — relative imports break direct script execution)
python -m hyspan.data.generation.data_generator
python -m hyspan.deep_models.TSTTD.Main
```

## Public API

Everything is importable from the top level:

```python
from hyspan import (
    # Detectors
    ACE_SCM, AMF_SCM, CEM_SCM, Kellys_SCM,
    ACE_GMM, AMF_GMM, CEM_GMM, Kellys_GMM,
    BaseDetector,
    # Orchestration
    Experiment,
    # Results
    ExperimentResult, ResultsStore, roc_auc,
    # Synthesis
    SynthesisEngine,
    # Utilities
    ts_generation,
)
```

Typical workflow:

```python
exp = Experiment(image, gt, name="sandiego")
exp.add(ACE_SCM()).add(ACE_GMM(n_components=10))
result = exp.run()           # fits, detects, computes AUC
print(result.summary())

store = ResultsStore("./results")
store.save(result)
store.compare()              # AUC table across all saved runs
```

## Architecture

### Tensor Convention

All HSI images use `(H, W, D)` shape where D is the spectral dimension. Flattened pixel arrays are `(H*W, D)`. All detectors use `@torch.no_grad()`.

### Package Layout

```
src/hyspan/
├── __init__.py                     # flat public API — all exports here
├── detectors/
│   ├── base.py                     # BaseDetector ABC (fit + detect interface)
│   ├── scm.py                      # CEM/AMF/ACE/Kellys _SCM variants
│   ├── gmm.py                      # CEM/AMF/ACE/Kellys _GMM variants
│   └── deep/                       # stub for future ML detectors
├── experiments/
│   └── runner.py                   # Experiment: orchestrates fit→detect→AUC
├── results/
│   ├── metrics.py                  # roc_curve(), roc_auc() — pure PyTorch
│   └── store.py                    # DetectorResult, ExperimentResult, ResultsStore
├── synthesis/
│   └── engine.py                   # SynthesisEngine: from_image() / from_labeled_image()
├── algorithms/
│   ├── classic.py                  # Low-level GLRT primitives (mf, glrt, osp)
│   ├── utils.py                    # ts_generation() — 9 target spectrum methods
│   └── target_detection/
│       ├── classic_scm.py          # ClassicDetectorSCM (low-level, used by scm.py)
│       └── classic_gmm.py          # ClassicDetectorGMM (low-level, used by gmm.py)
├── data/
│   ├── datasets.py                 # .mat file loaders (Sandiego, Pavia-U, Abu Airport)
│   └── generation/
│       ├── data_generator.py       # DataGeneratorModel + component types
│       ├── distribution_estimation.py  # build_data_model_from_gmm/stats (imports sklearn)
│       └── image_utils.py          # spatial comp maps, LMM weight generation
└── deep_models/
    ├── GMM/
    │   ├── TorchGmm.py             # TorchGMM — GPU EM-based GMM (primary)
    │   └── Gmm.py                  # legacy sklearn wrapper (import directly, not via __init__)
    └── TSTTD/                      # Triplet Spectralwise Transformer (TGRS 2023)
```

### Detector Architecture

**GLRT framework** in `algorithms/classic.py:glrt()` is the core math primitive, parameterised by `phi1`/`phi2`:
- **CEM**: `phi1=0, phi2=0` (energy minimisation)
- **AMF**: `phi1=1, phi2=0`
- **ACE**: `phi1=0, phi2=1`
- **Kelly's**: `phi1=N, phi2=1`

**Background estimation** is the only difference between SCM and GMM families:
- **SCM** (`ClassicDetectorSCM`): local k×k patch covariance per pixel at detect-time; no fit step needed.
- **GMM** (`ClassicDetectorGMM`): fit a GMM globally, assign each pixel to a component via neighbourhood majority vote, use that component's statistics.

**Adding a new detector:** subclass `BaseDetector`, implement `detect(image, target) → (H, W) tensor`. Override `fit(image, gt)` if the detector needs training. See `detectors/deep/__init__.py` for the stub.

### GMM

`TorchGMM` (`deep_models/GMM/TorchGmm.py`) runs full-covariance EM on GPU. Compatible with sklearn's API (`.means_`, `.covariances_`, `.weights_` numpy properties). Key design:
- Memory-safe E-step: loops over K components doing `(D, N)` triangular solves — avoids the O(K·D·N) peak allocation that kills HSI-scale fitting.
- `inv_cholesky_covariances_torch`: returns `U = L⁻ᵀ` (upper-triangular), matching sklearn's `precisions_cholesky_` convention so `ClassicDetectorGMM` works unchanged.
- `from_sklearn_gmm()` classmethod converts an existing sklearn model.

The legacy `Gmm` class (sklearn wrapper) must be imported directly: `from hyspan.deep_models.GMM.Gmm import Gmm`. It is NOT re-exported from `deep_models/GMM/__init__.py` to avoid pulling sklearn in at package load time.

### Synthesis

`SynthesisEngine` builds a `DataGeneratorModel` (mixture of Gaussian/StudentT/Uniform components) and samples synthetic images from it:
- `from_image(image, n_components, gt)` — fits `TorchGMM` to background pixels.
- `from_labeled_image(image, gt)` — uses empirical per-class statistics directly (faster, no GMM).
- `generate(H, W)` — unmixed: each pixel drawn from one component.
- `generate_lmm(H, W)` — LMM: spatially-smoothed convex combination of components.

**Known issue:** `synthesis/engine.py` imports `distribution_estimation` at module level, which imports sklearn. This causes the full package import to fail in the broken dev environment. Fix: move those imports inside `from_image()` and `from_labeled_image()` method bodies.

### Results Storage

`ExperimentResult.save(dir)` writes two files under `dir/<name>/`:
- `meta.json` — AUC scores, metadata, timestamp.
- `tensors.pt` — gt, target spectrum, and all score tensors.

`ResultsStore.compare()` prints a cross-experiment AUC table.

### Real Datasets

`.mat` files in `datasets/real/`: `Sandiego.mat`, `pavia-u.mat`, `abu-airport-2.mat`.

### Import Style

All internal imports within `src/hyspan/` use relative imports (e.g., `from ..classic import glrt`). Never use `src.hyspan.*` absolute imports.
