"""
hyspan — Hyperspectral Analysis Library
========================================

Quick-start
-----------
::
    from hyspan.detectors import ACE_SCM, ACE_GMM
    from hyspan.experiments import Experiment
    from hyspan.results import ResultsStore
    from hyspan.synthesis import SynthesisEngine

    # Run detectors and collect results
    exp = Experiment(image, gt, name="my_dataset")
    exp.add(ACE_SCM()).add(ACE_GMM(n_components=10))
    result = exp.run()
    print(result.summary())

    # Save / load / compare
    store = ResultsStore("./results")
    store.save(result)

    # Synthesise new data
    engine = SynthesisEngine.from_image(image, n_components=10, gt=gt)
    synth_image = engine.generate(128, 128, seed=0)
"""
from .detectors import (
    BaseDetector,
    CEM_SCM, AMF_SCM, ACE_SCM, Kellys_SCM,
    CEM_GMM, AMF_GMM, ACE_GMM, Kellys_GMM,
    LRT_GMM,
)
from .experiments import Experiment
from .results import ExperimentResult, ResultsStore, roc_auc
from .synthesis import SynthesisEngine
from .data import ts_generation

__all__ = [
    # Detectors
    "BaseDetector",
    "CEM_SCM", "AMF_SCM", "ACE_SCM", "Kellys_SCM",
    "CEM_GMM", "AMF_GMM", "ACE_GMM", "Kellys_GMM",
    "LRT_GMM",
    # Orchestration
    "Experiment",
    # Results
    "ExperimentResult", "ResultsStore", "roc_auc",
    # Synthesis
    "SynthesisEngine",
    # Data utilities
    "ts_generation",
]
