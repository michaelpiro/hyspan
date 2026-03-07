"""Structured storage for experiment results."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass
class DetectorResult:
    """Per-detector output from one experiment run."""
    name: str
    scores: torch.Tensor   # (H, W) detection scores
    auc: float             # AUC-ROC


@dataclass
class ExperimentResult:
    """
    All outputs from a single ``Experiment.run()`` call.

    Fields
    ------
    name           : Experiment identifier (used as directory name when saved).
    gt             : (H, W) binary ground-truth used for evaluation.
    target         : (D,)  target spectrum used during detection.
    detector_results: Ordered dict of detector_name → DetectorResult.
    metadata       : Free-form dict for any extra info (dataset path, ts_method, …).
    timestamp      : ISO-format string, set automatically at creation time.
    """
    name: str
    gt: torch.Tensor
    target: torch.Tensor
    detector_results: Dict[str, DetectorResult] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted AUC table string."""
        rows = [f"{'Detector':<30}  AUC", "-" * 42]
        for name, dr in self.detector_results.items():
            rows.append(f"{name:<30}  {dr.auc:.4f}")
        return "\n".join(rows)

    def auc_dict(self) -> Dict[str, float]:
        return {name: dr.auc for name, dr in self.detector_results.items()}

    def best(self) -> DetectorResult:
        """Return the DetectorResult with the highest AUC."""
        return max(self.detector_results.values(), key=lambda r: r.auc)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """
        Save this result under ``directory/<name>/``.

        Creates two files:
          meta.json    — AUC scores, metadata, timestamp (human-readable).
          tensors.pt   — gt, target, and all score tensors.
        """
        out = Path(directory) / self.name
        out.mkdir(parents=True, exist_ok=True)

        meta = {
            "name": self.name,
            "timestamp": self.timestamp,
            "auc": self.auc_dict(),
            "metadata": self.metadata,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))

        tensors = {
            "gt": self.gt.cpu(),
            "target": self.target.cpu(),
            "scores": {name: dr.scores.cpu() for name, dr in self.detector_results.items()},
        }
        torch.save(tensors, out / "tensors.pt")

    @classmethod
    def load(cls, directory: str, name: str) -> "ExperimentResult":
        """Load a previously saved result."""
        out = Path(directory) / name
        meta = json.loads((out / "meta.json").read_text())
        tensors = torch.load(out / "tensors.pt", map_location="cpu")

        detector_results = {
            det_name: DetectorResult(
                name=det_name,
                scores=tensors["scores"][det_name],
                auc=meta["auc"][det_name],
            )
            for det_name in meta["auc"]
        }

        return cls(
            name=meta["name"],
            gt=tensors["gt"],
            target=tensors["target"],
            detector_results=detector_results,
            metadata=meta.get("metadata", {}),
            timestamp=meta.get("timestamp", ""),
        )


class ResultsStore:
    """
    Directory-based store for ``ExperimentResult`` objects.

    Usage
    -----
    ::
        store = ResultsStore("./results")
        store.save(result)                     # saves to ./results/<result.name>/
        result = store.load("sandiego_test")
        store.compare()                        # prints AUC table across all saved runs
    """

    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: ExperimentResult) -> None:
        result.save(str(self.base_dir))

    def load(self, name: str) -> ExperimentResult:
        return ExperimentResult.load(str(self.base_dir), name)

    def list(self) -> List[str]:
        """Return names of all saved experiments."""
        return sorted(d.name for d in self.base_dir.iterdir() if d.is_dir())

    def compare(self, names: Optional[List[str]] = None) -> str:
        """
        Print a comparison table of AUC scores across experiments.

        Args:
            names: subset of experiment names to compare; defaults to all.
        """
        names = names or self.list()
        if not names:
            return "(no saved results)"

        results = {n: self.load(n) for n in names}

        # Collect all detector names across all experiments
        all_detectors = []
        for r in results.values():
            for d in r.detector_results:
                if d not in all_detectors:
                    all_detectors.append(d)

        col_w = 12
        header = f"{'Experiment':<30}" + "".join(f"{d[:col_w]:>{col_w}}" for d in all_detectors)
        rows = [header, "-" * len(header)]
        for exp_name, r in results.items():
            row = f"{exp_name:<30}"
            for det in all_detectors:
                auc = r.auc_dict().get(det, float("nan"))
                row += f"{auc:>{col_w}.4f}"
            rows.append(row)

        table = "\n".join(rows)
        print(table)
        return table
