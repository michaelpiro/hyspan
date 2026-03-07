"""Plotting utilities for hyspan experiment results."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import torch


def plot_roc_curves(
    result,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot ROC curves for all detectors in an ExperimentResult.

    Args:
        result    : ExperimentResult from Experiment.run().
        title     : plot title; defaults to result.name.
        save_path : if given, save figure to this path (e.g. 'roc.png').
        show      : call plt.show() when True.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from ..results.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(7, 6))

    for det_name, dr in result.detector_results.items():
        fpr, tpr = roc_curve(dr.scores, result.gt)
        ax.plot(fpr.tolist(), tpr.tolist(), label=f"{det_name} (AUC={dr.auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or result.name)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_detection_maps(
    result,
    cols: int = 4,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot per-detector detection score maps alongside the ground truth.

    Args:
        result    : ExperimentResult from Experiment.run().
        cols      : number of columns in the subplot grid.
        save_path : if given, save figure to this path.
        show      : call plt.show() when True.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    det_names = list(result.detector_results.keys())
    n_panels = len(det_names) + 1          # +1 for GT
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # Ground truth
    axes[0].imshow(result.gt.tolist(), cmap="gray", interpolation="nearest")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    for i, det_name in enumerate(det_names, start=1):
        scores = result.detector_results[det_name].scores.tolist()
        auc = result.detector_results[det_name].auc
        im = axes[i].imshow(scores, cmap="hot", interpolation="nearest")
        axes[i].set_title(f"{det_name}\nAUC={auc:.3f}", fontsize=8)
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Hide any unused subplot slots
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(result.name, fontsize=11, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_auc_bar(
    result,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Horizontal bar chart comparing AUC across detectors.

    Args:
        result    : ExperimentResult.
        save_path : optional save path.
        show      : call plt.show() when True.
    """
    import matplotlib.pyplot as plt

    names = list(result.detector_results.keys())
    aucs = [result.detector_results[n].auc for n in names]

    fig, ax = plt.subplots(figsize=(7, max(2, 0.5 * len(names))))
    bars = ax.barh(names, aucs, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUC-ROC")
    ax.set_title(f"Detector comparison — {result.name}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


__all__ = ["plot_roc_curves", "plot_detection_maps", "plot_auc_bar"]
