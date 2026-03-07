"""
ETB-Loss: Enhanced Triplet-BCE Loss for THANTD  (Sec. III-D / Eq. 9-11).

Combines:
  - Dual Triplet Loss  (Eq. 10): two cosine-distance triplet constraints
  - Binary Cross-Entropy Loss (Eq. 11): exploits binary label information
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ETBLoss(nn.Module):
    """
    Enhanced Triplet-BCE Loss.

    L_ETB = L_dual + lambda_bce * L_BCE

    Dual triplet uses cosine *similarity* (range [-1, 1], higher = closer).
    Distances in the paper's Eq. 9-10 are cosine similarities, so we want:
        d_pos > d_neg  (anchor closer to positive than to negative)

    Args:
        margin:     Triplet margin δ (default 0.3).
        lambda_bce: Weight for BCE term (default 0.5).
    """

    def __init__(self, margin: float = 0.3, lambda_bce: float = 0.5):
        super().__init__()
        self.margin     = margin
        self.lambda_bce = lambda_bce

    def forward(
        self,
        anchor_feat:   torch.Tensor,   # (B, d) — L2-normalised
        positive_feat: torch.Tensor,   # (B, d)
        negative_feat: torch.Tensor,   # (B, d)
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            loss:    scalar loss tensor
            metrics: dict with individual loss components and similarity stats
        """
        # Cosine similarities (Eq. 9)
        d_pos  = (anchor_feat   * positive_feat).sum(dim=-1)   # (B,) — d+
        d_neg  = (anchor_feat   * negative_feat).sum(dim=-1)   # (B,) — d-
        d_negp = (positive_feat * negative_feat).sum(dim=-1)   # (B,) — d-'

        # Dual Triplet Loss (Eq. 10)
        # Term 1: anchor-positive closer than anchor-negative
        # Term 2: anchor-positive closer than positive-negative
        t1 = torch.clamp(d_neg  - d_pos + self.margin, min=0.0)
        t2 = torch.clamp(d_negp - d_pos + self.margin, min=0.0)
        loss_dual = (t1 + t2).mean()

        # BCE Loss (Eq. 11)
        # Map cosine similarities from [-1, 1] → (0, 1) for cross-entropy
        s_pos = (d_pos + 1.0) * 0.5    # label = 1
        s_neg = (d_neg + 1.0) * 0.5    # label = 0
        eps   = 1e-7
        loss_bce = -0.5 * (
            torch.log(s_pos.clamp(min=eps)) +
            torch.log((1.0 - s_neg).clamp(min=eps))
        ).mean()

        loss = loss_dual + self.lambda_bce * loss_bce

        metrics = {
            "loss":      loss.item(),
            "loss_dual": loss_dual.item(),
            "loss_bce":  loss_bce.item(),
            "d_pos":     d_pos.mean().item(),
            "d_neg":     d_neg.mean().item(),
            "margin_viol": (t1 > 0).float().mean().item(),
        }
        return loss, metrics
