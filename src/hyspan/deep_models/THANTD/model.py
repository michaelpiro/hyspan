"""
THANTD: Triplet Hybrid Attention Network for Hyperspectral Target Detection
IEEE JSTARS 2025 — Liu et al.

Architecture (per paper Sec. III-B/C):
  SpectralEmbedder (GSE) → N×HAB → CLS token → L2-normalise
  Three weight-sharing branches for (anchor, positive, negative) during training.
  Inference: cosine similarity of pixel feature to the target anchor feature.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Channel Attention Module (CAM)  –  Fig. 3-4 in the paper
# ─────────────────────────────────────────────────────────────────────────────
class CAM(nn.Module):
    """
    Channel Attention Module.
    Two Conv1d (compress → expand, GELU) followed by SE-style channel gating.
    Operates on token-feature tensors of shape (B, seq, d_model).
    """

    def __init__(self, d_model: int, reduction: int = 4):
        super().__init__()
        mid = max(1, d_model // reduction)
        # Compress-expand over the feature (channel) dimension via 1×1 conv
        self.conv1 = nn.Conv1d(d_model, mid, kernel_size=1)
        self.act   = nn.GELU()
        self.conv2 = nn.Conv1d(mid, d_model, kernel_size=1)
        # SE-style gating  (Eq. 3-4)
        self.fc_down = nn.Linear(d_model, mid)
        self.fc_up   = nn.Linear(mid, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq, d)
        out = x.transpose(1, 2)                              # (B, d, seq)
        out = self.conv2(self.act(self.conv1(out)))          # (B, d, seq)
        out = out.transpose(1, 2)                            # (B, seq, d)
        # Channel gating: global-avg-pool → bottleneck FC → sigmoid scale
        z   = out.mean(dim=1)                                # (B, d)
        s   = torch.sigmoid(self.fc_up(F.relu(self.fc_down(z))))  # (B, d)
        return out * s.unsqueeze(1)                          # (B, seq, d)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Attention Block (HAB)  –  Fig. 3 / Eq. 5
# ─────────────────────────────────────────────────────────────────────────────
class HAB(nn.Module):
    """
    Hybrid Attention Block:
        XN  = LayerNorm(X)
        XM  = α·CAM(XN) + MSA(XN) + X        (Eq. 5, learnable α)
        out = XM + MLP(LayerNorm(XM))          (pre-norm MLP with residual)
    """

    def __init__(
        self,
        d_model:       int,
        n_heads:       int,
        mlp_ratio:     int   = 4,
        cam_reduction: int   = 4,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.msa   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cam   = CAM(d_model, cam_reduction)
        self.alpha = nn.Parameter(torch.tensor(0.1))   # learnable balance factor
        self.norm2 = nn.LayerNorm(d_model)
        dim_ff = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn          = self.norm1(x)
        msa_out, _  = self.msa(xn, xn, xn)
        cam_out     = self.cam(xn)
        xm          = self.alpha * cam_out + msa_out + x   # weighted fusion + residual
        return xm + self.mlp(self.norm2(xm))               # MLP block with residual


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Embedder (GSE)  –  Sec. III-B
# ─────────────────────────────────────────────────────────────────────────────
class SpectralEmbedder(nn.Module):
    """
    Groups adjacent spectral bands with an overlapping sliding window of size m
    (Eq. after Sec. III-B), then projects each group to d_model dimensions.
    Prepends a learnable CLS token and adds learnable positional embeddings.
    """

    def __init__(self, n_bands: int, d_model: int, window_size: int = 3):
        super().__init__()
        self.n_bands     = n_bands
        self.window_size = window_size
        self.proj        = nn.Linear(window_size, d_model)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_bands + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: (B, C)
        B  = s.shape[0]
        pad = self.window_size // 2
        # Reflect-pad at spectral boundaries then extract overlapping windows
        s_pad = F.pad(s, (pad, pad), mode='reflect')          # (B, C+2·pad)
        x     = s_pad.unfold(-1, self.window_size, 1)         # (B, C, m)
        z     = self.proj(x)                                   # (B, C, d)
        cls   = self.cls_token.expand(B, -1, -1)              # (B, 1, d)
        z     = torch.cat([cls, z], dim=1)                    # (B, C+1, d)
        return z + self.pos_embed                              # (B, C+1, d)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Backbone Encoder
# ─────────────────────────────────────────────────────────────────────────────
class THANTDEncoder(nn.Module):
    """
    Shared backbone used by all three triplet branches:
        SpectralEmbedder → N×HAB → LayerNorm → CLS token → L2-normalise
    """

    def __init__(
        self,
        n_bands:       int,
        d_model:       int   = 64,
        n_heads:       int   = 4,
        n_layers:      int   = 2,
        window_size:   int   = 3,
        mlp_ratio:     int   = 4,
        cam_reduction: int   = 4,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.embedder = SpectralEmbedder(n_bands, d_model, window_size)
        self.blocks   = nn.ModuleList([
            HAB(d_model, n_heads, mlp_ratio, cam_reduction, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """(B, C) → (B, d_model) L2-normalised CLS features."""
        x = self.embedder(s)
        for blk in self.blocks:
            x = blk(x)
        feat = self.norm(x)[:, 0]          # CLS token
        return F.normalize(feat, dim=-1)   # L2 normalise


# ─────────────────────────────────────────────────────────────────────────────
# Full THANTD model
# ─────────────────────────────────────────────────────────────────────────────
class THANTD(nn.Module):
    """
    THANTD: Triplet Hybrid Attention Network for Hyperspectral Target Detection.

    Training::
        model   = THANTD(n_bands=D)
        loss_fn = ETBLoss()
        a_feat, p_feat, n_feat = model(anchor, positive, negative)
        loss, metrics = loss_fn(a_feat, p_feat, n_feat)

    Detection::
        scores = model.detect(image, prior_spectrum)   # (H, W) cosine-sim map
    """

    def __init__(
        self,
        n_bands:       int,
        d_model:       int   = 64,
        n_heads:       int   = 4,
        n_layers:      int   = 2,
        window_size:   int   = 3,
        mlp_ratio:     int   = 4,
        cam_reduction: int   = 4,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.d_model = d_model
        self.encoder = THANTDEncoder(
            n_bands, d_model, n_heads, n_layers,
            window_size, mlp_ratio, cam_reduction, dropout,
        )

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        """Encode a batch of spectra → L2-normalised features. (B, C) → (B, d)"""
        return self.encoder(s)

    def forward(
        self,
        anchor:   torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Triplet forward — same encoder weights for all three branches."""
        return self.encode(anchor), self.encode(positive), self.encode(negative)

    @torch.no_grad()
    def detect(
        self,
        image:          torch.Tensor,
        prior_spectrum: torch.Tensor,
        batch_size:     int = 4096,
    ) -> torch.Tensor:
        """
        Run detection on a full hyperspectral image.

        Args:
            image:          (H, W, D) float tensor
            prior_spectrum: (D,) target spectrum
            batch_size:     pixels per forward-pass (memory control)
        Returns:
            scores: (H, W) cosine-similarity map in [-1, 1]
        """
        H, W, D = image.shape
        device  = next(self.parameters()).device
        pixels  = image.reshape(-1, D).to(device)
        prior   = prior_spectrum.to(device).unsqueeze(0)   # (1, D)

        self.eval()
        anchor_feat = self.encode(prior)                   # (1, d)
        scores = []
        for start in range(0, pixels.shape[0], batch_size):
            feat = self.encode(pixels[start:start + batch_size])   # (B, d)
            sim  = F.cosine_similarity(feat, anchor_feat.expand(feat.shape[0], -1))
            scores.append(sim.cpu())
        return torch.cat(scores).reshape(H, W)
