"""
Deep learning building blocks for HSI target detection models.

Organised by category. All modules follow PyTorch conventions.

Tensor convention throughout:
  spectral pixels : (B, D)         where D = number of spectral bands
  spectral tokens : (B, N, d)      N tokens of dimension d
  HSI cubes       : (B, H, W, D)   or (B, D, H, W) where noted

References are cited inline at each class.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Attention blocks
# ─────────────────────────────────────────────────────────────────────────────

class SpectralSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block adapted for the spectral dimension.

    Recalibrates band responses by learning a per-band importance weight
    from the global (spatial) context. When applied to spectral pixels (B, D),
    it learns which bands are most discriminative.

    Reference:
        Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
        Adapted for spectral attention: Li et al. (many HSI papers).

    Args:
        in_channels : D — number of spectral bands / feature channels.
        reduction   : bottleneck reduction ratio (default 4).
        activation  : gate activation ('sigmoid' or 'softmax').
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        activation: str = "sigmoid",
    ):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., D) — any leading batch dims are supported.
        Returns:
            (..., D) — same shape, band responses rescaled.
        """
        w = self.fc(x)
        if self.activation == "sigmoid":
            w = torch.sigmoid(w)
        else:
            w = torch.softmax(w, dim=-1)
        return x * w


class CBAMSpectralBlock(nn.Module):
    """
    Channel attention branch from CBAM for spectral feature maps.

    Uses both max-pool and avg-pool channel descriptors (more expressive
    than SE which uses only avg-pool).

    Reference:
        Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

    Args:
        in_channels : D — spectral dimension.
        reduction   : bottleneck ratio.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D)  →  (B, D)"""
        avg_w = self.shared_mlp(x.mean(0, keepdim=True))   # global avg over batch
        max_w = self.shared_mlp(x.max(0, keepdim=True).values)
        w = torch.sigmoid(avg_w + max_w)
        return x * w


class SelfAttention1D(nn.Module):
    """
    Multi-head self-attention over a 1D sequence (e.g. spectral tokens).

    A thin, einops-free wrapper around nn.MultiheadAttention that handles
    the (B, N, d) ↔ (N, B, d) transpose required by PyTorch's MHA.

    Args:
        embed_dim : token embedding dimension.
        num_heads : number of attention heads.
        dropout   : attention dropout.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x               : (B, N, d) input tokens.
            key_padding_mask: (B, N) bool mask (True = ignore).
        Returns:
            (B, N, d) with residual connection.
        """
        out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        return self.norm(x + out)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Convolutional blocks
# ─────────────────────────────────────────────────────────────────────────────

class SpectralConv1dBlock(nn.Module):
    """
    1D convolutional block operating along the spectral axis.

    Treats the D spectral bands as a 1D sequence and applies Conv1d.
    Optionally includes a residual connection.

    Reference:
        Hu et al., "Deep Convolutional Neural Networks for Hyperspectral
        Image Classification", Journal of Sensors 2015.
        Also used in: SSRN, HybridSN, and many 1D-CNN HSI papers.

    Args:
        in_channels  : input spectral channels (C_in).
        out_channels : output channels (C_out).
        kernel_size  : spectral kernel size (odd recommended).
        stride       : conv stride.
        residual     : add a 1×1 shortcut when shapes differ.
        dropout      : dropout after activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
            if residual and (in_channels != out_channels or stride != 1)
            else (nn.Identity() if residual else None)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L) — B samples, C_in channels, L spectral positions.
        Returns:
            (B, C_out, L') same spatial length if stride=1.
        """
        out = self.conv(x)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return self.act(out)


class SpatialConv2dBlock(nn.Module):
    """
    2D convolutional block operating on the spatial (H, W) plane.

    Used when the model processes local spatial patches of an HSI.
    Input channels typically correspond to spectral bands or feature maps.

    Args:
        in_channels  : C_in.
        out_channels : C_out.
        kernel_size  : spatial kernel (3 or 5).
        residual     : skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        residual: bool = True,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if residual and in_channels != out_channels
            else (nn.Identity() if residual else None)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W)  →  (B, C_out, H, W)"""
        out = self.conv(x)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return self.act(out)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transformer blocks
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Standard pre-norm Transformer encoder block (ViT / BERT style).

    Pre-LN layout: LN → Attn → residual → LN → FFN → residual.
    This is the dominant layout in vision transformers after ViT (2021).

    Reference:
        Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021.
        Used in SpectralFormer, SSFTT, TSTTD, and most recent HSI transformers.

    Args:
        dim        : token embedding dimension.
        num_heads  : attention heads.
        mlp_ratio  : FFN hidden dim = dim * mlp_ratio.
        dropout    : applied after attention and after FFN.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden     = int(dim * mlp_ratio)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, dim)  →  (B, N, dim)"""
        # attention sub-block
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop1(h)
        # FFN sub-block
        x = x + self.ffn(self.norm2(x))
        return x


class SpectralTransformerBlock(nn.Module):
    """
    Transformer block where each spectral band is one token.

    Treats an HSI spectrum (B, D) as a sequence of D scalar tokens, each
    projected to a small embedding. Cross-band self-attention then captures
    inter-band correlations.

    Reference:
        Hong et al., "SpectralFormer: Rethinking Hyperspectral Image
        Classification with Transformers", TGRS 2022.
        Also: TSTTD (Li et al., TGRS 2023).

    Args:
        n_bands    : D — number of spectral bands.
        embed_dim  : per-band embedding dimension (each band → R^embed_dim).
        num_heads  : attention heads.
        mlp_ratio  : FFN width factor.
        dropout    : dropout rate.
    """

    def __init__(
        self,
        n_bands: int,
        embed_dim: int = 16,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        # project each scalar band value to embed_dim
        self.band_embed = nn.Linear(1, embed_dim)
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_bands, embed_dim))
        self.block      = TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
        self.norm       = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) spectral pixel batch.
        Returns:
            (B, D, embed_dim) per-band context-aware embeddings.
        """
        # (B, D) → (B, D, 1) → (B, D, embed_dim)
        tokens = self.band_embed(x.unsqueeze(-1)) + self.pos_embed
        tokens = self.block(tokens)
        return self.norm(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Embedding and tokenisation
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """
    Sinusoidal positional encoding for 1D sequences (e.g. spectral bands).

    Reference:
        Vaswani et al., "Attention is All You Need", NeurIPS 2017.

    Args:
        d_model : embedding dimension (must be even).
        max_len : maximum sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)  —  adds PE in-place (no parameters)."""
        return x + self.pe[:, : x.size(1)]


class SpectrumEmbed(nn.Module):
    """
    Embed a raw spectral pixel (B, D) into a fixed-size vector (B, embed_dim).

    A simple learned linear projection with optional normalisation, useful as
    the first layer of most spectral-based DL detectors.

    Args:
        in_dim    : D — number of input spectral bands.
        embed_dim : output embedding dimension.
        norm      : apply LayerNorm to the embedding.
    """

    def __init__(self, in_dim: int, embed_dim: int, norm: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D)  →  (B, embed_dim)"""
        return self.norm(self.proj(x))


class SpatialPatchEmbed(nn.Module):
    """
    Extract and embed local spatial patches from an HSI.

    Takes a (H, W, D) hyperspectral image and extracts overlapping or
    non-overlapping spatial patches, projecting each to embed_dim.

    This enables ViT-style processing of HSI spatial context.

    Reference:
        Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021.
        Adapted for HSI: He et al. "Spatial-Spectral Transformer", TGRS 2021.

    Args:
        patch_size : spatial size of each patch (p × p pixels).
        in_channels: D — number of spectral bands.
        embed_dim  : output token embedding dimension.
    """

    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        # Conv2d as efficient non-overlapping patch extractor + projector
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, H, W) — note channel-first for Conv2d.
        Returns:
            (B, N_patches, embed_dim) where N_patches = (H/p) * (W/p).
        """
        B, D, H, W = x.shape
        out = self.proj(x)                          # (B, embed_dim, H/p, W/p)
        B, E, Hp, Wp = out.shape
        out = out.flatten(2).transpose(1, 2)        # (B, Hp*Wp, E)
        return self.norm(out)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MLP blocks
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Multi-layer perceptron with LayerNorm and optional residual.

    The workhorse building block for spectral-based detectors and classifiers.

    Args:
        in_dim    : input dimension.
        hidden_dims: hidden layer widths (list); e.g. [256, 128].
        out_dim   : output dimension.
        dropout   : applied after each hidden activation.
        residual  : add a linear skip from input to output when shapes match.
        norm      : apply LayerNorm before each linear (pre-norm).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        dropout: float = 0.0,
        residual: bool = False,
        norm: bool = True,
    ):
        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            if norm and i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i]))
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # no activation on final layer
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.skip = (
            nn.Linear(in_dim, out_dim, bias=False) if residual and in_dim != out_dim
            else (nn.Identity() if residual else None)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.skip is not None:
            out = out + self.skip(x)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Loss functions for target detection training
# ─────────────────────────────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.

    Minimises d(anchor, positive) and maximises d(anchor, negative).
    Used in TSTTD and other contrastive HSI target detection methods.

    Reference:
        Schroff et al., "FaceNet: A Unified Embedding for Face Recognition
        and Clustering", CVPR 2015.
        HSI application: Li et al., "TSTTD", TGRS 2023.

    Args:
        margin   : margin α in max(d(a,p) - d(a,n) + α, 0).
        distance : 'euclidean' or 'cosine'.
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        margin: float = 1.0,
        distance: str = "euclidean",
        reduction: str = "mean",
    ):
        super().__init__()
        self.margin    = margin
        self.distance  = distance
        self.reduction = reduction

    def _dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.distance == "cosine":
            return 1.0 - F.cosine_similarity(a, b, dim=-1)
        # euclidean
        return (a - b).pow(2).sum(-1).sqrt()

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor  : (B, d) anchor embeddings.
            positive: (B, d) positive (same class as anchor).
            negative: (B, d) negative (different class).
        Returns:
            Scalar loss.
        """
        d_ap = self._dist(anchor, positive)
        d_an = self._dist(anchor, negative)
        losses = F.relu(d_ap - d_an + self.margin)
        return losses.mean() if self.reduction == "mean" else losses.sum()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for binary-labelled pairs.

    Pulls together pairs with label=1, pushes apart pairs with label=0.

    Reference:
        Hadsell et al., "Dimensionality Reduction by Learning an Invariant
        Mapping", CVPR 2006.
        Used in HSI anomaly detection with siamese architectures.

    Args:
        margin   : margin for negative pairs.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin    = margin
        self.reduction = reduction

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z1, z2 : (B, d) pair embeddings.
            labels : (B,) — 1 = same class (positive pair), 0 = different.
        Returns:
            Scalar loss.
        """
        dist = (z1 - z2).pow(2).sum(-1).sqrt()
        pos_loss = labels * dist.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - dist).pow(2)
        losses   = 0.5 * (pos_loss + neg_loss)
        return losses.mean() if self.reduction == "mean" else losses.sum()


class FocalLoss(nn.Module):
    """
    Binary focal loss for imbalanced target detection.

    Down-weights easy background pixels so training focuses on hard/rare
    target pixels. Essential for real HSI datasets where targets are <1%
    of the image.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        alpha : weight for the positive (target) class.
        gamma : focusing parameter (0 = cross-entropy, 2 is typical).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B,) or (B, 1) — raw (unnormalised) predictions.
            targets: (B,)  — binary labels {0, 1}.
        Returns:
            Scalar focal loss.
        """
        logits  = logits.view(-1)
        targets = targets.view(-1).float()
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t     = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss    = alpha_t * (1 - p_t) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# 7. Autoencoder / reconstruction blocks
# ─────────────────────────────────────────────────────────────────────────────

class SpectralEncoder(nn.Module):
    """
    Spectral encoder: maps a pixel spectrum (B, D) to a latent vector (B, latent_dim).

    Standard backbone for autoencoder-based anomaly/target detection,
    where anomaly score = reconstruction error.

    Reference used in:
        Reed & Yu, "A Background Selection Method for Detecting Anomalies
        in Hyperspectral Images", IGARSS 1990 (classic).
        Deep variants: Xie et al., VAE-based HSI anomaly detection, TGRS 2020.

    Args:
        in_dim     : D — spectral bands.
        latent_dim : bottleneck size.
        hidden_dims: intermediate layer widths.
        dropout    : regularisation.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [max(latent_dim * 4, in_dim // 2), latent_dim * 2]
        self.net = MLP(in_dim, hidden_dims, latent_dim, dropout=dropout, norm=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D)  →  (B, latent_dim)"""
        return self.net(x)


class SpectralDecoder(nn.Module):
    """
    Spectral decoder: maps a latent vector (B, latent_dim) back to spectrum (B, D).

    Symmetric counterpart to SpectralEncoder.

    Args:
        latent_dim : bottleneck size.
        out_dim    : D — reconstructed spectral dimension.
        hidden_dims: intermediate layer widths (typically reversed from encoder).
        dropout    : regularisation.
    """

    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [latent_dim * 2, max(latent_dim * 4, out_dim // 2)]
        self.net = MLP(latent_dim, hidden_dims, out_dim, dropout=dropout, norm=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim)  →  (B, D)"""
        return self.net(z)


class VAEBottleneck(nn.Module):
    """
    Variational bottleneck (reparameterisation trick).

    Takes a deterministic encoder output and produces a stochastic latent
    sample z ~ N(mu, sigma²) with the KL divergence as a regulariser.

    Reference:
        Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014.
        HSI: Chen et al. "Anomaly Detection for Hyperspectral Images Based
        on Variational Auto-Encoder", JARS 2020.

    Args:
        in_dim     : dimension of the deterministic encoder output.
        latent_dim : latent dimension.
    """

    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.fc_mu  = nn.Linear(in_dim, latent_dim)
        self.fc_log_var = nn.Linear(in_dim, latent_dim)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, in_dim) deterministic encoder output.
        Returns:
            z    : (B, latent_dim) reparameterised sample.
            mu   : (B, latent_dim) mean.
            log_var: (B, latent_dim) log-variance.
        """
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        std     = torch.exp(0.5 * log_var)
        eps     = torch.randn_like(std)
        z       = mu + eps * std
        return z, mu, log_var

    @staticmethod
    def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """KL( N(mu, sigma²) || N(0, I) ) per sample, mean-reduced."""
        return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Attention
    "SpectralSEBlock",
    "CBAMSpectralBlock",
    "SelfAttention1D",
    # Convolutional
    "SpectralConv1dBlock",
    "SpatialConv2dBlock",
    # Transformer
    "TransformerBlock",
    "SpectralTransformerBlock",
    # Embedding
    "SinusoidalPE",
    "SpectrumEmbed",
    "SpatialPatchEmbed",
    # MLP
    "MLP",
    # Loss functions
    "TripletLoss",
    "ContrastiveLoss",
    "FocalLoss",
    # Autoencoder
    "SpectralEncoder",
    "SpectralDecoder",
    "VAEBottleneck",
]
