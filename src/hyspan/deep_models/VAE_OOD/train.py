"""
Training loop for SpectralVAE (background-only unsupervised training).

The model is trained on H0 (background) pixel spectra only.
No target labels are used.  This matches the paper's design:
  'the CVAE is fully unsupervised, operates natively on in-distribution
   data learned from target-free observations.'
"""
from __future__ import annotations

import torch
from torch.utils.data import TensorDataset, DataLoader

from .model import SpectralVAE
from .loss  import ELBOLoss


def train_vae_ood(
    model:          SpectralVAE,
    bg_pixels:      torch.Tensor,   # (N, D) background spectra
    epochs:         int   = 100,
    batch_size:     int   = 1024,
    lr:             float = 1e-3,
    weight_decay:   float = 1e-5,
    beta:           float = 1.0,
    device:         str   = "cuda",
    num_workers:    int   = 0,
    print_every:    int   = 10,
    checkpoint_path: str | None = None,
) -> list[dict]:
    """
    Train SpectralVAE on background pixels.

    Args:
        model:           SpectralVAE instance (moved to device internally)
        bg_pixels:       (N, D) float tensor of background spectra
        epochs:          training epochs
        batch_size:      pixels per batch
        lr:              initial learning rate (Adam)
        weight_decay:    L2 regularisation
        beta:            KL weight in ELBO (β-VAE)
        device:          'cuda', 'mps', or 'cpu'
        num_workers:     DataLoader workers
        print_every:     print interval
        checkpoint_path: save best checkpoint here (None = no saving)

    Returns:
        history: list of per-epoch metric dicts
    """
    model   = model.to(device)
    loss_fn = ELBOLoss(beta=beta)
    optim   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=lr * 0.01)

    # Always keep the dataset on CPU — DataLoader handles transfer to device.
    # bg_pixels may arrive on GPU (e.g. output of local_whiten_image with device='cuda'),
    # and pin_memory only works on CPU tensors, so we force .cpu() here.
    dataset = TensorDataset(bg_pixels.float().cpu())
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    history   = []
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        totals    = dict(loss=0.0, l_rec=0.0, l_kl=0.0)
        n_batches = 0

        for (x,) in loader:
            x = x.to(device)

            x_hat, mu, log_var = model(x)
            loss, metrics      = loss_fn(x, x_hat, mu, log_var)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            for k, v in metrics.items():
                totals[k] += v
            n_batches += 1

        sched.step()

        epoch_metrics = {k: v / max(1, n_batches) for k, v in totals.items()}
        epoch_metrics["epoch"] = epoch
        epoch_metrics["lr"]    = sched.get_last_lr()[0]
        history.append(epoch_metrics)

        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"loss={epoch_metrics['loss']:.4f}  "
                f"rec={epoch_metrics['l_rec']:.4f}  "
                f"kl={epoch_metrics['l_kl']:.4f}"
            )

        if checkpoint_path and epoch_metrics["loss"] < best_loss:
            best_loss = epoch_metrics["loss"]
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "loss":        best_loss,
                    "n_bands":     model.n_bands,
                    "latent_dim":  model.latent_dim,
                },
                checkpoint_path,
            )

    return history
