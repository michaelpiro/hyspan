"""Training loop for THANTD."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .model import THANTD
from .loss  import ETBLoss


def train_thantd(
    model:            THANTD,
    dataset,
    epochs:           int   = 50,
    batch_size:       int   = 256,
    lr:               float = 1e-4,
    weight_decay:     float = 1e-4,
    device:           str   = "cuda",
    margin:           float = 0.3,
    lambda_bce:       float = 0.5,
    num_workers:      int   = 0,
    print_every:      int   = 5,
    checkpoint_path:  str | None = None,
) -> list[dict]:
    """
    Train THANTD with ETB-Loss (cosine annealing LR schedule, AdamW optimiser).

    Args:
        model:           THANTD instance (will be moved to device)
        dataset:         TripletDataset (or any Dataset yielding anchor/pos/neg)
        epochs:          number of training epochs
        batch_size:      triplets per batch
        lr:              initial learning rate
        weight_decay:    L2 regularisation
        device:          'cuda', 'mps', or 'cpu'
        margin:          triplet margin δ
        lambda_bce:      weight for BCE component in ETB-Loss
        num_workers:     DataLoader workers
        print_every:     print progress every N epochs
        checkpoint_path: save best checkpoint here (None = no saving)

    Returns:
        history: list of per-epoch metric dicts
    """
    model   = model.to(device)
    loss_fn = ETBLoss(margin=margin, lambda_bce=lambda_bce)
    optim   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=lr * 0.01)

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
        totals    = dict(loss=0.0, loss_dual=0.0, loss_bce=0.0, d_pos=0.0, d_neg=0.0, margin_viol=0.0)
        n_batches = 0

        for anc, pos, neg in loader:
            anc = anc.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            a_feat, p_feat, n_feat = model(anc, pos, neg)
            loss, metrics          = loss_fn(a_feat, p_feat, n_feat)

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
                f"dual={epoch_metrics['loss_dual']:.4f}  "
                f"bce={epoch_metrics['loss_bce']:.4f}  "
                f"d+={epoch_metrics['d_pos']:.3f}  "
                f"d-={epoch_metrics['d_neg']:.3f}  "
                f"viol={epoch_metrics['margin_viol']:.2f}"
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
                    "d_model":     model.d_model,
                },
                checkpoint_path,
            )

    return history
