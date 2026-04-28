"""
FairVision - Training Module
=============================
Handles:
- Single epoch training loop
- Validation loop
- Full training with checkpointing and LR scheduling
- Training curve plotting
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List, Optional


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Returns:
        avg_loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, ages, _genders, _races in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        ages   = ages.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, ages)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct       += predicted.eq(ages).sum().item()
        total          += ages.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Run validation / evaluation pass.

    Returns:
        avg_loss, accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, ages, _genders, _races in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        ages   = ages.to(device, non_blocking=True)

        outputs = model(images)
        loss    = criterion(outputs, ages)

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct       += predicted.eq(ages).sum().item()
        total          += ages.size(0)

    return running_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int = 20,
    save_path: str = "models/best.pth",
    model_name: str = "Model"
) -> dict:
    """
    Full training loop with validation, checkpointing and LR scheduling.

    Returns:
        history dict with train_losses, val_losses, train_accs, val_accs
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    history = {
        "train_losses": [], "val_losses": [],
        "train_accs":   [], "val_accs":   []
    }
    best_val_acc = 0.0

    print(f"\n{'='*55}")
    print(f"  Training: {model_name}  |  Epochs: {epochs}  |  Device: {device}")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_losses"].append(tr_loss)
        history["val_losses"].append(vl_loss)
        history["train_accs"].append(tr_acc)
        history["val_accs"].append(vl_acc)

        # Checkpoint best model
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)
            flag = " ✓ saved"
        else:
            flag = ""

        print(
            f"  Epoch [{epoch:02d}/{epochs}]  "
            f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc*100:.2f}%  "
            f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc*100:.2f}%{flag}"
        )

    print(f"\n  Best Val Accuracy: {best_val_acc*100:.2f}%  →  saved to {save_path}")
    return history


def plot_training_curves(history: dict, title: str = "Training Curves",
                          save_path: Optional[str] = None):
    """Plot and optionally save loss + accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    epochs = range(1, len(history["train_losses"]) + 1)

    ax1.plot(epochs, history["train_losses"], "o-", label="Train Loss", color="steelblue")
    ax1.plot(epochs, history["val_losses"],   "o-", label="Val Loss",   color="coral")
    ax1.set_title(f"Loss — {title}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a*100 for a in history["train_accs"]], "o-", label="Train Acc", color="steelblue")
    ax2.plot(epochs, [a*100 for a in history["val_accs"]],   "o-", label="Val Acc",   color="coral")
    ax2.set_title(f"Accuracy — {title}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Curve saved → {save_path}")

    plt.show()
