"""Train SimpleCNN on RAF-DB (train/val split). Saves best checkpoint + logs + curves."""

# Author: Minh Quan Dang
# Created on: 16 May, 2026

from __future__ import annotations
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import Config, NORM_MEAN, NORM_STD
from model import SimpleCNN


# ---------- helpers ----------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rgb_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        img.load()   # force full decode before the file handle closes
    return img.convert("RGB")


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    return train_tf, eval_tf


def class_weights_from_dataset(dataset: ImageFolder, num_classes: int, device: str) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1.0."""
    counts = Counter(t for _, t in dataset.samples)
    freqs = np.array([counts[i] for i in range(num_classes)], dtype=np.float64)
    weights = freqs.sum() / (num_classes * freqs)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------- train / eval ----------

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


def plot_curves(history: dict, save_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:     cfg.num_epochs = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.lr is not None:         cfg.learning_rate = args.lr
    if args.no_class_weights:       cfg.use_class_weights = False

    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    print(f"Device: {cfg.device}")
    print(f"Train dir: {cfg.train_dir}")
    print(f"Val dir:   {cfg.val_dir}")

    train_tf, eval_tf = build_transforms(cfg.image_size)
    train_ds = ImageFolder(str(cfg.train_dir), transform=train_tf, loader=rgb_loader)
    val_ds   = ImageFolder(str(cfg.val_dir),   transform=eval_tf,  loader=rgb_loader)

    # ImageFolder sorts folder names alphabetically => "1".."7" map to indices 0..6.
    expected = ["1", "2", "3", "4", "5", "6", "7"]
    assert train_ds.classes == expected, f"Unexpected class order: {train_ds.classes}"

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    counts = Counter(t for _, t in train_ds.samples)
    dist = {i + 1: counts[i] for i in range(7)}
    print(f"Train class distribution (label 1..7): {dist}")

    model = SimpleCNN(num_classes=cfg.num_classes, dropout=cfg.dropout).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")

    if cfg.use_class_weights:
        weights = class_weights_from_dataset(train_ds, cfg.num_classes, cfg.device)
        print(f"Class weights: {weights.cpu().numpy().round(3).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    if cfg.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate,
                                    momentum=cfg.momentum, weight_decay=cfg.weight_decay,
                                    nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience, min_lr=cfg.min_lr,
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    epochs_since_improve = 0

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                          cfg.device, cfg.grad_clip)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, cfg.device)
        scheduler.step(vl_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(vl_acc)
        history["lr"].append(lr_now)

        improved = vl_acc > best_val_acc
        if improved:
            best_val_acc = vl_acc
            epochs_since_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "val_acc": vl_acc,
                "epoch": epoch,
                "classes": train_ds.classes,
            }, cfg.best_checkpoint)
        else:
            epochs_since_improve += 1

        flag = " *" if improved else ""
        print(f"Epoch {epoch:3d}/{cfg.num_epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {vl_loss:.4f} acc {vl_acc:.4f} | "
              f"lr {lr_now:.2e} | {time.time()-t0:.1f}s{flag}")

        if epochs_since_improve >= cfg.early_stopping_patience:
            print(f"Early stopping after {cfg.early_stopping_patience} epochs without val improvement.")
            break

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint saved to: {cfg.best_checkpoint}")

    with open(cfg.log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_curves(history, cfg.log_dir / "training_curves.png")
    print(f"Logs saved to: {cfg.log_dir}")


if __name__ == "__main__":
    main()