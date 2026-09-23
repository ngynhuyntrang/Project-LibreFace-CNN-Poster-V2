"""Hyperparameters and paths for the simple CNN baseline."""

# Author: Minh Quan Dang
# Created on: 15 May, 2026

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import torch


# Class mapping: 0-based labels (project standard).
CLASS_NAMES: dict[int, str] = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Sadness",
    5: "Surprise",
    6: "Neutral",
}

# ImageNet normalization stats. A reasonable default even for from-scratch training.
NORM_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
NORM_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class Config:
    # Image / data
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4

    pin_memory: bool = True

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"   # "adamw" or "sgd"
    momentum: float = 0.9       # used only when optimizer="sgd"

    # Loss
    label_smoothing: float = 0.05
    use_class_weights: bool = False

    # Scheduler (ReduceLROnPlateau on val loss)
    scheduler_factor: float = 0.5
    scheduler_patience: int = 8
    min_lr: float = 1e-6

    # Training loop
    num_epochs: int = 80
    early_stopping_patience: int = 20
    grad_clip: float = 1.0

    # Model
    num_classes: int = 7
    dropout: float = 0.3

    # Misc
    seed: int = 42
    device: str = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    # Derived paths (resolved in __post_init__)
    project_root: Path = field(init=False)
    data_root: Path = field(init=False)
    train_dir: Path = field(init=False)
    val_dir: Path = field(init=False)
    test_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    predictions_dir: Path = field(init=False)
    best_checkpoint: Path = field(init=False)
    predictions_csv: Path = field(init=False)

    def __post_init__(self) -> None:
        # config.py lives at: <root>/models/simple_cnn/config.py
        self.project_root = Path(__file__).resolve().parents[2]

        self.data_root = self.project_root / "data" / "raf_db"
        self.train_dir = self.data_root / "train"
        self.val_dir = self.data_root / "valid"
        self.test_dir = self.data_root / "test"

        self.model_dir = self.project_root / "models" / "simple_cnn"
        self.checkpoint_dir = self.model_dir / "checkpoints"
        self.log_dir = self.model_dir / "logs"
        self.predictions_dir = self.project_root / "predictions"

        self.best_checkpoint = self.checkpoint_dir / "simple_cnn_best.pt"
        self.predictions_csv = self.predictions_dir / "simple_cnn_raf_db.csv"

        for d in (self.checkpoint_dir, self.log_dir, self.predictions_dir):
            d.mkdir(parents=True, exist_ok=True)