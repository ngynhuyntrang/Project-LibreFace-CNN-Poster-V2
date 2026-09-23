"""Run inference on the RAF-DB test set and write predictions/simple_cnn.csv."""

# Author: Minh Quan Dang
# Created on: 17 May, 2026

from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import Config, NORM_MEAN, NORM_STD
from model import SimpleCNN

# The checkpoint was trained on data/raf_db/processed/ (folders 1-7, RAF-DB original ordering).
# ImageFolder sorted those as ["1".."7"] → internal idx 0-6 = Surprise,Fear,Disgust,Happiness,Sadness,Anger,Neutral.
# Project standard: 0=Anger 1=Disgust 2=Fear 3=Happiness 4=Sadness 5=Surprise 6=Neutral.
# MODEL_TO_PROJECT[i] gives the project label for model output index i.
MODEL_TO_PROJECT: list[int] = [5, 2, 1, 3, 4, 0, 6]


def rgb_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        img.load()   # force full decode before the file handle closes
    return img.convert("RGB")


def to_relative_path(abs_path: str, split: str) -> str:
    """
    Build a forward-slash relative path.
    Layout: data/raf_db/<split>/<class_folder>/<filename>
    """
    p = Path(abs_path)
    return f"data/raf_db/{split}/{p.parent.name}/{p.name}"


@torch.no_grad()
def run_inference(cfg: Config, split: str) -> pd.DataFrame:
    split_dir = {"train": cfg.train_dir, "val": cfg.val_dir, "test": cfg.test_dir}[split]

    eval_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    dataset = ImageFolder(str(split_dir), transform=eval_tf, loader=rgb_loader)
    expected = ["0", "1", "2", "3", "4", "5", "6"]
    assert dataset.classes == expected, f"Unexpected class order: {dataset.classes}"

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    model = SimpleCNN(num_classes=cfg.num_classes, dropout=cfg.dropout).to(cfg.device)
    ckpt = torch.load(cfg.best_checkpoint, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val acc {ckpt['val_acc']:.4f})")

    all_true: List[int] = []
    all_pred: List[int] = []
    all_probs: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(1)       # idx 0..6 = project labels 0-6
        all_pred.extend(preds.tolist())
        all_true.extend(y.numpy().tolist())
        all_probs.append(probs)

    probs_arr = np.concatenate(all_probs, axis=0)
    rel_paths = [to_relative_path(p, split) for p, _ in dataset.samples]

    # Remap model output indices to project-standard labels.
    pred_proj = [MODEL_TO_PROJECT[p] for p in all_pred]

    # Remap probability columns: new_probs[:, project_idx] = old_probs[:, model_idx]
    probs_proj = np.zeros_like(probs_arr)
    for model_idx, proj_idx in enumerate(MODEL_TO_PROJECT):
        probs_proj[:, proj_idx] = probs_arr[:, model_idx]

    df = pd.DataFrame({
        "image_path": rel_paths,
        "true_label": all_true,   # ImageFolder on test/0-6 → already project-standard
        "pred_label": pred_proj,
    })
    # prob_0..prob_6 (project standard: 0=Anger … 6=Neutral).
    for i in range(7):
        df[f"prob_{i}"] = probs_proj[:, i]
    return df


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path. Defaults to predictions/simple_cnn.csv")
    args = parser.parse_args()

    cfg = Config()
    out_path = Path(args.output) if args.output else cfg.predictions_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = run_inference(cfg, split=args.split)
    df.to_csv(out_path, index=False)

    acc = (df["true_label"] == df["pred_label"]).mean() * 100
    print(f"Saved {len(df)} predictions to: {out_path}")
    print(f"{args.split.capitalize()} accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()