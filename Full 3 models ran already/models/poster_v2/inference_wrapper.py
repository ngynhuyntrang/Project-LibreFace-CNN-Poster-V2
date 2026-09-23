"""
POSTER_V2 inference wrapper.
Usage:
    python inference_wrapper.py --config config.yaml

All paths in config.yaml are relative to this script's directory.
"""
import os
import sys
import glob
import argparse
import random

import numpy as np
import yaml
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from models.PosterV2_7cls import pyramid_trans_expr2

COLUMNS = [
    "image_path", "true_label", "pred_label",
    "prob_0", "prob_1", "prob_2", "prob_3", "prob_4", "prob_5", "prob_6",
]

# ---------------------------------------------------------------------------
# Pickle stubs — required so torch.load can deserialise POSTER_V2 checkpoints.
# main.py saves RecorderMeter and RecorderMeter1 instances inside the checkpoint
# dict. Pickle looks them up by name in __main__ when loading; they must exist
# here even though we only use state_dict and throw the rest away.
# ---------------------------------------------------------------------------
class RecorderMeter:        # noqa: E302
    pass

class RecorderMeter1:       # noqa: E302
    pass
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Checkpoint label remapping
# The official POSTER_V2 RAF-DB checkpoint was trained with RAF-DB's original
# 0-indexed class ordering (derived from its annotation file by subtracting 1):
#   Ckpt: 0=Surprise 1=Fear 2=Disgust 3=Happiness 4=Sadness 5=Anger 6=Neutral
# Our project standard:
#   Proj: 0=Anger 1=Disgust 2=Fear 3=Happiness 4=Sadness 5=Surprise 6=Neutral
#
# CKPT_TO_PROJECT[i] = the project index that checkpoint output i maps to.
# Applied to both argmax pred_label and softmax prob column ordering.
# ---------------------------------------------------------------------------
CKPT_TO_PROJECT = [5, 2, 1, 3, 4, 0, 6]


def resolve(path: str) -> str:
    """Resolve a config-relative path to absolute."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


def find_checkpoint(pattern: str) -> str:
    """Return first match for a glob pattern; raise if nothing found."""
    matches = sorted(glob.glob(resolve(pattern)))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for pattern: {pattern}\n"
            "Run training first, then update checkpoint in config.yaml."
        )
    return matches[-1]  # latest alphabetically (timestamps sort correctly)


def load_model(checkpoint_path: str, device: torch.device):
    model = pyramid_trans_expr2(img_size=224, num_classes=7)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def run_inference(model, valid_dir: str, device: torch.device, transform):
    records = []
    class_dirs = sorted(
        [d for d in os.listdir(valid_dir)
         if os.path.isdir(os.path.join(valid_dir, d)) and d.isdigit()],
        key=int,
    )
    if not class_dirs:
        raise RuntimeError(
            f"No numeric class subdirectories found in {valid_dir}.\n"
            "Expected layout: valid/0/, valid/1/, ..., valid/6/"
        )

    project_root = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))

    for cls_name in class_dirs:
        true_label = int(cls_name)
        cls_path = os.path.join(valid_dir, cls_name)
        images = [f for f in os.listdir(cls_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for img_name in tqdm(images, desc=f"Class {cls_name}", leave=False):
            img_path = os.path.join(cls_path, img_name)
            rel_path = os.path.relpath(img_path, start=project_root)

            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(tensor)
                    raw_probs = F.softmax(logits, dim=1).cpu().squeeze().tolist()
                    raw_pred  = int(torch.argmax(logits, dim=1).item())

                # Remap checkpoint indices → project indices.
                # Checkpoint uses RAF-DB original ordering; project uses ours.
                pred_label = CKPT_TO_PROJECT[raw_pred]
                probs = [0.0] * 7
                for ckpt_i, proj_i in enumerate(CKPT_TO_PROJECT):
                    probs[proj_i] = raw_probs[ckpt_i]

                record = {
                    "image_path": rel_path,
                    "true_label": true_label,
                    "pred_label": pred_label,
                }
                for i, p in enumerate(probs):
                    record[f"prob_{i}"] = round(float(p), 6)

                records.append(record)

            except Exception as e:
                print(f"Warning: skipping {img_path}: {e}")

    return records


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml (relative to this script)")
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print("Backend: Apple Metal Performance Shaders (MPS)")

    checkpoint_path = find_checkpoint(cfg["checkpoint"])
    print(f"Checkpoint: {checkpoint_path}")

    valid_dir = os.path.join(resolve(cfg["data_dir"]), cfg.get("valid_split", "valid"))
    print(f"Validation dir: {valid_dir}")

    output_csv = resolve(cfg["output_csv"])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    transform = get_transform()
    model = load_model(checkpoint_path, device)
    print(f"Model loaded from {checkpoint_path}")

    records = run_inference(model, valid_dir, device, transform)

    df = pd.DataFrame(records, columns=COLUMNS)
    df["true_label"] = df["true_label"].astype(int)
    df["pred_label"] = df["pred_label"].astype(int)
    df.to_csv(output_csv, index=False)

    acc = (df["true_label"] == df["pred_label"]).mean() * 100
    print(f"\nInference complete.")
    print(f"  Images  : {len(df)}")
    print(f"  Accuracy: {acc:.2f}%")
    print(f"  CSV     : {output_csv}")


if __name__ == "__main__":
    main()
