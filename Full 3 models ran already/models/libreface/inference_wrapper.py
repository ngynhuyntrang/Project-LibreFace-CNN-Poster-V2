"""
LibreFace inference wrapper for RAF-DB.

Usage:
    python inference_wrapper.py --config config.yaml

Requirements (minimal conda env):
    pip install libreface pyyaml pandas tqdm torch

Notes:
  - LibreFace returns only a hard predicted emotion string; no per-class probabilities
    are available from the public API.  prob_0..prob_6 are therefore one-hot vectors.
  - LibreFace is trained on AffectNet-8 (includes "Contempt").  "Contempt" is not in
    the 7-class project mapping and is remapped to Disgust (idx 1).
  - Aligned-face crops written to temp_dir are cleaned up after inference completes.
"""

import argparse
import contextlib
import io
import os
import random
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
SEED = 42


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))

COLUMNS = [
    "image_path", "true_label", "pred_label",
    "prob_0", "prob_1", "prob_2", "prob_3", "prob_4", "prob_5", "prob_6",
]

# LibreFace emotion string → project class index.
# Source: libreface/Facial_Expression_Recognition/inference.py::facial_expr_idx_to_class()
# LibreFace internal idx: 0=Neutral 1=Happiness 2=Sadness 3=Surprise 4=Fear 5=Disgust 6=Anger 7=Contempt
# Project standard:       0=Anger   1=Disgust   2=Fear    3=Happiness 4=Sadness 5=Surprise 6=Neutral
LIBREFACE_TO_PROJECT: dict[str, int] = {
    "Anger":     0,
    "Disgust":   1,
    "Fear":      2,
    "Happiness": 3,
    "Sadness":   4,
    "Surprise":  5,
    "Neutral":   6,
    "Contempt":  1,  # AffectNet-8 only; nearest project class is Disgust
}

# Original RAF-DB EmoLabel file uses 1-indexed labels with different ordering.
# Used only when data is in flat layout with the original RAF-DB label file.
_RAFDB_1IDX_TO_PROJECT = {1: 5, 2: 2, 3: 1, 4: 3, 5: 4, 6: 0, 7: 6}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve(path: str) -> str:
    """Resolve a path relative to this script's directory."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


def make_one_hot(idx: int, n: int = 7) -> list[float]:
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


@contextlib.contextmanager
def _quiet():
    """Suppress LibreFace's per-image 'Using device...' and MediaPipe prints."""
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# Image discovery — auto-detects layout
# ---------------------------------------------------------------------------
def discover_images(data_root: str) -> tuple[list[tuple[str, int]], str]:
    """
    Returns (items, layout_name) where items = list of (abs_image_path, true_label).

    Layout A — class-subdir (Khanh's organized format):
        data_root/0/*.jpg  data_root/1/*.jpg  ...  data_root/6/*.jpg
        true_label = int(subdirectory_name)

    Layout B — flat with RAF-DB label file:
        data_root/*.jpg  +  <data_root>/../EmoLabel/list_patition_label.txt
        Labels in label file are 1-indexed RAF-DB convention, remapped to project standard.
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    subdirs = sorted(
        [d for d in os.listdir(data_root)
         if os.path.isdir(os.path.join(data_root, d)) and d.isdigit()],
        key=int,
    )

    if subdirs:
        items: list[tuple[str, int]] = []
        for cls_name in subdirs:
            true_label = int(cls_name)
            cls_dir = os.path.join(data_root, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    items.append((os.path.join(cls_dir, fname), true_label))
        return items, "class-subdir"

    # --- Layout B: flat directory ---
    images = sorted(
        f for f in os.listdir(data_root)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not images:
        raise RuntimeError(
            f"No images and no numeric subdirectories found in: {data_root}\n"
            "Check that data_root in config.yaml points to the valid/ folder."
        )

    label_file = os.path.normpath(
        os.path.join(data_root, "..", "EmoLabel", "list_patition_label.txt")
    )
    if not os.path.isfile(label_file):
        raise RuntimeError(
            f"Flat image layout detected but no label file found at:\n  {label_file}\n"
            "Either reorganise data into class subdirs (0-6/) or provide the "
            "RAF-DB EmoLabel file."
        )

    label_lookup: dict[str, int] = {}
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                stem = os.path.splitext(parts[0])[0]
                label_lookup[stem] = _RAFDB_1IDX_TO_PROJECT[int(parts[1])]

    items = []
    for fname in images:
        stem = os.path.splitext(fname)[0]
        key = stem if stem in label_lookup else stem.replace("_aligned", "")
        if key in label_lookup:
            items.append((os.path.join(data_root, fname), label_lookup[key]))
        else:
            print(f"Warning: no label for {fname} — skipping")

    return items, "flat+labelfile"


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
def run_inference(
    items: list[tuple[str, int]],
    device: str,
    weights_dir: str,
) -> list[dict]:
    """
    Run LibreFace facial expression recognition on pre-aligned RAF-DB images.

    Uses libreface's internal solver_inference_image directly, bypassing the
    MediaPipe face-detection / alignment step.  RAF-DB images are already
    tightly-cropped aligned faces; MediaPipe requires surrounding context and
    raises "No face landmarks" on them.  The solver reads the image, resizes
    it to 224×224, and runs the RepVGG expression model — no detection needed.
    """
    from libreface.Facial_Expression_Recognition.inference import (
        ConfigObject,
        facial_expr_idx_to_class,
    )
    from libreface.Facial_Expression_Recognition.solver_inference_image import (
        solver_inference_image,
    )

    ckpt_path = os.path.join(
        weights_dir, "Facial_Expression_Recognition", "weights", "repvgg.pt"
    )
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # Mirror the exact config used by libreface internally (inference.py:get_facial_expression)
    opts = ConfigObject({
        "seed": 0,
        "train_csv": "training_filtered.csv",
        "test_csv": "validation_filtered.csv",
        "data_root": "",
        "ckpt_path": ckpt_path,
        "weights_download_id": "1yPBUjPhkwcIkRLt47-JJRLRD7rCwPIVU",
        "data": "AffectNet",
        "image_size": 224,
        "num_labels": 8,
        "dropout": 0.1,
        "hidden_dim": 128,
        "sigma": 10.0,
        "student_model_name": "repvgg",
        "student_model_choices": [
            "resnet_heatmap", "resnet", "swin", "mae", "emotionnet_mae", "gh_feat",
        ],
        "alpha": 1.0,
        "T": 1.0,
        "fm_distillation": True,
        "grad": True,
        "interval": 500,
        "threshold": 0.0,
        "loss": "unweighted",
        "num_epochs": 50,
        "batch_size": 256,
        "learning_rate": "3e-5",
        "weight_decay": "1e-4",
        "clip": 1.0,
        "when": 10,
        "patience": 10,
        "device": device,
    })

    print(f"Loading LibreFace expression model on {device!r} ...")
    print(f"  Weights : {ckpt_path}  (auto-downloaded ~50 MB if missing)")
    solver = solver_inference_image(opts).to(device)
    solver.student_model.eval()
    print("Model ready.\n")

    contempt_count = 0
    skipped = 0
    records: list[dict] = []

    for i, (img_path, true_label) in enumerate(tqdm(items, desc="LibreFace")):
        if i > 0 and i % 100 == 0:
            print(
                f"  [{i}/{len(items)}] processed  "
                f"({skipped} skipped, {contempt_count} Contempt→Disgust remaps)"
            )

        rel_path = os.path.relpath(img_path, start=PROJECT_ROOT)

        try:
            pred_int = solver.run(img_path)            # int 0-7 (LibreFace scale)
            emotion_str = facial_expr_idx_to_class(pred_int)

            if emotion_str not in LIBREFACE_TO_PROJECT:
                print(
                    f"Warning: unknown emotion '{emotion_str}' for "
                    f"{os.path.basename(img_path)}, defaulting to Neutral"
                )
                emotion_str = "Neutral"

            if emotion_str == "Contempt":
                contempt_count += 1

            pred_label = LIBREFACE_TO_PROJECT[emotion_str]
            # NOTE: LibreFace pip package returns only a single predicted emotion label, not per-class
            # probabilities. Prob columns are encoded as one-hot (1.0 for predicted class, 0.0 for others).
            # The unified evaluator uses these columns only for ROC/AUC — interpret LibreFace AUC results
            # with caution as they reflect hard decisions, not calibrated confidence scores.
            probs = make_one_hot(pred_label)

            record: dict = {
                "image_path": rel_path,
                "true_label": true_label,
                "pred_label": pred_label,
            }
            for j, p in enumerate(probs):
                record[f"prob_{j}"] = p

            records.append(record)

        except Exception as exc:
            skipped += 1
            print(f"Warning: skipping {os.path.basename(img_path)}: {exc}")

    print(
        f"\nDone. Processed: {len(records)} | "
        f"Skipped: {skipped} | "
        f"Contempt→Disgust remaps: {contempt_count}"
    )
    return records


# ---------------------------------------------------------------------------
# CSV validation
# ---------------------------------------------------------------------------
def validate_csv(df: pd.DataFrame) -> None:
    print("\n--- CSV Validation ---")
    expected_cols = COLUMNS
    assert list(df.columns) == expected_cols, (
        f"Column mismatch.\n  Got     : {list(df.columns)}\n"
        f"  Expected: {expected_cols}"
    )
    assert df["pred_label"].dtype == np.int64, (
        f"pred_label dtype is {df['pred_label'].dtype}, expected int64"
    )
    assert df["true_label"].dtype == np.int64, (
        f"true_label dtype is {df['true_label'].dtype}, expected int64"
    )
    assert df["pred_label"].between(0, 6).all(), "pred_label has values outside 0-6"
    assert df["true_label"].between(0, 6).all(), "true_label has values outside 0-6"

    prob_cols = [f"prob_{i}" for i in range(7)]
    sums = df[prob_cols].sum(axis=1)
    assert (sums - 1.0).abs().max() < 1e-4, "prob columns do not sum to 1.0"

    acc = (df["true_label"] == df["pred_label"]).mean() * 100
    print(f"Rows      : {len(df)}")
    print(f"Columns   : {list(df.columns)}")
    print(f"Dtypes    : true_label={df['true_label'].dtype}, pred_label={df['pred_label'].dtype}")
    print(f"Prob sums : min={sums.min():.4f}  max={sums.max():.4f}  (target: 1.0)")
    print(f"Accuracy  : {acc:.2f}%")
    print("\nSample rows (first 3):")
    print(
        df[["image_path", "true_label", "pred_label",
            "prob_0", "prob_1", "prob_2"]].head(3).to_string(index=False)
    )
    print("\nAll checks passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LibreFace on RAF-DB valid split and write a standard CSV."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config YAML (relative to this script or absolute).",
    )
    cli = parser.parse_args()

    set_seeds(SEED)

    cfg_path = resolve(cli.config)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_root    = resolve(cfg["data_root"])
    output_csv   = resolve(cfg["output_csv"])
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    temp_dir     = resolve(cfg.get("temp_dir", "./tmp"))
    weights_dir  = resolve(cfg.get("weights_dir", cfg.get("weights_download_dir", "./weights_libreface")))

    print(f"Data root  : {data_root}")
    print(f"Output CSV : {output_csv}")
    print(f"Device     : {device}")
    print(f"Temp dir   : {temp_dir}")
    print(f"Weights dir: {weights_dir}")

    items, layout = discover_images(data_root)
    print(f"Layout     : {layout}")
    print(f"Images     : {len(items)}")

    if not items:
        raise RuntimeError("No images found — check data_root in config.yaml.")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    # temp_dir is no longer used: we bypass get_facial_attributes() and call the
    # expression solver directly, so no aligned-crop temp files are written.
    records = run_inference(items, device, weights_dir)

    if not records:
        raise RuntimeError("No records produced — all images were skipped.")

    df = pd.DataFrame(records, columns=COLUMNS)
    df["true_label"] = df["true_label"].astype(int)
    df["pred_label"] = df["pred_label"].astype(int)
    df.to_csv(output_csv, index=False)

    validate_csv(df)
    print(f"\nCSV saved: {output_csv}")


if __name__ == "__main__":
    main()
