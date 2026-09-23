"""Constants, validation, and IO helpers for the unified FER evaluation framework."""

# Author: Minh Quan Dang
# Created on: 10 May 2026

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

# Canonical class mapping (1-7). Every model must follow this exactly.
CLASS_NAMES: dict[int, str] = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral"
}

NUM_CLASSES: int = len(CLASS_NAMES)
CLASS_LABELS: List[int] = sorted(CLASS_NAMES.keys())          # [1, 2, 3, 4, 5, 6, 7]
CLASS_NAME_LIST: List[str] = [CLASS_NAMES[i] for i in CLASS_LABELS]

# Standard CSV schema
PROB_COLUMNS: List[str] = [f"prob_{i}" for i in CLASS_LABELS]
REQUIRED_COLUMNS: List[str] = ["image_path", "true_label", "pred_label"] + PROB_COLUMNS

PathLike = Union[str, Path]


def validate_predictions_df(
    df: pd.DataFrame,
    *,
    strict_probs: bool = False,
    prob_tolerance: float = 1e-3,
) -> None:
    """Validate predictions DataFrame against the standard schema."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected: {REQUIRED_COLUMNS}")

    for col in ("true_label", "pred_label"):
        if not pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(int)

        out_of_range = df[~df[col].between(1, 7)]
        if len(out_of_range) > 0:
            raise ValueError(
                f"Column '{col}' contains values outside [1, 7]. "
                f"First offending row:\n{out_of_range.iloc[0].to_dict()}"
            )

    if strict_probs:
        prob_sums = df[PROB_COLUMNS].sum(axis=1).to_numpy()
        bad = np.where(~np.isclose(prob_sums, 1.0, atol=prob_tolerance))[0]
        if len(bad) > 0:
            first = bad[0]
            raise ValueError(
                f"{len(bad)} rows have probability sums outside 1.0 +/- {prob_tolerance}. "
                f"First row sum = {prob_sums[first]:.6f}"
            )


def load_predictions(
    csv_path: PathLike,
    *,
    strict_probs: bool = False,
) -> pd.DataFrame:
    """Load and validate a predictions CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    validate_predictions_df(df, strict_probs=strict_probs)
    return df


def ensure_dir(path: PathLike) -> Path:
    """Create directory if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def array_pair_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) arrays."""
    return df["true_label"].to_numpy(dtype=int), df["pred_label"].to_numpy(dtype=int)


def probabilities_from_df(df: pd.DataFrame) -> np.ndarray:
    """Extract probability matrix."""
    return df[PROB_COLUMNS].to_numpy(dtype=float)