"""Stateless metric functions for the unified FER evaluation framework."""

# Author: Minh Quan Dang
# Created on: 10 May 2026

from __future__ import annotations
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .utils import CLASS_LABELS, CLASS_NAME_LIST


def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1 classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean of per-class F1 scores."""
    return float(f1_score(y_true, y_pred, average="macro", labels=CLASS_LABELS, zero_division=0))


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Support-weighted mean of per-class F1 scores."""
    return float(f1_score(y_true, y_pred, average="weighted", labels=CLASS_LABELS, zero_division=0))


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-class precision, recall, F1, and support."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    return pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    }, index=pd.Index(CLASS_NAME_LIST, name="class"))


def confusion_matrix_raw(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Raw confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)


def confusion_matrix_normalized(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Row-normalised confusion matrix."""
    cm = confusion_matrix_raw(y_true, y_pred).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normed = np.where(row_sums > 0, cm / row_sums, 0.0)
    return normed


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute all metrics in one call."""
    return {
        "accuracy": overall_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "weighted_f1": weighted_f1(y_true, y_pred),
        "per_class": per_class_metrics(y_true, y_pred),
        "confusion_matrix_raw": confusion_matrix_raw(y_true, y_pred),
        "confusion_matrix_normalized": confusion_matrix_normalized(y_true, y_pred),
        "n_samples": int(len(y_true)),
    }


def cross_dataset_gap(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Dict[str, float]:
    """Compute performance gap between two datasets."""
    return {
        "accuracy_gap": metrics_a["accuracy"] - metrics_b["accuracy"],
        "macro_f1_gap": metrics_a["macro_f1"] - metrics_b["macro_f1"],
        "weighted_f1_gap": metrics_a["weighted_f1"] - metrics_b["weighted_f1"],
    }