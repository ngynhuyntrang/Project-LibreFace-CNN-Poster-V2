"""Unified FER evaluation framework."""

from .evaluator import Evaluator
from .metrics import (
    compute_all_metrics,
    confusion_matrix_normalized,
    confusion_matrix_raw,
    cross_dataset_gap,
    macro_f1,
    overall_accuracy,
    per_class_metrics,
    weighted_f1,
)
from .utils import (
    CLASS_LABELS,
    CLASS_NAME_LIST,
    CLASS_NAMES,
    NUM_CLASSES,
    PROB_COLUMNS,
    REQUIRED_COLUMNS,
    load_predictions,
    validate_predictions_df,
)

__all__ = [
    "Evaluator",
    "compute_all_metrics",
    "confusion_matrix_normalized",
    "confusion_matrix_raw",
    "cross_dataset_gap",
    "macro_f1",
    "overall_accuracy",
    "per_class_metrics",
    "weighted_f1",
    "CLASS_LABELS",
    "CLASS_NAMES",
    "CLASS_NAME_LIST",
    "NUM_CLASSES",
    "PROB_COLUMNS",
    "REQUIRED_COLUMNS",
    "load_predictions",
    "validate_predictions_df",
]