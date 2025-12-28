"""Evaluation metrics for model assessment."""
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels or probabilities.

    Returns:
        Accuracy score.
    """
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    return float(accuracy_score(y_true, y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Mean squared error.
    """
    return float(mean_squared_error(y_true, y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted") -> float:
    """Calculate precision.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging strategy. Defaults to "weighted".

    Returns:
        Precision score.
    """
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    return float(precision_score(y_true, y_pred, average=average, zero_division=0))


def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted") -> float:
    """Calculate recall.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging strategy. Defaults to "weighted".

    Returns:
        Recall score.
    """
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    return float(recall_score(y_true, y_pred, average=average, zero_division=0))


def f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted") -> float:
    """Calculate F1 score.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging strategy. Defaults to "weighted".

    Returns:
        F1 score.
    """
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def get_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
) -> Dict:
    """Get comprehensive classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels or probabilities.
        class_names: Optional list of class names.

    Returns:
        Dictionary containing overall and per-class metrics.
    """
    if y_pred.ndim > 1:
        y_pred_classes = np.argmax(y_pred, axis=-1)
    else:
        y_pred_classes = y_pred

    # Overall metrics
    overall_metrics = {
        "accuracy": accuracy(y_true, y_pred_classes),
        "precision": precision(y_true, y_pred_classes),
        "recall": recall(y_true, y_pred_classes),
        "f1": f1(y_true, y_pred_classes),
        "num_samples": len(y_true),
    }

    # Per-class metrics
    per_class_metrics = {}
    precision_vals, recall_vals, f1_vals, support_vals = precision_recall_fscore_support(
        y_true, y_pred_classes, average=None, zero_division=0
    )

    unique_classes = np.unique(y_true)
    for i, cls_idx in enumerate(unique_classes):
        cls_name = class_names[int(cls_idx)] if class_names else str(int(cls_idx))
        per_class_metrics[cls_name] = {
            "precision": float(precision_vals[i]),
            "recall": float(recall_vals[i]),
            "f1": float(f1_vals[i]),
            "num_samples": int(support_vals[i]),
        }

    return {
        "overall": overall_metrics,
        "per_class": per_class_metrics,
    }

