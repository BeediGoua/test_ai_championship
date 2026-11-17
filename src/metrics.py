"""Validated binary-classification metrics and ROC helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve


def _validate_binary_arrays(
    y_true: Any, values: Any, *, name: str
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    scores = np.asarray(values)

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and values must be one-dimensional.")
    if labels.size == 0 or labels.size != scores.size:
        raise ValueError("y_true and values must have the same non-zero length.")
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError("y_true must contain both binary labels 0 and 1.")

    try:
        numeric_values = scores.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values.") from error
    if not np.isfinite(numeric_values).all():
        raise ValueError(f"{name} must contain only finite values.")

    return labels.astype(int), numeric_values


def roc_metrics(y_true: Any, y_score: Any) -> dict[str, np.ndarray | float]:
    """Return ROC false/true positive rates, thresholds, and area under curve."""
    labels, scores = _validate_binary_arrays(y_true, y_score, name="y_score")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": float(auc(fpr, tpr)),
    }


def confusion_rates(y_true: Any, y_pred: Any) -> dict[str, float | int]:
    """Return FPR, TPR, and confusion counts for binary predictions."""
    labels, predictions = _validate_binary_arrays(
        y_true, y_pred, name="y_pred"
    )
    if not np.isin(predictions, [0, 1]).all():
        raise ValueError("y_pred must contain only binary labels 0 and 1.")

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions.astype(int), labels=[0, 1]
    ).ravel()
    return {
        "fpr": float(fp / (fp + tn)),
        "tpr": float(tp / (tp + fn)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
