"""Evaluation metric helpers for classification experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


ArrayLike = Sequence[int] | np.ndarray
FloatArrayLike = Sequence[Sequence[float]] | np.ndarray


def _as_int_array(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError("Expected a 1D label array.")
    return array


def _as_score_array(values: FloatArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Expected a 2D score array.")
    return array


def safe_macro_auc(y_true: ArrayLike, y_score: FloatArrayLike | None) -> float | None:
    """Compute one-vs-rest macro AUC when enough class variation exists."""
    if y_score is None:
        return None

    y_true_array = _as_int_array(y_true)
    y_score_array = _as_score_array(y_score)
    if y_score_array.shape[0] != y_true_array.shape[0]:
        raise ValueError("Label and score rows must match.")

    auc_values: list[float] = []
    for class_index in range(y_score_array.shape[1]):
        binary_true = (y_true_array == class_index).astype(np.int64)
        if binary_true.min() == binary_true.max():
            continue
        auc_values.append(float(roc_auc_score(binary_true, y_score_array[:, class_index])))

    if not auc_values:
        return None
    return float(np.mean(auc_values))


def safe_log_loss(y_true: ArrayLike, y_score: FloatArrayLike | None) -> float | None:
    """Compute log loss if class probabilities cover at least two labels."""
    if y_score is None:
        return None

    y_true_array = _as_int_array(y_true)
    y_score_array = _as_score_array(y_score)
    if y_score_array.shape[0] != y_true_array.shape[0]:
        raise ValueError("Label and score rows must match.")
    if np.unique(y_true_array).size < 2:
        return None

    labels = list(range(y_score_array.shape[1]))
    try:
        return float(log_loss(y_true_array, y_score_array, labels=labels))
    except ValueError:
        return None


def compute_classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: FloatArrayLike | None = None,
) -> dict[str, float | None]:
    """Return a JSON-friendly metric bundle for multiclass classification."""
    y_true_array = _as_int_array(y_true)
    y_pred_array = _as_int_array(y_pred)
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    return {
        "macro_f1": float(f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true_array, y_pred_array, average="weighted", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "macro_auc": safe_macro_auc(y_true_array, y_score),
        "log_loss": safe_log_loss(y_true_array, y_score),
    }
