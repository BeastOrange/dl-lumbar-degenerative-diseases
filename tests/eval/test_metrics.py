from __future__ import annotations

import numpy as np

from dl_lumbar_dd.eval import compute_classification_metrics, safe_log_loss, safe_macro_auc


def test_compute_classification_metrics_with_probability_scores() -> None:
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])
    y_score = np.array(
        [
            [0.90, 0.08, 0.02],
            [0.10, 0.80, 0.10],
            [0.15, 0.60, 0.25],
            [0.05, 0.90, 0.05],
            [0.70, 0.25, 0.05],
            [0.05, 0.15, 0.80],
        ]
    )

    metrics = compute_classification_metrics(y_true, y_pred, y_score)

    assert metrics["accuracy"] == 5 / 6
    assert metrics["macro_f1"] > 0.8
    assert metrics["weighted_f1"] > 0.8
    assert metrics["macro_auc"] is not None
    assert metrics["log_loss"] is not None


def test_safe_metrics_return_none_for_single_class_validation() -> None:
    y_true = np.array([1, 1, 1])
    y_score = np.array(
        [
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
        ]
    )

    assert safe_macro_auc(y_true, y_score) is None
    assert safe_log_loss(y_true, y_score) is None


def test_compute_classification_metrics_rejects_mismatched_shapes() -> None:
    y_true = np.array([0, 1])
    y_pred = np.array([0])

    try:
        compute_classification_metrics(y_true, y_pred)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched shapes.")
