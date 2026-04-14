"""Training metric helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, recall_score


def classification_metrics(
    targets: list[int],
    predictions: list[int],
    num_classes: int | None = None,
) -> dict[str, float]:
    if num_classes is None:
        num_classes = max([*targets, *predictions], default=-1) + 1

    metrics = (
        {
            "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(targets, predictions)),
        }
        if targets
        else {"macro_f1": 0.0, "accuracy": 0.0}
    )
    if num_classes <= 0:
        return metrics

    labels = list(range(num_classes))
    recalls = (
        recall_score(targets, predictions, labels=labels, average=None, zero_division=0)
        if targets
        else [0.0] * num_classes
    )
    prediction_counts = Counter(predictions)
    prediction_total = float(len(predictions))
    for class_index, recall in enumerate(recalls):
        metrics[f"recall_class_{class_index}"] = float(recall)
        metrics[f"pred_rate_{class_index}"] = (
            float(prediction_counts.get(class_index, 0) / prediction_total) if prediction_total else 0.0
        )
    return metrics


def metric_row(epoch: int, learning_rate: float, prefix_to_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {"epoch": epoch, "lr": learning_rate}
    for prefix, metrics in prefix_to_metrics.items():
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return row
