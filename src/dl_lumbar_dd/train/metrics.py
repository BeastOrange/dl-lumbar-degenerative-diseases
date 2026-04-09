"""Training metric helpers."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    if not targets:
        return {"macro_f1": 0.0, "accuracy": 0.0}
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(targets, predictions)),
    }


def metric_row(epoch: int, learning_rate: float, prefix_to_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {"epoch": epoch, "lr": learning_rate}
    for prefix, metrics in prefix_to_metrics.items():
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return row
