"""Plotting helpers for evaluation and experiment reporting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

from dl_lumbar_dd.eval.metrics import safe_macro_auc
from dl_lumbar_dd.utils.io import ensure_dir

sns.set_theme(style="whitegrid")


def save_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    output_path: str | Path,
    class_names: Sequence[str] | None = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> Path:
    """Render and save a confusion matrix figure."""
    labels = list(range(len(class_names))) if class_names is not None else None
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else "g",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_title(title)

    destination = Path(output_path)
    ensure_dir(destination.parent)
    figure.tight_layout()
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_multiclass_roc(
    y_true: Sequence[int],
    y_score: Sequence[Sequence[float]],
    output_path: str | Path,
    class_names: Sequence[str] | None = None,
    title: str = "One-vs-Rest ROC Curves",
) -> Path:
    """Render and save one-vs-rest ROC curves, skipping unsupported classes."""
    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_score_array = np.asarray(y_score, dtype=np.float64)
    names = list(class_names) if class_names is not None else [f"Class {idx}" for idx in range(y_score_array.shape[1])]

    figure, axis = plt.subplots(figsize=(7, 5))
    plotted = 0
    for class_index in range(y_score_array.shape[1]):
        binary_true = (y_true_array == class_index).astype(np.int64)
        if binary_true.min() == binary_true.max():
            continue
        fpr, tpr, _ = roc_curve(binary_true, y_score_array[:, class_index])
        axis.plot(fpr, tpr, label=names[class_index])
        plotted += 1

    if plotted == 0:
        axis.text(0.5, 0.5, "ROC is unavailable for single-class validation data.", ha="center", va="center")
    else:
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
        macro_auc = safe_macro_auc(y_true_array, y_score_array)
        if macro_auc is not None:
            axis.set_title(f"{title} (Macro AUC = {macro_auc:.3f})")
        else:
            axis.set_title(title)
        axis.legend(loc="lower right")

    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    if plotted == 0:
        axis.set_title(title)

    destination = Path(output_path)
    ensure_dir(destination.parent)
    figure.tight_layout()
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_training_history(
    history: Mapping[str, Sequence[float]],
    output_path: str | Path,
    title: str = "Training History",
) -> Path:
    """Render numeric training history series to a single figure."""
    numeric_series = {
        key: [float(value) for value in values]
        for key, values in history.items()
        if values and all(isinstance(value, (int, float)) for value in values)
    }

    figure, axis = plt.subplots(figsize=(8, 5))
    if not numeric_series:
        axis.text(0.5, 0.5, "No numeric history is available.", ha="center", va="center")
    else:
        for key, values in numeric_series.items():
            epochs = list(range(1, len(values) + 1))
            axis.plot(epochs, values, marker="o", linewidth=1.8, label=key)
        axis.legend(loc="best")

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Metric Value")
    axis.set_title(title)

    destination = Path(output_path)
    ensure_dir(destination.parent)
    figure.tight_layout()
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination
