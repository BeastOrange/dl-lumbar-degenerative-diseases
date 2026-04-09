"""High-level evaluation and comparison commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from dl_lumbar_dd.eval.comparison import build_ranking_table, save_ranking_table
from dl_lumbar_dd.utils.io import ensure_dir, write_json
from dl_lumbar_dd.visualization import save_confusion_matrix, save_multiclass_roc, save_training_history

matplotlib.use("Agg")


def run_evaluation(run_dir: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Build evaluation outputs for one run directory."""
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found in run directory: {run_path}")
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        raise ValueError(f"metrics.csv is empty: {metrics_path}")

    destination = ensure_dir(Path(output_root) / run_path.name)
    history_plot = save_training_history(
        _frame_to_series(metrics),
        destination / "training_history.png",
        title=f"Training History - {run_path.name}",
    )
    outputs: list[str] = [str(history_plot)]
    prediction_outputs = _maybe_build_prediction_outputs(run_path, destination)
    outputs.extend(prediction_outputs)

    summary = _build_run_summary(run_path, metrics, outputs)
    write_json(summary, destination / "evaluation_summary.json")
    return summary


def run_comparison(
    runs_root: str | Path,
    output_root: str | Path,
    primary_metric: str = "val_macro_f1",
) -> dict[str, Any]:
    """Build cross-run ranking outputs from all available runs."""
    ranking_table = build_ranking_table(runs_root, primary_metric=primary_metric)
    destination = ensure_dir(output_root)
    csv_path = destination / "model_ranking.csv"
    json_path = destination / "model_ranking.json"
    save_ranking_table(ranking_table, output_csv=csv_path, output_json=json_path)
    ranking_plot = _plot_ranking_bar(ranking_table, destination / "model_ranking.png", primary_metric)
    return {
        "runs_root": str(runs_root),
        "primary_metric": primary_metric,
        "count": len(ranking_table),
        "outputs": [str(csv_path), str(json_path), str(ranking_plot)],
    }


def _frame_to_series(frame: pd.DataFrame) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    for column in frame.columns:
        if column == "epoch":
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if numeric.empty:
            continue
        series[column] = numeric.astype(float).tolist()
    return series


def _maybe_build_prediction_outputs(run_path: Path, output_dir: Path) -> list[str]:
    prediction_path = run_path / "predictions.csv"
    if not prediction_path.exists():
        return []
    frame = pd.read_csv(prediction_path)
    required_columns = {"y_true", "y_pred"}
    if not required_columns.issubset(frame.columns):
        return []

    confusion = save_confusion_matrix(
        frame["y_true"].astype(int).tolist(),
        frame["y_pred"].astype(int).tolist(),
        output_dir / "confusion_matrix.png",
        class_names=["Normal/Mild", "Moderate", "Severe"],
        normalize=True,
        title="Confusion Matrix (Normalized)",
    )
    score_columns = [column for column in frame.columns if column.startswith("score_")]
    if not score_columns:
        return [str(confusion)]

    scores = frame[score_columns].to_numpy(dtype=float)
    roc = save_multiclass_roc(
        frame["y_true"].astype(int).tolist(),
        scores,
        output_dir / "roc_ovr.png",
        class_names=["Normal/Mild", "Moderate", "Severe"],
    )
    return [str(confusion), str(roc)]


def _build_run_summary(run_path: Path, metrics: pd.DataFrame, outputs: list[str]) -> dict[str, Any]:
    metric_name = "val_macro_f1" if "val_macro_f1" in metrics.columns else metrics.columns[-1]
    best_idx = metrics[metric_name].astype(float).idxmax()
    best_row = metrics.iloc[int(best_idx)].to_dict()
    final_row = metrics.iloc[-1].to_dict()
    return {
        "run_id": run_path.name,
        "run_dir": str(run_path),
        "best_epoch": int(best_row.get("epoch", 0)),
        "best_metric_name": metric_name,
        "best_metric_value": float(best_row.get(metric_name, 0.0)),
        "final_metrics": final_row,
        "outputs": outputs,
    }


def _plot_ranking_bar(ranking: list[dict[str, Any]], output_path: Path, primary_metric: str) -> Path:
    names = [item["run_id"] for item in ranking[:12]]
    scores = [float(item["primary_score"]) if item["primary_score"] is not None else 0.0 for item in ranking[:12]]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(names, scores, color="#2F6A8F")
    axis.set_title(f"Model Ranking by {primary_metric}")
    axis.set_xlabel("Run ID")
    axis.set_ylabel("Score")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path
