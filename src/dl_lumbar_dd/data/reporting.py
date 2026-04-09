"""EDA and preprocessing reporting outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dl_lumbar_dd.constants import TARGET_COLUMNS
from dl_lumbar_dd.data.dicom import build_three_view_tensor
from dl_lumbar_dd.data.ingest import RSNATables
from dl_lumbar_dd.data.splits import SplitManifests
from dl_lumbar_dd.utils.io import ensure_dir


def generate_eda_reports(bundle: RSNATables, figures_root: str | Path) -> list[Path]:
    """Create English figures and summary tables for dataset exploration."""
    root = ensure_dir(figures_root)
    outputs = [
        _plot_class_distribution(bundle.train, root / "class_distribution.png"),
        _plot_missing_values(bundle.train, root / "missing_values.png"),
        _plot_series_distribution(bundle.series, root / "series_distribution.png"),
        _write_eda_summary(bundle, root / "eda_summary.csv"),
    ]
    return outputs


def generate_preprocess_reports(
    bundle: RSNATables,
    manifests: SplitManifests,
    figures_root: str | Path,
    image_size: int = 224,
) -> list[Path]:
    """Create preprocessing visuals for split quality and representative tensors."""
    root = ensure_dir(figures_root)
    outputs = [_plot_split_distribution(manifests, root / "split_distribution.png")]
    preview_ids = manifests.train["study_id"].head(3).tolist()
    if preview_ids:
        outputs.append(_plot_preprocess_preview(bundle, preview_ids, root / "preprocessing_preview.png", image_size))
    return outputs


def _plot_class_distribution(train: pd.DataFrame, path: Path) -> Path:
    counts = train.loc[:, TARGET_COLUMNS].stack().fillna("Missing").value_counts().reset_index()
    counts.columns = ["severity", "count"]
    return _barplot(counts, x="severity", y="count", title="Class Distribution Across Targets", path=path)


def _plot_missing_values(train: pd.DataFrame, path: Path) -> Path:
    missing = train.loc[:, TARGET_COLUMNS].isna().mean().mul(100).sort_values(ascending=False).reset_index()
    missing.columns = ["target", "missing_pct"]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=missing, x="target", y="missing_pct", color="#D95F02")
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Missing Values (%)")
    plt.xlabel("Target")
    plt.title("Missing Values by Target")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _plot_series_distribution(series: pd.DataFrame, path: Path) -> Path:
    distribution = series["series_description"].value_counts().reset_index()
    distribution.columns = ["series_description", "count"]
    return _barplot(
        distribution,
        x="series_description",
        y="count",
        title="Series Distribution",
        path=path,
        color="#1B9E77",
    )


def _write_eda_summary(bundle: RSNATables, path: Path) -> Path:
    summary = pd.DataFrame(
        [
            {"metric": "study_count", "value": bundle.train["study_id"].nunique()},
            {"metric": "series_count", "value": len(bundle.series)},
            {"metric": "coordinate_count", "value": len(bundle.coordinates)},
            {"metric": "avg_series_per_study", "value": round(len(bundle.series) / bundle.train["study_id"].nunique(), 2)},
        ]
    )
    summary.to_csv(path, index=False)
    return path


def _plot_split_distribution(manifests: SplitManifests, path: Path) -> Path:
    split_counts = pd.DataFrame(
        {
            "split": ["train", "validation"],
            "study_count": [len(manifests.train), len(manifests.validation)],
        }
    )
    return _barplot(split_counts, x="split", y="study_count", title="Study Split Distribution", path=path, color="#7570B3")


def _plot_preprocess_preview(bundle: RSNATables, study_ids: list[int], path: Path, image_size: int) -> Path:
    figure, axes = plt.subplots(len(study_ids), 3, figsize=(9, 3 * len(study_ids)))
    axes_grid = axes if len(study_ids) > 1 else [axes]
    for row_index, study_id in enumerate(study_ids):
        tensor = build_three_view_tensor(bundle.dataset_root, study_id=study_id, series_table=bundle.series, image_size=image_size)
        row_axes = axes_grid[row_index]
        for col_index, axis in enumerate(row_axes):
            axis.imshow(tensor[col_index], cmap="gray")
            axis.set_title(f"Study {study_id} - View {col_index + 1}")
            axis.axis("off")
    figure.suptitle("Representative Three-View Preprocessing Preview", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _barplot(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    path: Path,
    color: str = "#4C78A8",
) -> Path:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=frame, x=x, y=y, color=color)
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path
