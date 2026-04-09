"""Command-level entry points for EDA and preprocessing."""

from __future__ import annotations

from pathlib import Path

from dl_lumbar_dd.data.ingest import build_study_index, load_rsna_tables
from dl_lumbar_dd.data.reporting import generate_eda_reports, generate_preprocess_reports
from dl_lumbar_dd.data.splits import build_split_manifests, save_split_manifests
from dl_lumbar_dd.utils.io import write_json


def run_eda(
    dataset_root: str | Path,
    figures_root: str | Path,
    metadata_root: str | Path,
    max_studies: int | None = None,
) -> dict[str, object]:
    """Generate dataset exploration artifacts for CLI or UI callers."""
    bundle = load_rsna_tables(dataset_root, max_studies=max_studies)
    outputs = generate_eda_reports(bundle, figures_root)
    summary = {"study_count": bundle.train["study_id"].nunique(), "output_count": len(outputs)}
    write_json(summary, Path(metadata_root) / "eda_summary.json")
    return {"study_count": summary["study_count"], "outputs": [str(path) for path in outputs]}


def run_preprocess(
    dataset_root: str | Path,
    output_root: str | Path,
    metadata_root: str | Path,
    figures_root: str | Path,
    max_studies: int | None = None,
    seed: int = 42,
    folds: int = 3,
    train_ratio: float = 0.8,
    image_size: int = 224,
) -> dict[str, object]:
    """Generate split manifests and preprocessing preview artifacts."""
    bundle = load_rsna_tables(dataset_root, max_studies=max_studies)
    study_index = build_study_index(bundle)
    manifests = build_split_manifests(study_index, seed=seed, train_ratio=train_ratio, folds=folds)
    outputs = save_split_manifests(manifests, output_root)
    outputs.extend(generate_preprocess_reports(bundle, manifests, figures_root, image_size=image_size))
    summary = _build_split_summary(manifests, folds=folds, seed=seed)
    write_json(summary, Path(metadata_root) / "split_summary.json")
    return {"study_count": len(study_index), "outputs": [str(path) for path in outputs], "summary": summary}


def _build_split_summary(manifests, folds: int, seed: int) -> dict[str, object]:
    fold_rows = [
        {
            "fold": index,
            "train_count": len(fold.train),
            "validation_count": len(fold.validation),
        }
        for index, fold in enumerate(manifests.folds)
    ]
    return {
        "seed": seed,
        "folds": folds,
        "train_count": len(manifests.train),
        "validation_count": len(manifests.validation),
        "fold_summary": fold_rows,
    }
