"""Study-level split generation with leakage prevention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from dl_lumbar_dd.utils.io import ensure_dir


@dataclass(frozen=True)
class FoldManifest:
    train: pd.DataFrame
    validation: pd.DataFrame


@dataclass(frozen=True)
class SplitManifests:
    train: pd.DataFrame
    validation: pd.DataFrame
    folds: list[FoldManifest]


def build_split_manifests(
    study_index: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    folds: int = 3,
) -> SplitManifests:
    """Create train/validation manifests and optional CV folds."""
    stratify_labels = _prepare_stratify_labels(study_index["stratify_key"], minimum_count=2)
    stratify_labels = _ensure_split_feasible(study_index, stratify_labels, train_ratio)
    train_frame, validation_frame = train_test_split(
        study_index,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_labels,
    )
    split_folds = _build_cross_validation_folds(study_index, seed=seed, folds=folds)
    return SplitManifests(
        train=_finalize_manifest(train_frame, split_name="train"),
        validation=_finalize_manifest(validation_frame, split_name="validation"),
        folds=split_folds,
    )


def save_split_manifests(manifests: SplitManifests, output_root: str | Path) -> list[Path]:
    """Persist manifests to CSV for later training stages."""
    root = ensure_dir(output_root)
    outputs = [
        _write_manifest(manifests.train, root / "train_manifest.csv"),
        _write_manifest(manifests.validation, root / "validation_manifest.csv"),
    ]
    for fold_index, fold in enumerate(manifests.folds):
        outputs.append(_write_manifest(fold.train, root / f"fold_{fold_index}_train.csv"))
        outputs.append(_write_manifest(fold.validation, root / f"fold_{fold_index}_validation.csv"))
    return outputs


def _build_cross_validation_folds(study_index: pd.DataFrame, seed: int, folds: int) -> list[FoldManifest]:
    if folds < 2:
        return []
    stratify_labels = _prepare_stratify_labels(study_index["stratify_key"], minimum_count=folds)
    splitter = _make_splitter(seed=seed, folds=folds, stratify_labels=stratify_labels)
    fold_manifests: list[FoldManifest] = []
    for fold_index, (train_idx, validation_idx) in enumerate(splitter.split(study_index, stratify_labels)):
        fold_train = _finalize_manifest(study_index.iloc[train_idx], split_name="train", fold=fold_index)
        fold_validation = _finalize_manifest(
            study_index.iloc[validation_idx], split_name="validation", fold=fold_index
        )
        fold_manifests.append(FoldManifest(train=fold_train, validation=fold_validation))
    return fold_manifests


def _prepare_stratify_labels(series: pd.Series, minimum_count: int) -> pd.Series | None:
    counts = series.value_counts()
    if counts.empty or counts.min() < minimum_count:
        return None
    return series


def _ensure_split_feasible(
    study_index: pd.DataFrame, stratify_labels: pd.Series | None, train_ratio: float
) -> pd.Series | None:
    if stratify_labels is None:
        return None
    sample_count = len(study_index)
    validation_count = max(int(round(sample_count * (1 - train_ratio))), 1)
    train_count = sample_count - validation_count
    class_count = int(stratify_labels.nunique())
    if class_count > validation_count or class_count > train_count:
        return None
    return stratify_labels


def _make_splitter(seed: int, folds: int, stratify_labels: pd.Series | None) -> KFold | StratifiedKFold:
    if stratify_labels is None:
        return KFold(n_splits=folds, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def _finalize_manifest(frame: pd.DataFrame, split_name: str, fold: int | None = None) -> pd.DataFrame:
    manifest = frame.sort_values("study_id").reset_index(drop=True).copy()
    manifest["split"] = split_name
    if fold is not None:
        manifest["fold"] = fold
    return manifest


def _write_manifest(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path
