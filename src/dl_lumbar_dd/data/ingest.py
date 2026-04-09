"""RSNA 2024 metadata ingestion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dl_lumbar_dd.constants import SERIES_TYPES, TARGET_COLUMNS


@dataclass(frozen=True)
class RSNATables:
    """Container for the core RSNA CSV tables."""

    dataset_root: Path
    train: pd.DataFrame
    series: pd.DataFrame
    coordinates: pd.DataFrame


def load_rsna_tables(dataset_root: str | Path, max_studies: int | None = None) -> RSNATables:
    """Read the training CSV files and optionally trim to a smoke subset."""
    root = Path(dataset_root)
    train = _read_csv(root / "train.csv", {"study_id", *TARGET_COLUMNS})
    series = _read_csv(root / "train_series_descriptions.csv", {"study_id", "series_id", "series_description"})
    coordinates = _read_csv(
        root / "train_label_coordinates.csv",
        {"study_id", "series_id", "instance_number", "condition", "level", "x", "y"},
    )
    train = _normalize_identifiers(train, include_series=False)
    series = _normalize_identifiers(series, include_series=True)
    coordinates = _normalize_identifiers(coordinates, include_series=True)
    if max_studies:
        allowed = sorted(train["study_id"].drop_duplicates().tolist())[:max_studies]
        train = train[train["study_id"].isin(allowed)].reset_index(drop=True)
        series = series[series["study_id"].isin(allowed)].reset_index(drop=True)
        coordinates = coordinates[coordinates["study_id"].isin(allowed)].reset_index(drop=True)
    return RSNATables(dataset_root=root, train=train, series=series, coordinates=coordinates)


def build_study_index(bundle: RSNATables) -> pd.DataFrame:
    """Aggregate patient-level metadata for split generation and preprocessing."""
    labels = bundle.train.copy()
    labels["moderate_count"] = labels.loc[:, TARGET_COLUMNS].eq("Moderate").sum(axis=1)
    labels["severe_count"] = labels.loc[:, TARGET_COLUMNS].eq("Severe").sum(axis=1)
    labels["missing_count"] = labels.loc[:, TARGET_COLUMNS].isna().sum(axis=1)
    labels["severity_burden"] = labels["moderate_count"] + (2 * labels["severe_count"])
    labels["stratify_key"] = labels.apply(_make_stratify_key, axis=1)

    series_counts = bundle.series.groupby("study_id").size().rename("series_count")
    series_presence = _build_series_presence(bundle.series)
    study_index = labels.merge(series_counts, on="study_id", how="left").merge(
        series_presence, on="study_id", how="left"
    )
    study_index["series_count"] = study_index["series_count"].fillna(0).astype(int)
    return study_index.sort_values("study_id").reset_index(drop=True)


def _read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required RSNA file is missing: {path}")
    frame = pd.read_csv(path)
    missing = required_columns - set(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns in {path.name}: {missing_list}")
    return frame


def _normalize_identifiers(frame: pd.DataFrame, include_series: bool) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["study_id"] = normalized["study_id"].astype(int)
    if include_series and "series_id" in normalized.columns:
        normalized["series_id"] = normalized["series_id"].astype(int)
    return normalized


def _make_stratify_key(row: pd.Series) -> str:
    burden = int(row["severity_burden"])
    if row["severe_count"] > 0:
        severity_bucket = "severe"
    elif row["moderate_count"] > 0:
        severity_bucket = "moderate"
    else:
        severity_bucket = "mild"
    burden_bucket = min(burden // 4, 4)
    return f"{severity_bucket}_{burden_bucket}"


def _build_series_presence(series: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        series.assign(series_available=1)
        .pivot_table(
            index="study_id",
            columns="series_description",
            values="series_available",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    for series_type in SERIES_TYPES:
        if series_type not in pivot.columns:
            pivot[series_type] = 0
    renamed = {
        series_type: f"series_{series_type.lower().replace('/', '_').replace(' ', '_')}_count"
        for series_type in SERIES_TYPES
    }
    return pivot.rename(columns=renamed)
