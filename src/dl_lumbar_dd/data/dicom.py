"""DICOM reading and three-view tensor preparation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

from dl_lumbar_dd.constants import SERIES_TYPES


def build_three_view_tensor(
    dataset_root: str | Path,
    study_id: int,
    series_table: pd.DataFrame,
    image_size: int = 224,
) -> np.ndarray:
    """Create a three-channel tensor from representative slices of each series type."""
    root = Path(dataset_root)
    study_series = series_table[series_table["study_id"] == study_id]
    views = [
        _load_view_image(root, study_id=study_id, study_series=study_series, series_type=series_type, image_size=image_size)
        for series_type in SERIES_TYPES
    ]
    return np.stack(views, axis=0).astype(np.float32)


def _load_view_image(
    dataset_root: Path,
    study_id: int,
    study_series: pd.DataFrame,
    series_type: str,
    image_size: int,
) -> np.ndarray:
    series_id = _select_series_id(study_series, series_type)
    if series_id is None:
        return np.zeros((image_size, image_size), dtype=np.float32)
    dicom_files = _list_dicom_files(dataset_root / "train_images" / str(study_id) / str(series_id))
    if not dicom_files:
        return np.zeros((image_size, image_size), dtype=np.float32)
    representative_file = dicom_files[len(dicom_files) // 2]
    pixels = _read_dicom_pixels(representative_file)
    normalized = _normalize_pixels(pixels)
    return cv2.resize(normalized, (image_size, image_size), interpolation=cv2.INTER_AREA)


def _select_series_id(study_series: pd.DataFrame, series_type: str) -> int | None:
    matches = study_series[study_series["series_description"] == series_type].sort_values("series_id")
    if matches.empty:
        return None
    return int(matches.iloc[0]["series_id"])


def _list_dicom_files(series_dir: Path) -> list[Path]:
    if not series_dir.exists():
        return []
    return sorted(series_dir.glob("*.dcm"), key=_dicom_sort_key)


def _dicom_sort_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)


def _read_dicom_pixels(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(str(path), force=True)
    pixels = dataset.pixel_array.astype(np.float32)
    if getattr(dataset, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        pixels = pixels.max() - pixels
    return pixels


def _normalize_pixels(pixels: np.ndarray) -> np.ndarray:
    minimum = float(np.min(pixels))
    maximum = float(np.max(pixels))
    if maximum <= minimum:
        return np.zeros_like(pixels, dtype=np.float32)
    return ((pixels - minimum) / (maximum - minimum)).astype(np.float32)
