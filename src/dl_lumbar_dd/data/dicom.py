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
    num_slices: int = 1,
) -> np.ndarray:
    """Create a tensor from representative slices of each series type.

    When num_slices=1: returns (3, H, W) — one grayscale slice per view.
    When num_slices>1: returns (3, num_slices, H, W) — multiple slices per view,
    suitable for feeding directly as multi-channel input to pretrained backbones.
    """
    root = Path(dataset_root)
    study_series = series_table[series_table["study_id"] == study_id]
    views = [
        _load_view_slices(root, study_id, study_series, st, image_size, num_slices)
        for st in SERIES_TYPES
    ]
    return np.stack(views, axis=0).astype(np.float32)


def _load_view_slices(
    dataset_root: Path,
    study_id: int,
    study_series: pd.DataFrame,
    series_type: str,
    image_size: int,
    num_slices: int,
) -> np.ndarray:
    series_id = _select_series_id(study_series, series_type)
    if series_id is None:
        if num_slices == 1:
            return np.zeros((image_size, image_size), dtype=np.float32)
        return np.zeros((num_slices, image_size, image_size), dtype=np.float32)

    dicom_files = _list_dicom_files(dataset_root / "train_images" / str(study_id) / str(series_id))
    if not dicom_files:
        if num_slices == 1:
            return np.zeros((image_size, image_size), dtype=np.float32)
        return np.zeros((num_slices, image_size, image_size), dtype=np.float32)

    indices = _select_slice_indices(len(dicom_files), num_slices)
    slices = [_read_and_resize(dicom_files[i], image_size) for i in indices]

    if num_slices == 1:
        return slices[0]
    return np.stack(slices, axis=0)


def _select_slice_indices(total: int, num_slices: int) -> list[int]:
    if num_slices == 1:
        return [total // 2]
    if total <= num_slices:
        indices = list(range(total))
        while len(indices) < num_slices:
            indices.append(indices[-1])
        return indices
    step = (total - 1) / (num_slices - 1)
    return [round(i * step) for i in range(num_slices)]


def _read_and_resize(path: Path, image_size: int) -> np.ndarray:
    pixels = _read_dicom_pixels(path)
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
