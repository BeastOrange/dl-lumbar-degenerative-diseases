"""DICOM reading and three-view tensor preparation."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pydicom

from dl_lumbar_dd.constants import SERIES_TYPES


@dataclass(frozen=True, slots=True)
class DicomUpload:
    name: str
    content: bytes


@dataclass(frozen=True, slots=True)
class UploadStudySummary:
    study_key: str
    study_label: str
    file_count: int
    recognized_series: tuple[str, ...]
    missing_series: tuple[str, ...]
    ready: bool


@dataclass(frozen=True, slots=True)
class UploadStudyGroup:
    summary: UploadStudySummary
    uploads: tuple[DicomUpload, ...]

    @property
    def study_key(self) -> str:
        return self.summary.study_key

    @property
    def study_label(self) -> str:
        return self.summary.study_label

    @property
    def file_count(self) -> int:
        return self.summary.file_count

    @property
    def recognized_series(self) -> tuple[str, ...]:
        return self.summary.recognized_series

    @property
    def missing_series(self) -> tuple[str, ...]:
        return self.summary.missing_series

    @property
    def ready(self) -> bool:
        return self.summary.ready


def build_three_view_tensor(
    dataset_root: str | Path,
    study_id: int,
    series_table: pd.DataFrame,
    image_size: int = 224,
    num_slices: int = 1,
    coord_lookup: dict[int, list[tuple[int, int, float, float]]] | None = None,
    roi_crop: bool = False,
    roi_padding: float = 0.3,
    context_slices: int = 0,
) -> np.ndarray:
    """Create a tensor from representative slices of each series type.

    When coord_lookup is provided, uses annotated instance_number and (x,y)
    instead of the blind middle slice. Supports ROI cropping and context slices.

    coord_lookup maps series_id -> [(instance_number, condition_idx, x, y), ...]
    """
    root = Path(dataset_root)
    study_series = series_table[series_table["study_id"] == study_id]
    resolved_series_ids = _resolve_required_series_ids(study_id, study_series)

    views = []
    for series_type in SERIES_TYPES:
        series_id = resolved_series_ids[series_type]
        coord_entries = coord_lookup.get(series_id, []) if coord_lookup else []

        view = _load_targeted_view(
            dataset_root=root,
            study_id=study_id,
            series_id=series_id,
            series_type=series_type,
            image_size=image_size,
            num_slices=num_slices,
            coord_entries=coord_entries,
            roi_crop=roi_crop,
            roi_padding=roi_padding,
            context_slices=context_slices,
        )
        views.append(view)

    return np.stack(views, axis=0).astype(np.float32)


def build_three_view_tensor_from_uploads(
    uploads: list[DicomUpload],
    image_size: int = 224,
) -> np.ndarray:
    if not uploads:
        raise ValueError("未读取到任何 DCM 文件，请重新选择包含 DCM 的目录。")

    grouped: dict[str, list[tuple[int, np.ndarray]]] = {series_type: [] for series_type in SERIES_TYPES}
    for upload in uploads:
        dataset = pydicom.dcmread(BytesIO(upload.content), force=True)
        series_type = _resolve_uploaded_series_type(dataset, upload.name)
        if series_type is None:
            continue
        pixels = _pixels_from_dataset(dataset)
        grouped[series_type].append((_instance_number(dataset), pixels))

    missing = [series_type for series_type, slices in grouped.items() if not slices]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"所选病例组文件不完整，仍缺少必要序列: {missing_text}。请重新选择包含完整 DCM 文件的目录。"
        )

    views = []
    for series_type in SERIES_TYPES:
        slices = sorted(grouped[series_type], key=lambda item: item[0])
        middle_pixels = slices[len(slices) // 2][1]
        normalized = _normalize_pixels(middle_pixels)
        resized = cv2.resize(normalized, (image_size, image_size), interpolation=cv2.INTER_AREA)
        views.append(resized)
    return np.stack(views, axis=0).astype(np.float32)


def split_uploads_by_study(uploads: list[DicomUpload]) -> list[UploadStudyGroup]:
    parsed_uploads = [_parse_upload_metadata(upload, index) for index, upload in enumerate(uploads)]
    groups: dict[str, list[_ParsedUpload]] = {}
    for parsed_upload in parsed_uploads:
        groups.setdefault(parsed_upload.study_key, []).append(parsed_upload)

    upload_groups: list[UploadStudyGroup] = []
    for parsed_group in groups.values():
        summary = _build_upload_study_summary(parsed_group)
        group_uploads = tuple(parsed.upload for parsed in parsed_group)
        upload_groups.append(UploadStudyGroup(summary=summary, uploads=group_uploads))

    return upload_groups


def inspect_upload_studies(uploads: list[DicomUpload]) -> list[UploadStudySummary]:
    return [group.summary for group in split_uploads_by_study(uploads)]


def group_dicom_uploads_by_study(uploads: list[DicomUpload]) -> list[UploadStudySummary]:
    return inspect_upload_studies(uploads)


def _load_targeted_view(
    dataset_root: Path,
    study_id: int,
    series_id: int,
    series_type: str,
    image_size: int,
    num_slices: int,
    coord_entries: list[tuple[int, int, float, float]],
    roi_crop: bool,
    roi_padding: float,
    context_slices: int,
) -> np.ndarray:
    series_dir = dataset_root / "train_images" / str(study_id) / str(series_id)
    dicom_files = _list_dicom_files(series_dir)
    if not dicom_files:
        raise ValueError(
            f"study_id={study_id} 缺少必要序列文件: {series_type} (series_id={series_id}) @ {series_dir}"
        )

    # Level 1: Targeted slice selection
    target_idx, crop_xy = _resolve_target_slice(dicom_files, coord_entries)

    # Level 3: Context slices (target ± N adjacent)
    if context_slices > 0:
        total = len(dicom_files)
        indices = []
        for offset in range(-context_slices, context_slices + 1):
            idx = max(0, min(total - 1, target_idx + offset))
            indices.append(idx)
        slices = [
            _read_normalize_crop_resize(dicom_files[i], image_size, crop_xy if roi_crop else None, roi_padding)
            for i in indices
        ]
        return np.stack(slices, axis=0)

    # Level 1+2: Single targeted slice with optional ROI crop
    if num_slices == 1:
        return _read_normalize_crop_resize(
            dicom_files[target_idx], image_size,
            crop_xy if roi_crop else None, roi_padding,
        )

    # Multi-slice mode (evenly spaced, fallback)
    indices = _select_slice_indices(len(dicom_files), num_slices)
    slices = [
        _read_normalize_crop_resize(dicom_files[i], image_size, crop_xy if roi_crop else None, roi_padding)
        for i in indices
    ]
    return np.stack(slices, axis=0)


def _resolve_target_slice(
    dicom_files: list[Path],
    coord_entries: list[tuple[int, int, float, float]],
) -> tuple[int, tuple[float, float] | None]:
    """Find the best slice index and optional (x, y) crop center."""
    if not coord_entries:
        return len(dicom_files) // 2, None

    # coord_entries: [(instance_number, condition_idx, x, y), ...]
    # Use the first entry (typically the target condition)
    instance_num, _, cx, cy = coord_entries[0]

    # Map instance_number to file index
    file_stems = {int(f.stem) if f.stem.isdigit() else -1: i for i, f in enumerate(dicom_files)}
    if instance_num in file_stems:
        return file_stems[instance_num], (cx, cy)

    # Fallback: find closest instance number
    closest_instance = min(file_stems.keys(), key=lambda k: abs(k - instance_num) if k >= 0 else 10**9)
    if closest_instance >= 0:
        return file_stems[closest_instance], (cx, cy)

    return len(dicom_files) // 2, None


def _read_normalize_crop_resize(
    path: Path,
    image_size: int,
    crop_center: tuple[float, float] | None,
    roi_padding: float,
) -> np.ndarray:
    """Read DICOM, normalize, optionally ROI-crop, then resize."""
    pixels = _read_dicom_pixels(path)
    normalized = _normalize_pixels(pixels)

    if crop_center is not None:
        normalized = _roi_crop(normalized, crop_center, roi_padding)

    return cv2.resize(normalized, (image_size, image_size), interpolation=cv2.INTER_AREA)


def _roi_crop(
    image: np.ndarray,
    center: tuple[float, float],
    padding: float,
) -> np.ndarray:
    """Crop a square region around (cx, cy) with padding relative to crop size."""
    h, w = image.shape[:2]
    cx, cy = center

    # Crop size: use the smaller dimension as base, with padding
    base_size = min(h, w) * 0.5
    half = int(base_size * (1.0 + padding) / 2)

    cx_int, cy_int = int(round(cx)), int(round(cy))
    x1 = max(0, cx_int - half)
    y1 = max(0, cy_int - half)
    x2 = min(w, cx_int + half)
    y2 = min(h, cy_int + half)

    if x2 <= x1 or y2 <= y1:
        return image

    return image[y1:y2, x1:x2]


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


def _resolve_required_series_ids(study_id: int, study_series: pd.DataFrame) -> dict[str, int]:
    resolved: dict[str, int] = {}
    missing: list[str] = []
    for series_type in SERIES_TYPES:
        series_id = _select_series_id(study_series, series_type)
        if series_id is None:
            missing.append(series_type)
            continue
        resolved[series_type] = series_id

    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"study_id={study_id} 缺少必要序列: {joined}")

    return resolved


def _list_dicom_files(series_dir: Path) -> list[Path]:
    if not series_dir.exists():
        return []
    return sorted(series_dir.glob("*.dcm"), key=_dicom_sort_key)


def _dicom_sort_key(path: Path) -> tuple[int, str]:
    instance_number = _read_instance_number_from_path(path)
    if instance_number is not None:
        return (instance_number, path.name)
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)


def _read_instance_number_from_path(path: Path) -> int | None:
    try:
        dataset = pydicom.dcmread(
            str(path),
            force=True,
            stop_before_pixels=True,
            specific_tags=["InstanceNumber"],
        )
    except Exception:
        return None

    raw_value = getattr(dataset, "InstanceNumber", None)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _read_dicom_pixels(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(str(path), force=True)
    return _pixels_from_dataset(dataset)


def _pixels_from_dataset(dataset: pydicom.Dataset) -> np.ndarray:
    pixels = dataset.pixel_array.astype(np.float32)
    if getattr(dataset, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        pixels = pixels.max() - pixels
    return pixels


def _instance_number(dataset: pydicom.Dataset) -> int:
    raw_value = getattr(dataset, "InstanceNumber", 0)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True, slots=True)
class _ParsedUpload:
    upload: DicomUpload
    study_key: str
    study_label: str
    series_type: str | None


def _parse_upload_metadata(upload: DicomUpload, fallback_index: int) -> _ParsedUpload:
    dataset = pydicom.dcmread(
        BytesIO(upload.content),
        force=True,
        stop_before_pixels=True,
        specific_tags=[
            "StudyInstanceUID",
            "StudyID",
            "PatientID",
            "AccessionNumber",
            "StudyDate",
            "SeriesDescription",
        ],
    )
    study_key, study_label = _resolve_study_identity(dataset, fallback_index)
    series_type = _resolve_uploaded_series_type(dataset, upload.name)
    return _ParsedUpload(
        upload=upload,
        study_key=study_key,
        study_label=study_label,
        series_type=series_type,
    )


def _resolve_study_identity(dataset: pydicom.Dataset, fallback_index: int) -> tuple[str, str]:
    study_instance_uid = _clean_metadata_value(getattr(dataset, "StudyInstanceUID", None))
    if study_instance_uid:
        suffix = study_instance_uid.split(".")[-1][-8:] or study_instance_uid[-8:]
        return (study_instance_uid, f"病例 {suffix}")

    patient_id = _clean_metadata_value(getattr(dataset, "PatientID", None))
    study_id = _clean_metadata_value(getattr(dataset, "StudyID", None))
    accession_number = _clean_metadata_value(getattr(dataset, "AccessionNumber", None))
    study_date = _clean_metadata_value(getattr(dataset, "StudyDate", None))
    fallback_parts = [
        ("study_id", study_id),
        ("patient_id", patient_id),
        ("accession", accession_number),
        ("study_date", study_date),
    ]
    populated_parts = [(name, value) for name, value in fallback_parts if value]
    if populated_parts:
        study_key = "fallback:" + "|".join(f"{name}={value}" for name, value in populated_parts)
        label_parts = [value for _, value in populated_parts]
        return (study_key, " / ".join(label_parts))

    fallback_key = f"fallback:index={fallback_index}"
    return (fallback_key, f"未识别病例 {fallback_index + 1}")


def _build_upload_study_summary(parsed_group: list[_ParsedUpload]) -> UploadStudySummary:
    first = parsed_group[0]
    recognized_series = tuple(
        series_type for series_type in SERIES_TYPES
        if any(parsed.series_type == series_type for parsed in parsed_group)
    )
    missing_series = tuple(series_type for series_type in SERIES_TYPES if series_type not in recognized_series)
    return UploadStudySummary(
        study_key=first.study_key,
        study_label=first.study_label,
        file_count=len(parsed_group),
        recognized_series=recognized_series,
        missing_series=missing_series,
        ready=not missing_series,
    )


def _clean_metadata_value(raw_value: Any) -> str:
    if raw_value is None:
        return ""
    value = str(raw_value).strip()
    return value


def _resolve_uploaded_series_type(dataset: pydicom.Dataset, filename: str) -> str | None:
    description = str(getattr(dataset, "SeriesDescription", "")).strip()
    normalized = description.lower()
    if description in SERIES_TYPES:
        return description
    if "sag" in normalized and "t1" in normalized:
        return "Sagittal T1"
    if "sag" in normalized and ("t2" in normalized or "stir" in normalized):
        return "Sagittal T2/STIR"
    if ("axial" in normalized or normalized.startswith("ax") or " ax " in f" {normalized} ") and "t2" in normalized:
        return "Axial T2"

    fallback = filename.lower()
    if "sag" in fallback and "t1" in fallback:
        return "Sagittal T1"
    if "sag" in fallback and ("t2" in fallback or "stir" in fallback):
        return "Sagittal T2/STIR"
    if ("axial" in fallback or fallback.startswith("ax") or "_ax_" in fallback or "-ax-" in fallback) and "t2" in fallback:
        return "Axial T2"
    return None


def _normalize_pixels(pixels: np.ndarray) -> np.ndarray:
    minimum = float(np.min(pixels))
    maximum = float(np.max(pixels))
    if maximum <= minimum:
        return np.zeros_like(pixels, dtype=np.float32)
    return ((pixels - minimum) / (maximum - minimum)).astype(np.float32)
