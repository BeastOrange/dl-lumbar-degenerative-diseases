from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, MRImageStorage


TARGET_COLUMNS = [
    "spinal_canal_stenosis_l1_l2",
    "spinal_canal_stenosis_l2_l3",
    "spinal_canal_stenosis_l3_l4",
    "spinal_canal_stenosis_l4_l5",
    "spinal_canal_stenosis_l5_s1",
    "left_neural_foraminal_narrowing_l1_l2",
    "left_neural_foraminal_narrowing_l2_l3",
    "left_neural_foraminal_narrowing_l3_l4",
    "left_neural_foraminal_narrowing_l4_l5",
    "left_neural_foraminal_narrowing_l5_s1",
    "right_neural_foraminal_narrowing_l1_l2",
    "right_neural_foraminal_narrowing_l2_l3",
    "right_neural_foraminal_narrowing_l3_l4",
    "right_neural_foraminal_narrowing_l4_l5",
    "right_neural_foraminal_narrowing_l5_s1",
    "left_subarticular_stenosis_l1_l2",
    "left_subarticular_stenosis_l2_l3",
    "left_subarticular_stenosis_l3_l4",
    "left_subarticular_stenosis_l4_l5",
    "left_subarticular_stenosis_l5_s1",
    "right_subarticular_stenosis_l1_l2",
    "right_subarticular_stenosis_l2_l3",
    "right_subarticular_stenosis_l3_l4",
    "right_subarticular_stenosis_l4_l5",
    "right_subarticular_stenosis_l5_s1",
]


def create_mock_rsna_dataset(root: Path, study_count: int = 6) -> Path:
    dataset_root = root / "mock-rsna"
    train_images_root = dataset_root / "train_images"
    train_images_root.mkdir(parents=True, exist_ok=True)

    study_ids = [1000 + index for index in range(study_count)]
    _write_train_csv(dataset_root / "train.csv", study_ids)
    _write_series_csv(dataset_root / "train_series_descriptions.csv", study_ids)
    _write_coordinates_csv(dataset_root / "train_label_coordinates.csv", study_ids)
    _write_sample_submission(dataset_root / "sample_submission.csv", study_ids)
    _write_test_series(dataset_root / "test_series_descriptions.csv")

    for study_id in study_ids:
        _write_study_dicoms(train_images_root, study_id)
    return dataset_root


def _write_train_csv(path: Path, study_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    severities = ["Normal/Mild", "Moderate", "Severe"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study_id", *TARGET_COLUMNS])
        for index, study_id in enumerate(study_ids):
            row = [study_id]
            for target_index, _ in enumerate(TARGET_COLUMNS):
                severity = severities[(index + target_index) % len(severities)]
                row.append(severity)
            writer.writerow(row)


def _write_series_csv(path: Path, study_ids: list[int]) -> None:
    series_types = ("Sagittal T1", "Sagittal T2/STIR", "Axial T2")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study_id", "series_id", "series_description"])
        for study_id in study_ids:
            for offset, series_type in enumerate(series_types, start=1):
                writer.writerow([study_id, int(f"{study_id}{offset}"), series_type])


def _write_coordinates_csv(path: Path, study_ids: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study_id", "series_id", "instance_number", "condition", "level", "x", "y"])
        for study_id in study_ids:
            writer.writerow(
                [study_id, int(f"{study_id}1"), 2, "Spinal Canal Stenosis", "L1/L2", 10.0, 12.0]
            )


def _write_sample_submission(path: Path, study_ids: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "normal_mild", "moderate", "severe"])
        for study_id in study_ids[:1]:
            writer.writerow([f"{study_id}_spinal_canal_stenosis_l1_l2", 0.33, 0.33, 0.34])


def _write_test_series(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study_id", "series_id", "series_description"])
        writer.writerow([9999, 99991, "Sagittal T1"])


def _write_study_dicoms(train_images_root: Path, study_id: int) -> None:
    series_lookup = {
        "Sagittal T1": int(f"{study_id}1"),
        "Sagittal T2/STIR": int(f"{study_id}2"),
        "Axial T2": int(f"{study_id}3"),
    }
    for series_index, (series_type, series_id) in enumerate(series_lookup.items(), start=1):
        series_dir = train_images_root / str(study_id) / str(series_id)
        series_dir.mkdir(parents=True, exist_ok=True)
        for instance in range(1, 4):
            pixels = np.full((16, 16), fill_value=series_index * instance * 50, dtype=np.uint16)
            _write_dicom(series_dir / f"{instance}.dcm", pixels, study_id, series_id, instance)


def _write_dicom(
    path: Path,
    pixels: np.ndarray,
    study_id: int,
    series_id: int,
    instance_number: int,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "Test^Patient"
    dataset.PatientID = str(study_id)
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.Modality = "MR"
    dataset.SeriesNumber = 1
    dataset.InstanceNumber = instance_number
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.HighBit = 15
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.ImagesInAcquisition = 3
    dataset.SeriesDescription = str(series_id)
    dataset.PixelData = pixels.tobytes()
    pydicom.dcmwrite(str(path), dataset, enforce_file_format=True)
