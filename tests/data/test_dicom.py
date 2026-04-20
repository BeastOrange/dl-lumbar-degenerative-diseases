from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from dl_lumbar_dd.data.dicom import DicomUpload, build_three_view_tensor, group_dicom_uploads_by_study


def test_build_three_view_tensor_reads_single_study_and_uses_middle_instance(tmp_path) -> None:
    dataset_root = tmp_path / "mock-rsna"
    study_id = 12001
    series_rows = [
        (study_id, 501, "Sagittal T1"),
        (study_id, 502, "Sagittal T2/STIR"),
        (study_id, 503, "Axial T2"),
    ]
    series_table = pd.DataFrame(series_rows, columns=["study_id", "series_id", "series_description"])

    _write_series(
        dataset_root,
        study_id=study_id,
        series_id=501,
        slices=[
            ("9.dcm", 1, _hotspot_pixels(8, 0, 0)),
            ("3.dcm", 2, _hotspot_pixels(8, 3, 2)),
            ("7.dcm", 3, _hotspot_pixels(8, 7, 7)),
        ],
    )
    _write_series(
        dataset_root,
        study_id=study_id,
        series_id=502,
        slices=[
            ("11.dcm", 1, _hotspot_pixels(8, 0, 7)),
            ("5.dcm", 2, _hotspot_pixels(8, 5, 1)),
            ("13.dcm", 3, _hotspot_pixels(8, 7, 0)),
        ],
    )
    _write_series(
        dataset_root,
        study_id=study_id,
        series_id=503,
        slices=[
            ("20.dcm", 1, _hotspot_pixels(8, 2, 6)),
            ("10.dcm", 2, _hotspot_pixels(8, 1, 4)),
            ("30.dcm", 3, _hotspot_pixels(8, 6, 6)),
        ],
    )

    tensor = build_three_view_tensor(
        dataset_root=dataset_root,
        study_id=study_id,
        series_table=series_table,
        image_size=8,
    )

    assert tensor.shape == (3, 8, 8)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    assert _argmax_xy(tensor[0]) == (3, 2)
    assert _argmax_xy(tensor[1]) == (5, 1)
    assert _argmax_xy(tensor[2]) == (1, 4)


def test_build_three_view_tensor_prefers_lowest_series_id_for_duplicate_series_types(tmp_path) -> None:
    dataset_root = tmp_path / "mock-rsna"
    study_id = 12002
    series_rows = [
        (study_id, 401, "Sagittal T1"),
        (study_id, 499, "Sagittal T1"),
        (study_id, 402, "Sagittal T2/STIR"),
        (study_id, 403, "Axial T2"),
    ]
    series_table = pd.DataFrame(series_rows, columns=["study_id", "series_id", "series_description"])

    _write_series(
        dataset_root,
        study_id=study_id,
        series_id=401,
        slices=[
            ("1.dcm", 1, _hotspot_pixels(8, 0, 0)),
            ("2.dcm", 2, _hotspot_pixels(8, 1, 1)),
            ("3.dcm", 3, _hotspot_pixels(8, 7, 7)),
        ],
    )
    _write_series(
        dataset_root,
        study_id=study_id,
        series_id=499,
        slices=[
            ("1.dcm", 1, _hotspot_pixels(8, 0, 0)),
            ("2.dcm", 2, _hotspot_pixels(8, 6, 6)),
            ("3.dcm", 3, _hotspot_pixels(8, 7, 7)),
        ],
    )
    for series_id, hotspot in (
        (402, (4, 2)),
        (403, (2, 5)),
    ):
        _write_series(
            dataset_root,
            study_id=study_id,
            series_id=series_id,
            slices=[
                ("1.dcm", 1, _hotspot_pixels(8, 0, 0)),
                ("2.dcm", 2, _hotspot_pixels(8, hotspot[0], hotspot[1])),
                ("3.dcm", 3, _hotspot_pixels(8, 7, 7)),
            ],
        )

    tensor = build_three_view_tensor(
        dataset_root=dataset_root,
        study_id=study_id,
        series_table=series_table,
        image_size=8,
    )

    assert _argmax_xy(tensor[0]) == (1, 1)
    assert _argmax_xy(tensor[1]) == (4, 2)
    assert _argmax_xy(tensor[2]) == (2, 5)


def test_build_three_view_tensor_raises_clear_error_when_required_series_is_missing(tmp_path) -> None:
    dataset_root = tmp_path / "mock-rsna"
    study_id = 12003
    series_rows = [
        (study_id, 601, "Sagittal T1"),
        (study_id, 602, "Sagittal T2/STIR"),
    ]
    series_table = pd.DataFrame(series_rows, columns=["study_id", "series_id", "series_description"])

    _write_series(
        dataset_root,
        study_id=study_id,
        series_id=601,
        slices=[
            ("1.dcm", 1, _hotspot_pixels(8, 0, 0)),
            ("2.dcm", 2, _hotspot_pixels(8, 1, 1)),
            ("3.dcm", 3, _hotspot_pixels(8, 2, 2)),
        ],
    )
    _write_series(
        dataset_root,
        study_id=study_id,
        series_id=602,
        slices=[
            ("1.dcm", 1, _hotspot_pixels(8, 3, 3)),
            ("2.dcm", 2, _hotspot_pixels(8, 4, 4)),
            ("3.dcm", 3, _hotspot_pixels(8, 5, 5)),
        ],
    )

    with pytest.raises(ValueError, match=rf"study_id={study_id}.*Axial T2"):
        build_three_view_tensor(
            dataset_root=dataset_root,
            study_id=study_id,
            series_table=series_table,
            image_size=8,
        )


def test_group_dicom_uploads_by_study_splits_multiple_studies() -> None:
    study_uid_a = generate_uid()
    study_uid_b = generate_uid()
    uploads = [
        _make_upload(study_instance_uid=study_uid_a, study_id="A100", patient_id="P001", series_description="Sagittal T1"),
        _make_upload(study_instance_uid=study_uid_b, study_id="B200", patient_id="P002", series_description="Axial T2"),
        _make_upload(study_instance_uid=study_uid_a, study_id="A100", patient_id="P001", series_description="Sagittal T2/STIR"),
    ]

    groups = group_dicom_uploads_by_study(uploads)

    assert [group.study_key for group in groups] == [study_uid_a, study_uid_b]
    assert [group.file_count for group in groups] == [2, 1]
    assert all(group.study_label for group in groups)


def test_group_dicom_uploads_by_study_reports_series_coverage_and_ready() -> None:
    incomplete_uid = generate_uid()
    ready_uid = generate_uid()
    uploads = [
        _make_upload(study_instance_uid=incomplete_uid, study_id="A300", patient_id="P003", series_description="Sagittal T1"),
        _make_upload(study_instance_uid=incomplete_uid, study_id="A300", patient_id="P003", series_description="Axial T2"),
        _make_upload(study_instance_uid=ready_uid, study_id="B400", patient_id="P004", series_description="Sagittal T1"),
        _make_upload(study_instance_uid=ready_uid, study_id="B400", patient_id="P004", series_description="Sagittal T2/STIR"),
        _make_upload(study_instance_uid=ready_uid, study_id="B400", patient_id="P004", series_description="Axial T2"),
    ]

    incomplete_group, ready_group = group_dicom_uploads_by_study(uploads)

    assert incomplete_group.recognized_series == ("Sagittal T1", "Axial T2")
    assert incomplete_group.missing_series == ("Sagittal T2/STIR",)
    assert incomplete_group.ready is False

    assert ready_group.recognized_series == (
        "Sagittal T1",
        "Sagittal T2/STIR",
        "Axial T2",
    )
    assert ready_group.missing_series == ()
    assert ready_group.ready is True


def test_group_dicom_uploads_by_study_uses_stable_fallback_when_study_uid_missing() -> None:
    uploads = [
        _make_upload(study_instance_uid=None, study_id="A500", patient_id="P005", study_date="20260420", series_description="Sagittal T1"),
        _make_upload(study_instance_uid=None, study_id="A500", patient_id="P005", study_date="20260420", series_description="Axial T2"),
        _make_upload(study_instance_uid=None, study_id="B600", patient_id="P006", study_date="20260420", series_description="Sagittal T1"),
    ]

    first_groups = group_dicom_uploads_by_study(uploads)
    second_groups = group_dicom_uploads_by_study(uploads)

    assert len(first_groups) == 2
    assert [group.file_count for group in first_groups] == [2, 1]
    assert first_groups[0].study_key.startswith("fallback:")
    assert [group.study_key for group in first_groups] == [group.study_key for group in second_groups]


def _write_series(
    dataset_root: Path,
    study_id: int,
    series_id: int,
    slices: list[tuple[str, int, np.ndarray]],
) -> None:
    series_dir = dataset_root / "train_images" / str(study_id) / str(series_id)
    series_dir.mkdir(parents=True, exist_ok=True)
    for filename, instance_number, pixels in slices:
        _write_dicom(series_dir / filename, pixels, study_id, series_id, instance_number)


def _make_upload(
    *,
    study_instance_uid: str | None,
    study_id: str,
    patient_id: str,
    series_description: str,
    study_date: str | None = None,
    instance_number: int = 1,
) -> DicomUpload:
    pixels = _hotspot_pixels(8, 2, 2)
    buffer = BytesIO()
    dataset = _build_dataset(
        identifier="upload.dcm",
        pixels=pixels,
        patient_id=patient_id,
        study_instance_uid=study_instance_uid,
        series_description=series_description,
        instance_number=instance_number,
        study_id=study_id,
        study_date=study_date,
        series_instance_uid=generate_uid(),
    )
    pydicom.dcmwrite(buffer, dataset, enforce_file_format=True)
    return DicomUpload(name="upload.dcm", content=buffer.getvalue())


def _write_dicom(
    path: Path,
    pixels: np.ndarray,
    study_id: int,
    series_id: int,
    instance_number: int,
) -> None:
    dataset = _build_dataset(
        identifier=str(path),
        pixels=pixels,
        patient_id=str(study_id),
        study_instance_uid=generate_uid(),
        series_description=str(series_id),
        instance_number=instance_number,
        series_id=series_id,
        series_instance_uid=generate_uid(),
    )
    pydicom.dcmwrite(str(path), dataset, enforce_file_format=True)


def _build_dataset(
    *,
    identifier: str,
    pixels: np.ndarray,
    patient_id: str,
    study_instance_uid: str | None,
    series_description: str,
    instance_number: int,
    study_id: str | int | None = None,
    study_date: str | None = None,
    series_id: int | None = None,
    series_instance_uid: str,
) -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(identifier, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "Test^Patient"
    dataset.PatientID = patient_id
    if study_id is not None:
        dataset.StudyID = str(study_id)
    if study_date is not None:
        dataset.StudyDate = study_date
    if study_instance_uid is not None:
        dataset.StudyInstanceUID = study_instance_uid
    dataset.SeriesInstanceUID = series_instance_uid
    dataset.Modality = "MR"
    dataset.SeriesDescription = series_description
    if series_id is not None:
        dataset.SeriesNumber = series_id
    dataset.InstanceNumber = instance_number
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.HighBit = 15
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    return dataset


def _hotspot_pixels(size: int, x: int, y: int) -> np.ndarray:
    pixels = np.zeros((size, size), dtype=np.uint16)
    pixels[y, x] = 4095
    return pixels


def _argmax_xy(image: np.ndarray) -> tuple[int, int]:
    y, x = np.unravel_index(int(np.argmax(image)), image.shape)
    return int(x), int(y)
