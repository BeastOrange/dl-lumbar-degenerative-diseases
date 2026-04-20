from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from dl_lumbar_dd.data.dicom import build_three_view_tensor


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
    dataset.SeriesDescription = str(series_id)
    dataset.InstanceNumber = instance_number
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.HighBit = 15
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    pydicom.dcmwrite(str(path), dataset, enforce_file_format=True)


def _hotspot_pixels(size: int, x: int, y: int) -> np.ndarray:
    pixels = np.zeros((size, size), dtype=np.uint16)
    pixels[y, x] = 4095
    return pixels


def _argmax_xy(image: np.ndarray) -> tuple[int, int]:
    y, x = np.unravel_index(int(np.argmax(image)), image.shape)
    return int(x), int(y)
