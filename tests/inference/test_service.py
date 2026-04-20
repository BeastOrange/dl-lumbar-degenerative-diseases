from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pydicom
import pytest
import torch
import yaml
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
from torch import nn

from dl_lumbar_dd.data.dicom import build_three_view_tensor_from_uploads
from dl_lumbar_dd.inference.service import DicomUpload, StudyInferenceService, find_latest_checkpoint_run
from tests.data.helpers import create_mock_rsna_dataset


def _make_upload(
    series_description: str,
    instance_number: int,
    *,
    bright_col: int,
    study_instance_uid: str | None = None,
    patient_id: str = "123",
    study_date: str = "20260420",
    name: str | None = None,
) -> DicomUpload:
    pixels = np.zeros((16, 16), dtype=np.uint16)
    pixels[:, bright_col: bright_col + 2] = 500
    pixels += np.arange(16, dtype=np.uint16).reshape(16, 1)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    buffer = BytesIO()
    dataset = FileDataset("upload.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "MR"
    dataset.PatientName = "Test^Patient"
    dataset.PatientID = patient_id
    dataset.StudyDate = study_date
    dataset.StudyInstanceUID = study_instance_uid or generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SeriesDescription = series_description
    dataset.InstanceNumber = instance_number
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.HighBit = 15
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.PixelData = pixels.tobytes()
    pydicom.dcmwrite(buffer, dataset, enforce_file_format=True)
    return DicomUpload(name=name or f"{series_description}-{instance_number}.dcm", content=buffer.getvalue())


def test_build_three_view_tensor_from_uploads_returns_normalized_tensor() -> None:
    uploads = [
        _make_upload("Axial T2", 1, bright_col=1),
        _make_upload("Sagittal T2/STIR", 1, bright_col=3),
        _make_upload("Sagittal T1", 1, bright_col=5),
    ]

    tensor = build_three_view_tensor_from_uploads(uploads, image_size=32)

    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_build_three_view_tensor_from_uploads_requires_all_three_series() -> None:
    uploads = [
        _make_upload("Sagittal T1", 1, bright_col=5),
        _make_upload("Axial T2", 1, bright_col=1),
    ]

    with pytest.raises(ValueError, match="文件不完整|缺少必要序列"):
        build_three_view_tensor_from_uploads(uploads, image_size=32)


def test_find_latest_checkpoint_run_returns_newest_run_with_best_checkpoint(tmp_path: Path) -> None:
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    missing = tmp_path / "missing"
    older.mkdir()
    newer.mkdir()
    missing.mkdir()
    (older / "best.ckpt").write_bytes(b"older")
    (newer / "best.ckpt").write_bytes(b"newer")

    older_time = 1_700_000_000
    newer_time = older_time + 100
    (older / "best.ckpt").touch()
    (newer / "best.ckpt").touch()
    import os
    os.utime(older, (older_time, older_time))
    os.utime(newer, (newer_time, newer_time))

    assert find_latest_checkpoint_run(tmp_path) == newer


@dataclass
class _ModelCall:
    pretrained: bool
    load_backbone_weights: bool


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = False

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        self.loaded = True
        return super().load_state_dict({}, strict=False)

    def eval(self):
        return self

    def to(self, device=None, dtype=None):
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1:] == (3, 1, 32, 32)
        return torch.tensor([[0.1, 2.0, 0.3]], dtype=torch.float32)


def _write_run_dir(root: Path, *, image_size: int = 32) -> Path:
    run_dir = root / "convnext_tiny_cbam-20260419-103601"
    run_dir.mkdir()
    (run_dir / "best.ckpt").write_bytes(b"checkpoint")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model_name": "convnext_tiny_cbam",
                "num_classes": 3,
                "fusion_enabled": True,
                "pretrained": True,
                "in_channels": 1,
                "dropout": 0.2,
                "image_size": image_size,
                "target_column": "spinal_canal_stenosis_l4_l5",
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return run_dir


def test_study_inference_service_predict_returns_chinese_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _write_run_dir(tmp_path)

    calls: list[_ModelCall] = []

    def _fake_create_model(**kwargs):
        calls.append(
            _ModelCall(
                pretrained=bool(kwargs["pretrained"]),
                load_backbone_weights=bool(kwargs["load_backbone_weights"]),
            )
        )
        return _FakeModel()

    monkeypatch.setattr("dl_lumbar_dd.inference.service.create_model", _fake_create_model)
    monkeypatch.setattr(
        "dl_lumbar_dd.inference.service.build_three_view_tensor_from_uploads",
        lambda uploads, image_size: np.ones((3, image_size, image_size), dtype=np.float32),
    )
    monkeypatch.setattr(
        "dl_lumbar_dd.inference.service.torch.load",
        lambda path, map_location="cpu": {"model_state": {}},
    )

    service = StudyInferenceService.from_run_dir(run_dir)
    result = service.predict([DicomUpload(name="case-1.dcm", content=b"fake")])

    assert calls == [_ModelCall(pretrained=True, load_backbone_weights=False)]
    assert result.predicted_label == "中度"
    assert result.target_name == "L4/L5 椎管狭窄"
    assert result.probabilities["中度"] > result.probabilities["轻度/正常"]


def test_study_inference_service_predict_dataset_discovers_studies_and_splits_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = create_mock_rsna_dataset(tmp_path, study_count=3)
    run_dir = _write_run_dir(tmp_path)
    skipped_study_id = 1001
    failed_study_id = 1002

    series_path = dataset_root / "train_series_descriptions.csv"
    series_frame = pd.read_csv(series_path)
    series_frame = series_frame[
        ~(
            (series_frame["study_id"] == skipped_study_id)
            & (series_frame["series_description"] == "Axial T2")
        )
    ]
    series_frame.to_csv(series_path, index=False)
    shutil.rmtree(dataset_root / "train_images" / str(skipped_study_id) / f"{skipped_study_id}3")

    model_calls: list[_ModelCall] = []
    checkpoint_loads: list[Path] = []
    built_studies: list[int] = []

    def _fake_create_model(**kwargs):
        model_calls.append(
            _ModelCall(
                pretrained=bool(kwargs["pretrained"]),
                load_backbone_weights=bool(kwargs["load_backbone_weights"]),
            )
        )
        return _FakeModel()

    def _fake_torch_load(path: Path, map_location: str = "cpu") -> dict[str, dict[str, torch.Tensor]]:
        checkpoint_loads.append(Path(path))
        return {"model_state": {}}

    def _fake_build_three_view_tensor(
        dataset_root: Path,
        study_id: int,
        series_table: pd.DataFrame,
        image_size: int,
    ) -> np.ndarray:
        built_studies.append(study_id)
        if study_id == failed_study_id:
            raise RuntimeError("bad dicom slice")
        return np.ones((3, image_size, image_size), dtype=np.float32)

    monkeypatch.setattr("dl_lumbar_dd.inference.service.DEFAULT_DATASET_ROOT", dataset_root)
    monkeypatch.setattr("dl_lumbar_dd.inference.service.create_model", _fake_create_model)
    monkeypatch.setattr("dl_lumbar_dd.inference.service.torch.load", _fake_torch_load)
    monkeypatch.setattr(
        "dl_lumbar_dd.inference.service.build_three_view_tensor",
        _fake_build_three_view_tensor,
    )

    service = StudyInferenceService.from_run_dir(run_dir)
    results = service.predict_dataset()

    assert [result.study_id for result in results] == [1000, skipped_study_id, failed_study_id]

    success_result, skipped_result, failed_result = results
    assert success_result.status == "成功"
    assert success_result.predicted_label == "中度"
    assert success_result.error_message is None

    assert skipped_result.status == "已跳过"
    assert skipped_result.predicted_label is None
    assert skipped_result.probabilities == {}
    assert skipped_result.error_message is not None
    assert "Axial T2" in skipped_result.error_message

    assert failed_result.status == "失败"
    assert failed_result.predicted_label is None
    assert failed_result.probabilities == {}
    assert failed_result.error_message == "bad dicom slice"

    assert built_studies == [1000, failed_study_id]
    assert model_calls == [_ModelCall(pretrained=True, load_backbone_weights=False)]
    assert checkpoint_loads == [run_dir / "best.ckpt"]


def test_inspect_upload_cases_groups_studies_and_reports_missing_series(tmp_path: Path) -> None:
    service = StudyInferenceService(
        run_dir=tmp_path,
        model=_FakeModel(),
        image_size=32,
        target_name="L4/L5 椎管狭窄",
        device="cpu",
    )
    study_a = generate_uid()
    study_b = generate_uid()
    uploads = [
        _make_upload("Sagittal T1", 1, bright_col=1, study_instance_uid=study_a, patient_id="case-a", study_date="20240101"),
        _make_upload("Sagittal T2/STIR", 2, bright_col=3, study_instance_uid=study_a, patient_id="case-a", study_date="20240101"),
        _make_upload("Axial T2", 3, bright_col=5, study_instance_uid=study_a, patient_id="case-a", study_date="20240101"),
        _make_upload("Sagittal T1", 1, bright_col=7, study_instance_uid=study_b, patient_id="case-b", study_date="20240202"),
        _make_upload("Axial T2", 2, bright_col=9, study_instance_uid=study_b, patient_id="case-b", study_date="20240202"),
    ]

    summaries = service.inspect_upload_cases(uploads)

    assert len(summaries) == 2
    assert summaries[0].study_key == study_a
    assert summaries[0].study_label == "病例 1（case-a / 2024-01-01）"
    assert summaries[0].file_count == 3
    assert summaries[0].recognized_series == ["Sagittal T1", "Sagittal T2/STIR", "Axial T2"]
    assert summaries[0].missing_series == []
    assert summaries[0].ready is True

    assert summaries[1].study_key == study_b
    assert summaries[1].study_label == "病例 2（case-b / 2024-02-02）"
    assert summaries[1].file_count == 2
    assert summaries[1].recognized_series == ["Sagittal T1", "Axial T2"]
    assert summaries[1].missing_series == ["Sagittal T2/STIR"]
    assert summaries[1].ready is False


def test_predict_upload_case_only_uses_selected_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built_upload_names: list[str] = []
    service = StudyInferenceService(
        run_dir=tmp_path,
        model=_FakeModel(),
        image_size=32,
        target_name="L4/L5 椎管狭窄",
        device="cpu",
    )
    study_a = generate_uid()
    study_b = generate_uid()
    uploads = [
        _make_upload("Sagittal T1", 1, bright_col=1, study_instance_uid=study_a, patient_id="case-a", name="a-sag-t1.dcm"),
        _make_upload("Sagittal T2/STIR", 2, bright_col=3, study_instance_uid=study_a, patient_id="case-a", name="a-sag-t2.dcm"),
        _make_upload("Axial T2", 3, bright_col=5, study_instance_uid=study_a, patient_id="case-a", name="a-ax-t2.dcm"),
        _make_upload("Sagittal T1", 1, bright_col=7, study_instance_uid=study_b, patient_id="case-b", name="b-sag-t1.dcm"),
        _make_upload("Sagittal T2/STIR", 2, bright_col=9, study_instance_uid=study_b, patient_id="case-b", name="b-sag-t2.dcm"),
        _make_upload("Axial T2", 3, bright_col=11, study_instance_uid=study_b, patient_id="case-b", name="b-ax-t2.dcm"),
    ]

    monkeypatch.setattr(
        "dl_lumbar_dd.inference.service.build_three_view_tensor_from_uploads",
        lambda case_uploads, image_size: built_upload_names.extend(upload.name for upload in case_uploads) or np.ones((3, image_size, image_size), dtype=np.float32),
    )

    study_key = service.inspect_upload_cases(uploads)[1].study_key
    result = service.predict_upload_case(uploads, study_key)

    assert built_upload_names == ["b-sag-t1.dcm", "b-sag-t2.dcm", "b-ax-t2.dcm"]
    assert result.predicted_label == "中度"


def test_predict_upload_case_raises_clear_error_when_study_key_missing(tmp_path: Path) -> None:
    service = StudyInferenceService(
        run_dir=tmp_path,
        model=_FakeModel(),
        image_size=32,
        target_name="L4/L5 椎管狭窄",
        device="cpu",
    )
    uploads = [_make_upload("Sagittal T1", 1, bright_col=1, study_instance_uid=generate_uid(), patient_id="case-a")]

    with pytest.raises(ValueError, match="未找到 study_key=missing-study 对应的上传病例"):
        service.predict_upload_case(uploads, "missing-study")
