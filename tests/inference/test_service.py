from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pydicom
import pytest
import torch
import yaml
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
from torch import nn

from dl_lumbar_dd.data.dicom import build_three_view_tensor_from_uploads
from dl_lumbar_dd.inference.service import DicomUpload, StudyInferenceService, find_latest_checkpoint_run


def _make_upload(series_description: str, instance_number: int, *, bright_col: int) -> DicomUpload:
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
    dataset.PatientID = "123"
    dataset.StudyInstanceUID = generate_uid()
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
    return DicomUpload(name=f"{series_description}-{instance_number}.dcm", content=buffer.getvalue())


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

    with pytest.raises(ValueError, match="Sagittal T2/STIR"):
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


def test_study_inference_service_predict_returns_chinese_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "convnext_tiny_cbam-20260419-103601"
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
                "image_size": 32,
                "target_column": "spinal_canal_stenosis_l4_l5",
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

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
