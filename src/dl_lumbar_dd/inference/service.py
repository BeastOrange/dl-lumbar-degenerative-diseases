"""Minimal inference service for uploaded DICOM studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from dl_lumbar_dd.constants import ARTIFACTS_DIR, DEFAULT_DATASET_ROOT, INDEX_TO_SEVERITY, SERIES_TYPES
from dl_lumbar_dd.data.dicom import DicomUpload, build_three_view_tensor, build_three_view_tensor_from_uploads
from dl_lumbar_dd.models import create_model

SEVERITY_TO_ZH = {
    "Normal/Mild": "轻度/正常",
    "Moderate": "中度",
    "Severe": "重度",
}
TARGET_PREFIX_TO_ZH = {
    "spinal_canal_stenosis": "椎管狭窄",
    "left_neural_foraminal_narrowing": "左侧神经孔狭窄",
    "right_neural_foraminal_narrowing": "右侧神经孔狭窄",
    "left_subarticular_stenosis": "左侧侧隐窝狭窄",
    "right_subarticular_stenosis": "右侧侧隐窝狭窄",
}
STATUS_SUCCESS = "成功"
STATUS_SKIPPED = "已跳过"
STATUS_FAILED = "失败"


@dataclass(frozen=True, slots=True)
class InferenceResult:
    run_dir: Path
    target_name: str
    predicted_index: int
    predicted_label: str
    probabilities: dict[str, float]


@dataclass(frozen=True, slots=True)
class BatchInferenceResult:
    study_id: int
    run_dir: Path
    target_name: str
    status: str
    predicted_index: int | None
    predicted_label: str | None
    probabilities: dict[str, float]
    error_message: str | None


def find_latest_checkpoint_run(runs_root: str | Path = ARTIFACTS_DIR / "runs") -> Path:
    root = Path(runs_root)
    candidates = [path for path in root.iterdir() if path.is_dir() and (path / "best.ckpt").exists()]
    if not candidates:
        raise FileNotFoundError("未找到可用的 best.ckpt，请先完成一次训练。")
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


class StudyInferenceService:
    def __init__(
        self,
        *,
        run_dir: Path,
        model: torch.nn.Module,
        image_size: int,
        target_name: str,
        device: str,
    ) -> None:
        self.run_dir = run_dir
        self.model = model
        self.image_size = image_size
        self.target_name = target_name
        self.device = device

    @classmethod
    def from_latest_run(cls, runs_root: str | Path = ARTIFACTS_DIR / "runs", device: str = "auto") -> "StudyInferenceService":
        return cls.from_run_dir(find_latest_checkpoint_run(runs_root), device=device)

    @classmethod
    def from_run_dir(cls, run_dir: str | Path, device: str = "auto") -> "StudyInferenceService":
        run_path = Path(run_dir)
        config = _load_run_config(run_path)
        target_name = _format_target_name(str(config.get("target_column", "spinal_canal_stenosis_l4_l5")))
        model = create_model(
            model_name=str(config["model_name"]),
            num_classes=int(config.get("num_classes", 3)),
            fusion_enabled=bool(config.get("fusion_enabled", True)),
            pretrained=bool(config.get("pretrained", False)),
            load_backbone_weights=False,
            in_channels=int(config.get("in_channels", 1)),
            dropout=float(config.get("dropout", 0.2)),
            image_size=int(config.get("image_size", 224)),
        )
        checkpoint = torch.load(run_path / "best.ckpt", map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        resolved_device = _resolve_device(device)
        model.to(resolved_device)
        model.eval()
        return cls(
            run_dir=run_path,
            model=model,
            image_size=int(config.get("image_size", 224)),
            target_name=target_name,
            device=resolved_device,
        )

    def predict(self, uploads: list[DicomUpload]) -> InferenceResult:
        tensor = build_three_view_tensor_from_uploads(uploads, image_size=self.image_size)
        predicted_index, predicted_label, probability_map = self._predict_tensor(tensor)
        return InferenceResult(
            run_dir=self.run_dir,
            target_name=self.target_name,
            predicted_index=predicted_index,
            predicted_label=predicted_label,
            probabilities=probability_map,
        )

    def predict_dataset(
        self,
        dataset_root: str | Path | None = None,
    ) -> list[BatchInferenceResult]:
        root = Path(dataset_root) if dataset_root is not None else DEFAULT_DATASET_ROOT
        series_table = _load_series_table(root)
        study_ids = sorted(series_table["study_id"].drop_duplicates().astype(int).tolist())
        return [
            self.predict_study(dataset_root=root, study_id=study_id, series_table=series_table)
            for study_id in study_ids
        ]

    def predict_study(
        self,
        *,
        dataset_root: str | Path,
        study_id: int,
        series_table: pd.DataFrame,
    ) -> BatchInferenceResult:
        root = Path(dataset_root)
        missing_series = _find_missing_required_series(root, study_id, series_table)
        if missing_series:
            return BatchInferenceResult(
                study_id=study_id,
                run_dir=self.run_dir,
                target_name=self.target_name,
                status=STATUS_SKIPPED,
                predicted_index=None,
                predicted_label=None,
                probabilities={},
                error_message=f"缺少必要序列: {', '.join(missing_series)}",
            )

        try:
            tensor = build_three_view_tensor(
                dataset_root=root,
                study_id=study_id,
                series_table=series_table,
                image_size=self.image_size,
            )
            predicted_index, predicted_label, probability_map = self._predict_tensor(tensor)
        except Exception as exc:
            return BatchInferenceResult(
                study_id=study_id,
                run_dir=self.run_dir,
                target_name=self.target_name,
                status=STATUS_FAILED,
                predicted_index=None,
                predicted_label=None,
                probabilities={},
                error_message=str(exc) or exc.__class__.__name__,
            )

        return BatchInferenceResult(
            study_id=study_id,
            run_dir=self.run_dir,
            target_name=self.target_name,
            status=STATUS_SUCCESS,
            predicted_index=predicted_index,
            predicted_label=predicted_label,
            probabilities=probability_map,
            error_message=None,
        )

    def _predict_tensor(self, tensor: np.ndarray) -> tuple[int, str, dict[str, float]]:
        inputs = torch.from_numpy(tensor).unsqueeze(0).unsqueeze(2).to(device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            logits = self.model(inputs)
            if logits.ndim == 3:
                logits = logits[:, 0, :]
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        predicted_index = int(np.argmax(probabilities))
        labels = [SEVERITY_TO_ZH[INDEX_TO_SEVERITY[index]] for index in range(probabilities.shape[0])]
        probability_map = {label: float(prob) for label, prob in zip(labels, probabilities, strict=True)}
        return predicted_index, labels[predicted_index], probability_map


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件：{config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"配置文件格式无效：{config_path}")
    return loaded


def _format_target_name(target_column: str) -> str:
    for prefix, label in TARGET_PREFIX_TO_ZH.items():
        prefix_token = f"{prefix}_"
        if not target_column.startswith(prefix_token):
            continue
        level = target_column.removeprefix(prefix_token).upper().replace("_", "/")
        return f"{level} {label}"
    return target_column


def _load_series_table(dataset_root: Path) -> pd.DataFrame:
    series_path = dataset_root / "train_series_descriptions.csv"
    if not series_path.exists():
        raise FileNotFoundError(f"未找到病例序列表：{series_path}")
    series_table = pd.read_csv(series_path)
    required_columns = {"study_id", "series_id", "series_description"}
    missing_columns = required_columns - set(series_table.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"病例序列表缺少必要列: {missing_list}")
    normalized = series_table.copy()
    normalized["study_id"] = normalized["study_id"].astype(int)
    normalized["series_id"] = normalized["series_id"].astype(int)
    normalized["series_description"] = normalized["series_description"].astype(str)
    return normalized


def _find_missing_required_series(dataset_root: Path, study_id: int, series_table: pd.DataFrame) -> list[str]:
    study_series = series_table[series_table["study_id"] == study_id]
    missing: list[str] = []
    for series_type in SERIES_TYPES:
        matches = study_series[study_series["series_description"] == series_type].sort_values("series_id")
        if matches.empty:
            missing.append(series_type)
            continue
        series_id = int(matches.iloc[0]["series_id"])
        series_dir = dataset_root / "train_images" / str(study_id) / str(series_id)
        if not series_dir.exists() or not any(series_dir.glob("*.dcm")):
            missing.append(series_type)
    return missing


def _resolve_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return normalized
