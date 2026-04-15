from __future__ import annotations

import csv
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dl_lumbar_dd.models import create_model
from dl_lumbar_dd.train import Trainer, TrainingConfig


class SyntheticLumbarDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, size: int | None = None, labels: list[int] | None = None) -> None:
        if labels is None and size is None:
            raise ValueError("Either size or labels must be provided")
        self.labels = labels if labels is not None else [index % 3 for index in range(int(size))]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label = int(self.labels[index])
        image = torch.full((3, 3, 32, 32), fill_value=float(label))
        return image, label


def test_trainer_runs_one_epoch_and_persists_artifacts(tmp_path: Path) -> None:
    train_loader = DataLoader(SyntheticLumbarDataset(9), batch_size=3, shuffle=False)
    val_loader = DataLoader(SyntheticLumbarDataset(6), batch_size=3, shuffle=False)
    model = create_model(
        model_name="convnext_tiny_cbam",
        num_classes=3,
        fusion_enabled=True,
        pretrained=False,
        in_channels=3,
        dropout=0.1,
    )
    config = TrainingConfig(
        model_name="convnext_tiny_cbam",
        fusion_enabled=True,
        runs_root=tmp_path,
        epochs=1,
        batch_size=3,
        num_workers=0,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer_name="adamw",
        scheduler_name="cosine",
        amp=False,
        device="cpu",
        seed=7,
    )

    trainer = Trainer(model=model, config=config)
    result = trainer.fit(train_loader=train_loader, val_loader=val_loader)

    assert result.best_checkpoint.exists()
    assert result.metrics_csv.exists()
    assert result.history_json.exists()

    with result.metrics_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["epoch"] == "1"
    assert "train_macro_f1" in rows[0]
    assert "val_accuracy" in rows[0]


def test_trainer_writes_predictions_csv_with_required_columns(tmp_path: Path) -> None:
    train_loader = DataLoader(SyntheticLumbarDataset(size=9), batch_size=3, shuffle=False)
    val_loader = DataLoader(SyntheticLumbarDataset(size=6), batch_size=3, shuffle=False)
    model = create_model(
        model_name="convnext_tiny_cbam",
        num_classes=3,
        fusion_enabled=True,
        pretrained=False,
        in_channels=3,
        dropout=0.1,
    )
    config = TrainingConfig(
        model_name="convnext_tiny_cbam",
        fusion_enabled=True,
        runs_root=tmp_path,
        epochs=1,
        batch_size=3,
        num_workers=0,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer_name="adamw",
        scheduler_name="cosine",
        amp=False,
        device="cpu",
        seed=7,
    )

    trainer = Trainer(model=model, config=config)
    result = trainer.fit(train_loader=train_loader, val_loader=val_loader)

    prediction_path = result.run_dir / "predictions.csv"
    assert prediction_path.exists(), "训练完成后应生成 predictions.csv"

    with prediction_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"y_true", "y_pred", "score_0", "score_1", "score_2"}
        assert reader.fieldnames is not None
        assert required_columns.issubset(set(reader.fieldnames))


def test_trainer_metrics_csv_contains_recall_and_prediction_rate_columns(tmp_path: Path) -> None:
    train_loader = DataLoader(SyntheticLumbarDataset(size=9), batch_size=3, shuffle=False)
    val_loader = DataLoader(SyntheticLumbarDataset(size=6), batch_size=3, shuffle=False)
    model = create_model(
        model_name="convnext_tiny_cbam",
        num_classes=3,
        fusion_enabled=True,
        pretrained=False,
        in_channels=3,
        dropout=0.1,
    )
    config = TrainingConfig(
        model_name="convnext_tiny_cbam",
        fusion_enabled=True,
        runs_root=tmp_path,
        epochs=1,
        batch_size=3,
        num_workers=0,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer_name="adamw",
        scheduler_name="cosine",
        amp=False,
        device="cpu",
        seed=7,
    )

    trainer = Trainer(model=model, config=config)
    result = trainer.fit(train_loader=train_loader, val_loader=val_loader)

    with result.metrics_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows, "metrics.csv 至少应包含一行指标"
    row = rows[-1]
    required_columns = {
        "train_recall_class_0",
        "train_recall_class_1",
        "train_recall_class_2",
        "val_recall_class_0",
        "val_recall_class_1",
        "val_recall_class_2",
        "train_pred_rate_0",
        "train_pred_rate_1",
        "train_pred_rate_2",
        "val_pred_rate_0",
        "val_pred_rate_1",
        "val_pred_rate_2",
    }
    missing_columns = required_columns.difference(row.keys())
    assert not missing_columns, f"metrics.csv 缺少诊断指标列: {sorted(missing_columns)}"

    for column_name in sorted(required_columns):
        value = row[column_name]
        assert value != ""
        parsed_value = float(value)
        assert 0.0 <= parsed_value <= 1.0


def test_trainer_uses_balanced_class_weights_for_cross_entropy(tmp_path: Path) -> None:
    config_fields = {field.name for field in fields(TrainingConfig)}
    assert "class_weight_mode" in config_fields, "TrainingConfig 需要支持 class_weight_mode"

    train_labels = [0, 0, 0, 0, 0, 0, 1, 1, 2]
    train_loader = DataLoader(SyntheticLumbarDataset(labels=train_labels), batch_size=3, shuffle=False)
    val_loader = DataLoader(SyntheticLumbarDataset(size=6), batch_size=3, shuffle=False)
    model = create_model(
        model_name="convnext_tiny_cbam",
        num_classes=3,
        fusion_enabled=True,
        pretrained=False,
        in_channels=3,
        dropout=0.1,
    )
    config = TrainingConfig(
        model_name="convnext_tiny_cbam",
        fusion_enabled=True,
        runs_root=tmp_path,
        epochs=1,
        batch_size=3,
        num_workers=0,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer_name="adamw",
        scheduler_name="cosine",
        amp=False,
        device="cpu",
        seed=7,
        class_weight_mode="balanced",
    )

    trainer = Trainer(model=model, config=config)
    trainer.fit(train_loader=train_loader, val_loader=val_loader)

    assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
    assert trainer.criterion.weight is not None

    weights = trainer.criterion.weight.detach().cpu()
    assert weights.numel() == 3
    assert float(weights[2]) > float(weights[1]) > float(weights[0])

    counts = torch.tensor([6.0, 2.0, 1.0], dtype=weights.dtype)
    products = weights * counts
    assert torch.allclose(products, products[0].expand_as(products), rtol=1e-3, atol=1e-3)


def test_trainer_builds_focal_loss_when_requested(tmp_path: Path) -> None:
    train_loader = DataLoader(SyntheticLumbarDataset(labels=[0, 0, 0, 1, 1, 2]), batch_size=3, shuffle=False)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 3 * 32 * 32, 3))
    config = SimpleNamespace(
        model_name="convnext_tiny_cbam",
        fusion_enabled=False,
        runs_root=tmp_path,
        epochs=1,
        batch_size=3,
        num_workers=0,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer_name="adamw",
        scheduler_name="cosine",
        amp=False,
        device="cpu",
        seed=7,
        class_weight_mode=None,
        sampler_mode="balanced",
        overfit_subset_size=None,
        early_stopping_patience=None,
        train_augment_mode=None,
        loss_name="focal",
        focal_gamma=2.0,
    )

    trainer = Trainer(model=model, config=config)
    criterion = trainer._build_criterion(train_loader)

    assert criterion.__class__.__name__ == "FocalLoss"


def test_trainer_rejects_combining_focal_loss_with_class_weights(tmp_path: Path) -> None:
    train_loader = DataLoader(SyntheticLumbarDataset(labels=[0, 0, 0, 1, 1, 2]), batch_size=3, shuffle=False)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 3 * 32 * 32, 3))
    config = SimpleNamespace(
        model_name="convnext_tiny_cbam",
        fusion_enabled=False,
        runs_root=tmp_path,
        epochs=1,
        batch_size=3,
        num_workers=0,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer_name="adamw",
        scheduler_name="cosine",
        amp=False,
        device="cpu",
        seed=7,
        class_weight_mode="balanced",
        sampler_mode="balanced",
        overfit_subset_size=None,
        early_stopping_patience=None,
        train_augment_mode=None,
        loss_name="focal",
        focal_gamma=2.0,
    )

    trainer = Trainer(model=model, config=config)

    with pytest.raises(ValueError, match="focal"):
        trainer._build_criterion(train_loader)


def test_trainer_metrics_include_per_class_recall_and_prediction_distribution(tmp_path: Path) -> None:
    train_loader = DataLoader(SyntheticLumbarDataset(labels=[0, 0, 0, 1, 1, 2]), batch_size=3, shuffle=False)
    val_loader = DataLoader(SyntheticLumbarDataset(labels=[0, 1, 2, 0, 1, 2]), batch_size=3, shuffle=False)
    model = create_model(
        model_name="convnext_tiny_cbam",
        num_classes=3,
        fusion_enabled=True,
        pretrained=False,
        in_channels=3,
        dropout=0.1,
    )
    config = TrainingConfig(
        model_name="convnext_tiny_cbam",
        fusion_enabled=True,
        runs_root=tmp_path,
        epochs=1,
        batch_size=3,
        num_workers=0,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer_name="adamw",
        scheduler_name="cosine",
        amp=False,
        device="cpu",
        seed=7,
    )

    trainer = Trainer(model=model, config=config)
    result = trainer.fit(train_loader=train_loader, val_loader=val_loader)

    with result.metrics_csv.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))

    required_columns = {
        "train_recall_class_0",
        "train_recall_class_1",
        "train_recall_class_2",
        "val_recall_class_0",
        "val_recall_class_1",
        "val_recall_class_2",
        "train_pred_rate_0",
        "train_pred_rate_1",
        "train_pred_rate_2",
        "val_pred_rate_0",
        "val_pred_rate_1",
        "val_pred_rate_2",
    }
    assert required_columns.issubset(row.keys())

    for column in required_columns:
        value = float(row[column])
        assert 0.0 <= value <= 1.0


def test_trainer_stops_early_when_validation_metric_stalls(tmp_path: Path, monkeypatch) -> None:
    config_fields = {field.name for field in fields(TrainingConfig)}
    assert "early_stopping_patience" in config_fields, "TrainingConfig 需要支持 early_stopping_patience"

    train_loader = DataLoader(SyntheticLumbarDataset(size=3), batch_size=1, shuffle=False)
    val_loader = DataLoader(SyntheticLumbarDataset(size=3), batch_size=1, shuffle=False)
    model = create_model(
        model_name="convnext_tiny_cbam",
        num_classes=3,
        fusion_enabled=True,
        pretrained=False,
        in_channels=3,
        dropout=0.1,
    )
    config = TrainingConfig(
        model_name="convnext_tiny_cbam",
        fusion_enabled=True,
        runs_root=tmp_path,
        epochs=5,
        batch_size=1,
        num_workers=0,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer_name="adamw",
        scheduler_name="cosine",
        amp=False,
        device="cpu",
        seed=7,
        early_stopping_patience=0,
    )
    trainer = Trainer(model=model, config=config)

    def fake_run_epoch(loader: object, training: bool, epoch: int, collect_scores: bool = False, tta_count: int = 1) -> tuple[dict[str, float], dict[str, list[object]]]:
        if training:
            return {"macro_f1": 0.5, "accuracy": 0.5, "loss": 1.0}, {"targets": [], "predictions": [], "scores": []}
        score = 0.6 if epoch == 1 else 0.4
        return {
            "macro_f1": score,
            "accuracy": score,
            "loss": 1.0,
        }, {"targets": [0], "predictions": [0], "scores": [[1.0, 0.0, 0.0]]}

    monkeypatch.setattr(trainer, "_run_epoch", fake_run_epoch)
    result = trainer.fit(train_loader=train_loader, val_loader=val_loader)

    with result.metrics_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2, "patience=0 时应在第一次验证退化后停止"
    assert rows[-1]["epoch"] == "2"
    assert result.best_checkpoint.exists()
    assert getattr(result, "best_epoch", None) == 1


def test_training_config_supports_early_stopping_patience() -> None:
    config_fields = {field.name for field in fields(TrainingConfig)}
    assert "early_stopping_patience" in config_fields, "TrainingConfig 需要支持 early_stopping_patience"


def test_trainer_stops_early_and_metrics_match_actual_epochs(tmp_path: Path, monkeypatch: object) -> None:
    config_fields = {field.name for field in fields(TrainingConfig)}
    assert "early_stopping_patience" in config_fields, "TrainingConfig 需要支持 early_stopping_patience"

    train_loader = DataLoader(SyntheticLumbarDataset(size=9), batch_size=3, shuffle=False)
    val_loader = DataLoader(SyntheticLumbarDataset(size=6), batch_size=3, shuffle=False)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 3 * 32 * 32, 3))
    config = TrainingConfig(
        model_name="convnext_tiny_cbam",
        fusion_enabled=False,
        runs_root=tmp_path,
        epochs=8,
        batch_size=3,
        num_workers=0,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer_name="adamw",
        scheduler_name="cosine",
        amp=False,
        device="cpu",
        seed=7,
        early_stopping_patience=1,
    )

    trainer = Trainer(model=model, config=config)
    val_macro_f1_by_epoch = [0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25]

    def fake_run_epoch(
        _loader: DataLoader[tuple[torch.Tensor, int]],
        training: bool,
        epoch: int,
        collect_scores: bool = False,
        tta_count: int = 1,
    ) -> tuple[dict[str, float], dict[str, list[float] | list[int] | list[list[float]]]]:
        if training:
            metrics = {
                "macro_f1": 0.5,
                "accuracy": 0.5,
                "loss": 1.0,
                "recall_class_0": 0.5,
                "recall_class_1": 0.5,
                "recall_class_2": 0.5,
                "pred_rate_0": 1 / 3,
                "pred_rate_1": 1 / 3,
                "pred_rate_2": 1 / 3,
            }
            return metrics, {"targets": [], "predictions": [], "scores": []}
        score = val_macro_f1_by_epoch[epoch - 1]
        metrics = {
            "macro_f1": score,
            "accuracy": score,
            "loss": 1.0,
            "recall_class_0": score,
            "recall_class_1": 0.0,
            "recall_class_2": 0.0,
            "pred_rate_0": 1.0,
            "pred_rate_1": 0.0,
            "pred_rate_2": 0.0,
        }
        outputs = {
            "targets": [0, 1, 2],
            "predictions": [0, 0, 0],
            "scores": [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.9, 0.05, 0.05]],
        }
        return metrics, outputs

    monkeypatch.setattr(trainer, "_run_epoch", fake_run_epoch)
    result = trainer.fit(train_loader=train_loader, val_loader=val_loader)

    with result.metrics_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) < config.epochs, "启用 early stopping 后，训练轮数应少于配置 epochs"
    assert int(rows[-1]["epoch"]) == len(rows), "metrics.csv 行数应与实际训练轮数一致"

    assert result.best_checkpoint.exists(), "提前停止不应导致 best checkpoint 丢失"
    checkpoint = torch.load(result.best_checkpoint, map_location="cpu")
    assert int(checkpoint["epoch"]) == 1, "best epoch 应保持为验证指标峰值对应的轮次"
