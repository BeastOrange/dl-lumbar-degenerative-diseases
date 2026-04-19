"""High-level training commands used by the CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from dl_lumbar_dd.config import load_yaml
from dl_lumbar_dd.constants import TARGET_COLUMNS
from dl_lumbar_dd.models import create_model
from dl_lumbar_dd.train.config import TrainingConfig
from dl_lumbar_dd.train.data import build_dataloaders
from dl_lumbar_dd.train.trainer import Trainer, set_global_seed
from dl_lumbar_dd.utils.io import write_json
from dl_lumbar_dd.visualization import save_training_history


def run_training(config_path: str | Path) -> dict[str, Any]:
    """Run one training job from YAML config and persist artifacts."""
    raw_config = load_yaml(config_path)
    training_config = _build_training_config(raw_config)
    set_global_seed(training_config.seed)

    stratify_col = _get_stratify_column(raw_config)
    target_columns_cfg = _get_target_columns(raw_config)
    num_tasks = len(target_columns_cfg) if target_columns_cfg else 1

    train_loader, validation_loader = build_dataloaders(
        dataset_root=str(raw_config["dataset_root"]),
        processed_root=str(raw_config["processed_root"]),
        target_column=stratify_col,
        image_size=training_config.image_size,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        seed=training_config.seed,
        folds=int(raw_config.get("folds", 3)),
        train_ratio=float(raw_config.get("train_ratio", 0.8)),
        max_studies=_as_optional_int(raw_config.get("max_studies")),
        max_train_samples=_as_optional_int(raw_config.get("max_train_samples")),
        max_val_samples=_as_optional_int(raw_config.get("max_val_samples")),
        sampler_mode=training_config.sampler_mode,
        overfit_subset_size=training_config.overfit_subset_size,
        train_augment_mode=training_config.train_augment_mode,
        target_columns=target_columns_cfg,
        num_slices=int(raw_config.get("num_slices", 1)),
    )
    model = create_model(
        model_name=training_config.model_name,
        num_classes=int(raw_config.get("num_classes", 3)),
        fusion_enabled=training_config.fusion_enabled,
        pretrained=bool(raw_config.get("pretrained", False)),
        in_channels=int(raw_config.get("in_channels", 1)),
        dropout=float(raw_config.get("dropout", 0.2)),
        image_size=training_config.image_size,
        num_tasks=num_tasks,
    )
    trainer = Trainer(model=model, config=training_config)
    result = trainer.fit(train_loader=train_loader, val_loader=validation_loader)
    _persist_run_config(raw_config, result.run_dir)
    history_plot = _persist_history_plot(result.history_json, result.run_dir)
    summary = {
        "run_dir": str(result.run_dir),
        "best_checkpoint": str(result.best_checkpoint),
        "best_epoch": result.best_epoch,
        "metrics_csv": str(result.metrics_csv),
        "history_json": str(result.history_json),
        "history_plot": str(history_plot),
        "model_name": training_config.model_name,
        "fusion_enabled": training_config.fusion_enabled,
        "num_tasks": num_tasks,
    }
    if result.predictions_csv is not None:
        summary["predictions_csv"] = str(result.predictions_csv)
    write_json(summary, result.run_dir / "run_summary.json")
    return summary


def _get_stratify_column(raw_config: dict[str, Any]) -> str:
    """Get the column used for stratified splitting. Used as single-task fallback."""
    return str(raw_config.get("target_column", "spinal_canal_stenosis_l4_l5"))


def _get_target_columns(raw_config: dict[str, Any]) -> list[str] | None:
    """Parse target_columns from config.

    Supports three forms:
      - null / "single": single-task mode (target_column used)
      - "all": all 25 TARGET_COLUMNS (multi-task)
      - [col1, col2, ...]: explicit list
    """
    value = raw_config.get("target_columns", None)
    if value is None:
        return None
    if isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower in ("", "null", "single"):
            return None
        if value_lower == "all":
            return list(TARGET_COLUMNS)
    if isinstance(value, list):
        return [str(v) for v in value]
    return None


def _build_training_config(config: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        model_name=str(config["model_name"]),
        fusion_enabled=bool(config.get("fusion_enabled", True)),
        runs_root=str(config["runs_root"]),
        epochs=int(config.get("epochs", 8)),
        batch_size=int(config.get("batch_size", 16)),
        num_workers=int(config.get("num_workers", 4)),
        learning_rate=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        optimizer_name=str(config.get("optimizer", "adamw")),
        scheduler_name=str(config.get("scheduler", "cosine")),
        amp=bool(config.get("amp", True)),
        device=str(config.get("device", "auto")),
        seed=int(config.get("seed", 42)),
        image_size=int(config.get("image_size", 224)),
        loss_name=str(config.get("loss_name", "cross_entropy")),
        focal_gamma=float(config.get("focal_gamma", 2.0)),
        class_weight_mode=_as_optional_str(config.get("class_weight_mode")),
        sampler_mode=_as_optional_str(config.get("sampler_mode")),
        overfit_subset_size=_as_optional_int(config.get("overfit_subset_size")),
        early_stopping_patience=_as_optional_int(config.get("early_stopping_patience")),
        train_augment_mode=_as_optional_str(config.get("train_augment_mode")),
        label_smoothing=float(config.get("label_smoothing", 0.0)),
        tta_count=int(config.get("tta_count", 1)),
    )


def _persist_run_config(config: dict[str, Any], run_dir: Path) -> None:
    output = run_dir / "config.yaml"
    output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _persist_history_plot(history_json: Path, run_dir: Path) -> Path:
    rows = json.loads(history_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        rows = []
    series = _history_rows_to_series(rows)
    return save_training_history(series, run_dir / "history_train_metrics.png")


def _history_rows_to_series(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key == "epoch" or not isinstance(value, (int, float)):
                continue
            series.setdefault(key, []).append(float(value))
    return series


def _as_optional_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    return int(value)


def _as_optional_str(value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    return str(value)
