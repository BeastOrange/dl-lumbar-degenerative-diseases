from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from dl_lumbar_dd.train import commands as train_commands


def test_build_training_config_supports_focal_loss_fields() -> None:
    config = train_commands._build_training_config(
        {
            "model_name": "convnext_tiny_cbam",
            "fusion_enabled": True,
            "runs_root": "./artifacts/runs",
            "loss_name": "focal",
            "focal_gamma": 1.5,
        }
    )

    assert hasattr(config, "loss_name"), "TrainingConfig 需要支持 loss_name"
    assert config.loss_name == "focal"
    assert hasattr(config, "focal_gamma"), "TrainingConfig 需要支持 focal_gamma"
    assert config.focal_gamma == 1.5


def test_run_training_seeds_before_model_creation(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_root": "./dataset",
                "processed_root": str(tmp_path / "processed"),
                "runs_root": str(tmp_path / "runs"),
                "metadata_root": str(tmp_path / "metadata"),
                "model_name": "convnext_tiny_cbam",
                "fusion_enabled": True,
                "target_column": "spinal_canal_stenosis_l4_l5",
                "num_classes": 3,
                "pretrained": False,
                "in_channels": 3,
                "dropout": 0.1,
                "batch_size": 2,
                "num_workers": 0,
                "epochs": 1,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "amp": False,
                "seed": 7,
                "image_size": 64,
                "device": "cpu",
                "folds": 3,
                "train_ratio": 0.8,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    captured_random_states: list[tuple[float, float]] = []
    run_counter = 0

    monkeypatch.setattr(train_commands, "build_dataloaders", lambda **_: ("train_loader", "val_loader"))

    def fake_create_model(**_: object) -> object:
        captured_random_states.append((random.random(), float(torch.rand(1).item())))
        return object()

    class DummyTrainer:
        def __init__(self, model: object, config: object) -> None:
            self.model = model
            self.config = config

        def fit(self, train_loader: object, val_loader: object) -> SimpleNamespace:
            nonlocal run_counter
            assert train_loader == "train_loader"
            assert val_loader == "val_loader"
            run_dir = tmp_path / f"run_{run_counter}"
            run_counter += 1
            run_dir.mkdir(parents=True, exist_ok=True)
            best_checkpoint = run_dir / "best.ckpt"
            best_checkpoint.write_text("checkpoint", encoding="utf-8")
            metrics_csv = run_dir / "metrics.csv"
            metrics_csv.write_text("epoch,val_macro_f1\n1,0.5\n", encoding="utf-8")
            history_json = run_dir / "history.json"
            history_json.write_text("[]", encoding="utf-8")
            return SimpleNamespace(
                run_dir=run_dir,
                best_checkpoint=best_checkpoint,
                metrics_csv=metrics_csv,
                history_json=history_json,
                best_epoch=1,
                predictions_csv=None,
            )

    monkeypatch.setattr(train_commands, "create_model", fake_create_model)
    monkeypatch.setattr(train_commands, "Trainer", DummyTrainer)
    monkeypatch.setattr(train_commands, "_persist_history_plot", lambda _history_json, run_dir: run_dir / "plot.png")

    random.seed(123)
    torch.manual_seed(123)
    train_commands.run_training(config_path)

    random.seed(999)
    torch.manual_seed(999)
    train_commands.run_training(config_path)

    assert len(captured_random_states) == 2
    assert captured_random_states[0] == captured_random_states[1]
