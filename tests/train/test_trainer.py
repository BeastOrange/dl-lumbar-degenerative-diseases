from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from dl_lumbar_dd.models import create_model
from dl_lumbar_dd.train import Trainer, TrainingConfig


class SyntheticLumbarDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = torch.full((3, 3, 32, 32), fill_value=float(index % 3))
        label = index % 3
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
