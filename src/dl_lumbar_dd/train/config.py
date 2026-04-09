"""Training configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainingConfig:
    model_name: str
    fusion_enabled: bool
    runs_root: str | Path
    epochs: int = 8
    batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    optimizer_name: str = "adamw"
    scheduler_name: str = "cosine"
    amp: bool = True
    device: str = "auto"
    seed: int = 42
    image_size: int = 224


@dataclass(slots=True)
class TrainingResult:
    run_dir: Path
    best_checkpoint: Path
    metrics_csv: Path
    history_json: Path
