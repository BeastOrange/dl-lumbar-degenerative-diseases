"""PyTorch training loop with artifact persistence."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from dl_lumbar_dd.train.config import TrainingConfig, TrainingResult
from dl_lumbar_dd.train.metrics import classification_metrics, metric_row
from dl_lumbar_dd.utils.io import ensure_dir


class Trainer:
    def __init__(self, model: nn.Module, config: TrainingConfig) -> None:
        self.config = config
        self._set_seed(config.seed)
        self.device = self._resolve_device(config.device)
        self.model = model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp and self.device.type == "cuda")
        self.run_dir = self._create_run_dir()

    def fit(self, train_loader: DataLoader[Any], val_loader: DataLoader[Any]) -> TrainingResult:
        best_score = float("-inf")
        history: list[dict[str, float | int]] = []
        best_checkpoint = self.run_dir / "best.ckpt"
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self._run_epoch(val_loader, training=False)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            history.append(metric_row(epoch, learning_rate, {"train": train_metrics, "val": val_metrics}))
            if val_metrics["macro_f1"] >= best_score:
                best_score = float(val_metrics["macro_f1"])
                self._save_checkpoint(best_checkpoint, epoch, history)
            if self.scheduler is not None:
                self.scheduler.step()
        metrics_csv = self._write_metrics_csv(history)
        history_json = self._write_history_json(history)
        return TrainingResult(self.run_dir, best_checkpoint, metrics_csv, history_json)

    def _run_epoch(self, loader: DataLoader[Any], training: bool) -> dict[str, float]:
        self.model.train(mode=training)
        losses: list[float] = []
        predictions: list[int] = []
        targets: list[int] = []
        grad_context = torch.enable_grad() if training else torch.no_grad()
        for batch in loader:
            inputs, labels = self._move_batch(batch)
            autocast_context = self._autocast_context()
            with grad_context, autocast_context:
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self._backward_step(loss)
            losses.append(float(loss.detach().cpu()))
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())
        metrics = classification_metrics(targets=targets, predictions=predictions)
        metrics["loss"] = sum(losses) / max(len(losses), 1)
        return metrics

    def _move_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            inputs = batch.get("images") or batch.get("image") or batch.get("views")
            labels = batch.get("labels") or batch.get("label")
        else:
            inputs, labels = batch
        if inputs is None or labels is None:
            raise ValueError("Batch must provide inputs and labels")
        return inputs.to(self.device), labels.to(self.device)

    def _backward_step(self, loss: torch.Tensor) -> None:
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            return
        loss.backward()
        self.optimizer.step()

    def _autocast_context(self) -> Any:
        if not self.config.amp or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.config.optimizer_name.lower()
        if name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay, momentum=0.9)
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        name = self.config.scheduler_name.lower()
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(self.config.epochs, 1))
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(self.config.epochs // 2, 1), gamma=0.1)
        return None

    def _create_run_dir(self) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        run_name = f"{self.config.model_name}-{timestamp}"
        return ensure_dir(Path(self.config.runs_root) / run_name)

    def _save_checkpoint(self, path: Path, epoch: int, history: list[dict[str, Any]]) -> None:
        torch.save(
            {
                "epoch": epoch,
                "config": asdict(self.config),
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": history,
            },
            path,
        )

    def _write_metrics_csv(self, history: list[dict[str, Any]]) -> Path:
        destination = self.run_dir / "metrics.csv"
        with destination.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        return destination

    def _write_history_json(self, history: list[dict[str, Any]]) -> Path:
        destination = self.run_dir / "history.json"
        destination.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return destination

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)
