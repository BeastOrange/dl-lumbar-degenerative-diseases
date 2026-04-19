"""PyTorch training loop with artifact persistence."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from dataclasses import asdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dl_lumbar_dd.train.config import TrainingConfig, TrainingResult
from dl_lumbar_dd.train.metrics import classification_metrics, metric_row
from dl_lumbar_dd.utils.io import ensure_dir


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-cross_entropy)
        loss = torch.pow(1.0 - pt, self.gamma) * cross_entropy
        return loss.mean()


class Trainer:
    def __init__(self, model: nn.Module, config: TrainingConfig) -> None:
        self.config = config
        set_global_seed(config.seed)
        self.device = self._resolve_device(config.device)
        self.model = model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        # criterion will be built in fit() after we know the dataset size
        self.criterion: nn.Module | list[nn.Module] | None = None
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp and self.device.type == "cuda")
        self.run_dir = self._create_run_dir()
        # Detect multi-task mode from model's num_tasks attribute
        self.num_tasks = getattr(self.model, "num_tasks", 1)

    def fit(self, train_loader: DataLoader[Any], val_loader: DataLoader[Any]) -> TrainingResult:
        best_score = float("-inf")
        best_epoch: int | None = None
        epochs_without_improvement = 0
        history: list[dict[str, float | int]] = []
        best_checkpoint = self.run_dir / "best.ckpt"
        predictions_csv: Path | None = None
        self.criterion = self._build_criterion(train_loader)
        for epoch in range(1, self.config.epochs + 1):
            train_metrics, _ = self._run_epoch(train_loader, training=True, epoch=epoch)
            val_metrics, val_outputs = self._run_epoch(
                val_loader, training=False, epoch=epoch, collect_scores=True,
                tta_count=int(getattr(self.config, "tta_count", 1)),
            )
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            history.append(metric_row(epoch, learning_rate, {"train": train_metrics, "val": val_metrics}))
            current_score = float(val_metrics["macro_f1"])
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint(best_checkpoint, epoch, history)
                predictions_csv = self._write_predictions_csv(
                    targets=val_outputs["targets"],
                    predictions=val_outputs["predictions"],
                    scores=val_outputs["scores"],
                )
            else:
                epochs_without_improvement += 1
            if self.scheduler is not None:
                self.scheduler.step()
            if self._should_stop_early(epochs_without_improvement):
                break
        metrics_csv = self._write_metrics_csv(history)
        history_json = self._write_history_json(history)
        return TrainingResult(self.run_dir, best_checkpoint, metrics_csv, history_json, best_epoch, predictions_csv)

    def _run_epoch(
        self,
        loader: DataLoader[Any],
        training: bool,
        epoch: int,
        collect_scores: bool = False,
        tta_count: int = 1,
    ) -> tuple[dict[str, float], dict[str, list[Any]]]:
        self.model.train(mode=training)
        losses: list[float] = []
        predictions: list[int] = []
        targets: list[int] = []
        scores: list[list[float]] = []
        grad_context = torch.enable_grad() if training else torch.no_grad()
        progress = tqdm(
            loader,
            total=len(loader),
            desc=self._progress_desc(training=training, epoch=epoch),
            unit="batch",
            dynamic_ncols=True,
            disable=not self._should_show_progress(),
        )
        from dl_lumbar_dd.train.data import apply_tta_augmentation

        accum_steps = max(getattr(self.config, "grad_accumulation_steps", 1), 1)
        mixup_alpha = getattr(self.config, "mixup_alpha", 0.0)

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(progress):
            inputs, labels = self._move_batch(batch)
            autocast_context = self._autocast_context()
            with grad_context, autocast_context:
                if training and mixup_alpha > 0.0:
                    inputs, labels_a, labels_b, lam = self._apply_mixup(inputs, labels, mixup_alpha)
                    logits = self.model(inputs)
                    loss = lam * self._compute_loss(logits, labels_a) + (1 - lam) * self._compute_loss(logits, labels_b)
                    metric_labels = labels_a
                elif training or tta_count <= 1:
                    logits = self.model(inputs)
                    loss = self._compute_loss(logits, labels)
                    metric_labels = labels
                else:
                    logits_list: list[torch.Tensor] = []
                    for t in range(tta_count):
                        aug_inputs = inputs if t == 0 else apply_tta_augmentation(inputs)
                        logits_list.append(self.model(aug_inputs))
                    logits = torch.stack(logits_list, dim=0).mean(dim=0)
                    loss = self._compute_loss(logits, labels)
                    metric_labels = labels

            if training:
                scaled_loss = loss / accum_steps if accum_steps > 1 else loss
                self._backward_only(scaled_loss)
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                    self._optimizer_step()
                    self.optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().cpu()))
            if logits.ndim == 3:
                task0_logits = logits[:, 0, :]
                task0_labels = metric_labels[:, 0]
            else:
                task0_logits = logits
                task0_labels = metric_labels
            probabilities = torch.softmax(task0_logits.detach(), dim=1)
            predictions.extend(probabilities.argmax(dim=1).cpu().tolist())
            targets.extend(task0_labels.detach().cpu().tolist())
            if collect_scores:
                scores.extend(probabilities.cpu().tolist())
            progress.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")
        progress.close()
        metrics = classification_metrics(
            targets=targets,
            predictions=predictions,
            num_classes=self._infer_num_classes(targets),
        )
        metrics["loss"] = sum(losses) / max(len(losses), 1)
        return metrics, {"targets": targets, "predictions": predictions, "scores": scores}

    def _move_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            inputs = batch.get("images") or batch.get("image") or batch.get("views")
            labels = batch.get("labels") or batch.get("label")
        else:
            inputs, labels = batch
        if inputs is None or labels is None:
            raise ValueError("Batch must provide inputs and labels")
        return inputs.to(self.device), labels.to(self.device)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for single-task or multi-task mode."""
        if self.num_tasks == 1:
            criteria = self.criterion
            assert criteria is not None
            return criteria(logits, labels)
        # Multi-task: one loss per task
        criteria = self.criterion
        assert isinstance(criteria, list)
        total_loss = torch.tensor(0.0, device=logits.device)
        for task_idx, crit in enumerate(criteria):
            task_logits = logits[:, task_idx, :]      # (batch, num_classes)
            task_labels = labels[:, task_idx]          # (batch,)
            total_loss = total_loss + crit(task_logits, task_labels)
        return total_loss

    def _backward_step(self, loss: torch.Tensor) -> None:
        self._backward_only(loss)
        self._optimizer_step()

    def _backward_only(self, loss: torch.Tensor) -> None:
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self) -> None:
        max_norm = getattr(self.config, "grad_clip_max_norm", None)
        if self.scaler.is_enabled():
            if max_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.optimizer.step()

    def _autocast_context(self) -> Any:
        if not self.config.amp or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    @staticmethod
    def _apply_mixup(
        inputs: torch.Tensor, labels: torch.Tensor, alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = float(np.random.beta(alpha, alpha))
        index = torch.randperm(inputs.size(0), device=inputs.device)
        mixed = lam * inputs + (1 - lam) * inputs[index]
        return mixed, labels, labels[index], lam

    def _build_criterion(self, train_loader: DataLoader[Any]) -> nn.Module | list[nn.Module]:
        """Build loss function(s). Returns a single Module for single-task, list for multi-task."""
        loss_name = getattr(self.config, "loss_name", "cross_entropy").strip().lower()
        mode = (self.config.class_weight_mode or "").strip().lower()
        ls = float(getattr(self.config, "label_smoothing", 0.0))

        if loss_name == "focal":
            if ls > 0.0:
                raise ValueError(
                    "label_smoothing > 0 is not compatible with focal loss. "
                    "Use loss_name='cross_entropy' when label_smoothing is enabled."
                )
            if mode:
                raise ValueError("focal loss does not support class_weight_mode")
            single = FocalLoss(gamma=float(getattr(self.config, "focal_gamma", 2.0)))
        elif loss_name in {"", "cross_entropy"}:
            if not mode and ls <= 0.0:
                single = nn.CrossEntropyLoss()
            elif mode and ls > 0.0:
                weights = self._build_balanced_class_weights(train_loader)
                single = nn.CrossEntropyLoss(
                    label_smoothing=ls,
                    weight=(weights.to(device=self.device, dtype=torch.float32) if weights is not None else None),
                )
            elif ls > 0.0:
                single = nn.CrossEntropyLoss(label_smoothing=ls)
            elif mode == "balanced":
                weights = self._build_balanced_class_weights(train_loader)
                single = nn.CrossEntropyLoss(
                    weight=weights.to(device=self.device, dtype=torch.float32) if weights is not None else None
                )
            else:
                raise ValueError(f"Unsupported class_weight_mode: {self.config.class_weight_mode}")
        else:
            raise ValueError(f"Unsupported loss_name: {getattr(self.config, 'loss_name', None)}")

        if self.num_tasks == 1:
            return single
        # Multi-task: one criterion per task (sharing the same config)
        return [single for _ in range(self.num_tasks)]

    def _build_balanced_class_weights(self, train_loader: DataLoader[Any]) -> torch.Tensor | None:
        label_indices = self._extract_label_indices(train_loader.dataset)
        if not label_indices:
            return None
        num_classes = self._infer_num_classes(label_indices)
        if num_classes <= 0:
            return None
        counts = torch.bincount(torch.tensor(label_indices, dtype=torch.long), minlength=num_classes).to(dtype=torch.float32)
        nonzero = counts > 0
        if not torch.any(nonzero):
            return None
        weights = torch.zeros_like(counts)
        denominator = float(num_classes)
        weights[nonzero] = counts[nonzero].sum() / (denominator * counts[nonzero])
        return weights

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.config.optimizer_name.lower()
        if name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay, momentum=0.9)
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        name = self.config.scheduler_name.lower()
        warmup = getattr(self.config, "warmup_epochs", 0)
        main_sched = self._build_main_scheduler(name, warmup)
        if warmup <= 0:
            return main_sched
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, total_iters=warmup,
        )
        if main_sched is None:
            return warmup_sched
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, main_sched],
            milestones=[warmup],
        )

    def _build_main_scheduler(self, name: str, warmup: int) -> torch.optim.lr_scheduler.LRScheduler | None:
        remaining = max(self.config.epochs - warmup, 1)
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=remaining)
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(remaining // 2, 1), gamma=0.1)
        return None

    def _create_run_dir(self) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        run_name = f"{self.config.model_name}-{timestamp}"
        return ensure_dir(Path(self.config.runs_root) / run_name)

    def _save_checkpoint(self, path: Path, epoch: int, history: list[dict[str, Any]]) -> None:
        torch.save(
            {
                "epoch": epoch,
                "config": self._serialized_config(),
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": history,
            },
            path,
        )

    def _serialized_config(self) -> dict[str, Any]:
        config_dict = asdict(self.config)
        for key, value in list(config_dict.items()):
            if isinstance(value, Path):
                config_dict[key] = str(value)
        return config_dict

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

    def _write_predictions_csv(
        self,
        targets: list[int],
        predictions: list[int],
        scores: list[list[float]],
    ) -> Path:
        destination = self.run_dir / "predictions.csv"
        score_count = max((len(row) for row in scores), default=self._infer_num_classes([]))
        fieldnames = ["y_true", "y_pred", *[f"score_{index}" for index in range(score_count)]]
        with destination.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for target, prediction, score_row in zip(targets, predictions, scores, strict=True):
                row: dict[str, int | float] = {"y_true": target, "y_pred": prediction}
                for index, score in enumerate(score_row):
                    row[f"score_{index}"] = float(score)
                writer.writerow(row)
        return destination

    def _should_stop_early(self, epochs_without_improvement: int) -> bool:
        patience = self.config.early_stopping_patience
        if patience is None:
            return False
        return epochs_without_improvement > patience

    def _extract_label_indices(self, dataset: Any) -> list[int]:
        direct = self._normalize_label_indices(getattr(dataset, "label_indices", None))
        if direct:
            return direct
        direct = self._normalize_label_indices(getattr(dataset, "labels", None))
        if direct:
            return direct
        direct = self._normalize_label_indices(getattr(dataset, "targets", None))
        if direct:
            return direct
        subset_indices = getattr(dataset, "indices", None)
        subset_source = getattr(dataset, "dataset", None)
        if subset_indices is None or subset_source is None:
            return []
        base_labels = self._extract_label_indices(subset_source)
        if not base_labels:
            return []
        return [base_labels[int(index)] for index in subset_indices if 0 <= int(index) < len(base_labels)]

    @staticmethod
    def _normalize_label_indices(raw_labels: Any) -> list[int]:
        if raw_labels is None:
            return []
        if isinstance(raw_labels, torch.Tensor):
            return [int(value) for value in raw_labels.detach().cpu().tolist()]
        try:
            return [int(value) for value in raw_labels]
        except TypeError:
            return []

    def _infer_num_classes(self, label_indices: list[int]) -> int:
        label_count = max(label_indices, default=-1) + 1
        model_head = getattr(self.model, "head", None)
        model_count = getattr(model_head, "out_features", 0)
        return max(int(label_count), int(model_count))

    def _progress_desc(self, training: bool, epoch: int) -> str:
        stage = "train" if training else "val"
        return f"Epoch {epoch}/{self.config.epochs} [{stage}]"

    @staticmethod
    def _should_show_progress() -> bool:
        override = os.getenv("LUMBAR_TQDM")
        if override is not None:
            return override.lower() not in {"0", "false", "no"}
        return sys.stdout.isatty() or sys.stderr.isatty()

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)
