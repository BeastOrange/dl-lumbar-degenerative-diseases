"""Model registry for lumbar classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dl_lumbar_dd.models.backbones import BACKBONE_FACTORIES
from dl_lumbar_dd.models.blocks import MultiViewFusionAdapter


@dataclass(slots=True)
class ModelSpec:
    model_name: str
    num_classes: int
    fusion_enabled: bool = True
    pretrained: bool = True
    load_backbone_weights: bool = True
    in_channels: int = 3
    dropout: float = 0.2
    image_size: int | None = 64
    num_tasks: int = 1  # 1 = single-task, >1 = multi-task with N heads


class _ImagenetNormalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=inputs.device, dtype=inputs.dtype)
        std = self.std.to(device=inputs.device, dtype=inputs.dtype)
        return (inputs - mean) / std


class _RepeatGrayscaleToRgb(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.repeat(1, 3, 1, 1)


class LumbarModel(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        encoder_factory = BACKBONE_FACTORIES[spec.model_name]
        self.encoder = encoder_factory(spec.pretrained and spec.load_backbone_weights, spec.image_size)
        self.input_adapter = nn.Identity()
        if spec.pretrained and spec.in_channels == 1:
            self.input_adapter = _RepeatGrayscaleToRgb()
        elif spec.in_channels != 3:
            self.input_adapter = nn.Conv2d(spec.in_channels, 3, kernel_size=1)
        self.input_normalizer = _ImagenetNormalize() if spec.pretrained else nn.Identity()
        self.fusion = MultiViewFusionAdapter(self.encoder.feature_dim) if spec.fusion_enabled else None
        self.norm = nn.LayerNorm(self.encoder.feature_dim)
        self.dropout = nn.Dropout(spec.dropout)
        self.num_tasks = spec.num_tasks
        self.num_classes = spec.num_classes
        # Multi-task: one head per task; single-task: one shared head
        self.heads = nn.ModuleList([
            nn.Linear(self.encoder.feature_dim, spec.num_classes)
            for _ in range(spec.num_tasks)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        views = self._prepare_views(inputs)
        features = torch.stack([self.encoder(view) for view in views], dim=1)
        pooled = self.fusion(features) if self.fusion is not None and features.size(1) > 1 else features.mean(dim=1)
        pooled = self.dropout(self.norm(pooled))
        if self.num_tasks == 1:
            return self.heads[0](pooled)
        # Multi-task: stack logits from all heads → (batch, num_tasks, num_classes)
        return torch.stack([head(pooled) for head in self.heads], dim=1)

    def _prepare_views(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        if inputs.ndim == 4:
            return [self.input_normalizer(self.input_adapter(inputs))]
        if inputs.ndim != 5:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {tuple(inputs.shape)}")
        return [self.input_normalizer(self.input_adapter(inputs[:, index])) for index in range(inputs.size(1))]


def available_models() -> tuple[str, ...]:
    return tuple(BACKBONE_FACTORIES.keys())


def create_model(
    model_name: str,
    num_classes: int,
    fusion_enabled: bool = True,
    pretrained: bool = True,
    load_backbone_weights: bool = True,
    in_channels: int = 3,
    dropout: float = 0.2,
    image_size: int | None = 64,
    num_tasks: int = 1,
) -> nn.Module:
    if model_name not in BACKBONE_FACTORIES:
        raise ValueError(f"Unsupported model_name: {model_name}")
    spec = ModelSpec(
        model_name=model_name,
        num_classes=num_classes,
        fusion_enabled=fusion_enabled,
        pretrained=pretrained,
        load_backbone_weights=load_backbone_weights,
        in_channels=in_channels,
        dropout=dropout,
        image_size=image_size,
        num_tasks=num_tasks,
    )
    return LumbarModel(spec)
