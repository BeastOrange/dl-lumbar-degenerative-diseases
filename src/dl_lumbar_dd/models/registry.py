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
    in_channels: int = 3
    dropout: float = 0.2
    image_size: int | None = 64


class LumbarModel(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        encoder_factory = BACKBONE_FACTORIES[spec.model_name]
        self.encoder = encoder_factory(spec.pretrained, spec.image_size)
        self.input_adapter = nn.Identity()
        if spec.in_channels != 3:
            self.input_adapter = nn.Conv2d(spec.in_channels, 3, kernel_size=1)
        self.fusion = MultiViewFusionAdapter(self.encoder.feature_dim) if spec.fusion_enabled else None
        self.norm = nn.LayerNorm(self.encoder.feature_dim)
        self.dropout = nn.Dropout(spec.dropout)
        self.head = nn.Linear(self.encoder.feature_dim, spec.num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        views = self._prepare_views(inputs)
        features = torch.stack([self.encoder(view) for view in views], dim=1)
        pooled = self.fusion(features) if self.fusion is not None and features.size(1) > 1 else features.mean(dim=1)
        pooled = self.dropout(self.norm(pooled))
        return self.head(pooled)

    def _prepare_views(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        if inputs.ndim == 4:
            return [self.input_adapter(inputs)]
        if inputs.ndim != 5:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {tuple(inputs.shape)}")
        return [self.input_adapter(inputs[:, index]) for index in range(inputs.size(1))]


def available_models() -> tuple[str, ...]:
    return tuple(BACKBONE_FACTORIES.keys())


def create_model(
    model_name: str,
    num_classes: int,
    fusion_enabled: bool = True,
    pretrained: bool = True,
    in_channels: int = 3,
    dropout: float = 0.2,
    image_size: int | None = 64,
) -> nn.Module:
    if model_name not in BACKBONE_FACTORIES:
        raise ValueError(f"Unsupported model_name: {model_name}")
    spec = ModelSpec(
        model_name=model_name,
        num_classes=num_classes,
        fusion_enabled=fusion_enabled,
        pretrained=pretrained,
        in_channels=in_channels,
        dropout=dropout,
        image_size=image_size,
    )
    return LumbarModel(spec)
