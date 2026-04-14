"""Backbone wrappers for model registry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torchvision import models as tv_models

from dl_lumbar_dd.models.blocks import (
    CBAM,
    DenseReuseProjection,
    FeatureVolume3D,
    HierarchicalFeatureFusion,
)


def _resize_if_needed(x: torch.Tensor, image_size: int | None) -> torch.Tensor:
    if image_size is None or x.shape[-1] == image_size:
        return x
    return torch.nn.functional.interpolate(
        x,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )


class ConvNeXtCBAMEncoder(nn.Module):
    feature_dim = 768

    def __init__(self, pretrained: bool, image_size: int | None) -> None:
        super().__init__()
        self.image_size = image_size
        backbone = _create_convnext_tiny_backbone(pretrained)
        self.features = backbone.features
        self.cbam = CBAM(self.feature_dim)
        self.avgpool = backbone.avgpool
        self.norm = backbone.classifier[0]
        self.flatten = backbone.classifier[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _resize_if_needed(x, self.image_size)
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = self.norm(x)
        return self.flatten(x)


class DenseNetDenseReuseEncoder(nn.Module):
    feature_dim = 1024

    def __init__(self, pretrained: bool, image_size: int | None) -> None:
        super().__init__()
        weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = tv_models.densenet121(weights=weights)
        self.image_size = image_size
        self.features = backbone.features
        self.reuse = DenseReuseProjection(self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _resize_if_needed(x, self.image_size)
        x = self.features(x)
        x = torch.relu(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
        x = torch.flatten(x, start_dim=1)
        return self.reuse(x)


class ResNet3DEncoder(nn.Module):
    feature_dim = 2048

    def __init__(self, pretrained: bool, image_size: int | None) -> None:
        super().__init__()
        weights = tv_models.ResNet101_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet101(weights=weights)
        self.image_size = image_size
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.volume = FeatureVolume3D(self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _resize_if_needed(x, self.image_size)
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.volume(x)


class SwinHierarchicalEncoder(nn.Module):
    feature_dim = 768

    def __init__(self, pretrained: bool, image_size: int | None) -> None:
        super().__init__()
        weights = tv_models.Swin_T_Weights.DEFAULT if pretrained else None
        backbone = tv_models.swin_t(weights=weights)
        backbone.head = nn.Identity()
        self.image_size = image_size
        self.backbone = backbone
        self.fusion = HierarchicalFeatureFusion(self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _resize_if_needed(x, self.image_size)
        return self.fusion(self.backbone(x))


class ViTPositionalEncoder(nn.Module):
    feature_dim = 768

    def __init__(self, pretrained: bool, image_size: int | None) -> None:
        super().__init__()
        model_size = image_size or 224
        weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained and model_size == 224 else None
        backbone = tv_models.vit_b_16(weights=weights, image_size=model_size)
        backbone.heads = nn.Identity()
        self.image_size = model_size
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _resize_if_needed(x, self.image_size)
        return self.backbone(x)


BACKBONE_FACTORIES: dict[str, Callable[[bool, int | None], nn.Module]] = {
    "convnext_tiny_cbam": ConvNeXtCBAMEncoder,
    "densenet121_dense_reuse": DenseNetDenseReuseEncoder,
    "resnet101_3d": ResNet3DEncoder,
    "swin_transformer": SwinHierarchicalEncoder,
    "vit_base_posenc": ViTPositionalEncoder,
}


def _create_convnext_tiny_backbone(pretrained: bool) -> nn.Module:
    if not pretrained:
        return tv_models.convnext_tiny(weights=None)
    weights_path = _resolve_local_weights_path("LUMBAR_CONVNEXT_TINY_WEIGHTS")
    if weights_path is None:
        return tv_models.convnext_tiny(weights=tv_models.ConvNeXt_Tiny_Weights.DEFAULT)
    model = tv_models.convnext_tiny(weights=None)
    state_dict = _load_torchvision_checkpoint(weights_path)
    model.load_state_dict(state_dict, strict=True)
    return model


def _resolve_local_weights_path(env_name: str) -> Path | None:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return None
    path = Path(raw_value).expanduser()
    if not path.is_file():
        return None
    return path


def _load_torchvision_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if isinstance(state_dict, dict):
            return state_dict
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format: {path}")
