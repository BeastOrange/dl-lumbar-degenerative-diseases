"""Model building blocks for lumbar classification."""

from __future__ import annotations

import math

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mlp(self.pool(x))


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_map, max_map], dim=1))
        return x * attention


class CBAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        return self.spatial(x)


class DenseReuseProjection(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        hidden = max(feature_dim // 2, 32)
        self.reuse = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
        )
        self.project = nn.Linear(feature_dim + hidden, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reused = self.reuse(x)
        return self.project(torch.cat([x, reused], dim=1))


class FeatureVolume3D(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        depth = 4
        height = 8
        width = math.ceil(feature_dim / (depth * height))
        padded_dim = depth * height * width
        self.feature_dim = feature_dim
        self.depth = depth
        self.height = height
        self.width = width
        self.padded_dim = padded_dim
        self.conv = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(4),
            nn.GELU(),
            nn.Conv3d(4, 1, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padded_dim != self.feature_dim:
            x = torch.nn.functional.pad(x, (0, self.padded_dim - self.feature_dim))
        volume = x.view(x.size(0), 1, self.depth, self.height, self.width)
        volume = self.conv(volume)
        flattened = volume.flatten(start_dim=1)
        return flattened[:, : self.feature_dim]


class HierarchicalFeatureFusion(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        hidden = max(feature_dim // 4, 32)
        self.local_branch = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden))
        self.global_branch = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
        )
        self.merge = nn.Linear(hidden * 2, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local_branch(x)
        global_context = self.global_branch(x)
        return self.merge(torch.cat([local, global_context], dim=1))


class MultiViewFusionAdapter(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        hidden = max(feature_dim // 4, 32)
        self.score = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.refine = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(features).squeeze(-1), dim=1).unsqueeze(-1)
        fused = (features * weights).sum(dim=1)
        return fused + self.refine(features.mean(dim=1))
