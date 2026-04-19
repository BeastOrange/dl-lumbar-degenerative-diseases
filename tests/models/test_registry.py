from __future__ import annotations

import pytest
import torch
from torch import nn

from dl_lumbar_dd.models import create_model
from dl_lumbar_dd.models import backbones
from dl_lumbar_dd.models.backbones import BACKBONE_FACTORIES


class _DummyEncoder(nn.Module):
    feature_dim = 8

    def __init__(self, pretrained: bool, image_size: int | None) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], self.feature_dim), device=x.device, dtype=x.dtype)


@pytest.mark.parametrize(
    ("model_name", "fusion_enabled"),
    [
        ("convnext_tiny_cbam", False),
        ("densenet121_dense_reuse", True),
        ("resnet101_3d", True),
        ("swin_transformer", False),
        ("vit_base_posenc", True),
    ],
)
def test_create_model_produces_logits_with_expected_shape(
    model_name: str,
    fusion_enabled: bool,
) -> None:
    model = create_model(
        model_name=model_name,
        num_classes=3,
        fusion_enabled=fusion_enabled,
        pretrained=False,
        in_channels=3,
        dropout=0.1,
    )
    batch = torch.randn(2, 3, 3, 64, 64)

    logits = model(batch)

    assert logits.shape == (2, 3)


def test_prepare_views_grayscale_pretrained_is_seed_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    model_name = "dummy_seed_stable"
    monkeypatch.setitem(BACKBONE_FACTORIES, model_name, _DummyEncoder)
    inputs = torch.rand(2, 1, 16, 16)

    torch.manual_seed(0)
    model_a = create_model(
        model_name=model_name,
        num_classes=3,
        fusion_enabled=False,
        pretrained=True,
        in_channels=1,
        dropout=0.1,
    )

    torch.manual_seed(1234)
    model_b = create_model(
        model_name=model_name,
        num_classes=3,
        fusion_enabled=False,
        pretrained=True,
        in_channels=1,
        dropout=0.1,
    )

    prepared_a = model_a._prepare_views(inputs)[0]
    prepared_b = model_b._prepare_views(inputs)[0]

    assert torch.allclose(
        prepared_a,
        prepared_b,
        atol=1e-6,
    ), "预训练灰度输入应做稳定通道复制，不应依赖随机初始化权重。"


def test_prepare_views_grayscale_pretrained_applies_imagenet_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = "dummy_imagenet_norm"
    monkeypatch.setitem(BACKBONE_FACTORIES, model_name, _DummyEncoder)
    model = create_model(
        model_name=model_name,
        num_classes=3,
        fusion_enabled=False,
        pretrained=True,
        in_channels=1,
        dropout=0.1,
    )
    inputs = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float32)

    prepared = model._prepare_views(inputs)[0]
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=prepared.dtype, device=prepared.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=prepared.dtype, device=prepared.device).view(1, 3, 1, 1)
    expected = (inputs.repeat(1, 3, 1, 1) - mean) / std

    assert prepared.shape == (1, 3, 8, 8)
    assert torch.allclose(
        prepared,
        expected,
        atol=1e-6,
    ), "预训练输入应使用 ImageNet mean/std 进行标准化。"


def test_prepare_views_can_keep_pretrained_pipeline_without_loading_backbone_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = "dummy_pretrained_pipeline_no_weights"
    monkeypatch.setitem(BACKBONE_FACTORIES, model_name, _DummyEncoder)
    model = create_model(
        model_name=model_name,
        num_classes=3,
        fusion_enabled=False,
        pretrained=True,
        load_backbone_weights=False,
        in_channels=1,
        dropout=0.1,
    )
    inputs = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float32)

    prepared = model._prepare_views(inputs)[0]
    expected = model.input_normalizer(inputs.repeat(1, 3, 1, 1))

    assert model.encoder.pretrained is False
    assert prepared.shape == (1, 3, 8, 8)
    assert torch.allclose(prepared, expected, atol=1e-6)


def test_convnext_encoder_uses_torchvision_default_weights_when_no_local_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    class _FakeFeatures(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            return torch.ones((batch_size, 768, 7, 7), dtype=x.dtype, device=x.device)

    class _FakeClassifier(nn.Sequential):
        def __init__(self) -> None:
            super().__init__(nn.Identity(), nn.Flatten(start_dim=1))

    class _FakeTVConvNeXt(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = _FakeFeatures()
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
            self.classifier = _FakeClassifier()

    monkeypatch.setattr(
        backbones,
        "_resolve_local_weights_path",
        lambda _env_name: None,
    )
    monkeypatch.setattr(
        backbones.tv_models,
        "convnext_tiny",
        lambda weights=None: calls.append(weights) or _FakeTVConvNeXt(),
    )

    encoder = backbones.ConvNeXtCBAMEncoder(pretrained=True, image_size=224)
    outputs = encoder(torch.randn(2, 3, 224, 224))

    assert calls == [backbones.tv_models.ConvNeXt_Tiny_Weights.DEFAULT]
    assert outputs.shape == (2, 768)


def test_convnext_encoder_prefers_local_weights_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: list[object] = []
    loaded_paths: list[str] = []
    weights_path = tmp_path / "convnext_tiny.pth"
    weights_path.write_bytes(b"fake")

    class _FakeFeatures(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            return torch.ones((batch_size, 768, 7, 7), dtype=x.dtype, device=x.device)

    class _FakeClassifier(nn.Sequential):
        def __init__(self) -> None:
            super().__init__(nn.Identity(), nn.Flatten(start_dim=1))

    class _FakeTVConvNeXt(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = _FakeFeatures()
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
            self.classifier = _FakeClassifier()

        def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
            loaded_paths.append(state_dict["loaded_from"])
            return super().load_state_dict({}, strict=False)

    monkeypatch.setattr(backbones, "_resolve_local_weights_path", lambda _env_name: weights_path)
    monkeypatch.setattr(
        backbones.tv_models,
        "convnext_tiny",
        lambda weights=None: calls.append(weights) or _FakeTVConvNeXt(),
    )
    monkeypatch.setattr(
        backbones,
        "_load_torchvision_checkpoint",
        lambda path: {"loaded_from": str(path)},
    )

    encoder = backbones.ConvNeXtCBAMEncoder(pretrained=True, image_size=224)
    outputs = encoder(torch.randn(2, 3, 224, 224))

    assert calls == [None]
    assert loaded_paths == [str(weights_path)]
    assert outputs.shape == (2, 768)
