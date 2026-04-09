from __future__ import annotations

import pytest
import torch

from dl_lumbar_dd.models import create_model


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
