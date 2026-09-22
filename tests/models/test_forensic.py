import pytest
import torch
from torch import nn

from forensight.models.forensic import (
    ForensicEncoder,
    ForensicOnlyDetector,
    NPRTransform,
    build_resnet18_forensic,
)


class DummyForensicBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(2, 3))


def test_npr_transform_is_deterministic_and_shape_preserving():
    transform = NPRTransform(scale_factor=0.5, mode="bilinear")
    image = torch.rand(2, 3, 16, 16)

    first = transform(image)
    second = transform(image)

    assert first.shape == image.shape
    assert torch.equal(first, second)


def test_npr_transform_validates_inputs():
    with pytest.raises(ValueError, match="scale_factor"):
        NPRTransform(scale_factor=1.5)
    with pytest.raises(ValueError, match="scale_factor"):
        NPRTransform(scale_factor=0.0)
    with pytest.raises(ValueError, match="mode"):
        NPRTransform(mode="nearest")


def test_forensic_detector_outputs_binary_logits_and_trains_backbone():
    backbone = DummyForensicBackbone()
    encoder = ForensicEncoder(
        backbone,
        feature_dim=4,
        projection_dim=256,
        npr=NPRTransform(scale_factor=0.5),
    )
    detector = ForensicOnlyDetector(encoder, hidden_dim=128, dropout=0.2)
    logits = detector(torch.rand(2, 3, 16, 16))
    logits.sum().backward()

    assert logits.shape == (2, 1)
    assert backbone.conv.weight.grad is not None
    assert detector.encoder.projection.weight.grad is not None
    assert any(p.grad is not None for p in detector.classifier.parameters())


def test_build_resnet18_forensic_offline():
    encoder = build_resnet18_forensic(projection_dim=256, npr_scale_factor=0.5)
    assert encoder.projection.out_features == 256
    x = torch.rand(2, 3, 32, 32)
    out = encoder(x)
    assert out.shape == (2, 256)
