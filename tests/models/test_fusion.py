import torch
from torch import nn

from forensight.models.forensic import ForensicEncoder, NPRTransform
from forensight.models.fusion import FusionDetector
from forensight.models.semantic import SemanticEncoder


class DummyClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(12, 8)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


class DummyForensic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(2, 3))


def test_fusion_outputs_binary_logit_and_keeps_clip_frozen():
    clip = DummyClip()
    semantic = SemanticEncoder(clip, feature_dim=8, projection_dim=256)
    forensic_backbone = DummyForensic()
    forensic = ForensicEncoder(
        forensic_backbone,
        feature_dim=4,
        projection_dim=256,
        npr=NPRTransform(scale_factor=0.5),
    )
    model = FusionDetector(semantic, forensic, hidden_dim=128, dropout=0.2)

    logits = model(
        torch.randn(2, 3, 2, 2),
        torch.rand(2, 3, 16, 16),
    )
    logits.sum().backward()

    assert logits.shape == (2, 1)
    assert model.fused_dim == 512
    assert all(parameter.grad is None for parameter in clip.parameters())
    assert semantic.projection.weight.grad is not None
    assert forensic_backbone.conv.weight.grad is not None
    assert forensic.projection.weight.grad is not None
    assert any(p.grad is not None for p in model.classifier.parameters())


def test_fusion_no_auxiliary_heads_strict_binary_classifier():
    clip = DummyClip()
    semantic = SemanticEncoder(clip, feature_dim=8, projection_dim=256)
    forensic_backbone = DummyForensic()
    forensic = ForensicEncoder(
        forensic_backbone,
        feature_dim=4,
        projection_dim=256,
        npr=NPRTransform(scale_factor=0.5),
    )
    model = FusionDetector(semantic, forensic, hidden_dim=128, dropout=0.2)

    # Invariant: R2 v1 fusion must NOT have auxiliary branch heads
    assert not hasattr(model, "aux_semantic"), "FusionDetector must not contain aux_semantic head"
    assert not hasattr(model, "aux_forensic"), "FusionDetector must not contain aux_forensic head"

    logits = model(
        torch.randn(2, 3, 2, 2),
        torch.rand(2, 3, 16, 16),
    )
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 1)
