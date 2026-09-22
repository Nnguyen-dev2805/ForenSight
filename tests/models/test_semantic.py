import torch
from torch import nn

from forensight.models.semantic import SemanticEncoder, SemanticOnlyDetector


class DummyClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(12, 8)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


def test_semantic_encoder_freezes_backbone_and_projects_to_256():
    backbone = DummyClip()
    encoder = SemanticEncoder(backbone, feature_dim=8, projection_dim=256)
    output = encoder(torch.randn(2, 3, 2, 2))

    assert output.shape == (2, 256)
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_semantic_encoder_stays_in_eval_mode_when_train_called():
    backbone = DummyClip()
    encoder = SemanticEncoder(backbone, feature_dim=8, projection_dim=256)
    encoder.train(True)

    assert not encoder.backbone.training
    assert encoder.projection.training
    assert encoder.norm.training


def test_semantic_detector_backpropagates_only_through_trainable_head():
    backbone = DummyClip()
    detector = SemanticOnlyDetector(
        SemanticEncoder(backbone, feature_dim=8, projection_dim=256),
        hidden_dim=128,
        dropout=0.2,
    )
    logits = detector(torch.randn(2, 3, 2, 2))
    logits.sum().backward()

    assert logits.shape == (2, 1)
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert any(
        parameter.grad is not None
        for parameter in detector.classifier.parameters()
    )
    assert detector.encoder.projection.weight.grad is not None
