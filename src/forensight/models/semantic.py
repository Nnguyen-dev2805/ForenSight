"""Semantic encoder and detector for ForenSight R2.

Wraps a frozen CLIP image backbone, projects high-level visual features to 256
dimensions with LayerNorm, and classifies with a small MLP for the Semantic-only
baseline.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn


class SemanticEncoder(nn.Module):
    """Frozen CLIP feature extractor with trainable 256-d linear projection."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: int,
        projection_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.projection = nn.Linear(feature_dim, projection_dim)
        self.norm = nn.LayerNorm(projection_dim)

    def train(self, mode: bool = True) -> SemanticEncoder:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone.encode_image(image)
        return self.norm(self.projection(features.float()))


class SemanticOnlyDetector(nn.Module):
    """Semantic-only binary detector: frozen CLIP -> 256 -> MLP -> logit."""

    def __init__(
        self,
        encoder: SemanticEncoder,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        projection_dim = encoder.projection.out_features
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(image))


def load_open_clip_semantic(
    *,
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    projection_dim: int = 256,
) -> tuple[SemanticEncoder, Callable[[Any], torch.Tensor]]:
    """Instantiate frozen OpenCLIP encoder and its corresponding preprocessing transform."""
    import open_clip

    backbone, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    feature_dim = int(backbone.visual.output_dim)
    encoder = SemanticEncoder(
        backbone,
        feature_dim=feature_dim,
        projection_dim=projection_dim,
    )
    return encoder, preprocess
