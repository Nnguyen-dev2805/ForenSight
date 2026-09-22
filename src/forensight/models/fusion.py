"""Concatenation fusion detector for ForenSight R2.

Fuses projected semantic (256-d) and forensic (256-d) representations via simple
concatenation into a 512-d feature vector, classified by a lightweight MLP:
    512 -> Linear(512, 128) -> ReLU -> Dropout -> Linear(128, 1) -> logit
"""

from __future__ import annotations

import torch
from torch import nn

from forensight.models.forensic import ForensicEncoder
from forensight.models.semantic import SemanticEncoder


class FusionDetector(nn.Module):
    """Simple feature concatenation fusion detector.

    Combines frozen CLIP semantic features and NPR-style ResNet18 forensic
    features via concatenation without attention or learned branch weighting.
    """

    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        forensic_encoder: ForensicEncoder,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.semantic_encoder = semantic_encoder
        self.forensic_encoder = forensic_encoder
        semantic_dim = semantic_encoder.projection.out_features
        forensic_dim = forensic_encoder.projection.out_features
        self.fused_dim = semantic_dim + forensic_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        clip_image: torch.Tensor,
        forensic_image: torch.Tensor,
    ) -> torch.Tensor:
        semantic = self.semantic_encoder(clip_image)
        forensic = self.forensic_encoder(forensic_image)
        fused = torch.cat([semantic, forensic], dim=-1)
        return self.classifier(fused)
