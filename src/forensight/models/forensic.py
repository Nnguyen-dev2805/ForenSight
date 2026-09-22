"""Forensic encoder and detector for ForenSight R2.

Implements the deterministic NPR-inspired residual transform and a trainable
ResNet18 feature extractor projecting to 256 dimensions with LayerNorm.

NOTE: This is a fixed ForenSight baseline inspired by NPR (Neighboring Pixel
Relationships; residual = input - upsample(downsample(input))), NOT a claim of
reproducing the full NPR paper.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


class NPRTransform(nn.Module):
    """Deterministic NPR-inspired low-level residual transform.

    Extracts high-frequency/residual artifacts by subtracting a low-pass
    downsampled-and-upsampled reconstruction from the input tensor:
        residual = image - upsample(downsample(image))
    """

    def __init__(self, *, scale_factor: float = 0.5, mode: str = "bilinear") -> None:
        super().__init__()
        if not 0.0 < scale_factor < 1.0:
            raise ValueError("scale_factor must be between 0 and 1.")
        if mode not in {"bilinear", "bicubic"}:
            raise ValueError("mode must be 'bilinear' or 'bicubic'.")
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        reduced = F.interpolate(
            image,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=False,
            recompute_scale_factor=False,
        )
        reconstructed = F.interpolate(
            reduced,
            size=image.shape[-2:],
            mode=self.mode,
            align_corners=False,
        )
        return image - reconstructed


class ForensicEncoder(nn.Module):
    """Forensic branch encoder: NPR transform -> ResNet backbone -> 256-d projection."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: int,
        projection_dim: int = 256,
        npr: NPRTransform | None = None,
    ) -> None:
        super().__init__()
        self.npr = npr or NPRTransform()
        self.backbone = backbone
        self.projection = nn.Linear(feature_dim, projection_dim)
        self.norm = nn.LayerNorm(projection_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residual = self.npr(image)
        features = self.backbone(residual)
        return self.norm(self.projection(features.float()))


class ForensicOnlyDetector(nn.Module):
    """Forensic-only binary detector: NPR -> ResNet18 -> 256 -> MLP -> logit."""

    def __init__(
        self,
        encoder: ForensicEncoder,
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


def build_resnet18_forensic(
    *,
    projection_dim: int = 256,
    npr_scale_factor: float = 0.5,
    npr_mode: str = "bilinear",
) -> ForensicEncoder:
    """Build a trainable ResNet18 forensic encoder with NPR preprocessing."""
    backbone = resnet18(weights=None)
    feature_dim = int(backbone.fc.in_features)
    backbone.fc = nn.Identity()
    return ForensicEncoder(
        backbone,
        feature_dim=feature_dim,
        projection_dim=projection_dim,
        npr=NPRTransform(scale_factor=npr_scale_factor, mode=npr_mode),
    )
