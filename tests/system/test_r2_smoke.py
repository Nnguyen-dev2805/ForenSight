"""End-to-end network-free system smoke test for ForenSight R2.

Validates the complete execution loop across Semantic-only, Forensic-only,
and Fusion detector variants:
    synthetic manifest -> R2ImageDataset -> train step (BCE loss + optimizer)
    -> checkpoint save -> checkpoint reload -> prediction export -> R0 evaluation.
Operates 100% offline using lightweight dummy backbones.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from forensight.data.r2_dataset import R2ImageDataset
from forensight.data.split import Manifest, ManifestRecord
from forensight.evaluation.runner import PredictionSet, evaluate_predictions
from forensight.models.forensic import ForensicEncoder, ForensicOnlyDetector, NPRTransform
from forensight.models.fusion import FusionDetector
from forensight.models.semantic import SemanticEncoder, SemanticOnlyDetector
from forensight.training.r2 import (
    load_checkpoint,
    predict_to_prediction_set,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)


class DummyClipBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(12, 16)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class DummyForensicBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(2, 3))


@pytest.fixture
def synthetic_r2_environment(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    paths = {}
    for name, color in [
        ("train_real", (10, 20, 30)),
        ("train_fake", (200, 210, 220)),
        ("val_real", (15, 25, 35)),
        ("val_fake", (190, 200, 210)),
        ("test_real", (12, 22, 32)),
        ("test_fake", (195, 205, 215)),
        ("ood_fake", (220, 230, 240)),
    ]:
        p = img_dir / f"{name}.png"
        Image.new("RGB", (16, 16), color=color).save(p)
        paths[name] = p

    train_manifest = Manifest([
        ManifestRecord(sample_id="tr1", image_path=str(paths["train_real"]), label=0, dataset="genimage", generator="nature", split="train"),
        ManifestRecord(sample_id="tf1", image_path=str(paths["train_fake"]), label=1, dataset="genimage", generator="sd14", split="train"),
    ])

    val_manifest = Manifest([
        ManifestRecord(sample_id="vr1", image_path=str(paths["val_real"]), label=0, dataset="genimage", generator="nature", split="val"),
        ManifestRecord(sample_id="vf1", image_path=str(paths["val_fake"]), label=1, dataset="genimage", generator="sd14", split="val"),
    ])

    eval_manifest = Manifest([
        ManifestRecord(sample_id="ter1", image_path=str(paths["test_real"]), label=0, dataset="genimage", generator="nature", split="in_domain_test"),
        ManifestRecord(sample_id="tef1", image_path=str(paths["test_fake"]), label=1, dataset="genimage", generator="sd14", split="in_domain_test"),
        ManifestRecord(sample_id="ood1", image_path=str(paths["ood_fake"]), label=1, dataset="genimage", generator="midjourney", split="cross_generator_ood"),
    ])

    return train_manifest, val_manifest, eval_manifest, tmp_path


@pytest.mark.parametrize("variant", ["semantic", "forensic", "fusion"])
def test_r2_end_to_end_offline_pipeline(synthetic_r2_environment, variant: str):
    train_manifest, val_manifest, eval_manifest, tmp_path = synthetic_r2_environment
    device = torch.device("cpu")

    # 1. Instantiate offline dummy models
    clip_backbone = DummyClipBackbone()
    semantic_encoder = SemanticEncoder(clip_backbone, feature_dim=16, projection_dim=32)

    forensic_backbone = DummyForensicBackbone()
    forensic_encoder = ForensicEncoder(
        forensic_backbone,
        feature_dim=8,
        projection_dim=32,
        npr=NPRTransform(scale_factor=0.5),
    )

    if variant == "semantic":
        model = SemanticOnlyDetector(semantic_encoder, hidden_dim=16)
        clip_transform = lambda img: torch.zeros(3, 2, 2)
        forensic_transform = None
    elif variant == "forensic":
        model = ForensicOnlyDetector(forensic_encoder, hidden_dim=16)
        clip_transform = None
        forensic_transform = lambda img: torch.zeros(3, 4, 4)
    else:
        model = FusionDetector(semantic_encoder, forensic_encoder, hidden_dim=16)
        clip_transform = lambda img: torch.zeros(3, 2, 2)
        forensic_transform = lambda img: torch.zeros(3, 4, 4)

    model.to(device)

    # 2. Build Datasets and Loaders
    train_dataset = R2ImageDataset(
        train_manifest,
        clip_transform=clip_transform,
        forensic_transform=forensic_transform,
    )
    val_dataset = R2ImageDataset(
        val_manifest,
        clip_transform=clip_transform,
        forensic_transform=forensic_transform,
    )
    eval_dataset = R2ImageDataset(
        eval_manifest,
        clip_transform=clip_transform,
        forensic_transform=forensic_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False)

    # 3. Train one epoch and validate
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    train_loss = train_one_epoch(model, train_loader, optimizer, device=device, variant=variant)
    val_loss = validate_one_epoch(model, val_loader, device=device, variant=variant)

    assert torch.isfinite(torch.tensor(train_loss))
    assert torch.isfinite(torch.tensor(val_loss))

    # 4. Checkpoint round-trip
    ckpt_path = tmp_path / f"{variant}_best.pt"
    save_checkpoint(ckpt_path, model, optimizer, epoch=1, config={"variant": variant})
    assert ckpt_path.exists()

    load_checkpoint(ckpt_path, model, optimizer=optimizer, map_location=device)

    # 5. Prediction export
    val_preds = predict_to_prediction_set(model, val_loader, device=device, variant=variant)
    eval_preds = predict_to_prediction_set(model, eval_loader, device=device, variant=variant)
    combined = PredictionSet([*val_preds.records, *eval_preds.records])
    assert len(combined) == 5

    # 6. R0 Leak-free evaluation
    report = evaluate_predictions(
        combined,
        val_split_name="val",
        threshold_strategy="f1",
        seed=42,
    )

    assert report.threshold_metadata["calibrated"] is True
    assert report.threshold_metadata["threshold_source"].startswith("val_")
    assert "in_domain_test" in report.by_split
    assert "cross_generator_ood" in report.by_split
    assert "midjourney" in report.by_generator
    assert report.overall.accuracy is not None
    assert report.overall.f1 is not None
