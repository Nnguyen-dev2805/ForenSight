import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from forensight.training.r2 import (
    forward_variant,
    load_checkpoint,
    load_r2_config,
    predict_to_prediction_set,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    validate_one_epoch,
)


class TinySemantic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(12, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.fc(image.flatten(1))


class TinyForensic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(12, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.fc(image.flatten(1))


class TinyFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(24, 1)

    def forward(self, clip_image: torch.Tensor, forensic_image: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([clip_image.flatten(1), forensic_image.flatten(1)], dim=1)
        return self.fc(combined)


def tiny_loader() -> DataLoader:
    samples = [
        {
            "clip_image": torch.zeros(3, 2, 2),
            "forensic_image": torch.zeros(3, 2, 2),
            "label": torch.tensor(0.0),
            "sample_id": "real",
            "split": "val",
            "generator": "nature",
            "dataset": "genimage",
        },
        {
            "clip_image": torch.ones(3, 2, 2),
            "forensic_image": torch.ones(3, 2, 2),
            "label": torch.tensor(1.0),
            "sample_id": "fake",
            "split": "val",
            "generator": "sd14",
            "dataset": "genimage",
        },
    ]
    return DataLoader(samples, batch_size=2, shuffle=False)


def test_set_seed():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_train_one_epoch_produces_finite_loss():
    model = TinySemantic()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_one_epoch(
        model,
        tiny_loader(),
        optimizer,
        device=torch.device("cpu"),
        variant="semantic",
    )
    assert torch.isfinite(torch.tensor(loss))


def test_validate_one_epoch_produces_finite_loss():
    model = TinySemantic()
    loss = validate_one_epoch(
        model,
        tiny_loader(),
        device=torch.device("cpu"),
        variant="semantic",
    )
    assert torch.isfinite(torch.tensor(loss))


def test_forward_variant_dispatch():
    sem_model = TinySemantic()
    forensic_model = TinyForensic()
    fusion_model = TinyFusion()
    batch = next(iter(tiny_loader()))

    out_sem = forward_variant(sem_model, batch, variant="semantic", device=torch.device("cpu"))
    out_forensic = forward_variant(forensic_model, batch, variant="forensic", device=torch.device("cpu"))
    out_fusion = forward_variant(fusion_model, batch, variant="fusion", device=torch.device("cpu"))

    assert out_sem.shape == (2, 1)
    assert out_forensic.shape == (2, 1)
    assert out_fusion.shape == (2, 1)

    with pytest.raises(ValueError, match="Unsupported R2 variant"):
        forward_variant(sem_model, batch, variant="unknown", device=torch.device("cpu"))


def test_prediction_export_matches_r0_schema():
    predictions = predict_to_prediction_set(
        TinySemantic(),
        tiny_loader(),
        device=torch.device("cpu"),
        variant="semantic",
    )
    assert len(predictions) == 2
    assert predictions[0].sample_id == "real"
    assert predictions[0].split == "val"
    assert predictions[0].generator == "nature"
    assert predictions[0].dataset == "genimage"
    assert 0.0 <= predictions[0].score <= 1.0


def test_checkpoint_round_trip(tmp_path: Path):
    path = tmp_path / "checkpoint.pt"
    model = TinySemantic()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(path, model, optimizer, epoch=3, config={"variant": "semantic"})

    restored = TinySemantic()
    state = load_checkpoint(path, restored, map_location="cpu")
    assert state["epoch"] == 3
    assert state["config"]["variant"] == "semantic"


def test_load_r2_config_validation(tmp_path: Path):
    cfg_file = tmp_path / "bad.json"
    cfg_file.write_text(json.dumps({"variant": "invalid"}))
    with pytest.raises(ValueError, match="Unsupported R2 variant"):
        load_r2_config(cfg_file)

    missing_sections = tmp_path / "missing.json"
    missing_sections.write_text(json.dumps({"variant": "semantic"}))
    with pytest.raises(ValueError, match="Missing required configuration section"):
        load_r2_config(missing_sections)
