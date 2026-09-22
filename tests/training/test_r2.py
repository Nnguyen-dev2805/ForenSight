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


def test_build_r2_model_with_dummy_backbones(monkeypatch):
    from forensight.models.forensic import ForensicEncoder, NPRTransform
    from forensight.models.semantic import SemanticEncoder
    from forensight.training.r2 import build_r2_model

    class MockClip(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)
        def encode_image(self, x):
            return self.linear(x)

    class MockForensicBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Identity()
        def forward(self, x):
            return torch.zeros(x.shape[0], 512)

    def mock_load_clip(**kwargs):
        encoder = SemanticEncoder(MockClip(), feature_dim=8, projection_dim=kwargs.get("projection_dim", 256))
        return encoder, lambda img: torch.zeros(3, 8, 8)

    def mock_build_forensic(**kwargs):
        return ForensicEncoder(MockForensicBackbone(), feature_dim=512, projection_dim=kwargs.get("projection_dim", 256), npr=NPRTransform(scale_factor=0.5))

    monkeypatch.setattr("forensight.training.r2.load_open_clip_semantic", mock_load_clip)
    monkeypatch.setattr("forensight.training.r2.build_resnet18_forensic", mock_build_forensic)

    base_model_cfg = {
        "clip_model": "ViT-L-14",
        "clip_pretrained": "openai",
        "projection_dim": 256,
        "hidden_dim": 128,
        "dropout": 0.2,
        "image_size": 224,
        "npr_scale_factor": 0.5,
        "npr_mode": "bilinear",
    }

    # Semantic variant
    sem_cfg = {"variant": "semantic", "model": base_model_cfg}
    sem_model, sem_clip_t, sem_for_t = build_r2_model(sem_cfg)
    assert sem_model is not None
    assert sem_clip_t is not None
    assert sem_for_t is None

    # Forensic variant
    for_cfg = {"variant": "forensic", "model": base_model_cfg}
    for_model, for_clip_t, for_for_t = build_r2_model(for_cfg)
    assert for_model is not None
    assert for_clip_t is None
    assert for_for_t is not None

    # Fusion variant
    fus_cfg = {"variant": "fusion", "model": base_model_cfg}
    fus_model, fus_clip_t, fus_for_t = build_r2_model(fus_cfg)
    assert fus_model is not None
    assert fus_clip_t is not None
    assert fus_for_t is not None


def test_select_device():
    from forensight.training.r2 import select_device

    device = select_device()
    assert isinstance(device, torch.device)


def test_train_r2_cli_parsing():
    import scripts.train_r2 as train_r2

    args = train_r2.parse_args([
        "--config", "configs/r2/semantic.json",
        "--train-manifest", "train.jsonl",
        "--val-manifest", "val.jsonl",
        "--output-dir", "results/r2/semantic/seed42",
    ])
    assert args.config == "configs/r2/semantic.json"
    assert args.train_manifest == "train.jsonl"
    assert args.val_manifest == "val.jsonl"
    assert args.output_dir == "results/r2/semantic/seed42"
    assert not hasattr(args, "test_manifest")


def test_evaluate_r2_cli_parsing():
    import scripts.evaluate_r2 as eval_r2

    args = eval_r2.parse_args([
        "--config", "configs/r2/semantic.json",
        "--checkpoint", "best.pt",
        "--val-manifest", "val.jsonl",
        "--eval-manifest", "test1.jsonl", "test2.jsonl",
        "--predictions-out", "preds.jsonl",
        "--report-json", "report.json",
    ])
    assert args.config == "configs/r2/semantic.json"
    assert args.checkpoint == "best.pt"
    assert args.val_manifest == "val.jsonl"
    assert args.eval_manifest == ["test1.jsonl", "test2.jsonl"]
    assert args.predictions_out == "preds.jsonl"
    assert args.report_json == "report.json"


def test_evaluate_r2_predictions_compatibility_with_r0_runner():
    from forensight.evaluation.runner import PredictionRecord, PredictionSet, evaluate_predictions

    records = [
        # Val split (used to calibrate threshold)
        PredictionRecord(sample_id="v1", label=0, score=0.2, split="val", generator="nature", dataset="genimage"),
        PredictionRecord(sample_id="v2", label=1, score=0.8, split="val", generator="sd14", dataset="genimage"),
        # Test splits (consume frozen threshold)
        PredictionRecord(sample_id="t1", label=0, score=0.1, split="in_domain_test", generator="nature", dataset="genimage"),
        PredictionRecord(sample_id="t2", label=1, score=0.9, split="in_domain_test", generator="sd14", dataset="genimage"),
        PredictionRecord(sample_id="o1", label=1, score=0.85, split="cross_generator_ood", generator="midjourney", dataset="genimage"),
    ]
    pset = PredictionSet(records)
    report = evaluate_predictions(pset, val_split_name="val", threshold_strategy="f1")

    assert report.threshold_metadata["calibrated"] is True
    assert report.threshold_metadata["threshold_source"].startswith("val_")
    assert "in_domain_test" in report.by_split
    assert "cross_generator_ood" in report.by_split
    assert "midjourney" in report.by_generator


