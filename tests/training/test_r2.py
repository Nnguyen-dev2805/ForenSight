import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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


class _FrozenStack(nn.Module):
    """Model with a frozen feature extractor and a trainable head that has BatchNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
        for parameter in self.frozen.parameters():
            parameter.requires_grad_(False)
        self.head = nn.Sequential(nn.Linear(8, 4), nn.BatchNorm1d(4), nn.Linear(4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.frozen(x))


def test_checkpoint_stores_only_trainable_subtrees(tmp_path: Path):
    torch.manual_seed(0)
    model = _FrozenStack()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "slim.pt"
    save_checkpoint(path, model, optimizer, epoch=2, config={"variant": "fusion"})

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_keys = set(checkpoint["model_state_dict"])
    assert checkpoint["state_dict_scope"] == "trainable_subtrees"
    # Frozen subtrees are rebuilt from config and must not be serialized.
    assert not any(key.startswith("frozen.") for key in saved_keys)
    # BatchNorm running statistics of the trainable head must be kept.
    assert "head.1.running_mean" in saved_keys

    restored = _FrozenStack()
    frozen_before = [p.detach().clone() for p in restored.frozen.parameters()]
    load_checkpoint(path, restored, map_location="cpu")

    assert torch.equal(restored.head[0].weight, checkpoint["model_state_dict"]["head.0.weight"])
    # Frozen parameters are not restored from disk; they come from construction.
    for before, after in zip(frozen_before, restored.frozen.parameters()):
        assert torch.equal(before, after)


def test_load_checkpoint_rejects_mismatched_trainable_keys(tmp_path: Path):
    model = _FrozenStack()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "slim.pt"
    save_checkpoint(path, model, optimizer, epoch=1, config={"variant": "fusion"})

    with pytest.raises(ValueError, match="does not match the trainable parameters"):
        load_checkpoint(path, nn.Linear(8, 8), map_location="cpu")


def test_set_seed_pins_cudnn_to_deterministic(monkeypatch):
    """Seed 42 must pin cuDNN, not just the RNGs.

    Without the deterministic flags, convolution algorithms are chosen by
    benchmark heuristics and two runs of the same seed can differ.
    """
    import torch.backends.cudnn as cudnn

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(cudnn, "deterministic", False)
    monkeypatch.setattr(cudnn, "benchmark", True)

    set_seed(42)

    assert cudnn.deterministic is True
    assert cudnn.benchmark is False


def test_report_configs_lock_seed42_five_epochs_and_openai_quickgelu():
    """Pin the seed-42 five-epoch reporting profile.

    The OpenAI ViT-L/14 checkpoint was trained with QuickGELU, so the plain
    `ViT-L-14` architecture (which uses GELU) is not weight-compatible with
    `pretrained="openai"`. The semantic branch must therefore name the
    QuickGELU variant explicitly, or the frozen encoder silently runs the
    wrong activation function.
    """
    expected = {
        "semantic": (42, 5, "ViT-L-14-quickgelu"),
        "forensic": (42, 5, "ViT-L-14"),
        "fusion": (42, 5, "ViT-L-14-quickgelu"),
    }
    for variant, (seed, epochs, clip_model) in expected.items():
        config = load_r2_config(Path("configs/r2") / f"{variant}.json")
        assert config["seed"] == seed, f"{variant}: seed must be {seed}"
        assert config["training"]["epochs"] == epochs, f"{variant}: epochs must be {epochs}"
        assert config["model"]["clip_model"] == clip_model, (
            f"{variant}: clip_model must be {clip_model}"
        )


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


def test_train_r2_cli_parsing_experiment_modes():
    import scripts.train_r2 as train_r2

    # 1. Single experiment
    args_single = train_r2.parse_args([
        "--experiment", "single",
        "--variant", "fusion",
        "--seed", "42",
        "--output-dir", "results/single_fusion",
    ])
    assert args_single.experiment == "single"
    assert args_single.variant == "fusion"
    assert args_single.seed == 42
    assert args_single.output_dir == "results/single_fusion"

    # 2. LOGO experiment
    args_logo = train_r2.parse_args([
        "--experiment", "logo",
        "--leave-out", "midjourney",
        "--variant", "semantic_only",
        "--output-dir", "results/logo_mj",
    ])
    assert args_logo.experiment == "logo"
    assert args_logo.leave_out == "midjourney"
    assert args_logo.variant == "semantic_only"

    # 3. All7 experiment
    args_all7 = train_r2.parse_args([
        "--experiment", "all7",
        "--variant", "forensic_only",
        "--output-dir", "results/all7",
    ])
    assert args_all7.experiment == "all7"
    assert args_all7.variant == "forensic_only"


def test_evaluate_r2_cli_parsing_experiment_modes():
    import scripts.evaluate_r2 as eval_r2

    # Single
    args_single = eval_r2.parse_args([
        "--config", "configs/r2/fusion.json",
        "--checkpoint", "best.pt",
        "--experiment", "single",
    ])
    assert args_single.experiment == "single"

    # LOGO
    args_logo = eval_r2.parse_args([
        "--config", "configs/r2/fusion.json",
        "--checkpoint", "best.pt",
        "--experiment", "logo",
        "--leave-out", "midjourney",
    ])
    assert args_logo.experiment == "logo"
    assert args_logo.leave_out == "midjourney"

    # All7
    args_all7 = eval_r2.parse_args([
        "--config", "configs/r2/fusion.json",
        "--checkpoint", "best.pt",
        "--experiment", "all7",
    ])
    assert args_all7.experiment == "all7"


def test_train_r2_preflight_leakage_blocks_training(tmp_path):
    import scripts.train_r2 as train_r2
    from forensight.data.split import Manifest, ManifestRecord
    from PIL import Image

    # Create dummy images so dataset loading doesn't fail on missing image
    img_path = tmp_path / "img1.png"
    Image.new("RGB", (224, 224), color=(100, 100, 100)).save(img_path)

    # Contaminate val with identical sample_id from train (leakage)
    train_record = ManifestRecord(
        sample_id="colliding_id_001",
        image_path=str(img_path),
        label=1,
        dataset="genimage",
        generator="sd15",
        split="train",
    )
    val_record = ManifestRecord(
        sample_id="colliding_id_001",  # Colliding ID!
        image_path=str(img_path),
        label=1,
        dataset="genimage",
        generator="sd15",
        split="val",
    )

    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    Manifest([train_record]).to_jsonl(train_p)
    Manifest([val_record]).to_jsonl(val_p)

    out_dir = tmp_path / "results_run"

    # Pre-flight check MUST abort before training begins
    with pytest.raises(RuntimeError, match="Pre-flight leakage audit FAILED"):
        train_r2.main([
            "--train-manifest", str(train_p),
            "--val-manifest", str(val_p),
            "--output-dir", str(out_dir),
            "--variant", "semantic",
            "--epochs", "1",
        ])

    # Invariant: No checkpoint was created
    assert not (out_dir / "checkpoint.pt").exists()
    assert not (out_dir / "train_history.json").exists()
    # Invariant: Audit artifact was persisted for post-mortem analysis
    audit_file = out_dir / "dataset_audit.json"
    assert audit_file.exists()
    with open(audit_file) as f:
        data = json.load(f)
    assert data["leakage"]["has_leakage"] is True


def test_train_r2_preflight_blocks_same_bytes_under_different_ids(tmp_path):
    import scripts.train_r2 as train_r2
    from forensight.data.split import Manifest, ManifestRecord
    from PIL import Image

    train_image = tmp_path / "train.png"
    val_image = tmp_path / "val.png"
    Image.new("RGB", (16, 16), color="red").save(train_image)
    val_image.write_bytes(train_image.read_bytes())
    train = ManifestRecord("train-id", str(train_image), 1, "genimage", "sd15", "train")
    val = ManifestRecord("val-id", str(val_image), 1, "genimage", "sd15", "val")
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    Manifest([train]).to_jsonl(train_path)
    Manifest([val]).to_jsonl(val_path)

    out_dir = tmp_path / "run"
    with patch("scripts.train_r2.build_r2_model", side_effect=AssertionError("model built before audit")):
        with pytest.raises(RuntimeError, match="Pre-flight leakage audit FAILED"):
            train_r2.main([
                "--train-manifest", str(train_path), "--val-manifest", str(val_path),
                "--variant", "forensic", "--output-dir", str(out_dir),
            ])

    assert not (out_dir / "checkpoint.pt").exists()
    audit = json.loads((out_dir / "dataset_audit.json").read_text())
    assert audit["leakage"]["content_hash_collisions"][0]["count"] == 1


def test_train_r2_rejects_config_for_another_variant(tmp_path):
    import scripts.train_r2 as train_r2

    with pytest.raises(ValueError, match="variant.*config"):
        train_r2.main([
            "--config", "configs/r2/semantic.json", "--variant", "forensic",
            "--train-manifest", str(tmp_path / "missing-train.jsonl"),
            "--val-manifest", str(tmp_path / "missing-val.jsonl"),
            "--output-dir", str(tmp_path / "run"),
        ])


def test_forensic_config_records_the_applied_npr_normalization(tmp_path):
    config = load_r2_config("configs/r2/forensic.json")
    assert config["model"]["npr_normalization"] == "spatial_std"

    config["model"]["npr_normalization"] = "none"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="npr_normalization"):
        load_r2_config(path)


def test_evaluate_r2_experiment_mode_persists_artifacts(tmp_path):
    import scripts.evaluate_r2 as eval_r2
    from forensight.data.split import Manifest, ManifestRecord
    from forensight.evaluation.runner import PredictionRecord, PredictionSet
    from PIL import Image

    # 1. Setup synthetic images and manifests under manifest_dir/single/
    manifest_dir = tmp_path / "manifests"
    single_dir = manifest_dir / "single"
    single_dir.mkdir(parents=True)

    images = [tmp_path / f"image_{i}.png" for i in range(4)]
    for i, image in enumerate(images):
        Image.new("RGB", (64, 64), color=(i * 40, 50, 50)).save(image)

    val_rec = ManifestRecord(sample_id="v1", image_path=str(images[0]), label=0, dataset="genimage", generator="nature", split="val")
    in_domain_rec = ManifestRecord(sample_id="t1", image_path=str(images[1]), label=1, dataset="genimage", generator="sd15", split="in_domain_test")
    ood_rec = ManifestRecord(sample_id="o1", image_path=str(images[2]), label=1, dataset="genimage", generator="midjourney", split="cross_generator_ood")
    train_rec = ManifestRecord(sample_id="tr1", image_path=str(images[3]), label=0, dataset="genimage", generator="nature", split="train")

    Manifest([val_rec]).to_jsonl(single_dir / "val.jsonl")
    Manifest([in_domain_rec]).to_jsonl(single_dir / "in_domain_test.jsonl")
    Manifest([ood_rec]).to_jsonl(single_dir / "cross_generator_ood.jsonl")
    Manifest([ood_rec]).to_jsonl(single_dir / "test_midjourney.jsonl")
    Manifest([train_rec]).to_jsonl(single_dir / "train.jsonl")

    # 2. Config & dummy checkpoint
    cfg_p = tmp_path / "config.json"
    cfg_p.write_text(json.dumps({
        "variant": "forensic",
        "model": {"name": "resnet18_forensic", "in_channels": 3},
        "training": {"batch_size": 2},
        "evaluation": {},
        "git_commit": None,
        "source_payload_sha256": "a" * 64,
    }))

    ckpt_p = tmp_path / "model.pt"
    ckpt_p.write_bytes(b"dummy_weights")

    out_dir = tmp_path / "results_eval"

    # Mock model assembly & predictions so this unit test runs cleanly and deterministically
    with patch("scripts.evaluate_r2.build_r2_model") as mock_build, \
         patch("scripts.evaluate_r2.torch.load", return_value={"model_state_dict": {}}), \
         patch("scripts.evaluate_r2.predict_to_prediction_set") as mock_pred:

        dummy_model = MagicMock()
        mock_build.return_value = (dummy_model, None, lambda x: torch.zeros(3, 16, 16))

        def fake_pred(model, loader, device, variant):
            records = [
                PredictionRecord(
                    sample_id=rec.sample_id,
                    label=rec.label,
                    score=0.2 if rec.label == 0 else 0.8,
                    split=rec.split,
                    generator=rec.generator,
                    dataset=rec.dataset,
                )
                for rec in loader.dataset.manifest
            ]
            return PredictionSet(records)

        mock_pred.side_effect = fake_pred

        eval_r2.main([
            "--experiment", "single",
            "--manifest-dir", str(manifest_dir),
            "--config", str(cfg_p),
            "--checkpoint", str(ckpt_p),
            "--out-dir", str(out_dir),
            "--dataset-revision", "synthetic-test-v1",
        ])

    assert (out_dir / "val_manifest.jsonl").exists()
    assert (out_dir / "test_manifest.jsonl").exists()
    assert (out_dir / "predictions.jsonl").exists()
    assert (out_dir / "evaluation.json").exists()
    assert (out_dir / "reproducibility.json").exists()
    assert (out_dir / "dataset_audit.json").exists()
    assert json.loads((out_dir / "reproducibility.json").read_text())["git_commit"] is None
    assert json.loads((out_dir / "evaluation.json").read_text())["run_metadata"]["dataset_revision"] == "synthetic-test-v1"
    test_ids = [r.sample_id for r in Manifest.from_jsonl(out_dir / "test_manifest.jsonl")]
    assert test_ids == ["t1", "o1"]


def _single_manifest_fixture(tmp_path):
    """Two-class single-protocol fixture: 1 val real, 1 in-domain fake, 1 OOD fake."""
    from forensight.data.split import Manifest, ManifestRecord
    from PIL import Image

    manifest_dir = tmp_path / "manifests"
    single_dir = manifest_dir / "single"
    single_dir.mkdir(parents=True)
    images = [tmp_path / f"image_{i}.png" for i in range(8)]
    for i, image in enumerate(images):
        Image.new("RGB", (64, 64), color=(i * 40, 50, 50)).save(image)

    # Validation must contain BOTH classes, otherwise calibration cannot run and
    # the report falls back to the default threshold.
    val_real = ManifestRecord(sample_id="v1", image_path=str(images[0]), label=0,
                              dataset="genimage", generator="nature", split="val")
    val_fake = ManifestRecord(sample_id="v2", image_path=str(images[1]), label=1,
                              dataset="genimage", generator="sd15", split="val")
    train_real = ManifestRecord(sample_id="tr1", image_path=str(images[2]), label=0,
                                dataset="genimage", generator="nature", split="train")
    # Both test splits need reals as well as fakes, otherwise the test cohort is
    # single-class and no binary metric (or threshold comparison) is defined.
    in_domain_fake = ManifestRecord(sample_id="t1", image_path=str(images[3]), label=1,
                                    dataset="genimage", generator="sd15", split="in_domain_test")
    in_domain_real = ManifestRecord(sample_id="t2", image_path=str(images[4]), label=0,
                                    dataset="genimage", generator="nature", split="in_domain_test")
    ood_fake = ManifestRecord(sample_id="o1", image_path=str(images[5]), label=1,
                              dataset="genimage", generator="midjourney", split="cross_generator_ood")
    ood_real = ManifestRecord(sample_id="o2", image_path=str(images[6]), label=0,
                              dataset="genimage", generator="nature", split="cross_generator_ood")

    Manifest([val_real, val_fake]).to_jsonl(single_dir / "val.jsonl")
    Manifest([in_domain_fake, in_domain_real]).to_jsonl(single_dir / "in_domain_test.jsonl")
    Manifest([ood_fake, ood_real]).to_jsonl(single_dir / "cross_generator_ood.jsonl")
    Manifest([train_real]).to_jsonl(single_dir / "train.jsonl")
    return manifest_dir


def _run_eval_with_fake_predictions(tmp_path, manifest_dir, out_dir):
    import scripts.evaluate_r2 as eval_r2
    from forensight.evaluation.runner import PredictionRecord, PredictionSet

    cfg_p = tmp_path / "config.json"
    cfg_p.write_text(json.dumps({
        "variant": "forensic", "model": {}, "training": {"batch_size": 2},
        "evaluation": {}, "git_commit": None, "source_payload_sha256": "a" * 64,
    }))
    ckpt_p = tmp_path / "model.pt"
    ckpt_p.write_bytes(b"dummy_weights")

    with patch("scripts.evaluate_r2.build_r2_model") as mock_build, \
         patch("scripts.evaluate_r2.torch.load", return_value={"model_state_dict": {}}), \
         patch("scripts.evaluate_r2.predict_to_prediction_set") as mock_pred:

        mock_build.return_value = (MagicMock(), None, lambda x: torch.zeros(3, 16, 16))

        def fake_pred(model, loader, device, variant):
            return PredictionSet([
                PredictionRecord(
                    sample_id=rec.sample_id, label=rec.label,
                    score=0.2 if rec.label == 0 else 0.8,
                    split=rec.split, generator=rec.generator, dataset=rec.dataset,
                    metadata=dict(rec.metadata),
                )
                for rec in loader.dataset.manifest
            ])

        mock_pred.side_effect = fake_pred
        eval_r2.main([
            "--experiment", "single", "--manifest-dir", str(manifest_dir),
            "--config", str(cfg_p), "--checkpoint", str(ckpt_p),
            "--out-dir", str(out_dir), "--dataset-revision", "synthetic-test-v1",
        ])


def test_evaluate_r2_writes_fixed_threshold_reference_report(tmp_path):
    """Both threshold reports must exist and share one score cohort.

    The seed-42 protocol requires a predeclared fixed-0.5 reference alongside the
    validation-calibrated report, produced from the same prediction scores, so
    the two are comparable and neither can be selected post hoc on test results.
    """
    manifest_dir = _single_manifest_fixture(tmp_path)
    out_dir = tmp_path / "results_eval"
    _run_eval_with_fake_predictions(tmp_path, manifest_dir, out_dir)

    calibrated = out_dir / "evaluation.json"
    fixed = out_dir / "evaluation_fixed_0_5.json"
    assert calibrated.exists()
    assert fixed.exists()

    cal = json.loads(calibrated.read_text())
    fix = json.loads(fixed.read_text())

    # Threshold provenance differs...
    assert cal["threshold_metadata"]["threshold_source"] == "val_optimal_f1"
    assert fix["threshold_metadata"]["threshold_source"] == "default_0.5"
    assert fix["threshold_metadata"]["threshold"] == pytest.approx(0.5)

    # ...but the evaluated cohort is identical, so the comparison is fair.
    assert cal["run_metadata"]["sample_set_hash"] == fix["run_metadata"]["sample_set_hash"]
    assert cal["overall"]["auroc"] == pytest.approx(fix["overall"]["auroc"])


def test_fixed_threshold_report_does_not_clobber_calibrated_decisions(tmp_path):
    """The fixed-threshold pass must not overwrite the calibrated binary decisions."""
    from forensight.evaluation.runner import PredictionSet

    manifest_dir = _single_manifest_fixture(tmp_path)
    out_dir = tmp_path / "results_eval"
    _run_eval_with_fake_predictions(tmp_path, manifest_dir, out_dir)

    pset = PredictionSet.from_jsonl(out_dir / "predictions.jsonl")
    cal = json.loads((out_dir / "evaluation.json").read_text())
    tau = cal["threshold_metadata"]["threshold"]

    # Each stored decision must match the calibrated threshold, not 0.5.
    for rec in pset:
        expected = 1 if rec.score >= tau else 0
        assert rec.prediction == expected, (
            f"{rec.sample_id}: prediction {rec.prediction} != calibrated decision {expected}"
        )


def test_evaluate_all7_does_not_repeat_combined_test_samples(tmp_path):
    import scripts.evaluate_r2 as eval_r2
    from forensight.data.split import Manifest, ManifestRecord
    from forensight.evaluation.runner import PredictionRecord, PredictionSet
    from PIL import Image

    all7_dir = tmp_path / "manifests" / "all7"
    all7_dir.mkdir(parents=True)
    val_image = tmp_path / "val.png"
    test_image = tmp_path / "test.png"
    Image.new("RGB", (16, 16), color="red").save(val_image)
    Image.new("RGB", (16, 16), color="blue").save(test_image)
    val = ManifestRecord("val", str(val_image), 0, "genimage", "nature", "val")
    test = ManifestRecord("test", str(test_image), 1, "genimage", "sd15", "in_domain_test")
    Manifest([val]).to_jsonl(all7_dir / "val.jsonl")
    Manifest([test]).to_jsonl(all7_dir / "test_all_combined.jsonl")
    Manifest([test]).to_jsonl(all7_dir / "test_sd15.jsonl")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"variant": "forensic", "model": {}, "training": {"batch_size": 1}, "evaluation": {}}))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"dummy")

    def fake_predict(model, loader, device, variant):
        return PredictionSet([
            PredictionRecord(
                sample_id=r.sample_id, label=r.label, score=0.8 if r.label else 0.2,
                split=r.split, generator=r.generator, dataset=r.dataset,
            )
            for r in loader.dataset.manifest
        ])

    with patch("scripts.evaluate_r2.build_r2_model", return_value=(MagicMock(), None, lambda x: torch.zeros(3, 16, 16))), \
         patch("scripts.evaluate_r2.torch.load", return_value={"model_state_dict": {}}), \
         patch("scripts.evaluate_r2.predict_to_prediction_set", side_effect=fake_predict):
        eval_r2.main([
            "--config", str(config), "--checkpoint", str(checkpoint),
            "--experiment", "all7", "--manifest-dir", str(tmp_path / "manifests"),
            "--output-dir", str(tmp_path / "output"),
            "--dataset-revision", "synthetic-test-v1",
        ])

    ids = [r.sample_id for r in Manifest.from_jsonl(tmp_path / "output" / "test_manifest.jsonl")]
    assert ids == ["test"]


def test_evaluate_r2_requires_dataset_revision_for_saved_report(tmp_path):
    import scripts.evaluate_r2 as eval_r2

    with pytest.raises(ValueError, match="--dataset-revision"):
        eval_r2.main([
            "--config", str(tmp_path / "missing.json"),
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--output-dir", str(tmp_path / "report"),
        ])


def test_evaluate_r2_rejects_byte_leakage_before_model_or_report(tmp_path):
    import scripts.evaluate_r2 as eval_r2
    from forensight.data.split import Manifest, ManifestRecord
    from PIL import Image

    val_image = tmp_path / "val.png"
    test_image = tmp_path / "test.png"
    Image.new("RGB", (16, 16), color="red").save(val_image)
    test_image.write_bytes(val_image.read_bytes())
    val = ManifestRecord("val-id", str(val_image), 0, "genimage", "nature", "val")
    test = ManifestRecord("test-id", str(test_image), 1, "genimage", "sd15", "in_domain_test")
    val_path = tmp_path / "val.jsonl"
    test_path = tmp_path / "test.jsonl"
    Manifest([val]).to_jsonl(val_path)
    Manifest([test]).to_jsonl(test_path)
    out_dir = tmp_path / "report"

    with patch("scripts.evaluate_r2.build_r2_model", side_effect=AssertionError("model built before audit")):
        with pytest.raises(RuntimeError, match="leakage"):
            eval_r2.main([
                "--config", "configs/r2/forensic.json", "--checkpoint", str(tmp_path / "unused.pt"),
                "--val-manifest", str(val_path), "--eval-manifest", str(test_path),
                "--dataset-revision", "synthetic-v1", "--output-dir", str(out_dir),
            ])

    assert not (out_dir / "evaluation.json").exists()
    assert not (out_dir / "predictions.jsonl").exists()


def test_evaluate_r2_rejects_checkpoint_with_different_frozen_model_config(tmp_path):
    import scripts.evaluate_r2 as eval_r2
    from forensight.data.split import Manifest, ManifestRecord
    from PIL import Image

    paths = [tmp_path / "val.png", tmp_path / "test.png"]
    Image.new("RGB", (16, 16), color="red").save(paths[0])
    Image.new("RGB", (16, 16), color="blue").save(paths[1])
    val_path, test_path = tmp_path / "val.jsonl", tmp_path / "test.jsonl"
    Manifest([ManifestRecord("v", str(paths[0]), 0, "genimage", "nature", "val")]).to_jsonl(val_path)
    Manifest([ManifestRecord("t", str(paths[1]), 1, "genimage", "sd15", "in_domain_test")]).to_jsonl(test_path)
    config = json.loads(Path("configs/r2/forensic.json").read_text())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    checkpoint_config = json.loads(json.dumps(config))
    checkpoint_config["model"]["npr_mode"] = "bicubic"

    with patch("scripts.evaluate_r2.build_r2_model", return_value=(MagicMock(), None, lambda x: torch.zeros(3, 16, 16))), \
         patch("scripts.evaluate_r2.torch.load", return_value={"model_state_dict": {}, "config": checkpoint_config}), \
         patch("scripts.evaluate_r2.predict_to_prediction_set", side_effect=AssertionError("inference started")):
        with pytest.raises(ValueError, match="model config"):
            eval_r2.main([
                "--config", str(config_path), "--checkpoint", str(tmp_path / "unused.pt"),
                "--val-manifest", str(val_path), "--eval-manifest", str(test_path),
            ])
