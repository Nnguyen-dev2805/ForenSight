"""Artifact Contract validation and compliance tests for ForenSight Milestone R2.

Enforces that every experiment run directory conforms strictly to the Artifact Contract:
results/r2/<experiment>/seed_<n>/
├── config.json
├── checkpoint.pt
├── train_history.json
├── train_manifest.jsonl
├── val_manifest.jsonl
├── test_manifest.jsonl
├── predictions.jsonl
├── evaluation.json
├── reproducibility.json
└── dataset_audit.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from forensight.data.split import Manifest, ManifestRecord
from forensight.evaluation.reproducibility import ReproducibilityRecord
from forensight.evaluation.runner import PredictionSet
import scripts.evaluate_r2 as evaluate_r2
import scripts.train_r2 as train_r2


def validate_run_artifact_contract(run_dir: str | Path) -> dict[str, Any]:
    """Validate all files and schemas in an experiment run directory against the Artifact Contract.

    Raises AssertionError if any required artifact is missing or fails schema invariants.
    """
    directory = Path(run_dir)
    assert directory.is_dir(), f"Run directory does not exist: {directory}"

    required_files = [
        "config.json",
        "checkpoint.pt",
        "train_history.json",
        "train_manifest.jsonl",
        "val_manifest.jsonl",
        "test_manifest.jsonl",
        "predictions.jsonl",
        "evaluation.json",
        "reproducibility.json",
        "dataset_audit.json",
    ]

    for fname in required_files:
        fpath = directory / fname
        assert fpath.exists(), f"Artifact Contract violation: missing required artifact '{fname}' in {directory}"
        assert fpath.stat().st_size > 0, f"Artifact '{fname}' is empty in {directory}"

    # 1. config.json
    from forensight.training.r2 import load_r2_config
    config = load_r2_config(directory / "config.json")
    assert isinstance(config, dict)
    assert config["variant"] in ("semantic", "forensic", "fusion")
    assert "model" in config
    assert "training" in config
    assert "evaluation" in config
    assert "seed" in config

    # 2. checkpoint.pt
    ckpt = torch.load(directory / "checkpoint.pt", map_location="cpu")
    assert isinstance(ckpt, dict)
    assert "epoch" in ckpt
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert "config" in ckpt

    # 3. train_history.json
    with (directory / "train_history.json").open("r", encoding="utf-8") as f:
        history = json.load(f)
    assert isinstance(history, list)
    assert len(history) > 0
    for entry in history:
        assert "epoch" in entry
        assert "train_loss" in entry
        assert "val_loss" in entry
        assert "val_auroc" in entry

    # 4. manifests
    train_m = Manifest.from_jsonl(directory / "train_manifest.jsonl")
    assert len(train_m) > 0
    val_m = Manifest.from_jsonl(directory / "val_manifest.jsonl")
    assert len(val_m) > 0
    test_m = Manifest.from_jsonl(directory / "test_manifest.jsonl")
    assert len(test_m) > 0
    # Invariant: No duplicate samples in test manifest
    test_ids = [r.sample_id for r in test_m]
    assert len(test_ids) == len(set(test_ids)), f"Test manifest contains duplicate IDs: {len(test_ids) - len(set(test_ids))}"

    # 5. predictions.jsonl
    pset = PredictionSet.from_jsonl(directory / "predictions.jsonl")
    assert len(pset) > 0
    pset_ids = [r.sample_id for r in pset]
    assert len(pset_ids) == len(set(pset_ids)), f"Predictions contains duplicate IDs: {len(pset_ids) - len(set(pset_ids))}"
    assert len(pset_ids) == len(test_ids), f"Prediction count ({len(pset_ids)}) != test manifest count ({len(test_ids)})"

    with (directory / "predictions.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert "sample_id" in rec
            assert "path" in rec
            assert rec["path"] is not None and len(str(rec["path"])) > 0
            # Ensure not ephemeral absolute path
            assert not str(rec["path"]).startswith("/tmp/")
            assert not str(rec["path"]).startswith("/kaggle/")
            assert "label" in rec
            assert rec["label"] in (0, 1)
            assert "score" in rec
            assert isinstance(rec["score"], (int, float))
            assert "prediction" in rec
            assert rec["prediction"] in (0, 1), f"Sample prediction must be 0 or 1, got {rec.get('prediction')}"
            assert "generator" in rec
            assert "split" in rec

    # 6. evaluation.json
    with (directory / "evaluation.json").open("r", encoding="utf-8") as f:
        eval_report = json.load(f)
    assert "overall" in eval_report
    assert "threshold_metadata" in eval_report
    assert "by_split" in eval_report
    assert "by_generator" in eval_report
    overall = eval_report["overall"]
    assert "auroc" in overall
    assert "accuracy" in overall
    assert "f1" in overall
    assert "confusion_matrix" in overall

    # 7. reproducibility.json
    repro = ReproducibilityRecord.load_json(directory / "reproducibility.json")
    val_errors = repro.validate(require_provenance=True)
    assert len(val_errors) == 0, f"ReproducibilityRecord validation errors: {val_errors}"
    assert repro.dataset is not None
    assert repro.dataset_revision is not None

    # 8. dataset_audit.json
    with (directory / "dataset_audit.json").open("r", encoding="utf-8") as f:
        audit = json.load(f)
    assert "val_samples" in audit
    assert "test_samples" in audit
    leakage = audit["leakage"]
    if "has_leakage" in leakage:
        assert leakage["has_leakage"] is False
    if "exact_duplicates" in leakage:
        assert leakage["exact_duplicates"] == 0

    return {
        "status": "VALID",
        "num_predictions": len(pset),
        "test_samples": len(test_m),
        "epochs": len(history),
    }


def test_artifact_contract_end_to_end(tmp_path: Path):
    """Run train_r2 and evaluate_r2 into the same run directory and verify all contract artifacts."""
    run_dir = tmp_path / "results" / "r2" / "test_exp" / "seed_42"
    run_dir.mkdir(parents=True)

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Generate synthetic images with relative paths
    train_paths = []
    for i in range(4):
        p = img_dir / f"train_{i}.png"
        Image.new("RGB", (32, 32), color=(i * 20, 100, 100)).save(p)
        train_paths.append(p)

    val_paths = []
    for i in range(4):
        p = img_dir / f"val_{i}.png"
        Image.new("RGB", (32, 32), color=(i * 30, 80, 80)).save(p)
        val_paths.append(p)

    test_paths = []
    for i in range(4):
        p = img_dir / f"test_{i}.png"
        Image.new("RGB", (32, 32), color=(i * 40, 60, 60)).save(p)
        test_paths.append(p)

    # Manifests with relative image_paths (relative to tmp_path)
    train_m = Manifest([
        ManifestRecord(sample_id="tr0", image_path=f"images/train_0.png", label=0, dataset="genimage", generator="nature", split="train"),
        ManifestRecord(sample_id="tr1", image_path=f"images/train_1.png", label=0, dataset="genimage", generator="nature", split="train"),
        ManifestRecord(sample_id="tf0", image_path=f"images/train_2.png", label=1, dataset="genimage", generator="sd14", split="train"),
        ManifestRecord(sample_id="tf1", image_path=f"images/train_3.png", label=1, dataset="genimage", generator="sd14", split="train"),
    ])
    val_m = Manifest([
        ManifestRecord(sample_id="vr0", image_path=f"images/val_0.png", label=0, dataset="genimage", generator="nature", split="val"),
        ManifestRecord(sample_id="vr1", image_path=f"images/val_1.png", label=0, dataset="genimage", generator="nature", split="val"),
        ManifestRecord(sample_id="vf0", image_path=f"images/val_2.png", label=1, dataset="genimage", generator="sd14", split="val"),
        ManifestRecord(sample_id="vf1", image_path=f"images/val_3.png", label=1, dataset="genimage", generator="sd14", split="val"),
    ])
    test_m = Manifest([
        ManifestRecord(sample_id="ter0", image_path=f"images/test_0.png", label=0, dataset="genimage", generator="nature", split="test"),
        ManifestRecord(sample_id="ter1", image_path=f"images/test_1.png", label=0, dataset="genimage", generator="nature", split="test"),
        ManifestRecord(sample_id="tef0", image_path=f"images/test_2.png", label=1, dataset="genimage", generator="midjourney", split="test"),
        ManifestRecord(sample_id="tef1", image_path=f"images/test_3.png", label=1, dataset="genimage", generator="midjourney", split="test"),
    ])

    train_m_path = tmp_path / "train.jsonl"
    val_m_path = tmp_path / "val.jsonl"
    test_m_path = tmp_path / "test.jsonl"
    train_m.to_jsonl(train_m_path)
    val_m.to_jsonl(val_m_path)
    test_m.to_jsonl(test_m_path)

    # Config using forensic variant (does not require external CLIP weights)
    config = {
        "variant": "forensic",
        "seed": 42,
        "model": {
            "projection_dim": 32,
            "hidden_dim": 16,
            "dropout": 0.1,
            "npr_scale_factor": 0.5,
            "npr_mode": "bicubic",
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "num_workers": 0,
        },
        "evaluation": {
            "threshold_strategy": "f1",
        },
    }
    config_path = tmp_path / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f)

    # 1. Execute train_r2
    train_args = [
        "--config", str(config_path),
        "--train-manifest", str(train_m_path),
        "--val-manifest", str(val_m_path),
        "--base-dir", str(tmp_path),
        "--output-dir", str(run_dir),
        "--device", "cpu",
    ]
    train_ret = train_r2.parse_args(train_args)
    # Monkey-patch sys.argv or run main logic
    from unittest.mock import patch
    with patch("sys.argv", ["train_r2.py"] + train_args):
        assert train_r2.main() == 0

    # 2. Execute evaluate_r2
    eval_args = [
        "--config", str(run_dir / "config.json"),
        "--checkpoint", str(run_dir / "checkpoint.pt"),
        "--val-manifest", str(val_m_path),
        "--eval-manifest", str(test_m_path),
        "--base-dir", str(tmp_path),
        "--output-dir", str(run_dir),
        "--dataset", "Tiny-GenImage",
        "--dataset-revision", "test-rev-123",
        "--device", "cpu",
    ]
    with patch("sys.argv", ["evaluate_r2.py"] + eval_args):
        assert evaluate_r2.main() == 0

    # 3. Assert all Artifact Contract requirements pass
    result = validate_run_artifact_contract(run_dir)
    assert result["status"] == "VALID"
    assert result["epochs"] == 1
    assert result["test_samples"] == 4


def test_kaggle_runner_artifact_contract(tmp_path: Path):
    """Verify that deploy/kaggle/main.py produces all Artifact Contract files and valid zip packaging for LOGO and All-in-One."""
    from deploy.kaggle.main import build_logo_splits, build_all_in_one_splits, train_and_eval_experiment, zip_results
    import hashlib
    from torchvision import transforms

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Generate synthetic records across multiple generators
    records = []
    gens = ["nature", "midjourney", "sd14", "adm"]
    counter = 0
    for g in gens:
        for split_role in ["train", "validation"]:
            for i in range(4):
                counter += 1
                lbl = 0 if g == "nature" else 1
                p = img_dir / f"{g}_{split_role}_{i}.png"
                img = Image.new("RGB", (32, 32), color=(counter * 7 % 255, (counter * 13) % 255, (counter * 19) % 255))
                img.save(p)
                raw_bytes = p.read_bytes()
                records.append({
                    "sample_id": f"tiny_genimage_{g}_{split_role}_{i:06d}",
                    "image_path": f"tiny_genimage/{g}/{split_role}/{g}_{split_role}_{i:06d}.png",
                    "abs_path": str(p),
                    "label": lbl,
                    "generator": g,
                    "raw_split": split_role,
                    "dataset": "TheKernel01/Tiny-GenImage",
                    "content_hash": hashlib.sha256(raw_bytes).hexdigest(),
                })

    # Test Canonical R2 splits (docs/plans/r2-forensic-perception-v1.md)
    from deploy.kaggle.main import build_canonical_r2_splits
    canonical_splits = build_canonical_r2_splits(records, n_val=2, seed=42)
    assert "train" in canonical_splits
    assert "val" in canonical_splits
    assert "in_domain_test" in canonical_splits
    assert "near_ood" in canonical_splits
    assert "cross_generator_ood" in canonical_splits

    # All test samples in canonical splits must be completely disjoint
    all_test_canonical = canonical_splits["in_domain_test"] + canonical_splits["near_ood"] + canonical_splits["cross_generator_ood"]
    canonical_test_ids = [r["sample_id"] for r in all_test_canonical]
    assert len(canonical_test_ids) == len(set(canonical_test_ids)), "Canonical test splits contain duplicate samples!"

    # Test LOGO splits
    logo_splits = build_logo_splits(records, leave_out_gen="midjourney", n_val_per_gen=2, seed=42)
    assert "cross_generator_ood" in logo_splits
    assert "in_domain_test" in logo_splits

    # Test All-In-One splits (MUST NOT have duplicate test samples)
    aio_splits = build_all_in_one_splits(records, n_val_per_gen=2, seed=42)
    aio_test_splits = [k for k in aio_splits.keys() if k.startswith("test_")]
    all_aio_test_records = []
    for s in aio_test_splits:
        all_aio_test_records.extend(aio_splits[s])
    aio_test_ids = [r["sample_id"] for r in all_aio_test_records]
    assert len(aio_test_ids) == len(set(aio_test_ids)), "All-in-One contains duplicate test samples!"

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, clip_img=None, foren_img=None):
            feat = clip_img[:, :1, 0, 0] if clip_img is not None else foren_img[:, :1, 0, 0]
            return self.linear(feat)

    t = transforms.ToTensor()

    # Verify all 3 variants with canonical R2 splits produce valid Artifact Contract outputs
    for v in ["semantic_only", "forensic_only", "fusion"]:
        exp_name = f"canonical_r2_{v}"
        train_and_eval_experiment(
            experiment_name=exp_name,
            model=DummyModel(),
            splits=canonical_splits,
            clip_transform=t,
            forensic_transform=t,
            device=torch.device("cpu"),
            output_dir=tmp_path,
            epochs=1,
            batch_size=2,
            num_workers=0,
            seed=42,
            git_commit="test_commit_sha",
            variant=v,
        )

        exp_run_dir = tmp_path / "results" / "r2" / exp_name / "seed_42"
        result = validate_run_artifact_contract(exp_run_dir)
        assert result["status"] == "VALID"
        assert result["epochs"] == 1

        # Verify predictions.jsonl vocabulary matches manifest
        with (exp_run_dir / "predictions.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                assert rec["split"] in ("in_domain_test", "near_ood", "cross_generator_ood")

    # Test zip packaging
    zip_path = tmp_path / "results_r2.zip"
    zip_results(tmp_path / "results", zip_path)
    assert zip_path.exists() and zip_path.stat().st_size > 0


def test_kaggle_audit_detects_train_val_content_duplicate():
    """A byte-identical image shared by train and val must fail the audit.

    Regression: the previous guard only compared sample_ids *within* the test splits,
    so a genuine train/val content collision passed silently and training continued.
    """
    import hashlib
    from deploy.kaggle.main import generate_dataset_audit

    shared = hashlib.sha256(b"identical-bytes").hexdigest()
    train = [
        {"sample_id": "t1", "label": 0, "content_hash": shared},
        {"sample_id": "t2", "label": 1, "content_hash": "train-only"},
    ]
    val = [
        {"sample_id": "v1", "label": 0, "content_hash": shared},
        {"sample_id": "v2", "label": 1, "content_hash": "val-only"},
    ]

    audit = generate_dataset_audit(train, val, [])
    assert audit["status"] == "FAILED"
    assert audit["leakage"]["train_val_overlap"] == 1
    assert audit["leakage"]["exact_duplicates"] == 1


def test_kaggle_audit_passes_on_disjoint_content_hashes():
    """Distinct content hashes across all three partitions must pass the audit."""
    from deploy.kaggle.main import generate_dataset_audit

    train = [{"sample_id": "t1", "label": 0, "content_hash": "h1"}]
    val = [{"sample_id": "v1", "label": 0, "content_hash": "h2"}]
    test = [{"sample_id": "e1", "label": 1, "content_hash": "h3"}]

    audit = generate_dataset_audit(train, val, test)
    assert audit["status"] == "PASS"
    assert audit["leakage"]["exact_duplicates"] == 0


def test_kaggle_runner_refuses_to_train_on_leaked_splits(tmp_path: Path):
    """Training must abort when the audit finds leakage, instead of logging and continuing.

    Regression: the runner previously logged `Dataset audit completed: FAILED` and then
    trained anyway, which is how a leaked run reached `results/`.
    """
    import pytest
    from deploy.kaggle.main import train_and_eval_experiment
    from torchvision import transforms

    class MockDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trainable = torch.nn.Linear(2, 1)

        def forward(self, clip_img=None, foren_img=None):
            return self.trainable(clip_img[:, :2, 0, 0])

    img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    img.save(tmp_path / "img.png")

    shared = {"sample_id": "dup", "label": 0, "generator": "sd14", "content_hash": "same"}
    leaked_splits = {
        "train": [dict(shared, image_path="img.png", abs_path=str(tmp_path / "img.png"))],
        "val": [dict(shared, image_path="img.png", abs_path=str(tmp_path / "img.png"))],
        "in_domain_test": [],
    }

    with pytest.raises(RuntimeError, match="leakage audit FAILED"):
        train_and_eval_experiment(
            experiment_name="test_leak",
            model=MockDetector(),
            splits=leaked_splits,
            clip_transform=transforms.ToTensor(),
            forensic_transform=transforms.ToTensor(),
            device=torch.device("cpu"),
            output_dir=tmp_path,
            epochs=1,
            batch_size=1,
            num_workers=0,
            variant="semantic_only",
        )


def test_kaggle_seed_isolation(tmp_path: Path):
    """Verify that model weights are deterministic whether run alone or as part of a suite."""
    from deploy.kaggle.main import build_detector, set_seed

    # Run in isolation
    set_seed(42)
    m_isolated = build_detector("fusion", model_name="ViT-B-32", pretrained="", hidden_dim=64, dropout=0.1)

    # Run after other variants consume RNG
    set_seed(42)
    _ = build_detector("semantic_only", model_name="ViT-B-32", pretrained="", hidden_dim=64, dropout=0.1)
    # Simulate RNG consumption
    _ = torch.randn(100, 100)

    set_seed(42)
    _ = build_detector("forensic_only", hidden_dim=64, dropout=0.1)
    _ = torch.randn(100, 100)

    set_seed(42)
    m_suite = build_detector("fusion", model_name="ViT-B-32", pretrained="", hidden_dim=64, dropout=0.1)

    for p1, p2 in zip(m_isolated.classifier.parameters(), m_suite.classifier.parameters()):
        assert torch.equal(p1, p2), "Model parameters differ! Seed isolation failed."


def test_kaggle_threshold_reproducibility():
    """Verify deploy/kaggle/main.py select_threshold matches R0 select_threshold exactly."""
    import numpy as np
    from deploy.kaggle.main import select_threshold as kaggle_select_threshold
    from forensight.evaluation.metrics import select_threshold as r0_select_threshold

    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=100)
    y_scores = rng.uniform(0.1, 0.9, size=100)

    for strat in ["f1", "accuracy", "youden"]:
        t_kaggle = kaggle_select_threshold(y_true, y_scores, strategy=strat)
        t_r0 = r0_select_threshold(y_true, y_scores, strategy=strat)
        assert np.isclose(t_kaggle, t_r0, atol=1e-6), f"Threshold mismatch for strategy '{strat}': {t_kaggle} != {t_r0}"


def test_kaggle_seed_control(tmp_path: Path):
    """Verify that different seeds in build_logo_splits produce different splits."""
    from deploy.kaggle.main import build_logo_splits

    records = []
    for i in range(20):
        records.append({
            "sample_id": f"samp_real_{i}",
            "image_path": f"img_real_{i}.png",
            "label": 0,
            "generator": "nature",
            "raw_split": "train" if i < 10 else "validation",
        })
    for g in ["sd14", "midjourney"]:
        for i in range(20):
            records.append({
                "sample_id": f"samp_{g}_{i}",
                "image_path": f"img_{g}_{i}.png",
                "label": 1,
                "generator": g,
                "raw_split": "train" if i < 10 else "validation",
            })

    splits_42 = build_logo_splits(records, leave_out_gen="midjourney", n_val_per_gen=2, seed=42)
    splits_99 = build_logo_splits(records, leave_out_gen="midjourney", n_val_per_gen=2, seed=99)

    order_42 = [r["sample_id"] for r in splits_42["train"]]
    order_99 = [r["sample_id"] for r in splits_99["train"]]
    assert order_42 != order_99, "Different seeds must produce different shuffle ordering!"


def test_kaggle_git_provenance():
    """Verify get_git_commit_sha() returns the live git commit when running inside repository."""
    import subprocess
    from deploy.kaggle.main import get_git_commit_sha

    sha = get_git_commit_sha()
    assert sha != "unversioned_kaggle_run"
    assert len(sha) == 40

    expected = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert sha == expected


def test_kaggle_optimization_protocol(tmp_path: Path):
    """Verify that Kaggle runner adheres strictly to canonical R2 optimization configs:
    - semantic: 1e-3
    - forensic & fusion: 1e-4
    - only trainable parameters (requires_grad=True) are optimized
    - explicit lr override is respected
    """
    from deploy.kaggle.main import train_and_eval_experiment
    from torchvision import transforms

    class MockDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Frozen parameter (e.g. CLIP backbone)
            self.frozen = torch.nn.Linear(2, 2)
            for p in self.frozen.parameters():
                p.requires_grad = False
            # Trainable parameter
            self.trainable = torch.nn.Linear(2, 1)

        def forward(self, clip_img=None, foren_img=None):
            return self.trainable(clip_img[:, :2, 0, 0])

    t = transforms.ToTensor()
    # Six distinct records: train / val / test must be mutually disjoint, otherwise the
    # leakage audit aborts before training.
    records = [
        {
            "sample_id": f"s_{i}",
            "image_path": f"img_{i}.png",
            "abs_path": str(tmp_path / f"img_{i}.png"),
            "label": i % 2,
            "generator": "sd14",
            "split": "train",
            "content_hash": f"hash_{i}",
        }
        for i in range(6)
    ]
    # Create one fake image per record
    for i in range(6):
        img = Image.new("RGB", (32, 32), color=(100, 100, 100))
        img.save(tmp_path / f"img_{i}.png")

    dummy_splits = {
        "train": records[0:2],
        "val": records[2:4],
        "in_domain_test": records[4:6],
    }

    # 1. Semantic variant default learning rate: 1e-3
    rep_sem = train_and_eval_experiment(
        experiment_name="test_sem",
        model=MockDetector(),
        splits=dummy_splits,
        clip_transform=t,
        forensic_transform=t,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        epochs=1,
        batch_size=2,
        num_workers=0,
        variant="semantic_only",
    )
    with (tmp_path / "results" / "r2" / "test_sem" / "seed_42" / "config.json").open("r", encoding="utf-8") as f:
        cfg_sem = json.load(f)
    assert cfg_sem["training"]["learning_rate"] == 0.001

    # 2. Forensic variant default learning rate: 1e-4
    rep_foren = train_and_eval_experiment(
        experiment_name="test_foren",
        model=MockDetector(),
        splits=dummy_splits,
        clip_transform=t,
        forensic_transform=t,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        epochs=1,
        batch_size=2,
        num_workers=0,
        variant="forensic_only",
    )
    with (tmp_path / "results" / "r2" / "test_foren" / "seed_42" / "config.json").open("r", encoding="utf-8") as f:
        cfg_foren = json.load(f)
    assert cfg_foren["training"]["learning_rate"] == 0.0001

    # 3. Explicit learning rate override
    train_and_eval_experiment(
        experiment_name="test_custom_lr",
        model=MockDetector(),
        splits=dummy_splits,
        clip_transform=t,
        forensic_transform=t,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        epochs=1,
        batch_size=2,
        num_workers=0,
        variant="fusion",
        learning_rate=0.0005,
    )
    with (tmp_path / "results" / "r2" / "test_custom_lr" / "seed_42" / "config.json").open("r", encoding="utf-8") as f:
        cfg_custom = json.load(f)
    assert cfg_custom["training"]["learning_rate"] == 0.0005


def test_kaggle_load_splits_from_manifests(tmp_path: Path):
    """Verify loading sealed R0 manifests and reconciling with extracted records."""
    from deploy.kaggle.main import load_splits_from_manifests

    # Generate synthetic image files
    img_dir = tmp_path / "extracted_images"
    img_dir.mkdir()
    extracted_records = []

    for i in range(10):
        p = img_dir / f"img_{i}.png"
        img = Image.new("RGB", (32, 32), color=(i * 10, i * 10, i * 10))
        img.save(p)
        extracted_records.append({
            "sample_id": f"sample_{i:04d}",
            "image_path": f"relative/img_{i}.png",
            "abs_path": str(p),
            "label": i % 2,
            "generator": "sd14" if i < 6 else "sd15",
            "content_hash": f"hash_{i}",
        })

    # Create sealed train, val, and test manifests
    train_m = tmp_path / "train_manifest.jsonl"
    val_m = tmp_path / "val_manifest.jsonl"
    test_m = tmp_path / "test_manifest.jsonl"

    with open(train_m, "w", encoding="utf-8") as f:
        for r in extracted_records[:4]:
            f.write(json.dumps({
                "sample_id": r["sample_id"],
                "image_path": r["image_path"],
                "label": r["label"],
                "generator": r["generator"],
                "split": "train",
                "dataset": "TheKernel01/Tiny-GenImage",
            }) + "\n")

    with open(val_m, "w", encoding="utf-8") as f:
        for r in extracted_records[4:6]:
            f.write(json.dumps({
                "sample_id": r["sample_id"],
                "image_path": r["image_path"],
                "label": r["label"],
                "generator": r["generator"],
                "split": "val",
                "dataset": "TheKernel01/Tiny-GenImage",
            }) + "\n")

    with open(test_m, "w", encoding="utf-8") as f:
        for r in extracted_records[6:8]:
            f.write(json.dumps({
                "sample_id": r["sample_id"],
                "image_path": r["image_path"],
                "label": r["label"],
                "generator": r["generator"],
                "split": "in_domain_test",
                "dataset": "TheKernel01/Tiny-GenImage",
                "metadata": {"eval_slice": "in_domain_test"},
            }) + "\n")
        for r in extracted_records[8:]:
            f.write(json.dumps({
                "sample_id": r["sample_id"],
                "image_path": r["image_path"],
                "label": r["label"],
                "generator": r["generator"],
                "split": "near_ood",
                "dataset": "TheKernel01/Tiny-GenImage",
                "metadata": {"eval_slice": "near_ood"},
            }) + "\n")

    # Load splits
    splits = load_splits_from_manifests(
        train_manifest=train_m,
        val_manifest=val_m,
        test_manifest=test_m,
        extracted_records=extracted_records,
    )

    assert "train" in splits
    assert "val" in splits
    assert "in_domain_test" in splits
    assert "near_ood" in splits
    assert len(splits["train"]) == 4
    assert len(splits["val"]) == 2
    assert len(splits["in_domain_test"]) == 2
    assert len(splits["near_ood"]) == 2

    # Check abs_path and content_hash reconciliation
    for k, v in splits.items():
        for r in v:
            assert Path(r["abs_path"]).exists(), f"Resolved abs_path does not exist: {r['abs_path']}"
            assert r["content_hash"] is not None and r["content_hash"].startswith("hash_")


def test_kaggle_dataset_sha_pinning(monkeypatch, tmp_path: Path):
    """Verify that prepare_tiny_genimage pins repo_files and downloads to resolved commit SHA."""
    from unittest.mock import MagicMock
    import deploy.kaggle.main as kaggle_main

    mock_sha = "c03f567890abcdef1234567890abcdef12345678"
    mock_api = MagicMock()
    mock_info = MagicMock()
    mock_info.sha = mock_sha
    mock_api.dataset_info.return_value = mock_info
    mock_api.list_repo_files.return_value = []

    monkeypatch.setattr(kaggle_main, "HfApi", lambda: mock_api)

    records, resolved_sha = kaggle_main.prepare_tiny_genimage(
        repo_id="test/repo",
        target_root=str(tmp_path),
        revision="v1.0",
    )

    assert resolved_sha == mock_sha
    mock_api.dataset_info.assert_called_once_with(repo_id="test/repo", revision="v1.0")
    # list_repo_files must be pinned to resolved_sha
    mock_api.list_repo_files.assert_called_once_with(repo_id="test/repo", revision=mock_sha, repo_type="dataset")
