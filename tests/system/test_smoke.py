"""Unit tests for ForenSight smoke-testing protocol and pipeline (Task 0.7).

Tests cover:
- Synthetic smoke dataset generation, folder structures, and manifest counts.
- Generator disjointness and zero leakage in synthetic partitions.
- Image integrity, format validity, and Pillow JPEG quantization tables.
- End-to-end execution of all 7 pipeline stages via run_smoke_pipeline().
- Multi-seed repeated run aggregation and preliminary status assertions.
- CLI script execution via main() and subprocess.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image
import pytest

from forensight.data.audit import estimate_jpeg_quality
from forensight.data.smoke import (
    DEFAULT_SYNSET_IDS,
    _assert_audit_leakage_clean,
    _resolve_classes,
    generate_smoke_dataset,
    run_smoke_pipeline,
)
from forensight.data.split import (
    Manifest,
    assert_generator_disjoint,
    validate_no_leakage,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_smoke_test import main


class TestGenerateSmokeDataset:
    """Tests for synthetic smoke dataset generation and directory layouts."""

    def test_default_generation_structure_and_counts(self, tmp_path: Path):
        manifest_paths, manifests = generate_smoke_dataset(tmp_path, num_classes=5, seed=42)

        expected_splits = {"train", "val", "cross_generator_ood"}
        assert set(manifest_paths.keys()) == expected_splits
        assert set(manifests.keys()) == expected_splits

        # Check directory layout requirements
        expected_dirs = [
            tmp_path / "genimage" / "sdv5" / "train" / "nature",
            tmp_path / "genimage" / "sdv5" / "train" / "ai",
            tmp_path / "genimage" / "sdv5" / "val" / "nature",
            tmp_path / "genimage" / "sdv5" / "val" / "ai",
            tmp_path / "genimage" / "midjourney" / "val" / "nature",
            tmp_path / "genimage" / "midjourney" / "val" / "ai",
        ]
        for d in expected_dirs:
            assert d.exists() and d.is_dir(), f"Expected directory missing: {d}"

        # 5 classes * 2 (real + fake) = 10 samples per split
        for split_name in expected_splits:
            m = manifests[split_name]
            assert len(m) == 10
            num_real = sum(1 for r in m if r.label == 0)
            num_fake = sum(1 for r in m if r.label == 1)
            assert num_real == 5
            assert num_fake == 5

            # Manifest file exists on disk and is readable
            p = manifest_paths[split_name]
            assert p.exists()
            reloaded = Manifest.from_jsonl(p)
            assert len(reloaded) == 10

    def test_custom_num_classes(self, tmp_path: Path):
        _, manifests_2 = generate_smoke_dataset(tmp_path / "c2", num_classes=2)
        assert len(manifests_2["train"]) == 4

        _, manifests_7 = generate_smoke_dataset(tmp_path / "c7", num_classes=7)
        assert len(manifests_7["train"]) == 14

    def test_resolve_classes_collision_free(self):
        classes_small = _resolve_classes(len(DEFAULT_SYNSET_IDS))
        assert len(classes_small) == len(DEFAULT_SYNSET_IDS)
        assert len(set(classes_small)) == len(DEFAULT_SYNSET_IDS)

        # Scale to 3500 classes to span beyond synset range and ensure zero duplicates
        classes_large = _resolve_classes(3500)
        assert len(classes_large) == 3500
        assert len(set(classes_large)) == 3500
        for cls_id in classes_large:
            assert cls_id.startswith("n")
            assert len(cls_id) == 9

    def test_invalid_num_classes_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="num_classes must be positive integer"):
            generate_smoke_dataset(tmp_path, num_classes=0)

        with pytest.raises(ValueError, match="num_classes must be positive integer"):
            generate_smoke_dataset(tmp_path, num_classes=-3)

    def test_image_integrity_and_quantization(self, tmp_path: Path):
        _, manifests = generate_smoke_dataset(tmp_path, num_classes=3, seed=42)
        sample = manifests["train"][0]

        img_path = Path(sample.image_path)
        assert img_path.exists()

        with Image.open(img_path) as img:
            assert img.format == "JPEG"
            assert img.mode == "RGB"
            assert img.size == (64, 64)
            qf = estimate_jpeg_quality(img)
            assert qf is not None
            assert 70 <= qf <= 95

    def test_determinism_across_identical_seeds(self, tmp_path: Path):
        _, m1 = generate_smoke_dataset(tmp_path / "run1", num_classes=3, seed=42)
        _, m2 = generate_smoke_dataset(tmp_path / "run2", num_classes=3, seed=42)

        img1_bytes = Path(m1["train"][0].image_path).read_bytes()
        img2_bytes = Path(m2["train"][0].image_path).read_bytes()

        hash1 = hashlib.sha256(img1_bytes).hexdigest()
        hash2 = hashlib.sha256(img2_bytes).hexdigest()
        assert hash1 == hash2

    def test_manifest_metadata_and_split_attributes(self, tmp_path: Path):
        _, manifests = generate_smoke_dataset(tmp_path, num_classes=3, seed=42)

        train_m = manifests["train"]
        val_m = manifests["val"]
        cross_m = manifests["cross_generator_ood"]

        # Check generators
        assert {r.generator for r in train_m if r.label == 1} == {"sd15"}
        assert {r.generator for r in val_m if r.label == 1} == {"sd15"}
        assert {r.generator for r in cross_m if r.label == 1} == {"midjourney"}

        # Real generators are always 'nature'
        assert {r.generator for r in train_m if r.label == 0} == {"nature"}

        # Invariant checks
        assert_generator_disjoint(train_m, [cross_m])
        validate_no_leakage(train_m, val_m, check_generators=False, strict=True)
        validate_no_leakage(train_m, cross_m, check_generators=True, strict=True)


class TestAuditLeakageGuard:
    """Tests that the Stage 4 audit-leakage guard reads the correct contract keys.

    Regression: the guard previously read non-existent keys ('leakage_detected',
    'generator_leakage_detected') instead of 'has_leakage', so it never raised
    even when audit_manifest_leakage reported real leakage.
    """

    def _leaking_audit_result(self) -> dict:
        return {
            "has_leakage": True,
            "sample_id_collisions": [
                {"eval_split": "cross_generator_ood", "count": 1, "colliding_sample_ids": ["s1"]}
            ],
            "image_path_collisions": [],
            "generator_overlaps": [
                {
                    "eval_split": "cross_generator_ood",
                    "train_generators": ["midjourney"],
                    "eval_generators": ["midjourney"],
                    "overlapping_generators": ["midjourney"],
                }
            ],
            "violations": [
                "Sample ID leakage between train and cross_generator_ood: 1 shared IDs.",
                "Generator leakage between train and cross_generator_ood: overlapping fake generator(s) ['midjourney'].",
            ],
        }

    def test_guard_raises_on_reported_leakage(self):
        with pytest.raises(ValueError, match="leakage"):
            _assert_audit_leakage_clean(self._leaking_audit_result())

    def test_guard_raises_on_leakage_without_violations_list(self):
        with pytest.raises(ValueError, match="leakage"):
            _assert_audit_leakage_clean({"has_leakage": True})

    def test_guard_passes_on_clean_audit(self):
        clean = {
            "has_leakage": False,
            "sample_id_collisions": [],
            "image_path_collisions": [],
            "generator_overlaps": [],
            "violations": [],
        }
        # Should not raise
        _assert_audit_leakage_clean(clean)


class TestRunSmokePipeline:
    """Tests for full end-to-end pipeline execution."""

    def test_pipeline_all_stages_succeed(self, tmp_path: Path):
        result = run_smoke_pipeline(output_dir=tmp_path, seed=42, num_classes=4)

        assert result["status"] == "success"
        assert result["duration_seconds"] > 0
        stages = result["stages"]

        expected_stages = [
            "1_inventory",
            "2_data_generation",
            "3_disjointness_and_leakage",
            "4_audit",
            "5_predictions",
            "6_evaluation",
            "7_aggregation",
        ]
        for st in expected_stages:
            assert st in stages
            assert stages[st]["status"] == "passed"

    def test_pipeline_aggregation_metrics(self, tmp_path: Path):
        result = run_smoke_pipeline(output_dir=tmp_path, seed=42, num_classes=5)

        agg = result["aggregated_report"]
        assert agg["num_runs"] == 3
        # 3 seeds ensures the benchmark is not marked preliminary
        assert agg["is_preliminary"] is False

        overall = agg["overall"]
        assert "auroc" in overall
        assert "accuracy" in overall
        assert "f1" in overall
        assert overall["auroc"]["mean"] is not None
        assert overall["accuracy"]["mean"] is not None

        # Verify breakdowns exist
        assert "by_split" in agg
        assert "val" in agg["by_split"]
        assert "cross_generator_ood" in agg["by_split"]

    def test_pipeline_artifacts_written_to_disk(self, tmp_path: Path):
        run_smoke_pipeline(output_dir=tmp_path, seed=42, num_classes=3)

        # Manifests
        assert (tmp_path / "manifests" / "train.jsonl").exists()
        assert (tmp_path / "manifests" / "cross_generator_ood.jsonl").exists()

        # Audit reports
        assert (tmp_path / "audit" / "train_audit.json").exists()
        assert (tmp_path / "audit" / "train_audit.md").exists()

        # Evaluation reports
        assert (tmp_path / "evaluation" / "report_seed_42.json").exists()
        assert (tmp_path / "evaluation" / "report_seed_43.json").exists()
        assert (tmp_path / "evaluation" / "report_seed_44.json").exists()
        assert (tmp_path / "evaluation" / "aggregated_report.json").exists()
        assert (tmp_path / "evaluation" / "aggregated_report.md").exists()

        # Summary
        summary_file = tmp_path / "smoke_pipeline_summary.json"
        assert summary_file.exists()
        loaded = json.loads(summary_file.read_text(encoding="utf-8"))
        assert loaded["status"] == "success"


class TestSmokeCLI:
    """Tests for scripts/run_smoke_test.py CLI interface."""

    def test_cli_help(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_cli_keep_artifacts(self, tmp_path: Path):
        out_dir = tmp_path / "cli_out"
        exit_code = main(["--output-dir", str(out_dir), "--keep-artifacts", "--quiet"])
        assert exit_code == 0
        assert (out_dir / "smoke_pipeline_summary.json").exists()

    def test_cli_temporary_run(self):
        exit_code = main(["--quiet"])
        assert exit_code == 0

    def test_cli_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        out_dir = tmp_path / "json_out"
        exit_code = main(["--output-dir", str(out_dir), "--json", "--num-classes", "3"])
        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "success"
        assert data["stages"]["7_aggregation"]["num_runs"] == 3

    def test_cli_subprocess_execution(self, tmp_path: Path):
        out_dir = tmp_path / "subp_out"
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/run_smoke_test.py",
                "--output-dir",
                str(out_dir),
                "--keep-artifacts",
                "--num-classes",
                "3",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert "All 7 ForenSight R0 stages completed successfully." in proc.stdout
        assert (out_dir / "smoke_pipeline_summary.json").exists()
