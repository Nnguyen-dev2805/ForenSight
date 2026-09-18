"""Unit and integration tests for ForenSight reproducibility records (Task 0.8).

Tests cover:
- ReproducibilityRecord dataclass initialization, validation, and serialization.
- Validation checks detecting missing, malformed, or invalid fields.
- JSON round-trip and disk persistence (to_dict, from_dict, to_json, from_json, save_json, load_json).
- Markdown report generation from ReproducibilityRecord.
- Git commit introspection and error handling.
- Runtime environment introspection (Python version, platform, dependencies, hardware).
- create_reproducibility_record() helper with explicit kwargs and EvaluationReport instances.
- EvaluationReport.to_reproducibility_record() convenience method.
- CLI integration via scripts/run_evaluation.py (--save-reproducibility, --split-version, --config-file, --notes).
- Subprocess execution verifying end-to-end CLI reproducibility output.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from forensight.evaluation.metrics import MetricResult
from forensight.evaluation.reproducibility import (
    ReproducibilityRecord,
    create_reproducibility_record,
    get_environment_info,
    get_git_commit,
)
from forensight.evaluation.runner import (
    EvaluationReport,
    PredictionRecord,
    PredictionSet,
    evaluate_predictions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_evaluation as cli_module


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_valid_record() -> ReproducibilityRecord:
    return ReproducibilityRecord(
        run_id="run_exp01_seed42",
        experiment_name="baseline_pilot",
        timestamp="2026-09-18T12:00:00+00:00",
        git_commit="abcdef1234567890abcdef1234567890abcdef12",
        seed=42,
        split_version="r0-smoke",
        config={"backbone": "clip-vit-b32", "batch_size": 32, "lr": 0.0001},
        threshold_source="val_f1",
        threshold_value=0.55,
        metrics={
            "overall": {
                "auroc": 0.9500,
                "accuracy": 0.9000,
                "f1": 0.8800,
            }
        },
        environment={
            "python_version": "3.14.5",
            "platform": "macOS-Darwin",
            "packages": {"numpy": "2.5.3", "pandas": "3.0.6"},
        },
        notes="Standard smoke test evaluation run.",
    )


@pytest.fixture
def sample_evaluation_report() -> EvaluationReport:
    preds = PredictionSet([
        PredictionRecord("v1", 0, 0.1, split="val", generator="nature"),
        PredictionRecord("v2", 1, 0.9, split="val", generator="sd14"),
        PredictionRecord("t1", 0, 0.2, split="test", generator="nature"),
        PredictionRecord("t2", 1, 0.8, split="test", generator="sd14"),
    ])
    return evaluate_predictions(
        predictions=preds,
        val_split_name="val",
        threshold_strategy="f1",
        run_metadata={"run_name": "eval_test_report"},
        seed=42,
    )


# =====================================================================
# 1. ReproducibilityRecord Dataclass & Validation Tests
# =====================================================================

class TestReproducibilityRecordDataclass:
    """Tests for dataclass fields, property helpers, and validation invariants."""

    def test_valid_record_passes_validation(self, sample_valid_record):
        errors = sample_valid_record.validate()
        assert errors == []
        assert sample_valid_record.is_valid is True
        # assert_valid should not raise
        sample_valid_record.assert_valid()

    def test_missing_or_empty_string_fields(self):
        rec = ReproducibilityRecord(
            run_id="",
            experiment_name="   ",
            timestamp="",
            git_commit=None,
            seed=None,
            split_version="",
            config={},
            threshold_source="   ",
            threshold_value=0.5,
            metrics={},
            environment={},
            notes="",
        )
        errors = rec.validate()
        assert rec.is_valid is False
        assert any("run_id" in e for e in errors)
        assert any("experiment_name" in e for e in errors)
        assert any("timestamp" in e for e in errors)
        assert any("split_version" in e for e in errors)
        assert any("threshold_source" in e for e in errors)

        with pytest.raises(ValueError, match="ReproducibilityRecord validation failed"):
            rec.assert_valid()

    def test_invalid_timestamp_format(self, sample_valid_record):
        sample_valid_record.timestamp = "not-a-timestamp-2026"
        errors = sample_valid_record.validate()
        assert any("timestamp" in e for e in errors)

    def test_invalid_threshold_values(self, sample_valid_record):
        sample_valid_record.threshold_value = float("nan")
        errors = sample_valid_record.validate()
        assert any("threshold_value" in e for e in errors)

        sample_valid_record.threshold_value = float("inf")
        errors = sample_valid_record.validate()
        assert any("threshold_value" in e for e in errors)

    def test_invalid_types_for_dict_fields(self, sample_valid_record):
        sample_valid_record.config = "not-a-dict"  # type: ignore
        sample_valid_record.metrics = ["not", "a", "dict"]  # type: ignore
        sample_valid_record.environment = 123  # type: ignore
        errors = sample_valid_record.validate()
        assert any("config" in e for e in errors)
        assert any("metrics" in e for e in errors)
        assert any("environment" in e for e in errors)

    def test_invalid_seed_types(self, sample_valid_record):
        sample_valid_record.seed = [42]  # type: ignore
        errors = sample_valid_record.validate()
        assert any("seed" in e for e in errors)

        sample_valid_record.seed = "   "
        errors = sample_valid_record.validate()
        assert any("seed" in e for e in errors)

        # Valid seeds
        sample_valid_record.seed = 42
        assert sample_valid_record.validate() == []
        sample_valid_record.seed = "seed_101"
        assert sample_valid_record.validate() == []
        sample_valid_record.seed = None
        assert sample_valid_record.validate() == []

    def test_invalid_git_commit(self, sample_valid_record):
        sample_valid_record.git_commit = 12345  # type: ignore
        errors = sample_valid_record.validate()
        assert any("git_commit" in e for e in errors)

        sample_valid_record.git_commit = "   "
        errors = sample_valid_record.validate()
        assert any("git_commit" in e for e in errors)

    def test_invalid_notes_type(self, sample_valid_record):
        sample_valid_record.notes = {"note": "abc"}  # type: ignore
        errors = sample_valid_record.validate()
        assert any("notes" in e for e in errors)


# =====================================================================
# 2. Serialization & Round-Trip Tests
# =====================================================================

class TestSerialization:
    """Tests for to_dict, from_dict, to_json, from_json, save_json, and load_json."""

    def test_dict_round_trip(self, sample_valid_record):
        data = sample_valid_record.to_dict()
        assert isinstance(data, dict)
        assert data["run_id"] == sample_valid_record.run_id
        assert data["threshold_value"] == 0.55

        reconstructed = ReproducibilityRecord.from_dict(data)
        assert reconstructed == sample_valid_record
        assert reconstructed.is_valid is True

    def test_json_round_trip(self, sample_valid_record):
        json_str = sample_valid_record.to_json()
        assert isinstance(json_str, str)

        reconstructed = ReproducibilityRecord.from_json(json_str)
        assert reconstructed == sample_valid_record
        assert reconstructed.is_valid is True

    def test_disk_save_and_load(self, sample_valid_record, tmp_path: Path):
        file_path = tmp_path / "sub" / "repro_record.json"
        sample_valid_record.save_json(file_path)
        assert file_path.exists()

        loaded = ReproducibilityRecord.load_json(file_path)
        assert loaded == sample_valid_record
        assert loaded.is_valid is True

    def test_from_dict_with_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected dict"):
            ReproducibilityRecord.from_dict("not-a-dict")  # type: ignore

    def test_from_json_with_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Invalid JSON string"):
            ReproducibilityRecord.from_json("invalid json {")

    def test_load_json_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ReproducibilityRecord.load_json(tmp_path / "non_existent.json")

    def test_generate_markdown(self, sample_valid_record):
        md = sample_valid_record.generate_markdown()
        assert "# ForenSight Reproducibility Record: `run_exp01_seed42`" in md
        assert "## 1. Experiment & Provenance Metadata" in md
        assert "## 2. Environment" in md
        assert "## 3. Notes" in md
        assert "## 4. Key Metrics Summary" in md
        assert "`abcdef1234`" in md
        assert "0.9500" in md


# =====================================================================
# 3. Environment & Git Introspection Tests
# =====================================================================

class TestIntrospection:
    """Tests for get_git_commit and get_environment_info."""

    def test_get_git_commit_in_repo(self):
        commit = get_git_commit()
        # Since this repository is git-tracked, HEAD commit must be non-empty hex SHA
        assert commit is not None
        assert isinstance(commit, str)
        assert len(commit) >= 7

    def test_get_git_commit_invalid_dir(self, tmp_path: Path):
        # tmp_path is outside git repository
        commit = get_git_commit(cwd=tmp_path)
        assert commit is None

    def test_get_environment_info_default(self):
        env = get_environment_info()
        assert "python_version" in env
        assert "platform" in env
        assert "packages" in env
        assert "hardware" in env

        packages = env["packages"]
        # Expected packages in ForenSight virtualenv
        assert "numpy" in packages
        assert packages["numpy"] is not None
        assert "pandas" in packages
        assert packages["pandas"] is not None
        assert "pytest" in packages
        assert packages["pytest"] is not None

    def test_get_environment_info_custom_packages(self):
        env = get_environment_info(packages=["numpy", "totally_non_existent_pkg_12345"])
        packages = env["packages"]
        assert len(packages) == 2
        assert packages["numpy"] is not None
        assert packages["totally_non_existent_pkg_12345"] is None


# =====================================================================
# 4. Helper create_reproducibility_record Tests
# =====================================================================

class TestCreateReproducibilityRecord:
    """Tests for create_reproducibility_record helper."""

    def test_create_with_defaults(self):
        rec = create_reproducibility_record(
            experiment_name="test_default_exp",
            split_version="r0-smoke",
        )
        assert rec.experiment_name == "test_default_exp"
        assert rec.split_version == "r0-smoke"
        assert rec.threshold_source == "default"
        assert rec.threshold_value == 0.5
        assert rec.git_commit is not None
        assert rec.environment["python_version"] is not None
        assert rec.is_valid is True

    def test_create_from_evaluation_report(self, sample_evaluation_report):
        rec = create_reproducibility_record(
            report=sample_evaluation_report,
            split_version="r0-eval-test",
            config={"model": "dummy_probe"},
            notes="Evaluated via EvaluationReport",
        )
        assert rec.run_id == "eval_test_report"
        assert rec.seed == 42
        assert rec.split_version == "r0-eval-test"
        assert rec.threshold_source == sample_evaluation_report.overall.threshold_source
        assert rec.threshold_value == pytest.approx(sample_evaluation_report.overall.threshold)
        assert "overall" in rec.metrics
        assert rec.config == {"model": "dummy_probe"}
        assert rec.notes == "Evaluated via EvaluationReport"
        assert rec.is_valid is True

    def test_create_passing_report_to_metrics(self, sample_evaluation_report):
        rec = create_reproducibility_record(
            metrics=sample_evaluation_report,
            split_version="r0-eval-test",
        )
        assert rec.threshold_source == sample_evaluation_report.overall.threshold_source
        assert rec.threshold_value == pytest.approx(sample_evaluation_report.overall.threshold)
        assert rec.is_valid is True

    def test_explicit_git_commit_none_not_overwritten(self):
        rec = create_reproducibility_record(
            git_commit=None,
            experiment_name="unversioned_run",
        )
        assert rec.git_commit is None
        assert rec.is_valid is True

    def test_evaluation_report_to_reproducibility_record_method(self, sample_evaluation_report):
        rec = sample_evaluation_report.to_reproducibility_record(
            split_version="v2.1",
            config={"opt": "sgd"},
            notes="Method call test",
        )
        assert isinstance(rec, ReproducibilityRecord)
        assert rec.run_id == "eval_test_report"
        assert rec.split_version == "v2.1"
        assert rec.config == {"opt": "sgd"}
        assert rec.notes == "Method call test"
        assert rec.is_valid is True


# =====================================================================
# 5. CLI Integration Tests
# =====================================================================

class TestCLIIntegration:
    """Integration tests for scripts/run_evaluation.py with reproducibility flags."""

    @pytest.fixture
    def test_csv_predictions(self, tmp_path: Path) -> Path:
        csv_path = tmp_path / "preds.csv"
        records = [
            PredictionRecord("v1", 0, 0.1, split="val", generator="nature"),
            PredictionRecord("v2", 1, 0.9, split="val", generator="sd14"),
            PredictionRecord("t1", 0, 0.2, split="test", generator="nature"),
            PredictionRecord("t2", 1, 0.8, split="test", generator="sd14"),
        ]
        PredictionSet(records).to_csv(csv_path)
        return csv_path

    @pytest.fixture
    def test_config_json(self, tmp_path: Path) -> Path:
        cfg_path = tmp_path / "config.json"
        cfg_data = {
            "model_architecture": "resnet50_linear_probe",
            "feature_layer": "layer4",
            "batch_size": 64,
            "optimizer": "adamw",
        }
        cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")
        return cfg_path

    def test_cli_save_reproducibility_basic(self, test_csv_predictions, tmp_path: Path):
        repro_path = tmp_path / "repro.json"
        exit_code = cli_module.main([
            "--predictions", str(test_csv_predictions),
            "--val-split", "val",
            "--threshold-strategy", "f1",
            "--seed", "42",
            "--split-version", "genimage_v1.0",
            "--save-reproducibility", str(repro_path),
            "--notes", "CLI basic test",
            "--quiet",
        ])
        assert exit_code == 0
        assert repro_path.exists()

        record = ReproducibilityRecord.load_json(repro_path)
        assert record.is_valid is True
        assert record.seed == 42
        assert record.split_version == "genimage_v1.0"
        assert record.notes == "CLI basic test"
        assert record.threshold_source == "val_optimal_f1"
        assert "overall" in record.metrics

    def test_cli_with_config_file(self, test_csv_predictions, test_config_json, tmp_path: Path):
        repro_path = tmp_path / "repro_cfg.json"
        exit_code = cli_module.main([
            "--predictions", str(test_csv_predictions),
            "--val-split", "val",
            "--config-file", str(test_config_json),
            "--save-reproducibility", str(repro_path),
            "--quiet",
        ])
        assert exit_code == 0
        assert repro_path.exists()

        record = ReproducibilityRecord.load_json(repro_path)
        assert record.config["model_architecture"] == "resnet50_linear_probe"
        assert record.config["feature_layer"] == "layer4"
        assert record.is_valid is True

    def test_cli_missing_config_file_returns_error(self, test_csv_predictions, tmp_path: Path):
        exit_code = cli_module.main([
            "--predictions", str(test_csv_predictions),
            "--config-file", str(tmp_path / "missing_config.json"),
            "--quiet",
        ])
        assert exit_code == 1

    def test_cli_malformed_config_file_returns_error(self, test_csv_predictions, tmp_path: Path):
        bad_cfg = tmp_path / "bad_config.json"
        bad_cfg.write_text("invalid json [", encoding="utf-8")

        exit_code = cli_module.main([
            "--predictions", str(test_csv_predictions),
            "--config-file", str(bad_cfg),
            "--quiet",
        ])
        assert exit_code == 1

    def test_cli_subprocess_execution(self, test_csv_predictions, tmp_path: Path):
        repro_path = tmp_path / "subproc_repro.json"
        out_json = tmp_path / "report.json"
        cmd = [
            sys.executable,
            "scripts/run_evaluation.py",
            "--predictions", str(test_csv_predictions),
            "--val-split", "val",
            "--output-json", str(out_json),
            "--save-reproducibility", str(repro_path),
            "--split-version", "r0-smoke-subproc",
            "--seed", "99",
            "--quiet",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert out_json.exists()
        assert repro_path.exists()

        record = ReproducibilityRecord.load_json(repro_path)
        assert record.seed == 99
        assert record.split_version == "r0-smoke-subproc"
        assert record.is_valid is True
