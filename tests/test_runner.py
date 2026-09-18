"""Comprehensive unit tests for ForenSight evaluation runner, prediction containers, and CLI."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

from forensight.evaluation.metrics import MetricResult
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
# 1. PredictionRecord Tests
# =====================================================================

class TestPredictionRecord:
    """Unit tests for atomic PredictionRecord initialization and validation."""

    def test_valid_record(self):
        rec = PredictionRecord(
            sample_id="img_001.jpg",
            label=1,
            score=0.85,
            split="in_domain_test",
            generator="sd14",
            dataset="genimage",
            metadata={"res": 512},
        )
        assert rec.sample_id == "img_001.jpg"
        assert rec.label == 1
        assert rec.score == 0.85
        assert rec.split == "in_domain_test"
        assert rec.generator == "sd14"
        assert rec.dataset == "genimage"
        assert rec.metadata == {"res": 512}

    def test_default_optional_fields(self):
        rec = PredictionRecord(sample_id="img_002", label=0, score=0.12)
        assert rec.split is None
        assert rec.generator is None
        assert rec.dataset is None
        assert rec.metadata == {}

    def test_type_normalization(self):
        rec = PredictionRecord(
            sample_id=12345,  # int coerced to str
            label=True,       # bool coerced to int 1
            score="0.75",     # string coerced to float
            split="  val  ",
        )
        assert rec.sample_id == "12345"
        assert rec.label == 1
        assert isinstance(rec.label, int)
        assert rec.score == 0.75
        assert isinstance(rec.score, float)
        assert rec.split == "val"

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="label must be 0 .* or 1"):
            PredictionRecord(sample_id="s1", label=2, score=0.5)
        with pytest.raises(ValueError, match="label must be 0 .* or 1"):
            PredictionRecord(sample_id="s1", label=-1, score=0.5)
        with pytest.raises(ValueError, match="label must be 0 .* or 1"):
            PredictionRecord(sample_id="s1", label="invalid", score=0.5)

    def test_invalid_score_raises(self):
        with pytest.raises(ValueError, match="score cannot be NaN or infinite"):
            PredictionRecord(sample_id="s1", label=1, score=float("nan"))
        with pytest.raises(ValueError, match="score cannot be NaN or infinite"):
            PredictionRecord(sample_id="s1", label=1, score=float("inf"))
        with pytest.raises(ValueError, match="score must be numerical"):
            PredictionRecord(sample_id="s1", label=1, score="not_a_number")

    def test_empty_sample_id_raises(self):
        with pytest.raises(ValueError, match="sample_id must be a non-empty string"):
            PredictionRecord(sample_id="", label=1, score=0.5)
        with pytest.raises(ValueError, match="sample_id must be a non-empty string"):
            PredictionRecord(sample_id="   ", label=1, score=0.5)
        with pytest.raises(ValueError, match="sample_id cannot be None"):
            PredictionRecord(sample_id=None, label=1, score=0.5)

    def test_non_dict_metadata_raises(self):
        with pytest.raises(ValueError, match="metadata must be a dictionary"):
            PredictionRecord(sample_id="s1", label=1, score=0.5, metadata=["not", "a", "dict"])

    def test_serialization_roundtrip(self):
        orig = PredictionRecord(
            sample_id="s_99",
            label=0,
            score=0.04,
            split="val",
            generator="nature",
            dataset="genimage",
            metadata={"seed": 42},
        )
        d = orig.to_dict()
        assert d["sample_id"] == "s_99"
        assert d["score"] == 0.04
        rebuilt = PredictionRecord.from_dict(d)
        assert rebuilt == orig

    def test_from_dict_with_column_aliases(self):
        # Test id / target / prob aliases
        d1 = {"id": "alias_1", "target": 1, "prob": 0.92, "split": "test"}
        r1 = PredictionRecord.from_dict(d1)
        assert r1.sample_id == "alias_1"
        assert r1.label == 1
        assert r1.score == 0.92

        # Test image_path / y_true / prediction aliases
        d2 = {"image_path": "path/img.png", "y_true": 0, "prediction": 0.15}
        r2 = PredictionRecord.from_dict(d2)
        assert r2.sample_id == "path/img.png"
        assert r2.label == 0
        assert r2.score == 0.15

        # Test extra keys captured in metadata
        d3 = {
            "sample_id": "s3",
            "label": 1,
            "score": 0.88,
            "custom_feature": "abc",
            "prompt": "a photo of a cat",
        }
        r3 = PredictionRecord.from_dict(d3)
        assert r3.metadata["custom_feature"] == "abc"
        assert r3.metadata["prompt"] == "a photo of a cat"


# =====================================================================
# 2. PredictionSet Tests
# =====================================================================

class TestPredictionSet:
    """Unit tests for PredictionSet collection container, slicing, and I/O."""

    @pytest.fixture
    def sample_records(self) -> list[PredictionRecord]:
        return [
            PredictionRecord("s1", 0, 0.10, split="val", generator="nature", dataset="genimage"),
            PredictionRecord("s2", 1, 0.90, split="val", generator="sd14", dataset="genimage"),
            PredictionRecord("s3", 0, 0.20, split="test", generator="nature", dataset="genimage"),
            PredictionRecord("s4", 1, 0.85, split="test", generator="midjourney", dataset="genimage"),
            PredictionRecord("s5", 1, 0.75, split="test", generator="midjourney", dataset="wildrf"),
        ]

    def test_initialization_from_records_and_dicts(self, sample_records):
        pset = PredictionSet(sample_records)
        assert len(pset) == 5

        # Dict initialization
        dicts = [r.to_dict() for r in sample_records]
        pset_from_dicts = PredictionSet(dicts)
        assert len(pset_from_dicts) == 5
        assert pset_from_dicts.sample_ids == pset.sample_ids

    def test_invalid_record_type_raises(self):
        with pytest.raises(TypeError, match="Expected PredictionRecord or dict"):
            PredictionSet([123])

    def test_indexing_and_slicing(self, sample_records):
        pset = PredictionSet(sample_records)
        first = pset[0]
        assert isinstance(first, PredictionRecord)
        assert first.sample_id == "s1"

        sliced = pset[1:4]
        assert isinstance(sliced, PredictionSet)
        assert len(sliced) == 3
        assert sliced.sample_ids == ["s2", "s3", "s4"]

    def test_properties(self, sample_records):
        pset = PredictionSet(sample_records)
        assert pset.sample_ids == ["s1", "s2", "s3", "s4", "s5"]
        assert np.array_equal(pset.y_true, np.array([0, 1, 0, 1, 1]))
        assert np.allclose(pset.y_scores, np.array([0.1, 0.9, 0.2, 0.85, 0.75]))
        assert pset.splits == {"val", "test"}
        assert pset.generators == {"nature", "sd14", "midjourney"}
        assert pset.datasets == {"genimage", "wildrf"}
        assert not pset.has_duplicates()

    def test_empty_set_properties(self):
        empty = PredictionSet([])
        assert len(empty) == 0
        assert empty.sample_ids == []
        assert len(empty.y_true) == 0
        assert len(empty.y_scores) == 0
        assert empty.splits == set()
        assert empty.generators == set()
        assert empty.datasets == set()
        assert not empty.has_duplicates()

    def test_duplicate_detection(self):
        recs = [
            PredictionRecord("dup_1", 0, 0.2),
            PredictionRecord("dup_1", 1, 0.8),
        ]
        pset = PredictionSet(recs)
        assert pset.has_duplicates()

    def test_filtering_methods(self, sample_records):
        pset = PredictionSet(sample_records)

        val_subset = pset.get_split("val")
        assert len(val_subset) == 2
        assert set(val_subset.sample_ids) == {"s1", "s2"}

        mj_subset = pset.get_generator("midjourney")
        assert len(mj_subset) == 2
        assert set(mj_subset.sample_ids) == {"s4", "s5"}

        wild_subset = pset.get_dataset("wildrf")
        assert len(wild_subset) == 1
        assert wild_subset.sample_ids == ["s5"]

        custom_filter = pset.filter(lambda r: r.score >= 0.8)
        assert len(custom_filter) == 2
        assert set(custom_filter.sample_ids) == {"s2", "s4"}

    def test_dataframe_roundtrip(self, sample_records):
        pset = PredictionSet(sample_records)
        df = pset.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ["sample_id", "label", "score", "split", "generator", "dataset", "metadata"]

        rebuilt = PredictionSet.from_dataframe(df)
        assert len(rebuilt) == 5
        assert rebuilt.sample_ids == pset.sample_ids
        assert np.array_equal(rebuilt.y_true, pset.y_true)
        assert np.allclose(rebuilt.y_scores, pset.y_scores)

    def test_csv_roundtrip(self, sample_records, tmp_path):
        csv_path = tmp_path / "test_preds.csv"
        pset = PredictionSet(sample_records)
        pset.to_csv(csv_path)
        assert csv_path.exists()

        loaded = PredictionSet.from_csv(csv_path)
        assert len(loaded) == len(pset)
        assert loaded.sample_ids == pset.sample_ids
        assert np.array_equal(loaded.y_true, pset.y_true)
        assert np.allclose(loaded.y_scores, pset.y_scores)
        assert loaded.splits == pset.splits

    def test_jsonl_roundtrip(self, sample_records, tmp_path):
        jsonl_path = tmp_path / "test_preds.jsonl"
        pset = PredictionSet(sample_records)
        pset.to_jsonl(jsonl_path)
        assert jsonl_path.exists()

        loaded = PredictionSet.from_jsonl(jsonl_path)
        assert len(loaded) == len(pset)
        assert loaded.sample_ids == pset.sample_ids
        assert np.array_equal(loaded.y_true, pset.y_true)
        assert np.allclose(loaded.y_scores, pset.y_scores)

    def test_file_not_found_raises(self, tmp_path):
        non_existent = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError):
            PredictionSet.from_csv(non_existent)
        with pytest.raises(FileNotFoundError):
            PredictionSet.from_jsonl(non_existent)


# =====================================================================
# 3. evaluate_predictions Tests
# =====================================================================

class TestEvaluatePredictions:
    """Unit tests for evaluate_predictions, threshold calibration, and invariants."""

    @pytest.fixture
    def benchmark_predictions(self) -> PredictionSet:
        """Construct synthetic prediction set with val, in_domain, and cross_gen splits."""
        records = [
            # Validation set (used for calibration: optimal F1 threshold should be ~0.50)
            PredictionRecord("v1", 0, 0.10, split="val", generator="nature", dataset="genimage"),
            PredictionRecord("v2", 0, 0.30, split="val", generator="nature", dataset="genimage"),
            PredictionRecord("v3", 1, 0.70, split="val", generator="sd14", dataset="genimage"),
            PredictionRecord("v4", 1, 0.90, split="val", generator="sd14", dataset="genimage"),
            # In-domain test set
            PredictionRecord("t1", 0, 0.15, split="in_domain_test", generator="nature", dataset="genimage"),
            PredictionRecord("t2", 1, 0.85, split="in_domain_test", generator="sd14", dataset="genimage"),
            # Cross-generator OOD test set
            PredictionRecord("t3", 0, 0.25, split="cross_generator_ood", generator="nature", dataset="genimage"),
            PredictionRecord("t4", 1, 0.65, split="cross_generator_ood", generator="midjourney", dataset="genimage"),
            PredictionRecord("t5", 1, 0.75, split="cross_generator_ood", generator="adm", dataset="genimage"),
        ]
        return PredictionSet(records)

    def test_evaluation_with_validation_calibration(self, benchmark_predictions):
        report = evaluate_predictions(
            predictions=benchmark_predictions,
            val_split_name="val",
            threshold_strategy="f1",
            run_metadata={"run_name": "test_run_01"},
        )

        assert isinstance(report, EvaluationReport)
        # Threshold should be calibrated from validation split
        assert report.threshold_metadata["calibrated"] is True
        assert report.threshold_metadata["strategy"] == "f1"
        assert report.threshold_metadata["threshold_source"] == "val_optimal_f1"
        assert report.threshold_metadata["val_samples_count"] == 4

        # Overall benchmark samples must EXCLUDE validation samples per evaluation protocol!
        # Total samples = 9, Val samples = 4 -> Evaluated overall = 5
        assert report.run_metadata["evaluated_samples"] == 5
        assert report.run_metadata["total_samples"] == 9

        # Verify overall metric calculations
        assert report.overall.auroc == 1.0  # Perfect separation on test
        assert report.overall.accuracy == 1.0
        assert report.overall.f1 == 1.0

        # Verify split breakdowns exist
        assert "val" in report.by_split
        assert "in_domain_test" in report.by_split
        assert "cross_generator_ood" in report.by_split

        # Verify generator breakdowns exist
        assert "nature" in report.by_generator
        assert "sd14" in report.by_generator
        assert "midjourney" in report.by_generator
        assert "adm" in report.by_generator

    def test_evaluation_without_validation_split_fallback(self):
        # Predictions with test splits only (no val split)
        records = [
            PredictionRecord("t1", 0, 0.20, split="test", generator="nature"),
            PredictionRecord("t2", 1, 0.80, split="test", generator="sd14"),
        ]
        pset = PredictionSet(records)

        report = evaluate_predictions(
            predictions=pset,
            val_split_name="val",  # 'val' split absent
            default_threshold=0.5,
        )

        assert report.threshold_metadata["calibrated"] is False
        assert report.threshold_metadata["threshold"] == 0.5
        assert report.threshold_metadata["threshold_source"] == "default_0.5"
        assert report.threshold_metadata["val_samples_count"] == 0
        assert report.run_metadata["evaluated_samples"] == 2
        assert report.overall.accuracy == 1.0

    def test_evaluation_with_separate_val_predictions(self):
        val_records = [
            PredictionRecord("v1", 0, 0.10),
            PredictionRecord("v2", 1, 0.90),
        ]
        test_records = [
            PredictionRecord("t1", 0, 0.15, generator="nature"),
            PredictionRecord("t2", 1, 0.85, generator="midjourney"),
        ]
        val_pset = PredictionSet(val_records)
        test_pset = PredictionSet(test_records)

        report = evaluate_predictions(
            predictions=test_pset,
            val_predictions=val_pset,
            threshold_strategy="accuracy",
        )

        assert report.threshold_metadata["calibrated"] is True
        assert report.threshold_metadata["val_samples_count"] == 2
        assert report.threshold_metadata["strategy"] == "accuracy"
        assert report.run_metadata["evaluated_samples"] == 2

    def test_explicit_eval_splits(self, benchmark_predictions):
        report = evaluate_predictions(
            predictions=benchmark_predictions,
            val_split_name="val",
            eval_splits=["in_domain_test"],
        )
        assert report.run_metadata["evaluated_samples"] == 2

    def test_single_class_validation_fallback(self):
        # Validation split has only fake images (class 1) -> cannot calibrate binary threshold
        records = [
            PredictionRecord("v1", 1, 0.90, split="val"),
            PredictionRecord("v2", 1, 0.85, split="val"),
            PredictionRecord("t1", 0, 0.10, split="test"),
            PredictionRecord("t2", 1, 0.95, split="test"),
        ]
        pset = PredictionSet(records)
        report = evaluate_predictions(pset, val_split_name="val", default_threshold=0.5)

        assert report.threshold_metadata["calibrated"] is False
        assert report.threshold_metadata["threshold"] == 0.5
        assert report.threshold_metadata["threshold_source"] == "default_0.5"

    def test_single_class_generator_slice_handling(self, benchmark_predictions):
        # Generator 'midjourney' has only fake samples (class 1)
        report = evaluate_predictions(benchmark_predictions, val_split_name="val")
        mj_metrics = report.by_generator["midjourney"]
        # AUROC should be None (undefined on 1 class), but secondary metrics computed without error
        assert mj_metrics.auroc is None
        assert mj_metrics.accuracy == 1.0
        assert mj_metrics.recall == 1.0

    def test_empty_predictions_raises(self):
        with pytest.raises(ValueError, match="Cannot evaluate empty PredictionSet"):
            evaluate_predictions(PredictionSet([]))

    def test_invalid_strategy_raises(self, benchmark_predictions):
        with pytest.raises(ValueError, match="Invalid threshold_strategy"):
            evaluate_predictions(benchmark_predictions, threshold_strategy="unknown_criterion")


# =====================================================================
# 4. EvaluationReport Serialization & Markdown Tests
# =====================================================================

class TestEvaluationReport:
    """Unit tests for EvaluationReport JSON and Markdown export."""

    @pytest.fixture
    def sample_report(self) -> EvaluationReport:
        records = [
            PredictionRecord("v1", 0, 0.1, split="val", generator="nature", dataset="genimage"),
            PredictionRecord("v2", 1, 0.9, split="val", generator="sd14", dataset="genimage"),
            PredictionRecord("t1", 0, 0.2, split="test", generator="nature", dataset="genimage"),
            PredictionRecord("t2", 1, 0.8, split="test", generator="midjourney", dataset="genimage"),
        ]
        return evaluate_predictions(PredictionSet(records), val_split_name="val")

    def test_to_dict_and_from_dict(self, sample_report):
        d = sample_report.to_dict()
        assert "overall" in d
        assert "by_split" in d
        assert "by_generator" in d
        assert "threshold_metadata" in d
        assert "run_metadata" in d

        rebuilt = EvaluationReport.from_dict(d)
        assert rebuilt.overall.accuracy == sample_report.overall.accuracy
        assert rebuilt.threshold_metadata == sample_report.threshold_metadata
        assert set(rebuilt.by_split.keys()) == set(sample_report.by_split.keys())

    def test_json_roundtrip_and_save(self, sample_report, tmp_path):
        json_str = sample_report.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "overall" in parsed

        rebuilt = EvaluationReport.from_json(json_str)
        assert rebuilt.overall.f1 == sample_report.overall.f1

        # Save to disk
        out_path = tmp_path / "report.json"
        sample_report.save_json(out_path)
        assert out_path.exists()
        disk_loaded = EvaluationReport.from_json(out_path.read_text(encoding="utf-8"))
        assert disk_loaded.overall.auroc == sample_report.overall.auroc

    def test_markdown_generation_and_save(self, sample_report, tmp_path):
        md = sample_report.generate_markdown()
        assert isinstance(md, str)
        # Check essential headers and invariant indicators
        assert "# ForenSight Evaluation Report" in md
        assert "## 1. Decision Threshold Provenance" in md
        assert "## 2. Overall Benchmark Performance" in md
        assert "## 3. Performance Breakdown by Partition (Split)" in md
        assert "## 4. Performance Breakdown by Generator Architecture" in md
        assert "VERIFIED" in md

        # Save markdown
        md_path = tmp_path / "report.md"
        sample_report.save_markdown(md_path)
        assert md_path.exists()
        saved_text = md_path.read_text(encoding="utf-8")
        assert "AUROC" in saved_text


# =====================================================================
# 5. CLI Execution Tests
# =====================================================================

class TestCLI:
    """Unit tests for scripts/run_evaluation.py CLI."""

    @pytest.fixture
    def test_csv_file(self, tmp_path) -> Path:
        csv_path = tmp_path / "preds.csv"
        records = [
            PredictionRecord("v1", 0, 0.1, split="val", generator="nature", dataset="genimage"),
            PredictionRecord("v2", 1, 0.9, split="val", generator="sd14", dataset="genimage"),
            PredictionRecord("t1", 0, 0.2, split="test", generator="nature", dataset="genimage"),
            PredictionRecord("t2", 1, 0.8, split="test", generator="sd14", dataset="genimage"),
        ]
        PredictionSet(records).to_csv(csv_path)
        return csv_path

    @pytest.fixture
    def test_jsonl_file(self, tmp_path) -> Path:
        jsonl_path = tmp_path / "preds.jsonl"
        records = [
            PredictionRecord("v1", 0, 0.1, split="val", generator="nature"),
            PredictionRecord("v2", 1, 0.9, split="val", generator="sd14"),
            PredictionRecord("t1", 0, 0.2, split="test", generator="nature"),
            PredictionRecord("t2", 1, 0.8, split="test", generator="midjourney"),
        ]
        PredictionSet(records).to_jsonl(jsonl_path)
        return jsonl_path

    def test_cli_main_with_csv(self, test_csv_file, tmp_path):
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"

        exit_code = cli_module.main([
            "--predictions", str(test_csv_file),
            "--val-split", "val",
            "--threshold-strategy", "f1",
            "--output-json", str(out_json),
            "--output-md", str(out_md),
            "--run-name", "cli_test_run",
            "--quiet",
        ])

        assert exit_code == 0
        assert out_json.exists()
        assert out_md.exists()

        report_data = json.loads(out_json.read_text(encoding="utf-8"))
        assert report_data["run_metadata"]["run_name"] == "cli_test_run"
        assert report_data["threshold_metadata"]["calibrated"] is True

    def test_cli_main_with_jsonl_and_separate_val(self, test_jsonl_file, tmp_path):
        val_path = tmp_path / "val.jsonl"
        val_records = [
            PredictionRecord("v1", 0, 0.05),
            PredictionRecord("v2", 1, 0.95),
        ]
        PredictionSet(val_records).to_jsonl(val_path)

        out_json = tmp_path / "separate_val_out.json"
        exit_code = cli_module.main([
            "--predictions", str(test_jsonl_file),
            "--val-predictions", str(val_path),
            "--threshold-strategy", "youden",
            "--output-json", str(out_json),
            "--quiet",
        ])

        assert exit_code == 0
        assert out_json.exists()

    def test_cli_error_handling_missing_file(self, tmp_path):
        exit_code = cli_module.main([
            "--predictions", str(tmp_path / "non_existent.csv"),
            "--quiet",
        ])
        assert exit_code == 1

    def test_cli_subprocess_execution(self, test_csv_file, tmp_path):
        out_json = tmp_path / "subproc_out.json"
        cmd = [
            sys.executable,
            "scripts/run_evaluation.py",
            "--predictions", str(test_csv_file),
            "--val-split", "val",
            "--output-json", str(out_json),
            "--quiet",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert out_json.exists()
