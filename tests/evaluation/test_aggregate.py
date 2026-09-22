"""Comprehensive unit tests for ForenSight repeated-run multi-seed evaluation aggregation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pytest

from forensight.evaluation.aggregate import (
    AggregatedEvaluationReport,
    AggregatedMetric,
    AggregatedSlice,
    aggregate_reports,
)
from forensight.evaluation.metrics import MetricResult
from forensight.evaluation.runner import (
    EvaluationReport,
    PredictionRecord,
    PredictionSet,
    evaluate_predictions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.aggregate_runs as cli_module


# =====================================================================
# Fixtures
# =====================================================================

def make_metric_result(
    auroc: float | None = 0.85,
    accuracy: float = 0.80,
    f1: float = 0.78,
    precision: float = 0.76,
    recall: float = 0.82,
    threshold: float = 0.50,
) -> MetricResult:
    """Helper to construct dummy MetricResult instances."""
    return MetricResult(
        auroc=auroc,
        accuracy=accuracy,
        f1=f1,
        precision=precision,
        recall=recall,
        threshold=threshold,
        threshold_source="val_optimal_f1",
        confusion_matrix={"tp": 82, "fp": 26, "tn": 74, "fn": 18},
    )


def make_dummy_report(
    seed: int | str = 42,
    overall_auc: float | None = 0.85,
    overall_acc: float = 0.80,
    split_auc: float | None = 0.88,
    gen_auc: float | None = None,
) -> EvaluationReport:
    """Helper to construct dummy EvaluationReport instances."""
    return EvaluationReport(
        overall=make_metric_result(auroc=overall_auc, accuracy=overall_acc),
        by_split={
            "in_domain_test": make_metric_result(auroc=split_auc, accuracy=0.82),
            "cross_generator_ood": make_metric_result(auroc=0.75, accuracy=0.70),
        },
        by_generator={
            "sd14": make_metric_result(auroc=0.89, accuracy=0.84),
            "midjourney": make_metric_result(auroc=gen_auc, accuracy=0.78),
        },
        threshold_metadata={
            "threshold": 0.50,
            "strategy": "f1",
            "threshold_source": "val_optimal_f1",
            "calibrated": True,
        },
        run_metadata={
            "run_name": f"run_seed_{seed}",
            "evaluated_samples": 200,
            "seed": seed,
        },
        by_dataset={
            "genimage": make_metric_result(auroc=0.84, accuracy=0.81),
        },
        seed=seed,
    )


# =====================================================================
# 1. AggregatedMetric Tests
# =====================================================================

class TestAggregatedMetric:
    """Unit tests for atomic AggregatedMetric statistical container."""

    def test_single_value_aggregation(self):
        m = AggregatedMetric.from_values([0.85])
        assert m is not None
        assert m.mean == 0.85
        assert m.std == 0.0
        assert m.min == 0.85
        assert m.max == 0.85
        assert m.n == 1
        assert m.formatted == "0.8500 ± 0.0000"
        assert str(m) == "0.8500 ± 0.0000"
        assert m.format(digits=2) == "0.85 ± 0.00"

    def test_three_values_aggregation_sample_std(self):
        # Values: 0.80, 0.85, 0.90 -> mean = 0.85, sample std (ddof=1) = 0.05
        values = [0.80, 0.85, 0.90]
        m = AggregatedMetric.from_values(values)
        assert m is not None
        assert m.mean == pytest.approx(0.85)
        assert m.std == pytest.approx(0.05)
        assert m.min == 0.80
        assert m.max == 0.90
        assert m.n == 3
        assert m.formatted == "0.8500 ± 0.0500"

    def test_empty_or_all_none_returns_none(self):
        assert AggregatedMetric.from_values([]) is None
        assert AggregatedMetric.from_values([None, None]) is None

    def test_partial_none_handling(self):
        values = [0.80, None, 0.90]
        m = AggregatedMetric.from_values(values)
        assert m is not None
        assert m.n == 2
        assert m.mean == pytest.approx(0.85)
        expected_std = float(np.std([0.80, 0.90], ddof=1))
        assert m.std == pytest.approx(expected_std)

    def test_nan_values_ignored(self):
        values = [0.80, float("nan"), 0.90]
        m = AggregatedMetric.from_values(values)
        assert m is not None
        assert m.n == 2
        assert m.mean == pytest.approx(0.85)

    def test_string_nan_and_none_filtering(self):
        # Cleanly filter out string "nan" and None
        values = [0.80, "nan", None]
        m = AggregatedMetric.from_values(values)
        assert m is not None
        assert m.n == 1
        assert m.mean == pytest.approx(0.80)
        assert m.std == 0.0

    def test_none_mean_formatted(self):
        m = AggregatedMetric(mean=None, std=None, min=None, max=None, n=0)
        assert m.formatted == "N/A"
        assert m.format(digits=2) == "N/A"

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="n cannot be negative"):
            AggregatedMetric(mean=0.5, std=0.1, min=0.4, max=0.6, n=-1)
        with pytest.raises(TypeError, match="n must be an integer"):
            AggregatedMetric(mean=0.5, std=0.1, min=0.4, max=0.6, n="invalid")
        with pytest.raises(TypeError, match="mean must be float or None"):
            AggregatedMetric(mean="not_a_float", std=0.1, min=0.4, max=0.6, n=1)

    def test_serialization_roundtrip(self):
        orig = AggregatedMetric(mean=0.885, std=0.012, min=0.871, max=0.899, n=3)
        d = orig.to_dict()
        assert d["mean"] == 0.885
        assert d["formatted"] == "0.8850 ± 0.0120"
        rebuilt = AggregatedMetric.from_dict(d)
        assert rebuilt == orig


# =====================================================================
# 2. AggregatedSlice Tests
# =====================================================================

class TestAggregatedSlice:
    """Unit tests for slice-level aggregated metric container."""

    def test_slice_from_metric_results(self):
        results = [
            make_metric_result(auroc=0.82, accuracy=0.79, f1=0.77),
            make_metric_result(auroc=0.85, accuracy=0.81, f1=0.79),
            make_metric_result(auroc=0.88, accuracy=0.83, f1=0.81),
        ]
        sl = AggregatedSlice.from_metric_results(results)

        assert sl.auroc is not None
        assert sl.auroc.n == 3
        assert sl.auroc.mean == pytest.approx(0.85)
        assert sl.accuracy.mean == pytest.approx(0.81)
        assert sl.f1.mean == pytest.approx(0.79)

        # Attribute and dict access
        assert sl["accuracy"].mean == pytest.approx(0.81)
        assert sl["auroc"].mean == pytest.approx(0.85)
        assert sl.get("accuracy").mean == pytest.approx(0.81)
        assert sl.get("unknown_metric", default="fallback") == "fallback"

    def test_slice_with_all_none_auroc(self):
        # Single-class slices (e.g. only fake samples) have AUROC = None across all runs
        results = [
            make_metric_result(auroc=None, accuracy=0.90, recall=0.90),
            make_metric_result(auroc=None, accuracy=0.92, recall=0.92),
            make_metric_result(auroc=None, accuracy=0.94, recall=0.94),
        ]
        sl = AggregatedSlice.from_metric_results(results)

        assert sl.auroc is None
        assert sl["auroc"] is None
        assert sl.get("auroc") is None
        assert sl.get("auroc", default="fallback") is None
        assert sl.get("nonexistent", default="fallback") == "fallback"
        assert sl.accuracy.mean == pytest.approx(0.92)
        assert sl.recall.mean == pytest.approx(0.92)

        # Dictionary export handles None auroc
        d = sl.to_dict()
        assert d["auroc"] is None
        assert d["accuracy"]["mean"] == pytest.approx(0.92)

        rebuilt = AggregatedSlice.from_dict(d)
        assert rebuilt.auroc is None
        assert rebuilt.accuracy.mean == pytest.approx(0.92)

    def test_slice_getitem_rejects_arbitrary_attributes(self):
        results = [make_metric_result(auroc=0.80, accuracy=0.75)]
        sl = AggregatedSlice.from_metric_results(results)
        with pytest.raises(KeyError, match="Metric 'to_dict' not found"):
            _ = sl["to_dict"]
        with pytest.raises(KeyError, match="Metric 'from_dict' not found"):
            _ = sl["from_dict"]

    def test_slice_empty_raises(self):
        with pytest.raises(ValueError, match="Cannot aggregate empty list"):
            AggregatedSlice.from_metric_results([])

    def test_slice_serialization_roundtrip(self):
        results = [
            make_metric_result(auroc=0.80, accuracy=0.75),
            make_metric_result(auroc=0.82, accuracy=0.77),
        ]
        sl = AggregatedSlice.from_metric_results(results)
        d = sl.to_dict()
        rebuilt = AggregatedSlice.from_dict(d)

        assert rebuilt.auroc.mean == sl.auroc.mean
        assert rebuilt.accuracy.mean == sl.accuracy.mean
        assert rebuilt.f1.mean == sl.f1.mean


# =====================================================================
# 3. aggregate_reports Tests
# =====================================================================

class TestAggregateReports:
    """Unit tests for multi-run report aggregation logic."""

    def test_empty_reports_raises(self):
        with pytest.raises(ValueError, match="Cannot aggregate empty list of reports"):
            aggregate_reports([])

    def test_one_seed_is_preliminary(self):
        r1 = make_dummy_report(seed=42, overall_auc=0.89, overall_acc=0.85)
        agg = aggregate_reports([r1])

        assert agg.num_runs == 1
        assert agg.seeds == [42]
        assert agg.is_preliminary is True
        assert agg.preliminary_result is True

        assert agg.overall.auroc.mean == 0.89
        assert agg.overall.auroc.std == 0.0
        assert agg.overall.accuracy.mean == 0.85
        assert agg.overall.accuracy.std == 0.0

    def test_two_seeds_is_preliminary(self):
        r1 = make_dummy_report(seed=10, overall_auc=0.80, overall_acc=0.75)
        r2 = make_dummy_report(seed=20, overall_auc=0.84, overall_acc=0.79)
        agg = aggregate_reports([r1, r2])

        assert agg.num_runs == 2
        assert agg.seeds == [10, 20]
        assert agg.is_preliminary is True
        assert agg.preliminary_result is True

        assert agg.overall.auroc.mean == pytest.approx(0.82)
        assert agg.overall.auroc.std > 0.0

    def test_three_seeds_is_sealed_benchmark(self):
        r1 = make_dummy_report(seed=101, overall_auc=0.84, overall_acc=0.78)
        r2 = make_dummy_report(seed=102, overall_auc=0.86, overall_acc=0.80)
        r3 = make_dummy_report(seed=103, overall_auc=0.88, overall_acc=0.82)
        agg = aggregate_reports([r1, r2, r3])

        assert agg.num_runs == 3
        assert agg.seeds == [101, 102, 103]
        assert agg.is_preliminary is False
        assert agg.preliminary_result is False

        assert agg.overall.auroc.mean == pytest.approx(0.86)
        assert agg.overall.auroc.std == pytest.approx(0.02)
        assert agg.overall.accuracy.mean == pytest.approx(0.80)
        assert agg.overall.accuracy.std == pytest.approx(0.02)

    def test_three_runs_with_duplicate_seeds_is_preliminary(self):
        # Even with 3 runs, if all seeds are identical (e.g. 42), the benchmark cannot be sealed
        r1 = make_dummy_report(seed=42)
        r2 = make_dummy_report(seed=42)
        r3 = make_dummy_report(seed=42)
        agg = aggregate_reports([r1, r2, r3])

        assert agg.num_runs == 3
        assert agg.seeds == [42, 42, 42]
        assert agg.is_preliminary is True
        assert agg.preliminary_result is True

    def test_three_runs_with_seed_none_is_preliminary(self):
        # 3 unseeded runs (fallback to run_0, run_1, run_2) cannot be sealed
        r1 = make_dummy_report(seed=None)
        r2 = make_dummy_report(seed=None)
        r3 = make_dummy_report(seed=None)
        # Ensure seed attribute and metadata are None
        for r in (r1, r2, r3):
            r.seed = None
            r.run_metadata.pop("seed", None)

        agg = aggregate_reports([r1, r2, r3])
        assert agg.num_runs == 3
        assert agg.seeds == ["run_0", "run_1", "run_2"]
        assert agg.is_preliminary is True
        assert agg.preliminary_result is True

    def test_per_split_and_per_generator_aggregation(self):
        r1 = make_dummy_report(seed=1, split_auc=0.85, gen_auc=None)
        r2 = make_dummy_report(seed=2, split_auc=0.87, gen_auc=None)
        r3 = make_dummy_report(seed=3, split_auc=0.89, gen_auc=None)
        agg = aggregate_reports([r1, r2, r3])

        # Per-split
        assert "in_domain_test" in agg.by_split
        assert "cross_generator_ood" in agg.by_split
        in_domain_auc = agg.by_split["in_domain_test"].auroc
        assert in_domain_auc.mean == pytest.approx(0.87)
        assert in_domain_auc.std == pytest.approx(0.02)

        # Per-generator
        assert "sd14" in agg.by_generator
        assert "midjourney" in agg.by_generator
        # midjourney has None AUROC (single-class)
        assert agg.by_generator["midjourney"].auroc is None
        # but secondary metrics exist
        assert agg.by_generator["midjourney"].accuracy.mean == pytest.approx(0.78)
        assert agg.by_generator["midjourney"].accuracy.n == 3

    def test_per_dataset_aggregation(self):
        r1 = make_dummy_report(seed=1)
        r2 = make_dummy_report(seed=2)
        agg = aggregate_reports([r1, r2])

        assert "genimage" in agg.by_dataset
        assert agg.by_dataset["genimage"].auroc.mean == pytest.approx(0.84)
        assert agg.by_dataset["genimage"].accuracy.mean == pytest.approx(0.81)

    def test_missing_seed_fallback(self):
        r1 = make_dummy_report(seed=1)
        r1.seed = None
        r1.run_metadata.pop("seed", None)

        agg = aggregate_reports([r1])
        assert agg.seeds == ["run_0"]


# =====================================================================
# 4. AggregatedEvaluationReport Serialization & Markdown Tests
# =====================================================================

class TestAggregatedEvaluationReport:
    """Unit tests for AggregatedEvaluationReport serialization and Markdown generation."""

    @pytest.fixture
    def sample_aggregated_report(self) -> AggregatedEvaluationReport:
        r1 = make_dummy_report(seed=1, overall_auc=0.85, overall_acc=0.80)
        r2 = make_dummy_report(seed=2, overall_auc=0.87, overall_acc=0.82)
        r3 = make_dummy_report(seed=3, overall_auc=0.89, overall_acc=0.84)
        return aggregate_reports([r1, r2, r3], metadata={"report_name": "Test Run SD14"})

    def test_to_dict_and_from_dict_roundtrip(self, sample_aggregated_report):
        d = sample_aggregated_report.to_dict()
        assert "seeds" in d
        assert "num_runs" in d
        assert "is_preliminary" in d
        assert "preliminary_result" in d
        assert "overall" in d
        assert "by_split" in d
        assert "by_generator" in d

        rebuilt = AggregatedEvaluationReport.from_dict(d)
        assert rebuilt.seeds == sample_aggregated_report.seeds
        assert rebuilt.num_runs == sample_aggregated_report.num_runs
        assert rebuilt.is_preliminary == sample_aggregated_report.is_preliminary
        assert rebuilt.overall.accuracy.mean == sample_aggregated_report.overall.accuracy.mean
        assert rebuilt.overall.auroc.std == sample_aggregated_report.overall.auroc.std

    def test_json_roundtrip_and_save(self, sample_aggregated_report, tmp_path):
        json_str = sample_aggregated_report.to_json()
        assert isinstance(json_str, str)

        rebuilt = AggregatedEvaluationReport.from_json(json_str)
        assert rebuilt.num_runs == 3
        assert rebuilt.is_preliminary is False

        # Save to disk
        out_file = tmp_path / "aggregated.json"
        sample_aggregated_report.save_json(out_file)
        assert out_file.exists()

        disk_loaded = AggregatedEvaluationReport.from_json(out_file.read_text(encoding="utf-8"))
        assert disk_loaded.overall.auroc.mean == sample_aggregated_report.overall.auroc.mean

    def test_markdown_generation_sealed(self, sample_aggregated_report, tmp_path):
        md = sample_aggregated_report.generate_markdown()
        assert "# ForenSight Aggregated Evaluation Report" in md
        assert "SEALED BENCHMARK" in md
        assert "## 1. Overall Benchmark Performance (Aggregated)" in md
        assert "## 2. Performance Breakdown by Partition (Split)" in md
        assert "## 3. Performance Breakdown by Generator Architecture" in md
        assert "## 4. Performance Breakdown by Dataset Source" in md
        assert "## 5. Evaluation Protocol Invariants" in md
        assert "0.8700 ± 0.0200" in md

        out_md = tmp_path / "aggregated.md"
        sample_aggregated_report.save_markdown(out_md)
        assert out_md.exists()
        assert "SEALED BENCHMARK" in out_md.read_text(encoding="utf-8")

    def test_markdown_generation_preliminary(self):
        r1 = make_dummy_report(seed=42)
        agg = aggregate_reports([r1])
        md = agg.generate_markdown()
        assert "PRELIMINARY RESULT" in md
        assert "⚠️" in md


# =====================================================================
# 5. CLI Execution Tests
# =====================================================================

class TestAggregateRunsCLI:
    """Integration unit tests for scripts/aggregate_runs.py CLI."""

    def test_cli_execution_with_three_reports(self, tmp_path):
        r1 = make_dummy_report(seed=1, overall_auc=0.84, overall_acc=0.80)
        r2 = make_dummy_report(seed=2, overall_auc=0.86, overall_acc=0.82)
        r3 = make_dummy_report(seed=3, overall_auc=0.88, overall_acc=0.84)

        p1 = tmp_path / "r1.json"
        p2 = tmp_path / "r2.json"
        p3 = tmp_path / "r3.json"

        r1.save_json(p1)
        r2.save_json(p2)
        r3.save_json(p3)

        out_json = tmp_path / "out_agg.json"
        out_md = tmp_path / "out_agg.md"

        cli_args = [
            "--reports",
            str(p1),
            str(p2),
            str(p3),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--report-name",
            "CLI Benchmark SD14",
        ]

        exit_code = cli_module.main(cli_args)
        assert exit_code == 0
        assert out_json.exists()
        assert out_md.exists()

        # Check loaded report
        loaded = AggregatedEvaluationReport.from_json(out_json.read_text(encoding="utf-8"))
        assert loaded.num_runs == 3
        assert loaded.is_preliminary is False
        assert loaded.overall.auroc.mean == pytest.approx(0.86)

    def test_cli_missing_report_file_exits_with_error(self, tmp_path):
        exit_code = cli_module.main(["--reports", str(tmp_path / "non_existent.json")])
        assert exit_code == 1

    def test_cli_print_markdown(self, tmp_path, capsys):
        r1 = make_dummy_report(seed=1)
        p1 = tmp_path / "r1.json"
        r1.save_json(p1)

        exit_code = cli_module.main(["--report", str(p1), "--print-markdown"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "# ForenSight Aggregated Evaluation Report" in captured.out
        assert "PRELIMINARY RESULT" in captured.out


# =====================================================================
# 6. EvaluationReport Individual Run Seed Invariant Tests
# =====================================================================

class TestEvaluationReportSeedInvariant:
    """Verify that individual runs track and preserve the seed attribute per protocol."""

    def test_evaluate_predictions_preserves_seed(self):
        records = [
            PredictionRecord("v1", 0, 0.1, split="val"),
            PredictionRecord("v2", 1, 0.9, split="val"),
            PredictionRecord("t1", 0, 0.2, split="test"),
            PredictionRecord("t2", 1, 0.8, split="test"),
        ]
        pset = PredictionSet(records)

        # Evaluate with explicit seed
        rep = evaluate_predictions(pset, val_split_name="val", seed=42)
        assert rep.seed == 42
        assert rep.run_metadata["seed"] == 42

        # Round-trip through dict
        d = rep.to_dict()
        assert d["seed"] == 42
        rebuilt = EvaluationReport.from_dict(d)
        assert rebuilt.seed == 42

        # In markdown
        md = rep.generate_markdown()
        assert "- **Seed:** `42`" in md
