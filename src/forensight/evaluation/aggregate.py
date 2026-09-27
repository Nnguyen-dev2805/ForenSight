"""Multi-seed repeated-run aggregation, uncertainty quantification, and report generation for ForenSight.

This module implements Task 0.6 of R0:
- AggregatedMetric: statistical container for mean, sample standard deviation (ddof=1),
  min, max, sample size n, and formatted string representation ("mean ± std").
- AggregatedSlice: collection of AggregatedMetric instances for a partition, generator,
  dataset, or overall benchmark evaluation.
- AggregatedEvaluationReport: structured multi-run container tracking seeds, preliminary status
  (preliminary_result: True if num_runs < 3), slice aggregations, and provenance.
- aggregate_reports: functional entrypoint aggregating a list of EvaluationReport objects.
- Markdown summary and machine-readable JSON export capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from forensight.evaluation.metrics import MetricResult
from forensight.evaluation.runner import EvaluationReport


@dataclass(frozen=True)
class AggregatedMetric:
    """Container for aggregated statistical metrics across repeated runs.

    Attributes:
        mean: Sample arithmetic mean, or None if undefined (e.g. single-class AUROC).
        std: Sample standard deviation with Bessel's correction (ddof=1 for n >= 2, 0.0 for n = 1),
             or None if undefined.
        min: Minimum value across observed runs, or None if undefined.
        max: Maximum value across observed runs, or None if undefined.
        n: Number of valid runs included in the aggregation.
    """

    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    n: int

    def __post_init__(self) -> None:
        """Validate metric values and types."""
        def _is_numeric(val: Any) -> bool:
            return isinstance(val, (int, float, np.floating, np.integer)) and not isinstance(val, bool)

        if self.mean is not None and not _is_numeric(self.mean):
            raise TypeError(f"mean must be float or None, got {type(self.mean)}")
        if self.std is not None and not _is_numeric(self.std):
            raise TypeError(f"std must be float or None, got {type(self.std)}")
        if self.min is not None and not _is_numeric(self.min):
            raise TypeError(f"min must be float or None, got {type(self.min)}")
        if self.max is not None and not _is_numeric(self.max):
            raise TypeError(f"max must be float or None, got {type(self.max)}")
        if not (isinstance(self.n, (int, np.integer)) and not isinstance(self.n, bool)):
            raise TypeError(f"n must be an integer, got {type(self.n)}")
        if self.n < 0:
            raise ValueError(f"n cannot be negative, got {self.n}")

    @property
    def formatted(self) -> str:
        """Return formatted mean ± std string (e.g. '0.8950 ± 0.0120')."""
        return self.format()

    def format(self, digits: int = 4) -> str:
        """Return formatted string with custom decimal precision.

        A single run has no measurable dispersion, so it is reported with its sample
        size instead of a `0.0000` standard deviation, which would falsely imply zero
        variance and overstate the precision of a preliminary result.
        """
        if self.mean is None or self.std is None:
            return "N/A"
        if self.n <= 1:
            return f"{self.mean:.{digits}f} (n={self.n}, preliminary)"
        return f"{self.mean:.{digits}f} ± {self.std:.{digits}f}"

    def __str__(self) -> str:
        return self.formatted

    def to_dict(self) -> dict[str, Any]:
        """Convert AggregatedMetric to serializable dictionary."""
        return {
            "mean": float(self.mean) if self.mean is not None else None,
            "std": float(self.std) if self.std is not None else None,
            "min": float(self.min) if self.min is not None else None,
            "max": float(self.max) if self.max is not None else None,
            "n": int(self.n),
            "formatted": self.formatted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregatedMetric:
        """Construct AggregatedMetric from dictionary."""
        mean = float(data["mean"]) if data.get("mean") is not None else None
        std = float(data["std"]) if data.get("std") is not None else None
        min_val = float(data["min"]) if data.get("min") is not None else None
        max_val = float(data["max"]) if data.get("max") is not None else None
        n = int(data.get("n", 0))
        return cls(mean=mean, std=std, min=min_val, max=max_val, n=n)

    @classmethod
    def from_values(
        cls,
        values: Sequence[float | None],
        ddof: int = 1,
    ) -> AggregatedMetric | None:
        """Calculate AggregatedMetric from a sequence of scalar values.

        Rules:
        - If values is empty or contains only None / NaN values, returns None.
        - For n = 1: std is set strictly to 0.0.
        - For n >= 2: sample standard deviation is computed with Bessel's correction (ddof=1).
        """
        valid: list[float] = []
        for v in values:
            if v is None:
                continue
            try:
                fv = float(v)
                if not np.isnan(fv):
                    valid.append(fv)
            except (ValueError, TypeError):
                continue

        if not valid:
            return None

        n = len(valid)
        mean_val = float(np.mean(valid))
        if n == 1:
            std_val = 0.0
        else:
            std_val = float(np.std(valid, ddof=ddof))

        min_val = float(np.min(valid))
        max_val = float(np.max(valid))

        return cls(
            mean=mean_val,
            std=std_val,
            min=min_val,
            max=max_val,
            n=n,
        )


KNOWN_METRIC_KEYS: set[str] = {
    "auroc",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "threshold",
}


@dataclass
class AggregatedSlice:
    """Collection of aggregated metrics for an evaluation slice (overall, split, or generator).

    Attributes:
        auroc: AggregatedMetric for AUROC, or None if AUROC was undefined for all runs.
        accuracy: AggregatedMetric for classification accuracy.
        f1: AggregatedMetric for F1 score.
        precision: AggregatedMetric for precision.
        recall: AggregatedMetric for recall.
        threshold: Optional AggregatedMetric for the decision threshold applied across runs.
        metrics: Dictionary mapping metric name to AggregatedMetric.
    """

    auroc: AggregatedMetric | None
    accuracy: AggregatedMetric
    f1: AggregatedMetric
    precision: AggregatedMetric
    recall: AggregatedMetric
    threshold: AggregatedMetric | None = None
    metrics: dict[str, AggregatedMetric] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Sync metrics dictionary with individual metric attributes."""
        if not self.metrics:
            m: dict[str, AggregatedMetric] = {
                "accuracy": self.accuracy,
                "f1": self.f1,
                "precision": self.precision,
                "recall": self.recall,
            }
            if self.auroc is not None:
                m["auroc"] = self.auroc
            if self.threshold is not None:
                m["threshold"] = self.threshold
            self.metrics = m

    def __getitem__(self, key: str) -> AggregatedMetric | None:
        """Access metric by name."""
        if key in KNOWN_METRIC_KEYS:
            return getattr(self, key)
        if key in self.metrics:
            return self.metrics[key]
        raise KeyError(f"Metric '{key}' not found in AggregatedSlice.")

    def get(self, key: str, default: Any = None) -> Any:
        """Safe metric lookup with default value for missing keys."""
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        """Convert AggregatedSlice to serializable dictionary."""
        return {
            "auroc": self.auroc.to_dict() if self.auroc is not None else None,
            "accuracy": self.accuracy.to_dict(),
            "f1": self.f1.to_dict(),
            "precision": self.precision.to_dict(),
            "recall": self.recall.to_dict(),
            "threshold": self.threshold.to_dict() if self.threshold is not None else None,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregatedSlice:
        """Construct AggregatedSlice from dictionary."""
        auroc = AggregatedMetric.from_dict(data["auroc"]) if data.get("auroc") is not None else None
        accuracy = AggregatedMetric.from_dict(data["accuracy"])
        f1 = AggregatedMetric.from_dict(data["f1"])
        precision = AggregatedMetric.from_dict(data["precision"])
        recall = AggregatedMetric.from_dict(data["recall"])
        thresh = AggregatedMetric.from_dict(data["threshold"]) if data.get("threshold") is not None else None
        metrics = {
            k: AggregatedMetric.from_dict(v) for k, v in data.get("metrics", {}).items()
        }
        return cls(
            auroc=auroc,
            accuracy=accuracy,
            f1=f1,
            precision=precision,
            recall=recall,
            threshold=thresh,
            metrics=metrics,
        )

    @classmethod
    def from_metric_results(cls, results: Sequence[MetricResult]) -> AggregatedSlice:
        """Aggregate a sequence of MetricResult objects into an AggregatedSlice.

        Handles cases where AUROC is None across all runs (e.g. single-class slices) gracefully.
        """
        if not results:
            raise ValueError("Cannot aggregate empty list of MetricResults.")

        auroc = AggregatedMetric.from_values([r.auroc for r in results])
        accuracy = AggregatedMetric.from_values([r.accuracy for r in results])
        f1 = AggregatedMetric.from_values([r.f1 for r in results])
        precision = AggregatedMetric.from_values([r.precision for r in results])
        recall = AggregatedMetric.from_values([r.recall for r in results])
        threshold = AggregatedMetric.from_values([r.threshold for r in results])

        # Required metrics are guaranteed non-None by MetricResult invariants
        assert accuracy is not None
        assert f1 is not None
        assert precision is not None
        assert recall is not None

        return cls(
            auroc=auroc,
            accuracy=accuracy,
            f1=f1,
            precision=precision,
            recall=recall,
            threshold=threshold,
        )


@dataclass
class AggregatedEvaluationReport:
    """Multi-run aggregated evaluation report with protocol compliance status.

    Attributes:
        seeds: List of seeds or identifiers for the evaluated runs.
        num_runs: Total number of runs aggregated.
        is_preliminary: True if num_runs < 3 (preliminary result), False if num_runs >= 3 (sealed).
        overall: AggregatedSlice for overall benchmark performance.
        by_split: Mapping from split name to AggregatedSlice.
        by_generator: Mapping from generator name to AggregatedSlice.
        by_dataset: Mapping from dataset name to AggregatedSlice.
        reports: List of underlying EvaluationReport instances.
        metadata: Execution metadata (e.g., aggregation timestamp, method name).
    """

    seeds: list[int | str]
    num_runs: int
    is_preliminary: bool
    overall: AggregatedSlice
    by_split: dict[str, AggregatedSlice]
    by_generator: dict[str, AggregatedSlice]
    by_dataset: dict[str, AggregatedSlice] = field(default_factory=dict)
    reports: list[EvaluationReport] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def preliminary_result(self) -> bool:
        """Alias for is_preliminary per ForenSight evaluation protocol specification."""
        return self.is_preliminary

    def to_dict(self) -> dict[str, Any]:
        """Convert aggregated report to standard serializable dictionary."""
        return {
            "seeds": self.seeds,
            "num_runs": self.num_runs,
            "is_preliminary": self.is_preliminary,
            "preliminary_result": self.is_preliminary,
            "overall": self.overall.to_dict(),
            "by_split": {k: v.to_dict() for k, v in self.by_split.items()},
            "by_generator": {k: v.to_dict() for k, v in self.by_generator.items()},
            "by_dataset": {k: v.to_dict() for k, v in self.by_dataset.items()},
            "reports": [r.to_dict() for r in self.reports],
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize aggregated report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path, indent: int = 2) -> None:
        """Save aggregated report to JSON file on disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregatedEvaluationReport:
        """Construct AggregatedEvaluationReport from dictionary."""
        seeds = list(data.get("seeds", []))
        num_runs = int(data.get("num_runs", len(seeds)))
        is_preliminary = bool(data.get("is_preliminary", data.get("preliminary_result", num_runs < 3)))
        overall = AggregatedSlice.from_dict(data["overall"])
        by_split = {
            k: AggregatedSlice.from_dict(v) for k, v in data.get("by_split", {}).items()
        }
        by_generator = {
            k: AggregatedSlice.from_dict(v) for k, v in data.get("by_generator", {}).items()
        }
        by_dataset = {
            k: AggregatedSlice.from_dict(v) for k, v in data.get("by_dataset", {}).items()
        }
        reports = [EvaluationReport.from_dict(r) for r in data.get("reports", [])]
        metadata = dict(data.get("metadata", {}))

        return cls(
            seeds=seeds,
            num_runs=num_runs,
            is_preliminary=is_preliminary,
            overall=overall,
            by_split=by_split,
            by_generator=by_generator,
            by_dataset=by_dataset,
            reports=reports,
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, json_str: str) -> AggregatedEvaluationReport:
        """Construct AggregatedEvaluationReport from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def generate_markdown(self) -> str:
        """Generate formatted Markdown summary report per ForenSight repeated-run protocol."""
        lines: list[str] = []

        report_name = self.metadata.get("report_name", "ForenSight Multi-Seed Evaluation")
        timestamp = self.metadata.get("timestamp", datetime.now(timezone.utc).isoformat())

        lines.append(f"# ForenSight Aggregated Evaluation Report: `{report_name}`\n")
        lines.append(f"- **Timestamp:** `{timestamp}`")
        lines.append(f"- **Total Repeated Runs:** {self.num_runs}")
        lines.append(f"- **Evaluated Seeds:** `{self.seeds}`")

        if self.is_preliminary:
            lines.append(
                "- **Benchmark Status:** ⚠️ **PRELIMINARY RESULT** "
                f"(N = {self.num_runs} < 3; insufficient seeds to seal baseline per evaluation protocol)"
            )
            warnings = self.metadata.get("aggregation_warnings")
            if warnings:
                for w in warnings:
                    lines.append(f"  - ⚠️ *Warning:* {w}")
        else:
            lines.append(
                "- **Benchmark Status:** ✅ **SEALED BENCHMARK** "
                f"(N = {self.num_runs} ≥ 3; conforms to repeated-run protocol invariant)"
            )

        lines.append(
            "- **Uncertainty Reporting:** `mean ± std` (sample standard deviation ddof=1 for N ≥ 2, 0.0 for N = 1)"
        )

        def _fmt(m: AggregatedMetric | None) -> str:
            if m is None or m.mean is None:
                return "N/A"
            return m.formatted

        def _fmt_bound(val: float | None) -> str:
            if val is None:
                return "N/A"
            return f"{val:.4f}"

        # 1. Overall Benchmark Performance Table
        lines.append("\n## 1. Overall Benchmark Performance (Aggregated)\n")
        lines.append("| Metric | Mean ± Std | Min | Max | Description |")
        lines.append("|:---|:---:|:---:|:---:|:---|")

        ov = self.overall
        auc_min = _fmt_bound(ov.auroc.min) if ov.auroc else "N/A"
        auc_max = _fmt_bound(ov.auroc.max) if ov.auroc else "N/A"
        lines.append(f"| **AUROC** | **{_fmt(ov.auroc)}** | {auc_min} | {auc_max} | Primary threshold-free generalization metric |")
        lines.append(f"| **Accuracy** | {_fmt(ov.accuracy)} | {_fmt_bound(ov.accuracy.min)} | {_fmt_bound(ov.accuracy.max)} | Classification accuracy at validation $\\tau^*$ |")
        lines.append(f"| **F1 Score** | {_fmt(ov.f1)} | {_fmt_bound(ov.f1.min)} | {_fmt_bound(ov.f1.max)} | Harmonic mean of precision and recall |")
        lines.append(f"| **Precision** | {_fmt(ov.precision)} | {_fmt_bound(ov.precision.min)} | {_fmt_bound(ov.precision.max)} | Proportion of true AI images among predicted AI |")
        lines.append(f"| **Recall (TPR)** | {_fmt(ov.recall)} | {_fmt_bound(ov.recall.min)} | {_fmt_bound(ov.recall.max)} | Sensitivity / detection rate for AI-generated images |")
        if ov.threshold is not None and ov.threshold.mean is not None:
            lines.append(f"| **Threshold ($\\tau^*$)** | {_fmt(ov.threshold)} | {_fmt_bound(ov.threshold.min)} | {_fmt_bound(ov.threshold.max)} | Frozen decision threshold from validation set |")

        # 2. Performance Breakdown by Split Table
        if self.by_split:
            lines.append("\n## 2. Performance Breakdown by Partition (Split)\n")
            lines.append("| Split | AUROC (Mean ± Std) | Accuracy | F1 | Precision | Recall | Runs |")
            lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
            for s_name in sorted(self.by_split.keys()):
                sl = self.by_split[s_name]
                n_runs = sl.accuracy.n
                lines.append(
                    f"| `{s_name}` | {_fmt(sl.auroc)} | {_fmt(sl.accuracy)} | "
                    f"{_fmt(sl.f1)} | {_fmt(sl.precision)} | {_fmt(sl.recall)} | {n_runs} |"
                )

        # 3. Performance Breakdown by Generator Architecture Table
        if self.by_generator:
            lines.append("\n## 3. Performance Breakdown by Generator Architecture\n")
            lines.append("| Generator | AUROC (Mean ± Std) | Accuracy | F1 | Precision | Recall (TPR) | Runs |")
            lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
            for g_name in sorted(self.by_generator.keys()):
                sl = self.by_generator[g_name]
                n_runs = sl.accuracy.n
                lines.append(
                    f"| `{g_name}` | {_fmt(sl.auroc)} | {_fmt(sl.accuracy)} | "
                    f"{_fmt(sl.f1)} | {_fmt(sl.precision)} | {_fmt(sl.recall)} | {n_runs} |"
                )

        # 4. Performance Breakdown by Dataset Source Table
        if self.by_dataset:
            lines.append("\n## 4. Performance Breakdown by Dataset Source\n")
            lines.append("| Dataset | AUROC (Mean ± Std) | Accuracy | F1 | Precision | Recall | Runs |")
            lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
            for d_name in sorted(self.by_dataset.keys()):
                sl = self.by_dataset[d_name]
                n_runs = sl.accuracy.n
                lines.append(
                    f"| `{d_name}` | {_fmt(sl.auroc)} | {_fmt(sl.accuracy)} | "
                    f"{_fmt(sl.f1)} | {_fmt(sl.precision)} | {_fmt(sl.recall)} | {n_runs} |"
                )

        # 5. Protocol Invariants and Guidance
        lines.append("\n## 5. Evaluation Protocol Invariants\n")
        lines.append(
            "- **3-Seed Protocol Rule:** Primary trained baselines require at least 3 distinct random seeds "
            "before any claim of generalization or superiority is sealed."
        )
        lines.append(
            "- **Preliminary Result Restriction:** Any run with fewer than 3 seeds is designated `preliminary` "
            "(`preliminary_result: True`) and small metric differences cannot be cited as evidence of improvement."
        )
        lines.append(
            "- **Single-Class AUROC Handling:** Slices containing only authentic or only AI-generated images report AUROC "
            "as `N/A` without discarding operational secondary metrics."
        )

        return "\n".join(lines) + "\n"

    def save_markdown(self, path: str | Path) -> None:
        """Save Markdown summary to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown())


def aggregate_reports(
    reports: list[EvaluationReport],
    metadata: dict[str, Any] | None = None,
) -> AggregatedEvaluationReport:
    """Aggregate multiple evaluation reports across seeds or runs.

    Enforces ForenSight repeated-run protocol invariants:
    - Minimum 3 seeds required to mark a baseline as sealed (is_preliminary=False).
    - Single-run or two-run aggregations are flagged as preliminary (is_preliminary=True).
    - Uncertainty is reported as sample standard deviation (ddof=1 when n >= 2, 0.0 when n = 1).
    - Single-class slices with undefined AUROC are handled gracefully.

    Args:
        reports: List of EvaluationReport instances to aggregate.
        metadata: Optional metadata dictionary to embed in the aggregated report.

    Returns:
        AggregatedEvaluationReport: Structured report with aggregated overall, per-split,
                                    per-generator, and per-dataset statistics.

    Raises:
        ValueError: If reports is empty.
    """
    if not reports:
        raise ValueError("Cannot aggregate empty list of reports.")

    # 1. Collect seeds and run identifiers
    seeds: list[int | str] = []
    aggregation_warnings: list[str] = []
    split_versions: set[str] = set()
    dataset_revisions: list[str | None] = []
    split_sets: list[set[str]] = []
    sample_set_hashes: list[str | None] = []
    val_cohort_hashes: list[tuple[bool, str | None]] = []
    threshold_strategies: list[str | None] = []
    model_names: list[str | None] = []
    experiment_names: list[str | None] = []
    overall_sample_counts: list[int] = []

    for idx, r in enumerate(reports):
        seed_val = getattr(r, "seed", None)
        if seed_val is None and hasattr(r, "run_metadata") and isinstance(r.run_metadata, dict):
            seed_val = r.run_metadata.get("seed")
        if seed_val is None:
            seed_val = f"run_{idx}"
        seeds.append(seed_val)

        sv = None
        if hasattr(r, "run_metadata") and isinstance(r.run_metadata, dict):
            sv = r.run_metadata.get("split_version")
        if sv:
            split_versions.add(str(sv))
        else:
            split_versions.add(None)
        dataset_revisions.append(r.run_metadata.get("dataset_revision") if isinstance(r.run_metadata, dict) else None)

        split_sets.append(set(r.by_split.keys()))

        # Extract cohort / sample set hash
        cohort_h = None
        if hasattr(r, "run_metadata") and isinstance(r.run_metadata, dict):
            cohort_h = (
                r.run_metadata.get("sample_set_hash")
                or r.run_metadata.get("cohort_hash")
                or r.run_metadata.get("test_cohort_hash")
            )
        sample_set_hashes.append(cohort_h)

        # Extract val_cohort_hash and calibration status
        calibrated = False
        val_h = None
        if hasattr(r, "threshold_metadata") and isinstance(r.threshold_metadata, dict):
            calibrated = bool(r.threshold_metadata.get("calibrated", False))
            val_h = r.threshold_metadata.get("val_cohort_hash")
        if val_h is None and hasattr(r, "run_metadata") and isinstance(r.run_metadata, dict):
            val_h = r.run_metadata.get("val_cohort_hash")
        val_cohort_hashes.append((calibrated, val_h))

        # Extract threshold strategy
        strat = None
        if hasattr(r, "threshold_metadata") and isinstance(r.threshold_metadata, dict):
            strat = r.threshold_metadata.get("strategy") or r.threshold_metadata.get("threshold_strategy")
        threshold_strategies.append(strat)

        # Extract model / architecture name
        m_name = None
        if hasattr(r, "run_metadata") and isinstance(r.run_metadata, dict):
            m_name = (
                r.run_metadata.get("model_name")
                or r.run_metadata.get("model")
                or r.run_metadata.get("architecture")
            )
        model_names.append(m_name)

        # Extract experiment name
        exp_name = None
        if hasattr(r, "run_metadata") and isinstance(r.run_metadata, dict):
            exp_name = r.run_metadata.get("experiment_name")
        experiment_names.append(exp_name)

        # Extract overall evaluated sample count
        cm = r.overall.confusion_matrix if hasattr(r.overall, "confusion_matrix") else {}
        cm_total = cm.get("tp", 0) + cm.get("fp", 0) + cm.get("tn", 0) + cm.get("fn", 0)
        if cm_total > 0:
            overall_n = cm_total
        elif hasattr(r, "run_metadata") and isinstance(r.run_metadata, dict) and "evaluated_samples" in r.run_metadata:
            overall_n = int(r.run_metadata["evaluated_samples"])
        else:
            overall_n = 0
        overall_sample_counts.append(overall_n)

    num_runs = len(reports)
    # A run requires at least 3 distinct, tracked seeds to be a sealed benchmark
    tracked_unique_seeds = {s for s in seeds if not str(s).startswith("run_")}
    is_preliminary = len(tracked_unique_seeds) < 3 or num_runs < 3

    # 1. Cohort and split version consistency checks
    if None in split_versions or len(split_versions) == 0:
        aggregation_warnings.append(
            "One or more reports lack a verified split_version; split identity cannot be proven."
        )
        is_preliminary = True
    elif len(split_versions) > 1:
        aggregation_warnings.append(
            f"Mismatched split_version across runs: {sorted(list(str(s) for s in split_versions))}."
        )
        is_preliminary = True

    known_revisions = {revision for revision in dataset_revisions if revision}
    if known_revisions and (len(known_revisions) > 1 or any(not revision for revision in dataset_revisions)):
        aggregation_warnings.append("Mismatched or missing dataset_revision across runs.")
        is_preliminary = True

    # 2. Evaluated split names consistency
    if any(s != split_sets[0] for s in split_sets[1:]):
        aggregation_warnings.append(
            "Mismatched evaluated splits across runs; reports evaluate disparate partitions."
        )
        is_preliminary = True

    # 3. Test cohort identity checks (MANDATORY: all runs must have verified identical sample_set_hash)
    if any(h is None or not str(h).strip() for h in sample_set_hashes):
        aggregation_warnings.append(
            "One or more reports lack a verified sample_set_hash; cohort identity cannot be proven."
        )
        is_preliminary = True
    elif len(set(sample_set_hashes)) > 1:
        aggregation_warnings.append(
            f"Mismatched sample_set_hash / cohort across runs: {sorted(list(set(str(h) for h in sample_set_hashes)))}. Evaluated cohorts differ."
        )
        is_preliminary = True

    # 4. Validation cohort identity checks (for calibrated runs)
    calibrated_pairs = [pair for pair in val_cohort_hashes if pair[0]]
    if calibrated_pairs:
        if any(h is None or not str(h).strip() for _, h in calibrated_pairs):
            aggregation_warnings.append(
                "One or more calibrated reports lack a verified val_cohort_hash; validation cohort identity cannot be proven."
            )
            is_preliminary = True
        elif len(set(h for _, h in calibrated_pairs)) > 1:
            aggregation_warnings.append(
                f"Mismatched val_cohort_hash across runs: {sorted(list(set(str(h) for _, h in calibrated_pairs)))}. Thresholds were calibrated on different validation cohorts."
            )
            is_preliminary = True

    # 5. Threshold strategy consistency
    unique_strats = {str(s) for s in threshold_strategies if s is not None}
    if any(s is None for s in threshold_strategies) or len(unique_strats) > 1:
        aggregation_warnings.append(
            f"Mismatched threshold strategies across runs: {sorted(list(unique_strats))}. All runs must use the identical threshold calibration protocol."
        )
        is_preliminary = True

    # 6. Model/Architecture/Variant consistency (MANDATORY: cannot aggregate disparate or anonymous models)
    if any(m is None or not str(m).strip() for m in model_names):
        aggregation_warnings.append(
            "One or more reports lack a verified model/variant/architecture identity; model identity cannot be proven."
        )
        is_preliminary = True
    else:
        valid_models = {str(m) for m in model_names}
        if len(valid_models) > 1:
            aggregation_warnings.append(
                f"Mismatched model/architecture across runs: {sorted(list(valid_models))}. Disparate models cannot be aggregated into a single sealed baseline."
            )
            is_preliminary = True

    # 7. Experiment name consistency (MANDATORY: cannot aggregate disparate or anonymous experiments)
    if any(e is None or not str(e).strip() for e in experiment_names):
        aggregation_warnings.append(
            "One or more reports lack a verified experiment_name; experiment identity cannot be proven."
        )
        is_preliminary = True
    else:
        valid_exps = {str(e) for e in experiment_names}
        if len(valid_exps) > 1:
            aggregation_warnings.append(
                f"Mismatched experiment_name across runs: {sorted(list(valid_exps))}. Different experiments cannot be aggregated into a single sealed baseline."
            )
            is_preliminary = True

    # 8. Overall sample count consistency
    if len(set(overall_sample_counts)) > 1:
        aggregation_warnings.append(
            f"Mismatched overall sample counts across runs: {overall_sample_counts}. Reports must evaluate the same cohort to seal benchmark."
        )
        is_preliminary = True

    # 9. Per-split sample count consistency
    split_keys = sorted(set().union(*(r.by_split.keys() for r in reports)))
    for s_name in split_keys:
        split_counts = []
        for r in reports:
            if s_name in r.by_split:
                cm = r.by_split[s_name].confusion_matrix if hasattr(r.by_split[s_name], "confusion_matrix") else {}
                n = cm.get("tp", 0) + cm.get("fp", 0) + cm.get("tn", 0) + cm.get("fn", 0)
                split_counts.append(n)
        if len(set(split_counts)) > 1:
            aggregation_warnings.append(
                f"Mismatched sample counts for split '{s_name}' across runs: {split_counts}. Evaluated cohorts differ."
            )
            is_preliminary = True

    # 2. Overall benchmark aggregation
    overall = AggregatedSlice.from_metric_results([r.overall for r in reports])

    # 3. Per-split aggregation
    by_split: dict[str, AggregatedSlice] = {}
    for k in split_keys:
        split_results = [r.by_split[k] for r in reports if k in r.by_split]
        if split_results:
            by_split[k] = AggregatedSlice.from_metric_results(split_results)

    # 4. Per-generator aggregation
    gen_keys = sorted(set().union(*(r.by_generator.keys() for r in reports)))
    by_generator: dict[str, AggregatedSlice] = {}
    for k in gen_keys:
        gen_results = [r.by_generator[k] for r in reports if k in r.by_generator]
        if gen_results:
            by_generator[k] = AggregatedSlice.from_metric_results(gen_results)

    # 5. Per-dataset aggregation
    ds_keys = sorted(
        set().union(*(r.by_dataset.keys() for r in reports if getattr(r, "by_dataset", None)))
    )
    by_dataset: dict[str, AggregatedSlice] = {}
    for k in ds_keys:
        ds_results = [
            r.by_dataset[k]
            for r in reports
            if getattr(r, "by_dataset", None) and k in r.by_dataset
        ]
        if ds_results:
            by_dataset[k] = AggregatedSlice.from_metric_results(ds_results)

    # 6. Aggregate metadata
    meta: dict[str, Any] = {
        "report_name": "Aggregated Multi-Seed Evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_runs": num_runs,
        "seeds": seeds,
        "is_preliminary": is_preliminary,
    }
    if aggregation_warnings:
        meta["aggregation_warnings"] = aggregation_warnings
    if metadata:
        meta.update(metadata)

    return AggregatedEvaluationReport(
        seeds=seeds,
        num_runs=num_runs,
        is_preliminary=is_preliminary,
        overall=overall,
        by_split=by_split,
        by_generator=by_generator,
        by_dataset=by_dataset,
        reports=reports,
        metadata=meta,
    )
