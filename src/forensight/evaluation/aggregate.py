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
        if self.mean is None or self.std is None:
            return "N/A"
        return f"{self.mean:.4f} ± {self.std:.4f}"

    def format(self, digits: int = 4) -> str:
        """Return formatted string with custom decimal precision."""
        if self.mean is None or self.std is None:
            return "N/A"
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
    for idx, r in enumerate(reports):
        seed_val = getattr(r, "seed", None)
        if seed_val is None and hasattr(r, "run_metadata") and isinstance(r.run_metadata, dict):
            seed_val = r.run_metadata.get("seed")
        if seed_val is None:
            seed_val = f"run_{idx}"
        seeds.append(seed_val)

    num_runs = len(reports)
    # A run requires at least 3 distinct, tracked seeds to be a sealed benchmark
    tracked_unique_seeds = {s for s in seeds if not str(s).startswith("run_")}
    is_preliminary = len(tracked_unique_seeds) < 3 or num_runs < 3

    # 2. Overall benchmark aggregation
    overall = AggregatedSlice.from_metric_results([r.overall for r in reports])

    # 3. Per-split aggregation
    split_keys = sorted(set().union(*(r.by_split.keys() for r in reports)))
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
