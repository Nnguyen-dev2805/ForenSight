"""Model-agnostic evaluation runner, prediction containers, and report generation for ForenSight.

This module implements Task 0.5 of R0:
- PredictionRecord: atomic model prediction with ground-truth, continuous score, and slice metadata.
- PredictionSet: flexible collection container supporting CSV, JSONL, DataFrame, and dict records.
- EvaluationReport: structured container for overall, per-split, per-generator, and per-dataset
  metrics with complete threshold calibration provenance.
- evaluate_predictions: calibrates decision threshold strictly on validation partition and evaluates
  all evaluation/test partitions with frozen threshold.
- Markdown summary and machine-readable JSON export capabilities.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
import numpy as np
import pandas as pd

from forensight.evaluation.metrics import (
    MetricResult,
    VALID_STRATEGIES,
    ThresholdStrategy,
    compute_metrics,
    select_threshold,
)
from forensight.evaluation.reproducibility import _SENTINEL


@dataclass
class PredictionRecord:
    """Atomic prediction record for binary AI-generated image detection.

    Attributes:
        sample_id: Unique identifier for the sample (e.g. image path or hash).
        label: Binary ground-truth target (0 = real, 1 = fake/ai-generated).
        score: Continuous prediction score or probability in [0.0, 1.0] (higher indicates fake).
        split: Optional evaluation partition (e.g. 'train', 'val', 'in_domain_test', 'cross_generator_ood').
        generator: Optional generative model identifier (e.g. 'sd14', 'midjourney', 'nature').
        dataset: Optional source dataset name (e.g. 'genimage', 'wildrf').
        metadata: Arbitrary additional metadata dictionary.
    """

    sample_id: str
    label: int
    score: float
    split: str | None = None
    generator: str | None = None
    dataset: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields, normalize types, and ensure data integrity."""
        # Validate sample_id
        if self.sample_id is None:
            raise ValueError("PredictionRecord sample_id cannot be None.")
        sample_id_str = str(self.sample_id).strip()
        if not sample_id_str:
            raise ValueError("PredictionRecord sample_id must be a non-empty string.")
        self.sample_id = sample_id_str

        # Validate label
        try:
            lbl_f = float(self.label)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"PredictionRecord label must be 0 (real) or 1 (fake), got: {self.label}"
            ) from exc

        if lbl_f not in (0.0, 1.0):
            raise ValueError(
                f"PredictionRecord label must be 0 (real) or 1 (fake), got: {self.label}"
            )
        self.label = int(lbl_f)

        # Validate score
        try:
            sc = float(self.score)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"PredictionRecord score must be numerical, got: {self.score}"
            ) from exc

        if np.isnan(sc) or np.isinf(sc):
            raise ValueError(
                f"PredictionRecord score cannot be NaN or infinite, got: {sc}"
            )
        self.score = sc

        # Normalize optional strings
        def _clean_str(val: Any) -> str | None:
            if val is None or pd.isna(val):
                return None
            s = str(val).strip()
            return s if s and s.lower() != "nan" else None

        self.split = _clean_str(self.split)
        self.generator = _clean_str(self.generator)
        self.dataset = _clean_str(self.dataset)

        # Validate metadata
        if not isinstance(self.metadata, dict):
            raise ValueError(
                f"PredictionRecord metadata must be a dictionary, got {type(self.metadata)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction record to dictionary."""
        return {
            "sample_id": self.sample_id,
            "label": self.label,
            "score": self.score,
            "split": self.split,
            "generator": self.generator,
            "dataset": self.dataset,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionRecord:
        """Construct PredictionRecord from dictionary, supporting common column aliases."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data)}")

        def _find_val(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
            for k in keys:
                v = d.get(k)
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    return v
            return None

        def _clean_str(val: Any) -> str | None:
            if val is None or pd.isna(val):
                return None
            s = str(val).strip()
            return s if s and s.lower() != "nan" else None

        sample_id = _find_val(data, ("sample_id", "id", "image_path"))
        label = _find_val(data, ("label", "target", "y_true"))
        score = _find_val(data, ("score", "prob", "prediction"))

        if sample_id is None:
            raise ValueError(f"Missing required sample_id in prediction record: {data}")
        if label is None:
            raise ValueError(f"Missing required label in prediction record: {data}")
        if score is None:
            raise ValueError(f"Missing required score in prediction record: {data}")

        split = _clean_str(data.get("split"))
        generator = _clean_str(data.get("generator"))
        dataset = _clean_str(data.get("dataset"))

        # Parse metadata
        meta = data.get("metadata", {})
        if meta is None or (not isinstance(meta, dict) and pd.isna(meta)):
            meta = {}
        elif isinstance(meta, str):
            try:
                meta = json.loads(meta) if meta.strip() else {}
            except json.JSONDecodeError:
                meta = {"raw_metadata_str": meta}
        elif not isinstance(meta, dict):
            meta = {"value": meta}
        else:
            meta = dict(meta)

        # Capture leftover keys into metadata
        standard_keys = {
            "sample_id", "id", "image_path",
            "label", "target", "y_true",
            "score", "prob", "prediction",
            "split", "generator", "dataset", "metadata",
        }
        for k, v in data.items():
            if k not in standard_keys and k not in meta:
                meta[k] = v

        return cls(
            sample_id=str(sample_id),
            label=label,
            score=score,
            split=split,
            generator=generator,
            dataset=dataset,
            metadata=meta,
        )


class PredictionSet:
    """Immutable-style collection container for PredictionRecords.

    Supports ingestion from CSV, JSONL, DataFrame, or sequences of records/dicts,
    filtering by split/generator/dataset, and vectorized label/score extraction.
    """

    def __init__(self, records: Sequence[PredictionRecord | dict[str, Any]] | None = None) -> None:
        """Initialize a PredictionSet from a sequence of records or dicts."""
        self._records: list[PredictionRecord] = []
        if records:
            for r in records:
                if isinstance(r, PredictionRecord):
                    self._records.append(r)
                elif isinstance(r, dict):
                    self._records.append(PredictionRecord.from_dict(r))
                else:
                    raise TypeError(f"Expected PredictionRecord or dict, got {type(r)}")

    @property
    def records(self) -> list[PredictionRecord]:
        """Return shallow copy of the list of PredictionRecords."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[PredictionRecord]:
        return iter(self._records)

    def __getitem__(self, index: int | slice) -> PredictionRecord | PredictionSet:
        if isinstance(index, slice):
            return PredictionSet(self._records[index])
        return self._records[index]

    @property
    def sample_ids(self) -> list[str]:
        """Return list of sample IDs."""
        return [r.sample_id for r in self._records]

    @property
    def y_true(self) -> np.ndarray:
        """Return 1D numpy array of integer ground truth labels."""
        if not self._records:
            return np.array([], dtype=int)
        return np.array([r.label for r in self._records], dtype=int)

    @property
    def y_scores(self) -> np.ndarray:
        """Return 1D numpy array of float64 prediction scores."""
        if not self._records:
            return np.array([], dtype=np.float64)
        return np.array([r.score for r in self._records], dtype=np.float64)

    @property
    def splits(self) -> set[str]:
        """Return unique non-null split names present in the records."""
        return {r.split for r in self._records if r.split is not None}

    @property
    def generators(self) -> set[str]:
        """Return unique non-null generator names present in the records."""
        return {r.generator for r in self._records if r.generator is not None}

    @property
    def datasets(self) -> set[str]:
        """Return unique non-null dataset names present in the records."""
        return {r.dataset for r in self._records if r.dataset is not None}

    def has_duplicates(self) -> bool:
        """Return True if duplicate sample_ids exist within the set."""
        ids = self.sample_ids
        return len(ids) != len(set(ids))

    def filter(self, predicate: Callable[[PredictionRecord], bool]) -> PredictionSet:
        """Filter records by predicate function and return a new PredictionSet."""
        return PredictionSet([r for r in self._records if predicate(r)])

    def get_split(self, split_name: str) -> PredictionSet:
        """Return a new PredictionSet containing only records for the given split."""
        target = str(split_name).strip()
        return self.filter(lambda r: r.split == target)

    def get_generator(self, generator_name: str) -> PredictionSet:
        """Return a new PredictionSet containing only records for the given generator."""
        target = str(generator_name).strip()
        return self.filter(lambda r: r.generator == target)

    def get_dataset(self, dataset_name: str) -> PredictionSet:
        """Return a new PredictionSet containing only records for the given dataset."""
        target = str(dataset_name).strip()
        return self.filter(lambda r: r.dataset == target)

    def to_records(self) -> list[PredictionRecord]:
        """Return list of PredictionRecords."""
        return list(self._records)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return list of dictionaries."""
        return [r.to_dict() for r in self._records]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert PredictionSet to pandas DataFrame."""
        rows = []
        for r in self._records:
            row = {
                "sample_id": r.sample_id,
                "label": r.label,
                "score": r.score,
                "split": r.split,
                "generator": r.generator,
                "dataset": r.dataset,
                "metadata": json.dumps(r.metadata) if r.metadata else "{}",
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def to_csv(self, path: str | Path) -> None:
        """Save PredictionSet to CSV file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["sample_id", "label", "score", "split", "generator", "dataset", "metadata"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._records:
                writer.writerow(
                    {
                        "sample_id": r.sample_id,
                        "label": r.label,
                        "score": r.score,
                        "split": r.split or "",
                        "generator": r.generator or "",
                        "dataset": r.dataset or "",
                        "metadata": json.dumps(r.metadata) if r.metadata else "",
                    }
                )

    def to_jsonl(self, path: str | Path) -> None:
        """Save PredictionSet to JSONL file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            for r in self._records:
                f.write(json.dumps(r.to_dict()) + "\n")

    @classmethod
    def from_records(
        cls, records: Sequence[PredictionRecord | dict[str, Any]]
    ) -> PredictionSet:
        """Construct PredictionSet from sequence of PredictionRecords or dicts."""
        return cls(records)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> PredictionSet:
        """Construct PredictionSet from pandas DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
        records = [PredictionRecord.from_dict(row) for row in df.to_dict(orient="records")]
        return cls(records)

    @classmethod
    def from_csv(cls, path: str | Path) -> PredictionSet:
        """Load PredictionSet from CSV file."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Predictions CSV file does not exist: {target}")

        records: list[PredictionRecord] = []
        with open(target, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(PredictionRecord.from_dict(row))
        return cls(records)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> PredictionSet:
        """Load PredictionSet from JSONL file."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Predictions JSONL file does not exist: {target}")

        records: list[PredictionRecord] = []
        with open(target, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Failed to parse JSONL line {line_idx} in {target}: {exc}"
                    ) from exc
                records.append(PredictionRecord.from_dict(data))
        return cls(records)


@dataclass
class EvaluationReport:
    """Container for complete evaluation results, breakdown slices, and threshold provenance.

    Attributes:
        overall: MetricResult computed across all evaluated evaluation/test records.
        by_split: Mapping from split name to MetricResult.
        by_generator: Mapping from generator name to MetricResult.
        threshold_metadata: Detailed record of threshold value, optimization strategy,
                            and validation partition provenance.
        run_metadata: Execution metadata (e.g., run_name, timestamp, total samples, evaluated samples).
        by_dataset: Optional mapping from dataset name to MetricResult.
        seed: Optional random seed associated with this evaluation run.
    """

    overall: MetricResult
    by_split: dict[str, MetricResult]
    by_generator: dict[str, MetricResult]
    threshold_metadata: dict[str, Any]
    run_metadata: dict[str, Any]
    by_dataset: dict[str, MetricResult] = field(default_factory=dict)
    seed: int | str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert report to standard serializable dictionary."""
        return {
            "overall": self.overall.to_dict(),
            "by_split": {k: v.to_dict() for k, v in self.by_split.items()},
            "by_generator": {k: v.to_dict() for k, v in self.by_generator.items()},
            "by_dataset": {k: v.to_dict() for k, v in self.by_dataset.items()},
            "threshold_metadata": dict(self.threshold_metadata),
            "run_metadata": dict(self.run_metadata),
            "seed": self.seed,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path, indent: int = 2) -> None:
        """Save report to JSON file on disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))

    def to_reproducibility_record(
        self,
        split_version: str = "r0-default",
        config: dict[str, Any] | None = None,
        notes: str = "",
        git_commit: Any = _SENTINEL,
    ) -> Any:
        """Convert evaluation report to a full ReproducibilityRecord."""
        from forensight.evaluation.reproducibility import create_reproducibility_record

        return create_reproducibility_record(
            report=self,
            split_version=split_version,
            config=config,
            notes=notes,
            git_commit=git_commit,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationReport:
        """Construct EvaluationReport from dictionary."""
        overall = MetricResult.from_dict(data["overall"])
        by_split = {
            k: MetricResult.from_dict(v) for k, v in data.get("by_split", {}).items()
        }
        by_generator = {
            k: MetricResult.from_dict(v) for k, v in data.get("by_generator", {}).items()
        }
        by_dataset = {
            k: MetricResult.from_dict(v) for k, v in data.get("by_dataset", {}).items()
        }
        seed = data.get("seed")
        if seed is None and isinstance(data.get("run_metadata"), dict):
            seed = data["run_metadata"].get("seed")

        return cls(
            overall=overall,
            by_split=by_split,
            by_generator=by_generator,
            threshold_metadata=dict(data.get("threshold_metadata", {})),
            run_metadata=dict(data.get("run_metadata", {})),
            by_dataset=by_dataset,
            seed=seed,
        )

    @classmethod
    def from_json(cls, json_str: str) -> EvaluationReport:
        """Construct EvaluationReport from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def generate_markdown(self) -> str:
        """Generate formatted Markdown summary report per ForenSight evaluation protocol."""
        lines: list[str] = []

        run_name = self.run_metadata.get("run_name", "evaluation_run")
        timestamp = self.run_metadata.get("timestamp", datetime.now(timezone.utc).isoformat())
        total_samples = self.run_metadata.get("total_samples", "N/A")
        eval_samples = self.run_metadata.get("evaluated_samples", "N/A")

        lines.append(f"# ForenSight Evaluation Report: `{run_name}`\n")
        lines.append(f"- **Timestamp:** `{timestamp}`")
        if self.seed is not None:
            lines.append(f"- **Seed:** `{self.seed}`")
        lines.append(f"- **Total Samples:** {total_samples}")
        lines.append(f"- **Evaluated Samples (Overall):** {eval_samples}")

        # Threshold Provenance Section
        lines.append("\n## 1. Decision Threshold Provenance\n")
        thresh = self.threshold_metadata.get("threshold", self.overall.threshold)
        strategy = self.threshold_metadata.get("strategy", "unknown")
        source = self.threshold_metadata.get("threshold_source", self.overall.threshold_source)
        calibrated = self.threshold_metadata.get("calibrated", False)
        val_split = self.threshold_metadata.get("val_split_name")
        val_samples = self.threshold_metadata.get("val_samples_count", 0)

        invariant_status = (
            "✅ **VERIFIED** (Calibrated strictly on validation data; frozen for test distributions)"
            if calibrated
            else "⚠️ **DEFAULT FALLBACK** (Uncalibrated; default threshold applied)"
        )

        lines.append(f"- **Threshold ($\\tau^*$):** `{thresh:.4f}`")
        lines.append(f"- **Selection Strategy:** `{strategy}`")
        lines.append(f"- **Provenance Source:** `{source}`")
        if val_split:
            lines.append(f"- **Validation Split:** `{val_split}` (N = {val_samples:,})")
        lines.append(f"- **Protocol Invariant:** {invariant_status}")

        def _fmt_metric(val: float | None) -> str:
            if val is None:
                return "N/A"
            return f"{val:.4f}"

        def _fmt_cm(cm: dict[str, int]) -> str:
            tp = cm.get("tp", 0)
            fp = cm.get("fp", 0)
            tn = cm.get("tn", 0)
            fn = cm.get("fn", 0)
            total = tp + fp + tn + fn
            return f"TP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,} (N: {total:,})"

        # Overall Metrics Table
        lines.append("\n## 2. Overall Benchmark Performance\n")
        lines.append(
            "| Metric | Value | Description |"
        )
        lines.append("|:---|:---|:---|")
        lines.append(f"| **AUROC** | **{_fmt_metric(self.overall.auroc)}** | Primary threshold-free generalization metric |")
        lines.append(f"| **Accuracy** | {_fmt_metric(self.overall.accuracy)} | Classification accuracy at $\\tau^*$ |")
        lines.append(f"| **F1 Score** | {_fmt_metric(self.overall.f1)} | Harmonic mean of precision and recall for AI-generated class |")
        lines.append(f"| **Precision** | {_fmt_metric(self.overall.precision)} | Proportion of true AI images among predicted AI images |")
        lines.append(f"| **Recall (TPR)** | {_fmt_metric(self.overall.recall)} | Sensitivity / detection rate for AI-generated images |")
        lines.append(f"| **Confusion Matrix** | `{_fmt_cm(self.overall.confusion_matrix)}` | Raw classification counts at $\\tau^*$ |")

        # Breakdown by Split Table
        if self.by_split:
            lines.append("\n## 3. Performance Breakdown by Partition (Split)\n")
            lines.append(
                "| Split | AUROC | Accuracy | F1 | Precision | Recall | TP | FP | TN | FN | Total |"
            )
            lines.append(
                "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
            )
            for s_name in sorted(self.by_split.keys()):
                m = self.by_split[s_name]
                cm = m.confusion_matrix
                n_total = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
                lines.append(
                    f"| `{s_name}` | {_fmt_metric(m.auroc)} | {_fmt_metric(m.accuracy)} | "
                    f"{_fmt_metric(m.f1)} | {_fmt_metric(m.precision)} | {_fmt_metric(m.recall)} | "
                    f"{cm['tp']:,} | {cm['fp']:,} | {cm['tn']:,} | {cm['fn']:,} | {n_total:,} |"
                )

        # Breakdown by Generator Table
        if self.by_generator:
            lines.append("\n## 4. Performance Breakdown by Generator Architecture\n")
            lines.append(
                "| Generator | AUROC | Accuracy | F1 | Precision | Recall (TPR) | N Samples |"
            )
            lines.append(
                "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"
            )
            for g_name in sorted(self.by_generator.keys()):
                m = self.by_generator[g_name]
                cm = m.confusion_matrix
                n_total = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
                lines.append(
                    f"| `{g_name}` | {_fmt_metric(m.auroc)} | {_fmt_metric(m.accuracy)} | "
                    f"{_fmt_metric(m.f1)} | {_fmt_metric(m.precision)} | {_fmt_metric(m.recall)} | "
                    f"{n_total:,} |"
                )

        # Breakdown by Dataset Table
        if self.by_dataset:
            lines.append("\n## 5. Performance Breakdown by Dataset Source\n")
            lines.append(
                "| Dataset | AUROC | Accuracy | F1 | Precision | Recall | N Samples |"
            )
            lines.append(
                "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"
            )
            for d_name in sorted(self.by_dataset.keys()):
                m = self.by_dataset[d_name]
                cm = m.confusion_matrix
                n_total = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
                lines.append(
                    f"| `{d_name}` | {_fmt_metric(m.auroc)} | {_fmt_metric(m.accuracy)} | "
                    f"{_fmt_metric(m.f1)} | {_fmt_metric(m.precision)} | {_fmt_metric(m.recall)} | "
                    f"{n_total:,} |"
                )

        # Notes
        lines.append("\n## 6. Evaluation Protocol Compliance Notes\n")
        lines.append(
            "- **Invariant Confirmation:** The decision threshold $\\tau^*$ applied across all partitions "
            "was selected without accessing any test samples."
        )
        lines.append(
            "- **AUROC Undefined on Single Class:** Subsets containing only one class (e.g. single-generator fake slices) "
            "report AUROC as `N/A`, while secondary classification metrics reflect exact operational decisions."
        )

        return "\n".join(lines) + "\n"

    def save_markdown(self, path: str | Path) -> None:
        """Save Markdown summary to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown())


def evaluate_predictions(
    predictions: PredictionSet,
    val_split_name: str = "val",
    threshold_strategy: ThresholdStrategy | str = "f1",
    default_threshold: float = 0.5,
    val_predictions: PredictionSet | None = None,
    eval_splits: Sequence[str] | None = None,
    run_metadata: dict[str, Any] | None = None,
    seed: int | str | None = None,
) -> EvaluationReport:
    """Evaluate predictions, calibrating decision threshold on validation data.

    Enforces ForenSight's evaluation protocol invariants:
    - Decision threshold is selected strictly on the validation partition (or default if absent).
    - Threshold is frozen when evaluating all test/evaluation partitions.
    - Overall benchmark evaluation excludes validation and training partitions when partitions exist.
    - Complete provenance of threshold selection is tracked in the resulting report.

    Args:
        predictions: PredictionSet containing model predictions to evaluate.
        val_split_name: Name of the validation partition inside predictions (default: 'val').
        threshold_strategy: Criterion for threshold selection ('f1', 'accuracy', or 'youden').
        default_threshold: Fallback threshold when validation data is unavailable or invalid (default: 0.5).
        val_predictions: Optional separate PredictionSet dedicated to validation threshold tuning.
        eval_splits: Optional explicit list of split names to include in overall benchmark.
        run_metadata: Optional additional metadata to embed in the evaluation report.
        seed: Optional random seed associated with this evaluation run.

    Returns:
        EvaluationReport: Comprehensive report with overall, split, and generator breakdowns.

    Raises:
        ValueError: If predictions is empty, or threshold strategy is invalid.
    """
    if len(predictions) == 0:
        raise ValueError("Cannot evaluate empty PredictionSet.")

    strat = threshold_strategy.lower().strip()
    if strat not in VALID_STRATEGIES:
        raise ValueError(
            f"Invalid threshold_strategy '{threshold_strategy}'. Supported strategies: {sorted(VALID_STRATEGIES)}"
        )

    # 1. Identify validation partition
    val_set: PredictionSet | None = None
    if val_predictions is not None and len(val_predictions) > 0:
        val_set = val_predictions
    elif val_split_name in predictions.splits:
        val_set = predictions.get_split(val_split_name)

    # 2. Calibrate decision threshold on validation partition
    calibrated = False
    val_samples_count = 0
    if val_set is not None and len(val_set) > 0:
        unique_val_classes = np.unique(val_set.y_true)
        if len(unique_val_classes) >= 2:
            tau_star = select_threshold(
                val_set.y_true,
                val_set.y_scores,
                strategy=strat,
                default_threshold=default_threshold,
            )
            threshold_source = f"val_optimal_{strat}"
            calibrated = True
            val_samples_count = len(val_set)
        else:
            # Single-class validation partition cannot calibrate binary threshold
            tau_star = float(default_threshold)
            threshold_source = f"default_{default_threshold}"
            val_samples_count = len(val_set)
    else:
        # No validation partition available: fallback to default threshold
        tau_star = float(default_threshold)
        threshold_source = f"default_{default_threshold}"

    val_cohort_hash: str | None = None
    if val_set is not None and len(val_set) > 0:
        val_tuples = sorted(
            f"{r.sample_id}:{r.label}:{r.split}:{r.generator}:{r.dataset}"
            for r in val_set
        )
        val_cohort_hash = hashlib.sha256("\n".join(val_tuples).encode("utf-8")).hexdigest()[:16]

    threshold_metadata: dict[str, Any] = {
        "threshold": float(tau_star),
        "threshold_value": float(tau_star),
        "strategy": strat if calibrated else "default",
        "threshold_strategy": strat if calibrated else "default",
        "threshold_source": threshold_source,
        "calibrated": calibrated,
        "val_split_name": val_split_name if (calibrated and val_predictions is None) else (
            "separate_val_predictions" if calibrated else None
        ),
        "val_samples_count": val_samples_count,
        "val_cohort_hash": val_cohort_hash,
        "default_threshold": float(default_threshold),
        "invariant_preserved": True,
    }

    # 3. Determine overall evaluation partition
    # Protocol rule: Test/benchmark overall metrics MUST NOT evaluate on validation or train samples
    # if partitions are present.
    excluded_splits = {"train", "training"}
    if val_split_name:
        excluded_splits.add(val_split_name)
    excluded_splits.update({"val", "validation", "valid"})

    if eval_splits is not None:
        target_splits = set(eval_splits)
        disallowed = target_splits & excluded_splits
        if disallowed:
            raise ValueError(
                f"eval_splits cannot include training or validation splits: {sorted(list(disallowed))}"
            )
        overall_records = predictions.filter(lambda r: r.split in target_splits)
        if len(overall_records) == 0:
            raise ValueError(f"No samples found for specified eval_splits: {eval_splits}")
    else:
        candidate_records = predictions.filter(
            lambda r: r.split is None or r.split not in excluded_splits
        )
        if len(candidate_records) == 0:
            raise ValueError(
                f"No evaluation records found after excluding training and validation splits {sorted(list(excluded_splits))}."
            )
        overall_records = candidate_records

    # 4. Compute overall benchmark metrics
    overall_result = compute_metrics(
        overall_records.y_true,
        overall_records.y_scores,
        threshold=tau_star,
        threshold_source=threshold_source,
    )

    # 5. Compute per-split metrics
    by_split: dict[str, MetricResult] = {}
    for s in sorted(predictions.splits):
        sub_split = predictions.get_split(s)
        if len(sub_split) > 0:
            by_split[s] = compute_metrics(
                sub_split.y_true,
                sub_split.y_scores,
                threshold=tau_star,
                threshold_source=threshold_source,
            )

    # 6. Compute per-generator metrics within evaluation scope
    # Protocol rule: Generator slicing must strictly operate within the evaluation scope (overall_records),
    # preventing validation and training samples from contaminating per-generator benchmark metrics.
    eval_reals = [r for r in overall_records if r.label == 0]

    by_generator: dict[str, MetricResult] = {}
    for g in sorted(overall_records.generators):
        sub_gen = overall_records.get_generator(g)
        if len(sub_gen) == 0:
            continue

        unique_labels = set(sub_gen.y_true)
        if len(unique_labels) == 1 and 1 in unique_labels and eval_reals:
            # Synthetic generator with only fake images: combine with reference reals from evaluation scope
            paired_records = list(sub_gen) + eval_reals
            gen_y_true = np.array([r.label for r in paired_records], dtype=int)
            gen_y_scores = np.array([r.score for r in paired_records], dtype=np.float64)
            by_generator[g] = compute_metrics(
                gen_y_true,
                gen_y_scores,
                threshold=tau_star,
                threshold_source=threshold_source,
            )
        else:
            by_generator[g] = compute_metrics(
                sub_gen.y_true,
                sub_gen.y_scores,
                threshold=tau_star,
                threshold_source=threshold_source,
            )

    # 7. Compute per-dataset metrics within evaluation scope
    by_dataset: dict[str, MetricResult] = {}
    for d in sorted(overall_records.datasets):
        sub_ds = overall_records.get_dataset(d)
        if len(sub_ds) > 0:
            by_dataset[d] = compute_metrics(
                sub_ds.y_true,
                sub_ds.y_scores,
                threshold=tau_star,
                threshold_source=threshold_source,
            )

    # 8. Build run metadata
    seed_val = seed
    if seed_val is None and run_metadata:
        seed_val = run_metadata.get("seed")

    eval_tuples = sorted(
        f"{r.sample_id}:{r.label}:{r.split}:{r.generator}:{r.dataset}"
        for r in overall_records
    )
    sample_set_hash = hashlib.sha256("\n".join(eval_tuples).encode("utf-8")).hexdigest()[:16]

    run_meta: dict[str, Any] = {
        "run_name": (run_metadata.get("run_name") if run_metadata else None) or "evaluation_run",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(predictions),
        "evaluated_samples": len(overall_records),
        "sample_set_hash": sample_set_hash,
        "cohort_hash": sample_set_hash,
        "test_cohort_hash": sample_set_hash,
        "val_cohort_hash": val_cohort_hash,
        "splits": sorted(list(predictions.splits)),
        "generators": sorted(list(overall_records.generators)),
        "datasets": sorted(list(overall_records.datasets)),
    }
    if seed_val is not None:
        run_meta["seed"] = seed_val
    if run_metadata:
        for k, v in run_metadata.items():
            if k not in run_meta:
                run_meta[k] = v

    return EvaluationReport(
        overall=overall_result,
        by_split=by_split,
        by_generator=by_generator,
        threshold_metadata=threshold_metadata,
        run_metadata=run_meta,
        by_dataset=by_dataset,
        seed=seed_val,
    )
