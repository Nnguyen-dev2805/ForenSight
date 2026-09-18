"""Evaluation utilities and metrics for ForenSight."""

from forensight.evaluation.aggregate import (
    AggregatedEvaluationReport,
    AggregatedMetric,
    AggregatedSlice,
    aggregate_reports,
)
from forensight.evaluation.metrics import (
    MetricResult,
    VALID_STRATEGIES,
    calculate_auroc,
    compute_metrics,
    evaluate_with_validation_threshold,
    select_threshold,
)
from forensight.evaluation.runner import (
    EvaluationReport,
    PredictionRecord,
    PredictionSet,
    evaluate_predictions,
)

__all__ = [
    "AggregatedEvaluationReport",
    "AggregatedMetric",
    "AggregatedSlice",
    "EvaluationReport",
    "MetricResult",
    "PredictionRecord",
    "PredictionSet",
    "VALID_STRATEGIES",
    "aggregate_reports",
    "calculate_auroc",
    "compute_metrics",
    "evaluate_predictions",
    "evaluate_with_validation_threshold",
    "select_threshold",
]

