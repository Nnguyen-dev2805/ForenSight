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

__all__ = [
    "AggregatedEvaluationReport",
    "AggregatedMetric",
    "AggregatedSlice",
    "EvaluationReport",
    "MetricResult",
    "PredictionRecord",
    "PredictionSet",
    "ReproducibilityRecord",
    "VALID_STRATEGIES",
    "aggregate_reports",
    "calculate_auroc",
    "compute_metrics",
    "create_reproducibility_record",
    "evaluate_predictions",
    "evaluate_with_validation_threshold",
    "get_environment_info",
    "get_git_commit",
    "select_threshold",
]

