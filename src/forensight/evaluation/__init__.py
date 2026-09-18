"""Evaluation utilities and metrics for ForenSight."""

from forensight.evaluation.metrics import (
    MetricResult,
    VALID_STRATEGIES,
    calculate_auroc,
    compute_metrics,
    evaluate_with_validation_threshold,
    select_threshold,
)

__all__ = [
    "MetricResult",
    "VALID_STRATEGIES",
    "calculate_auroc",
    "compute_metrics",
    "evaluate_with_validation_threshold",
    "select_threshold",
]
