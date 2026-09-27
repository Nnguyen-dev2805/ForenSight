"""Metric calculation, threshold policy, and evaluation utilities for ForenSight.

This module implements Task 0.4 of R0:
- MetricResult: container for evaluation results with threshold provenance and serialization.
- calculate_auroc: threshold-free primary metric calculation.
- select_threshold: optimal decision threshold selection on validation data ("f1", "accuracy", "youden").
- compute_metrics: secondary metric calculation (Accuracy, F1, Precision, Recall, Confusion Matrix).
- evaluate_with_validation_threshold: validation-to-test threshold transference guaranteeing test-invariance.
- Strict input validation against NaN, Inf, length mismatch, empty arrays, and invalid class labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Literal
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


ThresholdStrategy = Literal["f1", "accuracy", "youden"]
VALID_STRATEGIES: set[str] = {"f1", "accuracy", "youden"}


def _validate_and_convert_inputs(
    y_true: Iterable[Any] | np.ndarray,
    y_scores: Iterable[Any] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert ground-truth targets and prediction scores.

    Args:
        y_true: Binary ground-truth labels (0 for real, 1 for fake/ai-generated).
        y_scores: Model prediction scores or probabilities.

    Returns:
        tuple[np.ndarray, np.ndarray]: Validated 1D numpy arrays (y_true as int, y_scores as float64).

    Raises:
        ValueError: If inputs are empty, dimensions are incompatible, lengths mismatch,
                    scores contain NaN or infinite values, or targets contain invalid labels.
    """
    # Convert inputs to numpy arrays
    try:
        y_t = np.asarray(y_true, dtype=np.float64)
        y_s = np.asarray(y_scores, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Failed to convert inputs to numerical arrays: {exc}") from exc

    # Handle shape / dimensionality
    if y_t.ndim > 1:
        y_t = np.squeeze(y_t)
    if y_s.ndim > 1:
        y_s = np.squeeze(y_s)

    if y_t.ndim != 1 or y_s.ndim != 1:
        raise ValueError(
            f"Inputs must be 1-dimensional after squeezing. Got y_true ndim={y_t.ndim}, y_scores ndim={y_s.ndim}."
        )

    # Check for empty inputs
    if len(y_t) == 0 or len(y_s) == 0:
        raise ValueError("Inputs cannot be empty.")

    # Check length alignment
    if len(y_t) != len(y_s):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_t)} samples, but y_scores has {len(y_s)} samples."
        )

    # Check for NaN / Inf in scores
    if np.isnan(y_s).any():
        raise ValueError("y_scores contains NaN values.")
    if np.isinf(y_s).any():
        raise ValueError("y_scores contains infinite values.")

    # Check for NaN in labels
    if np.isnan(y_t).any():
        raise ValueError("y_true contains NaN values.")

    # Check binary label validity {0, 1}
    unique_labels = np.unique(y_t)
    for lbl in unique_labels:
        if lbl not in (0.0, 1.0):
            raise ValueError(
                f"y_true must contain only binary labels (0 for real, 1 for fake). Found invalid label: {lbl}"
            )

    y_t = y_t.astype(int)
    return y_t, y_s


@dataclass(frozen=True)
class MetricResult:
    """Immutable container for evaluation metrics, threshold metadata, and confusion counts.

    Attributes:
        auroc: Area Under ROC Curve (primary metric), or None if only a single class is present.
        accuracy: Proportion of correctly classified samples.
        f1: Harmonic mean of precision and recall for the positive class (fake).
        precision: True positive rate among predicted positives.
        recall: True positive rate among actual positives (sensitivity).
        threshold: Decision threshold applied to scores (samples with score >= threshold are positive).
        threshold_source: Provenance of the threshold (e.g. 'default_0.5', 'val_optimal_f1').
        confusion_matrix: Dictionary containing counts for 'tp', 'fp', 'tn', and 'fn'.
    """

    auroc: float | None
    accuracy: float
    f1: float
    precision: float
    recall: float
    threshold: float
    threshold_source: str
    confusion_matrix: dict[str, int]
    tpr_at_1pct_fpr: float | None = None
    tpr_at_01pct_fpr: float | None = None
    eer: float | None = None
    pr_auc: float | None = None

    def __post_init__(self) -> None:
        """Validate result fields and types."""
        def _is_numeric(val: Any) -> bool:
            return isinstance(val, (int, float)) and not isinstance(val, bool)

        if self.auroc is not None and not _is_numeric(self.auroc):
            raise TypeError(f"auroc must be float or None, got {type(self.auroc)}")
        if not _is_numeric(self.accuracy):
            raise TypeError(f"accuracy must be float, got {type(self.accuracy)}")
        if not _is_numeric(self.f1):
            raise TypeError(f"f1 must be float, got {type(self.f1)}")
        if not _is_numeric(self.precision):
            raise TypeError(f"precision must be float, got {type(self.precision)}")
        if not _is_numeric(self.recall):
            raise TypeError(f"recall must be float, got {type(self.recall)}")
        if not _is_numeric(self.threshold):
            raise TypeError(f"threshold must be float, got {type(self.threshold)}")
        if not isinstance(self.threshold_source, str) or not self.threshold_source.strip():
            raise ValueError("threshold_source must be a non-empty string.")

        for attr in ("tpr_at_1pct_fpr", "tpr_at_01pct_fpr", "eer", "pr_auc"):
            val = getattr(self, attr)
            if val is not None and not _is_numeric(val):
                raise TypeError(f"{attr} must be float or None, got {type(val)}")

        required_cm_keys = {"tp", "fp", "tn", "fn"}
        if not isinstance(self.confusion_matrix, dict) or set(self.confusion_matrix.keys()) != required_cm_keys:
            raise ValueError(f"confusion_matrix must be a dict with keys {required_cm_keys}")

        for k, v in self.confusion_matrix.items():
            if not (isinstance(v, (int, np.integer)) and not isinstance(v, bool)):
                raise TypeError(f"confusion_matrix['{k}'] must be an integer, got {type(v)}")

    def to_dict(self) -> dict[str, Any]:
        """Convert metric result to a standard serializable dictionary."""
        return {
            "auroc": float(self.auroc) if self.auroc is not None else None,
            "accuracy": float(self.accuracy),
            "f1": float(self.f1),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "threshold": float(self.threshold),
            "threshold_source": str(self.threshold_source),
            "confusion_matrix": {
                "tp": int(self.confusion_matrix["tp"]),
                "fp": int(self.confusion_matrix["fp"]),
                "tn": int(self.confusion_matrix["tn"]),
                "fn": int(self.confusion_matrix["fn"]),
            },
            "tpr_at_1pct_fpr": float(self.tpr_at_1pct_fpr) if self.tpr_at_1pct_fpr is not None else None,
            "tpr_at_01pct_fpr": float(self.tpr_at_01pct_fpr) if self.tpr_at_01pct_fpr is not None else None,
            "eer": float(self.eer) if self.eer is not None else None,
            "pr_auc": float(self.pr_auc) if self.pr_auc is not None else None,
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize metric result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path, indent: int = 2) -> None:
        """Save metric result to a JSON file on disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricResult:
        """Construct MetricResult from dictionary."""
        return cls(
            auroc=float(data["auroc"]) if data.get("auroc") is not None else None,
            accuracy=float(data["accuracy"]),
            f1=float(data["f1"]),
            precision=float(data["precision"]),
            recall=float(data["recall"]),
            threshold=float(data["threshold"]),
            threshold_source=str(data["threshold_source"]),
            confusion_matrix={
                "tp": int(data["confusion_matrix"]["tp"]),
                "fp": int(data["confusion_matrix"]["fp"]),
                "tn": int(data["confusion_matrix"]["tn"]),
                "fn": int(data["confusion_matrix"]["fn"]),
            },
            tpr_at_1pct_fpr=float(data["tpr_at_1pct_fpr"]) if data.get("tpr_at_1pct_fpr") is not None else None,
            tpr_at_01pct_fpr=float(data["tpr_at_01pct_fpr"]) if data.get("tpr_at_01pct_fpr") is not None else None,
            eer=float(data["eer"]) if data.get("eer") is not None else None,
            pr_auc=float(data["pr_auc"]) if data.get("pr_auc") is not None else None,
        )

    @classmethod
    def from_json(cls, json_str: str) -> MetricResult:
        """Construct MetricResult from JSON string."""
        return cls.from_dict(json.loads(json_str))


def calculate_auroc(
    y_true: Iterable[Any] | np.ndarray,
    y_scores: Iterable[Any] | np.ndarray,
) -> float | None:
    """Calculate the Area Under the Receiver Operating Characteristic Curve (AUROC).

    AUROC is ForenSight's primary evaluation metric. It evaluates discriminative ability
    across all classification thresholds without fixing a single decision boundary.

    If only a single class is present in y_true, AUROC is mathematically undefined and
    returns None.

    Args:
        y_true: Ground truth binary labels (0 = real, 1 = fake).
        y_scores: Continuous decision scores or probabilities. Higher score indicates fake.

    Returns:
        float | None: AUROC score in [0.0, 1.0], or None if only one class exists in y_true.

    Raises:
        ValueError: On empty arrays, length mismatch, NaNs, or invalid labels.
    """
    y_t, y_s = _validate_and_convert_inputs(y_true, y_scores)
    unique_classes = np.unique(y_t)
    if len(unique_classes) < 2:
        return None

    return float(roc_auc_score(y_t, y_s))


def calculate_tpr_at_fpr(
    y_true: Iterable[Any] | np.ndarray,
    y_scores: Iterable[Any] | np.ndarray,
    target_fpr: float = 0.01,
    *,
    min_negatives: int | None = None,
) -> float | None:
    """Calculate the True Positive Rate (TPR) at a fixed False Positive Rate (FPR).

    Used in forensic detection and biometric security evaluations (e.g. TPR@1%FPR, TPR@0.1%FPR).
    Returns None if only a single class is present in y_true, or if the number of negative samples
    is insufficient to resolve the requested target_fpr (< ceil(1.0 / target_fpr)).
    """
    y_t, y_s = _validate_and_convert_inputs(y_true, y_scores)
    if len(np.unique(y_t)) < 2:
        return None
    if not (0.0 <= target_fpr <= 1.0):
        raise ValueError(f"target_fpr must be between 0.0 and 1.0, got {target_fpr}")

    n_neg = int(np.sum(y_t == 0))
    required_neg = (
        min_negatives
        if min_negatives is not None
        else (int(np.ceil(1.0 / target_fpr)) if target_fpr > 0 else 1)
    )
    if n_neg < required_neg:
        return None

    fpr, tpr, _ = roc_curve(y_t, y_s)
    return float(np.interp(target_fpr, fpr, tpr))


def calculate_eer(
    y_true: Iterable[Any] | np.ndarray,
    y_scores: Iterable[Any] | np.ndarray,
) -> float | None:
    """Calculate Equal Error Rate (EER) where FPR == FNR (1 - TPR).

    Lower is better. Returns None if only a single class is present in y_true.
    Computes exact zero-crossing of (fpr - fnr) via linear interpolation.
    """
    y_t, y_s = _validate_and_convert_inputs(y_true, y_scores)
    if len(np.unique(y_t)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_t, y_s)
    fnr = 1.0 - tpr
    diff = fpr - fnr

    # Exact zero match
    zero_indices = np.where(diff == 0.0)[0]
    if len(zero_indices) > 0:
        return float(fpr[zero_indices[0]])

    # Crossing point where diff transitions from <= 0 to > 0
    pos_indices = np.where(diff > 0.0)[0]
    if len(pos_indices) == 0:
        return 0.0
    idx = int(pos_indices[0])
    if idx == 0:
        return float(fpr[0])

    d0 = float(diff[idx - 1])
    d1 = float(diff[idx])
    if d1 - d0 == 0:
        return float(fpr[idx])
    t = -d0 / (d1 - d0)
    eer = float(fpr[idx - 1] + t * (fpr[idx] - fpr[idx - 1]))
    return float(np.clip(eer, 0.0, 1.0))


def calculate_pr_auc(
    y_true: Iterable[Any] | np.ndarray,
    y_scores: Iterable[Any] | np.ndarray,
) -> float | None:
    """Calculate Area Under the Precision-Recall Curve (PR-AUC / Average Precision).

    Evaluates detection performance under class imbalance.
    Returns None if only a single class is present in y_true.
    """
    y_t, y_s = _validate_and_convert_inputs(y_true, y_scores)
    if len(np.unique(y_t)) < 2:
        return None

    return float(average_precision_score(y_t, y_s))



def select_threshold(
    y_true: Iterable[Any] | np.ndarray,
    y_scores: Iterable[Any] | np.ndarray,
    strategy: ThresholdStrategy | str = "f1",
    default_threshold: float = 0.5,
) -> float:
    """Select the optimal decision threshold on a validation dataset.

    In accordance with ForenSight's evaluation protocol invariants, decision thresholds
    MUST be selected on validation data and NEVER optimized on test distributions.

    Candidate thresholds are generated from unique score boundaries and midpoints.
    Ties in metric performance are broken deterministically by selecting the threshold
    closest to the default boundary (0.5), maximizing generalization margin.

    Supported strategies:
        - "f1": Maximizes F1 score for the positive class (fake).
        - "accuracy": Maximizes overall sample classification accuracy.
        - "youden": Maximizes Youden's J statistic (Sensitivity + Specificity - 1).

    Args:
        y_true: Validation ground-truth binary labels (0 = real, 1 = fake).
        y_scores: Validation continuous prediction scores.
        strategy: Optimization criterion ('f1', 'accuracy', or 'youden').
        default_threshold: Baseline reference point for tie-breaking (default 0.5).

    Returns:
        float: The chosen optimal threshold value.

    Raises:
        ValueError: If strategy is unsupported, if y_true contains only a single class,
                    or if inputs fail validation.
    """
    strat = strategy.lower().strip()
    if strat not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown threshold selection strategy '{strategy}'. Supported strategies: {sorted(VALID_STRATEGIES)}"
        )

    y_t, y_s = _validate_and_convert_inputs(y_true, y_scores)
    if len(np.unique(y_t)) < 2:
        raise ValueError(
            "Threshold selection requires both positive (1) and negative (0) classes in y_true."
        )

    # Sort scores to evaluate all operational decision boundaries efficiently
    order = np.argsort(y_s)
    y_s_sorted = y_s[order]
    y_t_sorted = y_t[order]
    n_samples = len(y_s_sorted)

    # Build candidate thresholds: unique observed scores + midpoints + upper bound beyond max
    unique_scores = np.unique(y_s_sorted)
    upper_bound = np.nextafter(unique_scores.max(), np.inf)
    if len(unique_scores) > 1:
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        candidate_list = [unique_scores, midpoints, [upper_bound]]
        # Include default_threshold if it falls within data range
        if unique_scores.min() <= default_threshold <= upper_bound:
            candidate_list.append([default_threshold])
        candidates = np.unique(np.concatenate(candidate_list))
    else:
        candidates = np.unique(np.concatenate([unique_scores, [upper_bound]]))

    # Vectorized confusion matrix computation for all candidate thresholds
    # idx is the first index where y_s_sorted >= candidate
    idx = np.searchsorted(y_s_sorted, candidates, side="left")
    cum_pos = np.cumsum(y_t_sorted)
    total_pos = cum_pos[-1]
    total_neg = n_samples - total_pos

    # Positive samples before index idx
    pos_before = np.where(idx > 0, cum_pos[idx - 1], 0)
    tp = total_pos - pos_before
    fp = (n_samples - idx) - tp
    fn = total_pos - tp
    tn = total_neg - fp

    # Evaluate metric array based on strategy
    if strat == "f1":
        denom_f1 = total_pos + tp + fp
        metric_vals = (2.0 * tp) / denom_f1
    elif strat == "accuracy":
        metric_vals = (tp + tn) / n_samples
    elif strat == "youden":
        # Youden's J = Sensitivity (TPR) - FPR
        tpr = tp / total_pos
        fpr = fp / total_neg
        metric_vals = tpr - fpr
    else:
        raise ValueError(f"Unhandled strategy: {strat}")

    # Identify candidate(s) maximizing the metric
    max_val = np.max(metric_vals)
    best_mask = np.isclose(metric_vals, max_val, rtol=0.0, atol=1e-12)
    tied_candidates = candidates[best_mask]

    # Deterministic tie-breaking:
    # 1. Closest distance to default_threshold (generalization margin preference)
    # 2. Higher threshold value (conservative defense against false alarms)
    best_threshold = min(
        tied_candidates,
        key=lambda c: (abs(c - default_threshold), -c),
    )

    return float(best_threshold)


def compute_metrics(
    y_true: Iterable[Any] | np.ndarray,
    y_scores: Iterable[Any] | np.ndarray,
    threshold: float = 0.5,
    threshold_source: str = "default_0.5",
) -> MetricResult:
    """Compute binary classification metrics given ground-truth labels and prediction scores.

    Computes:
        - AUROC (threshold-free primary metric)
        - Accuracy, F1, Precision, Recall at the specified threshold
        - Confusion matrix counts (tp, fp, tn, fn)

    Args:
        y_true: Binary ground-truth labels (0 = real, 1 = fake).
        y_scores: Continuous prediction scores or probabilities.
        threshold: Decision threshold for secondary metrics (y_score >= threshold is positive).
        threshold_source: Explicit provenance record of the threshold value.

    Returns:
        MetricResult: Dataclass containing all metrics, confusion matrix, and threshold metadata.

    Raises:
        ValueError: On empty arrays, length mismatch, NaNs, or invalid labels.
    """
    y_t, y_s = _validate_and_convert_inputs(y_true, y_scores)
    thresh = float(threshold)

    # Secondary metric classifications
    y_pred = (y_s >= thresh).astype(int)

    tp = int(np.sum((y_t == 1) & (y_pred == 1)))
    fp = int(np.sum((y_t == 0) & (y_pred == 1)))
    tn = int(np.sum((y_t == 0) & (y_pred == 0)))
    fn = int(np.sum((y_t == 1) & (y_pred == 0)))

    n_samples = len(y_t)
    accuracy = float((tp + tn) / n_samples) if n_samples > 0 else 0.0

    denom_prec = tp + fp
    precision = float(tp / denom_prec) if denom_prec > 0 else 0.0

    denom_rec = tp + fn
    recall = float(tp / denom_rec) if denom_rec > 0 else 0.0

    denom_f1 = precision + recall
    f1 = float((2.0 * precision * recall) / denom_f1) if denom_f1 > 0 else 0.0

    # Primary & threshold-independent metrics
    auroc = calculate_auroc(y_t, y_s)
    tpr_1pct = calculate_tpr_at_fpr(y_t, y_s, target_fpr=0.01)
    tpr_01pct = calculate_tpr_at_fpr(y_t, y_s, target_fpr=0.001)
    eer = calculate_eer(y_t, y_s)
    pr_auc = calculate_pr_auc(y_t, y_s)

    return MetricResult(
        auroc=auroc,
        accuracy=accuracy,
        f1=f1,
        precision=precision,
        recall=recall,
        threshold=thresh,
        threshold_source=str(threshold_source),
        confusion_matrix={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        tpr_at_1pct_fpr=tpr_1pct,
        tpr_at_01pct_fpr=tpr_01pct,
        eer=eer,
        pr_auc=pr_auc,
    )


def evaluate_with_validation_threshold(
    val_true: Iterable[Any] | np.ndarray,
    val_scores: Iterable[Any] | np.ndarray,
    test_true: Iterable[Any] | np.ndarray,
    test_scores: Iterable[Any] | np.ndarray,
    strategy: ThresholdStrategy | str = "f1",
) -> tuple[MetricResult, float]:
    """Calibrate decision threshold on validation set and evaluate on test set.

    Guarantees ForenSight's core evaluation invariant: decision thresholds are chosen
    strictly on validation data and applied unchanged to the test distribution.
    The test data is never inspected or utilized during threshold selection.

    Args:
        val_true: Validation ground-truth binary labels.
        val_scores: Validation continuous prediction scores.
        test_true: Test ground-truth binary labels.
        test_scores: Test continuous prediction scores.
        strategy: Threshold optimization strategy on validation ('f1', 'accuracy', 'youden').

    Returns:
        tuple[MetricResult, float]:
            - MetricResult computed on the test distribution using the fixed validation threshold.
            - The selected threshold value (float).

    Raises:
        ValueError: If validation or test inputs fail validation checks, or strategy is invalid.
    """
    strat = strategy.lower().strip()
    if strat not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown threshold selection strategy '{strategy}'. Supported strategies: {sorted(VALID_STRATEGIES)}"
        )

    # 1. Select optimal threshold on validation set
    selected_thresh = select_threshold(val_true, val_scores, strategy=strat)

    # 2. Apply fixed validation threshold to test set
    threshold_source = f"val_optimal_{strat}"
    test_result = compute_metrics(
        test_true,
        test_scores,
        threshold=selected_thresh,
        threshold_source=threshold_source,
    )

    return test_result, float(selected_thresh)
