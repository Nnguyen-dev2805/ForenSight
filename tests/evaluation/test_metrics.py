"""Tests for ForenSight evaluation metrics, threshold policy, and test-invariance guarantees."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from forensight.evaluation.metrics import (
    MetricResult,
    VALID_STRATEGIES,
    _validate_and_convert_inputs,
    calculate_auroc,
    compute_metrics,
    evaluate_with_validation_threshold,
    select_threshold,
)


class TestInputValidation:
    """Unit tests for input validation and type conversions."""

    def test_valid_numpy_arrays(self):
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.1, 0.8, 0.2, 0.9])
        yt, ys = _validate_and_convert_inputs(y_true, y_scores)
        assert yt.dtype == int
        assert ys.dtype == np.float64
        assert len(yt) == 4
        assert len(ys) == 4

    def test_valid_python_lists_and_tuples(self):
        y_true = [0, 1, 0, 1]
        y_scores = (0.2, 0.7, 0.3, 0.85)
        yt, ys = _validate_and_convert_inputs(y_true, y_scores)
        assert isinstance(yt, np.ndarray)
        assert isinstance(ys, np.ndarray)
        assert len(yt) == 4

    def test_valid_pandas_series(self):
        y_true = pd.Series([0, 1, 1, 0])
        y_scores = pd.Series([0.15, 0.95, 0.65, 0.35])
        yt, ys = _validate_and_convert_inputs(y_true, y_scores)
        assert len(yt) == 4
        assert np.array_equal(yt, np.array([0, 1, 1, 0]))

    def test_valid_2d_column_vector_squeezed(self):
        y_true = np.array([[0], [1], [0], [1]])
        y_scores = np.array([[0.1], [0.8], [0.2], [0.9]])
        yt, ys = _validate_and_convert_inputs(y_true, y_scores)
        assert yt.ndim == 1
        assert ys.ndim == 1
        assert len(yt) == 4

    def test_higher_dimensional_array_raises(self):
        y_true = np.zeros((4, 2))
        y_scores = np.zeros((4, 2))
        with pytest.raises(ValueError, match="Inputs must be 1-dimensional"):
            _validate_and_convert_inputs(y_true, y_scores)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="Inputs cannot be empty"):
            _validate_and_convert_inputs([], [])
        with pytest.raises(ValueError, match="Inputs cannot be empty"):
            _validate_and_convert_inputs([0, 1], [])
        with pytest.raises(ValueError, match="Inputs cannot be empty"):
            _validate_and_convert_inputs([], [0.1, 0.9])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            _validate_and_convert_inputs([0, 1, 0], [0.1, 0.9])

    def test_nan_in_scores_raises(self):
        with pytest.raises(ValueError, match="y_scores contains NaN values"):
            _validate_and_convert_inputs([0, 1], [0.5, np.nan])

    def test_inf_in_scores_raises(self):
        with pytest.raises(ValueError, match="y_scores contains infinite values"):
            _validate_and_convert_inputs([0, 1], [0.5, np.inf])
        with pytest.raises(ValueError, match="y_scores contains infinite values"):
            _validate_and_convert_inputs([0, 1], [-np.inf, 0.5])

    def test_nan_in_labels_raises(self):
        with pytest.raises(ValueError, match="y_true contains NaN values"):
            _validate_and_convert_inputs([0, np.nan], [0.2, 0.8])

    def test_invalid_labels_raises(self):
        with pytest.raises(ValueError, match="must contain only binary labels"):
            _validate_and_convert_inputs([0, 2], [0.1, 0.9])
        with pytest.raises(ValueError, match="must contain only binary labels"):
            _validate_and_convert_inputs([-1, 1], [0.1, 0.9])
        with pytest.raises(ValueError, match="must contain only binary labels"):
            _validate_and_convert_inputs([0.5, 0.5], [0.1, 0.9])

    def test_boolean_labels_accepted(self):
        y_true = [False, True, False, True]
        y_scores = [0.1, 0.9, 0.2, 0.8]
        yt, _ = _validate_and_convert_inputs(y_true, y_scores)
        assert np.array_equal(yt, np.array([0, 1, 0, 1]))

    def test_string_inputs_raise_value_error(self):
        with pytest.raises(ValueError, match="Failed to convert inputs to numerical arrays"):
            _validate_and_convert_inputs(["real", "fake"], [0.1, 0.9])
        with pytest.raises(ValueError, match="Failed to convert inputs to numerical arrays"):
            _validate_and_convert_inputs([0, 1], ["low", "high"])
        with pytest.raises(ValueError, match="Failed to convert inputs to numerical arrays"):
            compute_metrics(["real", "fake"], [0.1, 0.9])


class TestAUROCCalculation:
    """Unit tests for AUROC calculation and edge cases."""

    def test_perfect_classifier(self):
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.8, 0.9]
        auroc = calculate_auroc(y_true, y_scores)
        assert auroc == 1.0

    def test_inverted_classifier(self):
        y_true = [0, 0, 1, 1]
        y_scores = [0.9, 0.8, 0.2, 0.1]
        auroc = calculate_auroc(y_true, y_scores)
        assert auroc == 0.0

    def test_constant_prediction_scores(self):
        y_true = [0, 0, 1, 1]
        y_scores = [0.5, 0.5, 0.5, 0.5]
        auroc = calculate_auroc(y_true, y_scores)
        assert auroc == 0.5

    def test_random_scores_agreement_with_sklearn(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100)
        y_scores = rng.uniform(0.0, 1.0, size=100)
        expected = float(roc_auc_score(y_true, y_scores))
        actual = calculate_auroc(y_true, y_scores)
        assert actual == pytest.approx(expected, abs=1e-6)

    def test_single_class_all_zeros_returns_none(self):
        y_true = [0, 0, 0, 0]
        y_scores = [0.1, 0.4, 0.3, 0.2]
        auroc = calculate_auroc(y_true, y_scores)
        assert auroc is None

    def test_single_class_all_ones_returns_none(self):
        y_true = [1, 1, 1, 1]
        y_scores = [0.8, 0.7, 0.9, 0.6]
        auroc = calculate_auroc(y_true, y_scores)
        assert auroc is None


class TestThresholdSelection:
    """Unit tests for validation threshold selection strategies and tie-breaking."""

    def test_optimal_f1_selection(self):
        # 2 reals, 2 fakes
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.3, 0.7, 0.9]
        # Any threshold in (0.3, 0.7] gives F1 = 1.0
        # Candidate set includes 0.5, which is closest to 0.5
        thresh = select_threshold(y_true, y_scores, strategy="f1")
        assert thresh == pytest.approx(0.5, abs=1e-5)

        # Apply to check F1 is indeed 1.0
        res = compute_metrics(y_true, y_scores, threshold=thresh)
        assert res.f1 == 1.0
        assert res.accuracy == 1.0

    def test_optimal_accuracy_selection(self):
        y_true = [0, 0, 0, 1]
        y_scores = [0.1, 0.2, 0.4, 0.8]
        # At threshold 0.6, predicts [0, 0, 0, 1], accuracy = 1.0
        thresh = select_threshold(y_true, y_scores, strategy="accuracy")
        res = compute_metrics(y_true, y_scores, threshold=thresh)
        assert res.accuracy == 1.0

    def test_optimal_youden_selection(self):
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.25, 0.75, 0.9]
        thresh = select_threshold(y_true, y_scores, strategy="youden")
        res = compute_metrics(y_true, y_scores, threshold=thresh)
        assert res.accuracy == 1.0
        # Youden's J = TPR - FPR = 1.0 - 0.0 = 1.0
        assert res.recall == 1.0

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown threshold selection strategy"):
            select_threshold([0, 1], [0.2, 0.8], strategy="maximize_revenue")

    def test_single_class_validation_raises(self):
        with pytest.raises(ValueError, match="requires both positive .* and negative .* classes"):
            select_threshold([0, 0, 0], [0.1, 0.2, 0.3], strategy="f1")
        with pytest.raises(ValueError, match="requires both positive .* and negative .* classes"):
            select_threshold([1, 1, 1], [0.7, 0.8, 0.9], strategy="accuracy")

    def test_deterministic_tie_breaking_centered(self):
        # Symmetrical separation: gap between 0.2 and 0.8
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.8, 0.9]
        thresh = select_threshold(y_true, y_scores, strategy="f1")
        # 0.5 is candidate and closest to 0.5
        assert thresh == pytest.approx(0.5, abs=1e-5)

    def test_all_identical_scores(self):
        y_true = [0, 1]
        y_scores = [0.6, 0.6]
        thresh = select_threshold(y_true, y_scores, strategy="accuracy")
        assert thresh == pytest.approx(0.6, abs=1e-5)

    def test_select_threshold_f1_no_runtime_warning_when_negatives_highest(self):
        # Inverted scores where negative samples (reals) have highest scores
        y_true = [0, 0, 0, 1, 1]
        y_scores = [0.95, 0.85, 0.75, 0.25, 0.15]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            thresh = select_threshold(y_true, y_scores, strategy="f1")
        assert isinstance(thresh, float)

    def test_threshold_tie_breaking_rtol_zero(self):
        # Construct large dataset where two candidate thresholds differ in metric by < 1e-5.
        # With default rtol=1e-5, np.isclose would treat them as tied and pick the candidate closer to 0.5.
        # With rtol=0.0, the candidate with strictly higher metric is correctly selected.
        n_samples = 200000
        y_true = np.zeros(n_samples, dtype=int)
        y_scores = np.zeros(n_samples, dtype=float)
        y_true[100000:] = 1  # 100k reals, 100k fakes

        # Reals: 99998 at 0.1, 1 at 0.6, 1 at 0.7
        y_scores[:99998] = 0.1
        y_scores[99998] = 0.6
        y_scores[99999] = 0.7

        # Fakes: 1 at 0.4, 99999 at 0.9
        y_scores[100000] = 0.4
        y_scores[100001:] = 0.9

        # At threshold 0.8: accuracy = 199999 / 200000 = 0.999995
        # At threshold 0.5: accuracy = 199997 / 200000 = 0.999985
        # |0.8 - 0.5| = 0.3, |0.5 - 0.5| = 0.0
        thresh = select_threshold(y_true, y_scores, strategy="accuracy")
        assert thresh == pytest.approx(0.8, abs=1e-5)

    def test_candidate_upper_bound_reaches_all_negative_boundary(self):
        # 9 reals with scores 0.1..0.9, 1 fake with score 0.05
        # If we threshold above max score (0.9), all 10 are predicted 0 -> accuracy = 9/10 = 0.90
        # If thresholding at 0.9, real with 0.9 is predicted 1 -> accuracy = 8/10 = 0.80
        y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        y_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.05]
        thresh = select_threshold(y_true, y_scores, strategy="accuracy")
        assert thresh > 0.9
        res = compute_metrics(y_true, y_scores, threshold=thresh)
        assert res.accuracy == 0.9
        assert res.confusion_matrix == {"tp": 0, "fp": 0, "tn": 9, "fn": 1}

    def test_logit_scores_outside_unit_interval(self):
        # Model producing unbounded logits
        y_true = [0, 0, 1, 1]
        y_scores = [-4.5, -1.2, 2.3, 6.8]
        thresh = select_threshold(y_true, y_scores, strategy="f1")
        res = compute_metrics(y_true, y_scores, threshold=thresh)
        assert res.auroc == 1.0
        assert res.accuracy == 1.0
        assert res.f1 == 1.0
        assert -1.2 < thresh <= 2.3


class TestComputeMetrics:
    """Unit tests for compute_metrics calculation, confusion matrix, and edge cases."""

    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.8, 0.9]
        res = compute_metrics(y_true, y_scores, threshold=0.5, threshold_source="test_source")
        assert res.auroc == 1.0
        assert res.accuracy == 1.0
        assert res.f1 == 1.0
        assert res.precision == 1.0
        assert res.recall == 1.0
        assert res.threshold == 0.5
        assert res.threshold_source == "test_source"
        assert res.confusion_matrix == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}

    def test_zero_positives_predicted_graceful_division(self):
        # All scores below threshold -> no positive predictions
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.3, 0.4]
        res = compute_metrics(y_true, y_scores, threshold=0.5)
        assert res.confusion_matrix == {"tp": 0, "fp": 0, "tn": 2, "fn": 2}
        assert res.precision == 0.0
        assert res.recall == 0.0
        assert res.f1 == 0.0
        assert res.accuracy == 0.5

    def test_all_positives_predicted(self):
        # All scores above threshold -> predict 1 for everything
        y_true = [0, 0, 1, 1]
        y_scores = [0.6, 0.7, 0.8, 0.9]
        res = compute_metrics(y_true, y_scores, threshold=0.5)
        assert res.confusion_matrix == {"tp": 2, "fp": 2, "tn": 0, "fn": 0}
        assert res.precision == 0.5
        assert res.recall == 1.0
        assert res.f1 == pytest.approx(2 * (0.5 * 1.0) / 1.5, abs=1e-5)
        assert res.accuracy == 0.5

    def test_single_class_all_reals_metrics(self):
        y_true = [0, 0, 0, 0]
        y_scores = [0.1, 0.2, 0.3, 0.4]
        res = compute_metrics(y_true, y_scores, threshold=0.5)
        assert res.auroc is None
        assert res.accuracy == 1.0
        assert res.precision == 0.0
        assert res.recall == 0.0
        assert res.f1 == 0.0
        assert res.confusion_matrix == {"tp": 0, "fp": 0, "tn": 4, "fn": 0}

    def test_single_class_all_fakes_metrics(self):
        y_true = [1, 1, 1, 1]
        y_scores = [0.6, 0.7, 0.8, 0.9]
        res = compute_metrics(y_true, y_scores, threshold=0.5)
        assert res.auroc is None
        assert res.accuracy == 1.0
        assert res.precision == 1.0
        assert res.recall == 1.0
        assert res.f1 == 1.0
        assert res.confusion_matrix == {"tp": 4, "fp": 0, "tn": 0, "fn": 0}


class TestValidationToTestTransference:
    """Tests verifying threshold selection on validation transferred to test without leakage."""

    def test_evaluate_with_validation_threshold_f1(self):
        val_true = [0, 0, 1, 1]
        val_scores = [0.1, 0.2, 0.8, 0.9]

        test_true = [0, 0, 1, 1]
        test_scores = [0.15, 0.25, 0.75, 0.85]

        test_result, selected_thresh = evaluate_with_validation_threshold(
            val_true, val_scores, test_true, test_scores, strategy="f1"
        )

        assert isinstance(test_result, MetricResult)
        assert isinstance(selected_thresh, float)
        assert test_result.threshold == selected_thresh
        assert test_result.threshold_source == "val_optimal_f1"
        assert test_result.accuracy == 1.0
        assert test_result.f1 == 1.0

    def test_test_data_cannot_leak_into_threshold(self):
        val_true = [0, 0, 1, 1]
        val_scores = [0.1, 0.2, 0.8, 0.9]

        # Test set 1: standard
        test_true_1 = [0, 1]
        test_scores_1 = [0.2, 0.8]
        _, thresh_1 = evaluate_with_validation_threshold(
            val_true, val_scores, test_true_1, test_scores_1, strategy="f1"
        )

        # Test set 2: drastically different labels and scores (e.g. inverted, shifted)
        test_true_2 = [1, 0]
        test_scores_2 = [0.01, 0.99]
        _, thresh_2 = evaluate_with_validation_threshold(
            val_true, val_scores, test_true_2, test_scores_2, strategy="f1"
        )

        # Threshold depends solely on validation data
        assert thresh_1 == thresh_2
        assert thresh_1 == select_threshold(val_true, val_scores, strategy="f1")

    def test_all_strategies_supported_in_workflow(self):
        val_true = [0, 0, 1, 1]
        val_scores = [0.1, 0.3, 0.7, 0.9]
        test_true = [0, 1]
        test_scores = [0.2, 0.8]

        for strat in VALID_STRATEGIES:
            res, thresh = evaluate_with_validation_threshold(
                val_true, val_scores, test_true, test_scores, strategy=strat
            )
            assert res.threshold_source == f"val_optimal_{strat}"
            assert res.threshold == thresh


class TestMetricResultSerialization:
    """Unit tests for MetricResult serialization, deserialization, and immutability."""

    def test_to_dict_and_from_dict_roundtrip(self):
        cm = {"tp": 10, "fp": 2, "tn": 15, "fn": 1}
        original = MetricResult(
            auroc=0.945,
            accuracy=0.893,
            f1=0.869,
            precision=0.833,
            recall=0.909,
            threshold=0.523,
            threshold_source="val_optimal_f1",
            confusion_matrix=cm,
        )

        d = original.to_dict()
        assert d["auroc"] == pytest.approx(0.945)
        assert d["accuracy"] == pytest.approx(0.893)
        assert d["threshold_source"] == "val_optimal_f1"
        assert d["confusion_matrix"] == cm

        reconstructed = MetricResult.from_dict(d)
        assert reconstructed == original

    def test_to_json_and_from_json_roundtrip(self):
        cm = {"tp": 5, "fp": 0, "tn": 5, "fn": 0}
        original = MetricResult(
            auroc=1.0,
            accuracy=1.0,
            f1=1.0,
            precision=1.0,
            recall=1.0,
            threshold=0.5,
            threshold_source="default_0.5",
            confusion_matrix=cm,
        )

        json_str = original.to_json()
        parsed = json.loads(json_str)
        assert parsed["auroc"] == 1.0
        assert parsed["confusion_matrix"]["tp"] == 5

        reconstructed = MetricResult.from_json(json_str)
        assert reconstructed == original

    def test_save_json_and_disk_roundtrip(self, tmp_path: Path):
        file_path = tmp_path / "metrics" / "eval_res.json"
        original = MetricResult(
            auroc=None,  # test None serialization
            accuracy=0.75,
            f1=0.667,
            precision=0.5,
            recall=1.0,
            threshold=0.4,
            threshold_source="val_optimal_accuracy",
            confusion_matrix={"tp": 1, "fp": 1, "tn": 2, "fn": 0},
        )

        original.save_json(file_path)
        assert file_path.exists()

        content = file_path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert parsed["auroc"] is None

        reloaded = MetricResult.from_json(content)
        assert reloaded == original
        assert reloaded.auroc is None

    def test_dataclass_frozen_immutability(self):
        res = MetricResult(
            auroc=0.9,
            accuracy=0.8,
            f1=0.8,
            precision=0.8,
            recall=0.8,
            threshold=0.5,
            threshold_source="default_0.5",
            confusion_matrix={"tp": 8, "fp": 2, "tn": 8, "fn": 2},
        )
        with pytest.raises(FrozenInstanceError):
            res.accuracy = 0.99  # type: ignore[misc]

    def test_post_init_validation(self):
        with pytest.raises(ValueError, match="threshold_source must be a non-empty string"):
            MetricResult(
                auroc=0.9,
                accuracy=0.8,
                f1=0.8,
                precision=0.8,
                recall=0.8,
                threshold=0.5,
                threshold_source="",
                confusion_matrix={"tp": 1, "fp": 0, "tn": 1, "fn": 0},
            )

        with pytest.raises(ValueError, match="confusion_matrix must be a dict with keys"):
            MetricResult(
                auroc=0.9,
                accuracy=0.8,
                f1=0.8,
                precision=0.8,
                recall=0.8,
                threshold=0.5,
                threshold_source="default_0.5",
                confusion_matrix={"tp": 1, "fp": 0},  # missing tn, fn
            )

    def test_boolean_types_rejected_in_post_init(self):
        cm = {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
        # Booleans should not pass as floats
        with pytest.raises(TypeError, match="accuracy must be float"):
            MetricResult(
                auroc=0.9,
                accuracy=True,  # type: ignore[arg-type]
                f1=0.8,
                precision=0.8,
                recall=0.8,
                threshold=0.5,
                threshold_source="default_0.5",
                confusion_matrix=cm,
            )

        # Booleans should not pass as confusion matrix ints
        with pytest.raises(TypeError, match="confusion_matrix\\['tp'\\] must be an integer"):
            MetricResult(
                auroc=0.9,
                accuracy=0.8,
                f1=0.8,
                precision=0.8,
                recall=0.8,
                threshold=0.5,
                threshold_source="default_0.5",
                confusion_matrix={"tp": True, "fp": 0, "tn": 1, "fn": 0},  # type: ignore[dict-item]
            )


class TestExtendedForensicMetrics:
    """Unit tests for TPR@FPR, EER, and PR-AUC."""

    def test_calculate_tpr_at_fpr_perfect(self):
        from forensight.evaluation.metrics import calculate_tpr_at_fpr

        # With 100 negatives, target_fpr=0.01 is empirically valid
        y_true = [0] * 100 + [1] * 100
        y_scores = [0.1] * 100 + [0.9] * 100
        tpr = calculate_tpr_at_fpr(y_true, y_scores, target_fpr=0.01)
        assert tpr == 1.0

    def test_calculate_tpr_at_fpr_sample_guard(self):
        from forensight.evaluation.metrics import calculate_tpr_at_fpr

        # 4 negatives cannot resolve 1% FPR (requires >= 100 negatives)
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_scores = [0.1, 0.2, 0.15, 0.25, 0.8, 0.85, 0.9, 0.95]
        assert calculate_tpr_at_fpr(y_true, y_scores, target_fpr=0.01) is None
        # But can resolve 25% FPR (requires >= 4 negatives)
        assert calculate_tpr_at_fpr(y_true, y_scores, target_fpr=0.25) == 1.0

    def test_calculate_tpr_at_fpr_single_class(self):
        from forensight.evaluation.metrics import calculate_tpr_at_fpr

        assert calculate_tpr_at_fpr([1, 1, 1], [0.8, 0.9, 0.7]) is None
        assert calculate_tpr_at_fpr([0, 0, 0], [0.1, 0.2, 0.3]) is None

    def test_calculate_tpr_at_fpr_invalid_range(self):
        from forensight.evaluation.metrics import calculate_tpr_at_fpr

        with pytest.raises(ValueError, match="target_fpr must be between"):
            calculate_tpr_at_fpr([0, 1], [0.2, 0.8], target_fpr=-0.1)
        with pytest.raises(ValueError, match="target_fpr must be between"):
            calculate_tpr_at_fpr([0, 1], [0.2, 0.8], target_fpr=1.5)

    def test_calculate_eer_perfect(self):
        from forensight.evaluation.metrics import calculate_eer

        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.8, 0.9]
        eer = calculate_eer(y_true, y_scores)
        assert eer == 0.0

    def test_calculate_eer_linear_interpolation(self):
        from forensight.evaluation.metrics import calculate_eer

        # Non-separable with crossing between discrete steps
        y_true = [0, 0, 0, 1, 1, 1]
        y_scores = [0.1, 0.4, 0.6, 0.3, 0.5, 0.9]
        eer = calculate_eer(y_true, y_scores)
        assert eer is not None
        assert 0.0 < eer < 1.0

    def test_calculate_eer_single_class(self):
        from forensight.evaluation.metrics import calculate_eer

        assert calculate_eer([1, 1], [0.8, 0.9]) is None

    def test_calculate_pr_auc_perfect(self):
        from forensight.evaluation.metrics import calculate_pr_auc

        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.8, 0.9]
        pr_auc = calculate_pr_auc(y_true, y_scores)
        assert pr_auc == 1.0

    def test_calculate_pr_auc_single_class(self):
        from forensight.evaluation.metrics import calculate_pr_auc

        assert calculate_pr_auc([0, 0], [0.1, 0.2]) is None

    def test_compute_metrics_includes_forensic_metrics(self):
        # With sufficient samples (>= 1000 negatives), both TPR@1% and TPR@0.1% are valid
        y_true = [0] * 1000 + [1] * 1000
        y_scores = [0.1] * 1000 + [0.9] * 1000
        res = compute_metrics(y_true, y_scores)
        assert res.auroc == 1.0
        assert res.eer == 0.0
        assert res.pr_auc == 1.0
        assert res.tpr_at_1pct_fpr == 1.0
        assert res.tpr_at_01pct_fpr == 1.0

    def test_compute_metrics_small_sample_guards_tpr(self):
        # Small sample size (2 negatives) gracefully sets TPR@FPR to None while computing other metrics
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.8, 0.9]
        res = compute_metrics(y_true, y_scores)
        assert res.auroc == 1.0
        assert res.eer == 0.0
        assert res.pr_auc == 1.0
        assert res.tpr_at_1pct_fpr is None
        assert res.tpr_at_01pct_fpr is None
