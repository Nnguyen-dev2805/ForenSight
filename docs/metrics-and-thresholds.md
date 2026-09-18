# Metrics and Threshold Policy

## 1. Overview and Protocol Invariants

ForenSight investigates generalizable and explainable AI-generated image detection.
Evaluation integrity is paramount: detection metrics must be unambiguous, reproducible,
and strictly protected against data snooping, leakage, and threshold optimization on test sets.

In accordance with [`docs/evaluation-protocol.md`](evaluation-protocol.md), this document
formalizes:
1. **Primary metric:** Area Under the Receiver Operating Characteristic Curve (**AUROC**).
2. **Secondary metrics:** **Accuracy**, **F1-Score**, **Precision**, and **Recall** at a fixed decision threshold.
3. **Threshold policy:** Decision thresholds for secondary metrics **MUST** be chosen exclusively
   from the validation split (`val`) and frozen prior to test evaluation. Thresholds must **NEVER**
   be optimized, tuned, or swept on any test distribution.
4. **Threshold provenance:** Every evaluation result explicitly records its threshold value
   and threshold source (e.g., `"val_optimal_f1"`, `"default_0.5"`).
5. **Model-agnostic evaluation API:** Standardized interface consuming 1D numerical sequences
   of binary ground-truth labels and continuous prediction scores.

---

## 2. Primary Metric: AUROC

### Motivation
AUROC is ForenSight's **primary metric** for benchmark comparisons across generators, datasets,
and perturbations because:
- It is **threshold-free**: it evaluates the ranking quality across all possible operational operating points.
- It is **invariant to class imbalance**: changing real-to-fake proportions does not shift the expected ROC curve.
- It directly reflects the probability that a randomly chosen fake (AI-generated) sample receives a higher score than a randomly chosen real (authentic) sample.

### Formulation
For binary classification where label $y \in \{0, 1\}$ (0 = real, 1 = fake) and score $s \in \mathbb{R}$:

$$\text{AUROC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(u)) \, du$$

Equivalently, via the non-parametric Wilcoxon-Mann-Whitney $U$-statistic:

$$\text{AUROC} = \frac{1}{N_{\text{fake}} \cdot N_{\text{real}}} \sum_{i \in \mathcal{P}_{\text{fake}}} \sum_{j \in \mathcal{N}_{\text{real}}} \left[ \mathbb{I}(s_i > s_j) + \frac{1}{2} \mathbb{I}(s_i = s_j) \right]$$

### Edge Cases and Singularities
- **Single-class distribution:** If a partition contains only real ($y_i = 0$) or only fake ($y_i = 1$) images, AUROC is mathematically undefined. In this scenario, `calculate_auroc()` safely returns `None`.
- **Constant scores:** If a model predicts identical scores for all samples, `AUROC = 0.5` (random guess baseline).
- **Perfect inversion:** If a model assigns lower scores to fakes than reals, `AUROC = 0.0`.
- **Perfect discrimination:** If all fakes score strictly higher than all reals, `AUROC = 1.0`.

---

## 3. Secondary Metrics

At a chosen decision threshold $\tau \in \mathbb{R}$, continuous scores are binarized:

$$\hat{y}_i = \begin{cases} 1 & \text{if } s_i \ge \tau \\ 0 & \text{if } s_i < \tau \end{cases}$$

Samples are categorized into standard confusion counts:
- **True Positive (TP):** $y_i = 1 \land \hat{y}_i = 1$ (fake correctly identified as fake)
- **False Positive (FP):** $y_i = 0 \land \hat{y}_i = 1$ (real falsely flagged as fake)
- **True Negative (TN):** $y_i = 0 \land \hat{y}_i = 0$ (real correctly identified as real)
- **False Negative (FN):** $y_i = 1 \land \hat{y}_i = 0$ (fake falsely missed as real)

### Metric Definitions

#### Accuracy
Proportion of all samples correctly classified:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

#### Precision (Positive Predictive Value)
Proportion of predicted fakes that are genuinely fake:

$$\text{Precision} = \begin{cases} \frac{\text{TP}}{\text{TP} + \text{FP}} & \text{if } \text{TP} + \text{FP} > 0 \\ 0.0 & \text{otherwise} \end{cases}$$

#### Recall (Sensitivity / True Positive Rate)
Proportion of actual fakes that are detected:

$$\text{Recall} = \begin{cases} \frac{\text{TP}}{\text{TP} + \text{FN}} & \text{if } \text{TP} + \text{FN} > 0 \\ 0.0 & \text{otherwise} \end{cases}$$

#### F1-Score
Harmonic mean of precision and recall for the positive (fake) class:

$$\text{F1} = \begin{cases} 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}} & \text{if } \text{Precision} + \text{Recall} > 0 \\ 0.0 & \text{otherwise} \end{cases}$$

Zero-denominator edge cases default cleanly to `0.0`.

---

## 4. Threshold Policy and Test-Invariance Guarantee

### The Invariant
> **Threshold Rule:** Decision thresholds applied to any evaluation split (`in_domain_test`, `cross_generator_ood`, `modern_external`, `real_world_external`) **MUST** be fixed from the validation set (`val`) or set to an a-priori default (`0.5`).
> **Under NO circumstances is a threshold tuned, searched, or optimized on test data.**

Optimizing thresholds on test data introduces data leakage, inflates reported metrics, and invalidates scientific comparisons across baselines.

### Calibration & Evaluation Workflow

```
[Validation Split] ────────► select_threshold(val_true, val_scores, strategy)
                                            │
                                            ▼
                                  Optimal Threshold τ*
                                            │
                                            ▼
[Test Split]       ────────► compute_metrics(test_true, test_scores, threshold=τ*,
                                             threshold_source="val_optimal_<strategy>")
```

To guarantee this separation in code, `evaluate_with_validation_threshold()` orchestrates the two-phase evaluation:
1. Optimizes $\tau^*$ on validation data using the selected strategy.
2. Applies $\tau^*$ to the test data without modifying $\tau^*$.
3. Returns a `MetricResult` tagged with `threshold_source=f"val_optimal_{strategy}"` and the exact float $\tau^*$.

---

## 5. Threshold Selection Strategies

The `select_threshold(y_true, y_scores, strategy=...)` function supports three optimization criteria:

| Strategy | Optimization Objective | Mathematical Formula | Recommended Use |
|---|---|---|---|
| `"f1"` | Maximize F1-Score | $\arg\max_\tau F_1(\tau)$ | Imbalanced datasets, prioritizing fake detection precision & recall |
| `"accuracy"` | Maximize Accuracy | $\arg\max_\tau \frac{\text{TP}(\tau) + \text{TN}(\tau)}{N}$ | Balanced validation sets, equal cost for false alarms and misses |
| `"youden"` | Maximize Youden's J | $\arg\max_\tau [\text{TPR}(\tau) - \text{FPR}(\tau)]$ | Diagnostic sensitivity-specificity balance independent of prior class prevalence |

### Candidate Thresholds and Vectorized Search
Rather than using arbitrary grid searches, candidate thresholds $\mathcal{C}$ are constructed directly from observed score distributions:
1. All distinct score values: $\{ s_{(1)}, s_{(2)}, \dots, s_{(m)} \}$
2. All midpoints between consecutive distinct scores: $\left\{ \frac{s_{(k)} + s_{(k+1)}}{2} \right\}$ (providing maximum separation margin between adjacent predictions)
3. Canonical reference threshold: $\{ 0.5 \}$ (if within the range $[\min(s), \max(s)]$)

For $N$ validation samples, exact metric evaluation across all candidate thresholds is executed in $\mathcal{O}(N \log N)$ time via `np.searchsorted` and cumulative positive counts.

### Deterministic Tie-Breaking
When multiple candidate thresholds yield the identical maximal metric value (common in well-separated or plateau regions):
1. **Primary rule:** Minimize absolute distance to default reference: $|\tau - 0.5|$.
   *Rationale:* Prefers thresholds centered between real and fake clusters, maximizing generalization margin and robustness to minor score shifts on unseen test data.
2. **Secondary rule:** Choose higher threshold value ($\tau$).
   *Rationale:* Conservative bias against false positives (minimizing false accusations of authenticity).

---

## 6. Input Validation and Defense-in-Depth

The evaluation module strictly validates all input vectors:
- **Empty input detection:** Raises `ValueError("Inputs cannot be empty.")` if $N = 0$.
- **Shape and dimension alignment:** Flattens single-dimensional slices (e.g. $(N, 1) \rightarrow (N,)$) and validates $N_{\text{true}} = N_{\text{scores}}$.
- **Finite numeric enforcement:** Rejects any `NaN` or `Inf` in scores or targets with explicit `ValueError`.
- **Binary label validation:** Requires $y_i \in \{0, 1\}$. Non-binary values (e.g. -1, 2, continuous floats) trigger `ValueError`.
- **Single-class safety:**
  - `calculate_auroc` safely returns `None`.
  - `compute_metrics` computes valid confusion matrix and secondary metrics.
  - `select_threshold` raises `ValueError` (optimal thresholding requires both classes to define a separation boundary).
- **Type tolerance:** Seamlessly accepts Python lists, tuples, NumPy arrays, and pandas Series.

---

## 7. Python API Reference

```python
from forensight.evaluation import (
    MetricResult,
    calculate_auroc,
    compute_metrics,
    select_threshold,
    evaluate_with_validation_threshold,
)

# 1. Compute AUROC directly
auroc = calculate_auroc(y_true=[0, 0, 1, 1], y_scores=[0.1, 0.2, 0.8, 0.9])
# auroc = 1.0

# 2. Select optimal threshold on validation data
val_thresh = select_threshold(
    y_true=val_labels,
    y_scores=val_scores,
    strategy="f1",  # "f1" | "accuracy" | "youden"
)

# 3. Compute metrics at fixed threshold
result = compute_metrics(
    y_true=test_labels,
    y_scores=test_scores,
    threshold=val_thresh,
    threshold_source="val_optimal_f1",
)

# 4. Or run end-to-end validation-to-test evaluation
test_res, applied_thresh = evaluate_with_validation_threshold(
    val_true=val_labels,
    val_scores=val_scores,
    test_true=test_labels,
    test_scores=test_scores,
    strategy="f1",
)

# 5. Serialization
res_dict = test_res.to_dict()
res_json = test_res.to_json(indent=2)
test_res.save_json("artifacts/eval_results.json")
```
