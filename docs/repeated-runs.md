# Repeated-Run Protocol & Uncertainty Quantification

## 1. Motivation and Purpose

Deep neural networks trained for AI-generated image detection exhibit non-trivial variance across training runs due to:
- Random weight initialization;
- Training batch permutation and stochastic augmentation;
- Non-deterministic GPU floating-point operations;
- Optimizer trajectories and early stopping variations.

In research on generalizable detection (**RQ1**) and explanation fidelity (**RQ2**), reported performance differences between methods (e.g. semantic-only vs. forensic-only vs. multimodal fusion) are often on the order of 1–3% AUROC. Attributing research progress to architectural differences is invalid if the measured effect is comparable to random run-to-run variance.

In accordance with [`docs/evaluation-protocol.md`](evaluation-protocol.md) and [`AGENTS.md`](../AGENTS.md), this document establishes the protocol invariants for repeated evaluations, uncertainty quantification, and baseline sealing.

---

## 2. Protocol Invariants

### 2.1 The 3-Seed Requirement
- **Target:** All primary trained baselines must target at least **3 distinct random seeds** when compute permits.
- **Reproducibility:** Every individual run must explicitly record its `seed` in its schema (`EvaluationReport.seed` and `run_metadata["seed"]`).
- **Independent Threshold Calibration:** For each seed run, the decision threshold $\tau^*$ is calibrated independently on that run's validation partition (`val`) and held fixed across all test distributions. Test distributions are never pooled across runs to re-fit thresholds.

### 2.2 Sealed Baselines vs. Preliminary Results
To uphold research integrity without blocking rapid exploration during active development, ForenSight enforces a strict distinction between **preliminary** and **sealed** evaluations:

| Status | Criterion | Allowed Claims | Schema Flag |
|:---|:---|:---|:---|
| **Preliminary Result** | $N < 3$ runs (e.g. 1-seed or 2-seed runs) | Development sanity checks, debugging, hypothesis formation. **Forbidden:** claiming a method is "better", "superior", or "generalizable" based on small gains. | `is_preliminary: True`<br>`preliminary_result: True` |
| **Sealed Baseline** | $N \ge 3$ independent runs across distinct seeds | Official benchmark claims, hypothesis testing, comparative analysis in papers and reports. | `is_preliminary: False`<br>`preliminary_result: False` |

> [!IMPORTANT]
> A 1-seed or 2-seed result cannot be cited to claim method superiority over an existing baseline. Any comparative table or chart containing preliminary runs must explicitly display the `⚠️ PRELIMINARY` indicator.

---

## 3. Uncertainty Reporting: $\text{mean} \pm \text{std}$

### 3.1 Sample Mean
For a metric $m$ evaluated across $N$ valid runs:

$$\bar{m} = \frac{1}{N} \sum_{i=1}^N m_i$$

### 3.2 Sample Standard Deviation (Bessel's Correction)
Uncertainty across repeated runs is reported as the sample standard deviation with degrees of freedom $N - 1$ (`ddof=1`):

$$s = \sqrt{\frac{1}{N - 1} \sum_{i=1}^N (m_i - \bar{m})^2} \quad \text{for } N \ge 2$$

For single-seed runs ($N = 1$), standard deviation is defined strictly as $0.0$:

$$s = 0.0000 \quad \text{for } N = 1$$

### 3.3 Formatted String Representation
All tabular reports and console displays format aggregated metrics as:

$$\text{mean} \pm \text{std} \quad \text{(e.g. } `0.8950 \pm 0.0120` \text{)}$$

Single-seed preliminary runs display with zero uncertainty: `0.8950 ± 0.0000`.

---

## 4. Aggregation Granularity

Aggregation is performed systematically across all ForenSight evaluation axes:

1. **Overall Benchmark:** Aggregated across test/evaluation partitions (excluding training and validation samples).
2. **Per-Partition (Split):** In-domain (`in_domain_test`), near-OOD (`sd15`), cross-generator OOD (`cross_generator_ood`), external benchmarks (`genimage_plus`, `wildrf`).
3. **Per-Generator Architecture:** Individual generator architectures (e.g. `midjourney`, `adm`, `glide`, `wukong`, `vqdm`, `biggan`).
4. **Per-Dataset Source:** Source datasets (e.g. `genimage`, `genimage_plus`, `wildrf`).

---

## 5. Edge Cases and Singularity Handling

### 5.1 Single-Class Partitions (Undefined AUROC)
When evaluating slices that contain only authentic images (real) or only synthetic images (fake) — such as generator-specific test subsets lacking matched authentic samples:
- `MetricResult.auroc` is mathematically undefined and evaluates to `None`.
- In `AggregatedSlice`, `auroc` evaluates cleanly to `None`.
- `formatted` string property outputs `"N/A"`.
- Secondary metrics (**Accuracy**, **Recall**, **Precision**, **F1**) are computed and aggregated normally without throwing exceptions.

### 5.2 Disjoint Slices Across Runs
If certain optional splits or external datasets appear in a subset of runs, `aggregate_reports()` aggregates over the intersection of valid runs for each specific slice, tracking the sample size $N$ per metric independently in `AggregatedMetric.n`.

---

## 6. Python API Reference

### 6.1 `AggregatedMetric`
Statistical container representing an aggregated scalar metric:
```python
from forensight.evaluation import AggregatedMetric

metric = AggregatedMetric.from_values([0.892, 0.901, 0.908])
print(metric.formatted)  # "0.9003 ± 0.0081"
print(metric.mean)       # 0.900333...
print(metric.std)        # 0.008082...
print(metric.min)        # 0.892
print(metric.max)        # 0.908
print(metric.n)          # 3
```

### 6.2 `AggregatedSlice`
Collection of `AggregatedMetric` instances for an evaluation slice:
```python
from forensight.evaluation import AggregatedSlice

# Access attributes directly or via dictionary indexing
print(agg_slice.accuracy.formatted)   # "0.8650 ± 0.0075"
print(agg_slice["f1"].formatted)        # "0.8520 ± 0.0090"
if agg_slice.auroc is not None:
    print(agg_slice.auroc.formatted)  # "0.9120 ± 0.0045"
```

### 6.3 `AggregatedEvaluationReport` & `aggregate_reports`
Aggregate multiple `EvaluationReport` objects:
```python
from forensight.evaluation import aggregate_reports, EvaluationReport

# Load single-run reports
report1 = EvaluationReport.from_json(path1.read_text())
report2 = EvaluationReport.from_json(path2.read_text())
report3 = EvaluationReport.from_json(path3.read_text())

# Aggregate runs
agg_report = aggregate_reports([report1, report2, report3])

print(agg_report.num_runs)          # 3
print(agg_report.seeds)             # [42, 43, 44]
print(agg_report.is_preliminary)    # False (sealed benchmark)

# Serialization and Markdown summary
agg_report.save_json("results/aggregated.json")
agg_report.save_markdown("results/aggregated.md")
```

---

## 7. CLI Reference: `scripts/aggregate_runs.py`

Command-line tool to aggregate multiple single-run evaluation reports:

```bash
# Basic aggregation of 3 seeds
python scripts/aggregate_runs.py \
    --reports results/sd14_seed42.json results/sd14_seed43.json results/sd14_seed44.json \
    --output-json results/sd14_aggregated.json \
    --output-md results/sd14_aggregated.md

# Inspect aggregated Markdown report on stdout
python scripts/aggregate_runs.py \
    --report results/s1.json results/s2.json \
    --print-markdown
```
