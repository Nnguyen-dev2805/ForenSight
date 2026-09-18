# Experiment Reproducibility Record & Provenance Protocol

## 1. Motivation and Purpose

In research on generalizable AI-generated image detection (**RQ1**) and explanation fidelity (**RQ2**), reported improvements are frequently within the margin of random seed variance (1–3% AUROC) or susceptible to subtle evaluation protocol discrepancies. Without rigorous tracking of the exact data partition, hyperparameter configuration, decision threshold calibration provenance, runtime environment, and codebase revision, scientific claims cannot be verified or reproduced.

In accordance with [`docs/evaluation-protocol.md` Section 12](evaluation-protocol.md) and [`AGENTS.md`](../AGENTS.md), ForenSight mandates that **every evaluation run must preserve a complete, machine-readable reproducibility record**.

---

## 2. Invariants & Required Fields

Section 12 of the ForenSight Evaluation Protocol specifies that every run must record:
1. **`run_id`**: Unique identifier for the individual run.
2. **`experiment_name`**: Logical baseline family or experiment identifier.
3. **`timestamp`**: UTC timestamp in strict ISO 8601 format (e.g. `2026-09-18T12:30:00.000000+00:00`).
4. **`git_commit`**: Git commit hash (`HEAD`) at execution time (or `None` if unversioned).
5. **`seed`**: Random seed for weight initialization, data shuffling, and sampling (or `None` for unseeded runs).
6. **`split_version`**: Dataset manifest or partition version tag (e.g. `r0-smoke`, `genimage_v1.0`).
7. **`config`**: Complete dictionary of hyperparameters, model arguments, and evaluation options.
8. **`threshold_source`**: Source description of the decision threshold (e.g. `val_f1`, `val_youden`, `default`).
9. **`threshold_value`**: Frozen numerical decision threshold $\tau^*$ calibrated on validation data.
10. **`metrics`**: Comprehensive metrics dictionary (overall performance and partition/generator breakdowns).
11. **`environment`**: Operating system, Python runtime version, package dependency versions, and hardware.
12. **`notes`**: Optional free-text annotations, operational context, or hypothesis tracking.

---

## 3. Schema Specification

The `ReproducibilityRecord` dataclass in [`src/forensight/evaluation/reproducibility.py`](../src/forensight/evaluation/reproducibility.py) formalizes this schema:

| Field | Type | Default | Validation Invariant | Description |
|:---|:---|:---|:---|:---|
| `run_id` | `str` | Required | Non-empty string | Unique execution identifier |
| `experiment_name` | `str` | Required | Non-empty string | Experiment family or baseline identifier |
| `timestamp` | `str` | Required | Valid ISO 8601 string | UTC execution timestamp |
| `git_commit` | `str \| None` | `None` | Non-empty if specified | Git commit SHA (HEAD) |
| `seed` | `int \| str \| None`| `None` | Int, str, or None | Random seed |
| `split_version` | `str` | Required | Non-empty string | Dataset split or manifest version tag |
| `config` | `dict[str, Any]` | `{}` | Must be a `dict` | Hyperparameters and run configuration |
| `threshold_source`| `str` | Required | Non-empty string | Strategy used to select decision threshold |
| `threshold_value` | `float` | Required | Finite float (no NaN/Inf) | Calibrated decision threshold $\tau^*$ |
| `metrics` | `dict[str, Any]` | `{}` | Must be a `dict` | Evaluation metrics (overall & slices) |
| `environment` | `dict[str, Any]` | `{}` | Must be a `dict` | Runtime OS, Python, packages, hardware |
| `notes` | `str` | `""` | Must be a `str` | Free-text annotations |

### 3.1 Validation Method: `validate() -> list[str]`
The `validate()` method verifies all invariants and returns a list of human-readable error messages. If the record satisfies all protocol requirements, it returns an empty list `[]`.
- `record.is_valid`: Boolean property indicating whether `len(record.validate()) == 0`.
- `record.assert_valid()`: Helper raising `ValueError` containing all error messages if validation fails.

---

## 4. Runtime Environment Introspection

The helper `get_environment_info(packages=None)` automatically introspects:
- **Python Runtime:** Version (`platform.python_version()`), implementation (`platform.python_implementation()`), and interpreter executable path (`sys.executable`).
- **Operating System & Architecture:** Platform string (`platform.platform()`), system name (`platform.system()`), and machine architecture (`platform.machine()`).
- **Core Dependencies:** Introspects versions using standard-library `importlib.metadata.version()` without forcing heavy module imports:
  - `torch`, `torchvision` (if installed)
  - `numpy`
  - `pandas`
  - `scipy`
  - `scikit-learn`
  - `pillow`
  - `pytest`
- **Hardware Acceleration:** CPU core count (`os.cpu_count()`), CUDA availability and device names (if PyTorch loaded), and Apple Silicon MPS availability.

The helper `get_git_commit(cwd=None)` queries `git rev-parse HEAD` with a 5-second timeout, gracefully returning `None` if running outside a Git repository or in restricted environments.

---

## 5. Usage & Integration

### 5.1 CLI Integration (`scripts/run_evaluation.py`)

Run evaluations and automatically persist a reproducibility record alongside standard JSON and Markdown reports:

```bash
# Basic evaluation with reproducibility record
python scripts/run_evaluation.py \
    --predictions data/preds/baseline_seed42.jsonl \
    --val-split val \
    --threshold-strategy f1 \
    --seed 42 \
    --split-version "genimage_v1.0" \
    --save-reproducibility results/baseline_seed42_repro.json

# Using an external config file
python scripts/run_evaluation.py \
    --predictions data/preds/resnet50_seed101.csv \
    --val-split val \
    --config-file configs/resnet50_baseline.json \
    --split-version "r0-pilot" \
    --save-reproducibility results/resnet50_seed101_repro.json \
    --notes "ResNet-50 frozen backbone with linear probe on SD1.4"
```

### 5.2 Programmatic Python API

```python
from forensight.evaluation.reproducibility import create_reproducibility_record
from forensight.evaluation.runner import evaluate_predictions, PredictionSet

# 1. Run model evaluation
preds = PredictionSet.from_jsonl("data/preds/run_42.jsonl")
report = evaluate_predictions(predictions=preds, val_split_name="val", threshold_strategy="f1")

# 2. Generate reproducibility record from evaluation report
record = create_reproducibility_record(
    experiment_name="baseline_pilot",
    split_version="r0-smoke",
    config={"backbone": "clip-vit-b32", "batch_size": 32, "lr": 1e-4},
    report=report,
    notes="Initial pilot run across 5 synset classes",
)

# 3. Validate and persist
record.assert_valid()
record.save_json("results/run_42_repro.json")

# 4. Reload and inspect
loaded = ReproducibilityRecord.load_json("results/run_42_repro.json")
assert loaded.is_valid
print(loaded.generate_markdown())
```

### 5.3 Direct Conversion from `EvaluationReport`

Any `EvaluationReport` can directly export its reproducibility record:

```python
record = report.to_reproducibility_record(
    split_version="genimage_v1.0",
    config={"model": "dinov2_vitb14"},
    notes="Pre-trained DINOv2 evaluation",
)
record.save_json("results/dinov2_record.json")
```

---

## 6. Multi-Seed Aggregation Workflow

When conducting 3-seed benchmark runs under the repeated-run protocol ([`docs/repeated-runs.md`](repeated-runs.md)):
1. Each seed evaluation (`seed=42`, `seed=43`, `seed=44`) generates its own `EvaluationReport` and `ReproducibilityRecord`.
2. Each run's threshold $\tau^*_i$ is calibrated strictly on its validation partition.
3. The multi-seed aggregator (`scripts/aggregate_runs.py` / `aggregate_reports()`) aggregates the evaluation reports into an `AggregatedEvaluationReport`.
4. Auditability is preserved because each seed's exact configuration, environment, and threshold provenance are tracked in its reproducibility record.
