# ForenSight Evaluation Runner

This document specifies the architecture, input schemas, calibration policies, CLI commands, and programmatic APIs for the model-agnostic ForenSight evaluation runner (`forensight.evaluation.runner`).

## 1. Overview & Evaluation Invariants

In accordance with [`docs/evaluation-protocol.md`](evaluation-protocol.md) and R0 requirements, the evaluation runner operates under strict research invariants:

1. **Model-Agnostic Interface:** The runner processes prediction records (`sample_id`, `label`, `score`, metadata) rather than PyTorch models or raw tensors. Any detector model (CNN, ViT, frequency-based, fusion, or MLLM) can be benchmarked without coupling code to model implementations.
2. **Threshold Isolation Invariant:** The decision threshold $\tau^*$ is calibrated strictly on the validation partition (`val`) using the designated criterion (`f1`, `accuracy`, or `youden`). Once selected, $\tau^*$ is frozen and applied unchanged to all evaluation, in-domain, and out-of-distribution (OOD) test sets. Test data is never inspected during threshold selection.
3. **Multi-Slice Breakdown:** Results are computed across overall benchmarks and systematically partitioned by evaluation split (`in_domain_test`, `near_ood`, `cross_generator_ood`, `modern_external`, `real_world_external`), generator architecture (`sd14`, `sd15`, `midjourney`, `adm`, `glide`, `flux`, etc.), and dataset source (`genimage`, `wildrf`).
4. **Reproducibility & Provenance:** Every report logs full threshold provenance, sample counts, partition metadata, and timestamp in both machine-readable JSON and human-readable Markdown.

---

## 2. Input Schema & Data Formats

### 2.1 Conceptual Schema (`PredictionRecord`)

| Field | Type | Required | Description / Supported Values |
|:---|:---|:---:|:---|
| `sample_id` | `str` | Yes | Unique identifier (image path, hash, or ID). |
| `label` | `int` | Yes | Ground-truth binary target: `0` (real) or `1` (fake/AI-generated). |
| `score` | `float` | Yes | Continuous prediction score or probability in $[0.0, 1.0]$. Higher score indicates fake. |
| `split` | `str` | Optional | Partition name (`train`, `val`, `in_domain_test`, `near_ood`, `cross_generator_ood`, etc.). |
| `generator` | `str` | Optional | Generative model or source (`sd14`, `midjourney`, `nature`, `social_wild`, etc.). |
| `dataset` | `str` | Optional | Dataset origin (`genimage`, `wildrf`, `genimage_plus_plus`). |
| `metadata` | `dict` | Optional | Arbitrary metadata (resolution, prompt, class synset, inference time, etc.). |

Common aliases are automatically mapped during ingestion:
- `sample_id`: `id`, `image_id`, `path`, `image_path`
- `label`: `target`, `y_true`, `ground_truth`
- `score`: `prob`, `probability`, `pred`, `prediction`, `y_score`, `y_pred`

### 2.2 CSV Format

```csv
sample_id,label,score,split,generator,dataset,metadata
sample_0001.jpg,0,0.12,val,nature,genimage,"{""resolution"": [512, 512]}"
sample_0002.jpg,1,0.88,val,sd14,genimage,"{""resolution"": [512, 512]}"
sample_0003.jpg,1,0.76,cross_generator_ood,midjourney,genimage,"{}"
sample_0004.jpg,0,0.05,in_domain_test,nature,genimage,"{}"
```

### 2.3 JSONL Format

```json
{"sample_id": "s1", "label": 0, "score": 0.08, "split": "val", "generator": "nature", "dataset": "genimage"}
{"sample_id": "s2", "label": 1, "score": 0.94, "split": "val", "generator": "sd14", "dataset": "genimage"}
{"sample_id": "s3", "label": 1, "score": 0.81, "split": "cross_generator_ood", "generator": "adm", "dataset": "genimage"}
```

---

## 3. Programmatic API

### 3.1 Loading Predictions (`PredictionSet`)

```python
from forensight.evaluation import PredictionSet, PredictionRecord

# 1. Ingestion from CSV or JSONL
preds = PredictionSet.from_csv("results/predictions.csv")
preds = PredictionSet.from_jsonl("results/predictions.jsonl")

# 2. Ingestion from pandas DataFrame
import pandas as pd
df = pd.read_parquet("results/predictions.parquet")
preds = PredictionSet.from_dataframe(df)

# 3. Collection slicing and filtering
val_set = preds.get_split("val")
midjourney_set = preds.get_generator("midjourney")
filtered = preds.filter(lambda r: r.score > 0.5)

# 4. Numpy arrays
y_true = preds.y_true    # 1D int array
y_scores = preds.y_scores # 1D float64 array
```

### 3.2 Running Evaluation (`evaluate_predictions`)

```python
from forensight.evaluation import evaluate_predictions

report = evaluate_predictions(
    predictions=preds,
    val_split_name="val",         # Partition used to calibrate threshold
    threshold_strategy="f1",      # 'f1', 'accuracy', or 'youden'
    default_threshold=0.5,        # Fallback if validation split is absent
    run_metadata={"model": "clip_vit_b16", "seed": 42},
)

# Access overall metrics
print(f"AUROC: {report.overall.auroc:.4f}")
print(f"Accuracy: {report.overall.accuracy:.4f}")
print(f"F1 Score: {report.overall.f1:.4f}")
print(f"Threshold: {report.threshold_metadata['threshold']:.4f}")
print(f"Provenance: {report.threshold_metadata['threshold_source']}")

# Access per-split metrics
for split_name, metrics in report.by_split.items():
    print(f"Split [{split_name}]: AUROC={metrics.auroc}, F1={metrics.f1:.4f}")

# Access per-generator metrics
for gen_name, metrics in report.by_generator.items():
    print(f"Generator [{gen_name}]: AUROC={metrics.auroc}, Recall={metrics.recall:.4f}")

# Export artifacts
report.save_json("results/report.json")
report.save_markdown("results/report.md")
```

### 3.3 Separate Validation Predictions

When validation predictions are stored in a distinct file from test predictions:

```python
test_preds = PredictionSet.from_csv("results/test_predictions.csv")
val_preds = PredictionSet.from_csv("results/val_predictions.csv")

report = evaluate_predictions(
    predictions=test_preds,
    val_predictions=val_preds,
    threshold_strategy="youden",
)
```

---

## 4. CLI Usage

The CLI script `scripts/run_evaluation.py` allows executing evaluations from terminal or automation pipelines.

### 4.1 Basic Evaluation (Single File)

```bash
python scripts/run_evaluation.py \
    --predictions results/preds.jsonl \
    --val-split val \
    --threshold-strategy f1 \
    --output-json results/report.json \
    --output-md results/report.md \
    --run-name clip_vit_b16_sd14
```

### 4.2 Separate Validation File

```bash
python scripts/run_evaluation.py \
    --predictions results/test_cross_gen.csv \
    --val-predictions results/val_sd14.csv \
    --threshold-strategy youden \
    --output-json results/cross_gen_report.json
```

### 4.3 Default Threshold (No Validation Set)

```bash
python scripts/run_evaluation.py \
    --predictions results/external_wildrf.csv \
    --default-threshold 0.5 \
    --output-md results/wildrf_report.md
```

### 4.4 CLI Options Reference

| Argument | Shorthand | Default | Description |
|:---|:---|:---|:---|
| `--predictions` | `-p` | *Required* | Path to prediction file (CSV or JSONL). |
| `--val-predictions` | | `None` | Path to separate validation prediction file. |
| `--val-split` | | `"val"` | Name of validation split in `--predictions`. |
| `--threshold-strategy` | `-s` | `"f1"` | Optimization criterion (`f1`, `accuracy`, `youden`). |
| `--default-threshold` | `-t` | `0.5` | Fallback threshold when validation split is absent. |
| `--eval-splits` | | `None` | Specific splits to include in overall metrics. |
| `--output-json` | `-j` | `None` | Path to save machine-readable JSON report. |
| `--output-md` | `-o` | `None` | Path to save human-readable Markdown report. |
| `--run-name` | `-n` | Stem of file | Identifier for this evaluation run. |
| `--print-markdown` | | `False` | Print full markdown document to stdout. |
| `--quiet` | `-q` | `False` | Suppress standard stdout summary. |

---

## 5. Machine-Readable JSON Schema

The exported JSON report adheres to the following structure:

```json
{
  "overall": {
    "auroc": 0.9412,
    "accuracy": 0.8850,
    "f1": 0.8795,
    "precision": 0.8920,
    "recall": 0.8675,
    "threshold": 0.5420,
    "threshold_source": "val_optimal_f1",
    "confusion_matrix": {
      "tp": 867,
      "fp": 105,
      "tn": 903,
      "fn": 125
    }
  },
  "by_split": {
    "in_domain_test": { "auroc": 0.9850, "accuracy": 0.9400, "f1": 0.9380, ... },
    "cross_generator_ood": { "auroc": 0.8920, "accuracy": 0.8300, "f1": 0.8210, ... }
  },
  "by_generator": {
    "sd14": { "auroc": 0.9850, "accuracy": 0.9400, ... },
    "midjourney": { "auroc": null, "accuracy": 0.8400, "recall": 0.8400, ... }
  },
  "by_dataset": {
    "genimage": { ... }
  },
  "threshold_metadata": {
    "threshold": 0.5420,
    "strategy": "f1",
    "threshold_source": "val_optimal_f1",
    "calibrated": true,
    "val_split_name": "val",
    "val_samples_count": 1000,
    "default_threshold": 0.5,
    "invariant_preserved": true
  },
  "run_metadata": {
    "run_name": "clip_vit_b16_sd14",
    "timestamp": "2026-09-18T17:00:00.000000+00:00",
    "total_samples": 3000,
    "evaluated_samples": 2000,
    "splits": ["cross_generator_ood", "in_domain_test", "val"],
    "generators": ["adm", "midjourney", "nature", "sd14"],
    "datasets": ["genimage"]
  }
}
```

---

## 6. Single-Class Subset Handling

When evaluating generator-specific slices containing exclusively fake images (e.g., `midjourney` samples without matched real images):
- **AUROC:** Mathematically undefined for single-class distributions; safely reported as `None` (JSON `null`, Markdown `N/A`).
- **Accuracy & Recall:** Accurately measure operational detection performance (e.g. proportion of fake images successfully flagged above $\tau^*$).
- The runner never raises an unhandled exception or crashes on single-class slices.
