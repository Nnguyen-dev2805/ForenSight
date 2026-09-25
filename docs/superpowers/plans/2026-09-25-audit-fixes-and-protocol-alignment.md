# Implementation Plan: Audit Fixes & Protocol Alignment

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all 8 verified audit findings across the ForenSight evaluation engine, leakage auditor, repeated-run aggregator, per-generator AUROC computation, and Kaggle runtime protocol alignment to restore research integrity and reproducibility.

**Architecture:** Strengthen core R0 invariants across `src/forensight/data/audit.py`, `src/forensight/evaluation/runner.py`, `src/forensight/evaluation/aggregate.py`, and `scripts/evaluate_r2.py`. Eliminate silent data contamination and fallback distortions, align per-generator AUROC calculations with standard forensic benchmarks (pairing generator fakes with authentic reference images), and standardize Kaggle execution provenance.

**Tech Stack:** Python 3.10+, PyTorch, scikit-learn, numpy, pytest.

---

## User Review Required

> [!IMPORTANT]
> **Key Architectural Decisions in this Plan:**
> 1. **Per-Generator AUROC Calculation (Issue 5):** Currently, `predictions.get_generator("midjourney")` yields 100% fake samples because real images have `generator="nature"`. We will update `evaluate_predictions()` so that when calculating per-generator slices, each fake generator `g` is evaluated against the shared authentic/nature images from the same evaluation scope, restoring standard dual-class AUROC calculation.
> 2. **Pairwise Leakage Audit (Issue 3):** `audit_manifest_leakage` will now enforce `eval_i ∩ eval_j = ∅` across all evaluation manifests (e.g. `val ↔ test` and `val ↔ ood`), failing the audit if validation samples appear in test sets.
> 3. **Corrupt Image Policy (Issue 7):** Replacing corrupted images with flat gray images silently distorts NPR residuals ($I - \text{resize}(I) = 0$). We will replace silent substitution with fail-fast validation and structured corruption logging.

---

## Proposed Changes

### Component 1: Evaluation Runner & CLI Threshold Fixes (Issues 1 & 2)

#### [MODIFY] `src/forensight/evaluation/runner.py`
- In `threshold_metadata`: Include both `"threshold"` and `"threshold_value"` keys for backward and forward compatibility.
- In `evaluate_predictions()` partition filtering: Ensure `excluded_splits` ALWAYS includes `val_split_name` (and variants `"val"`, `"validation"`), preventing validation records from leaking into `overall_records` even when `val_predictions` is provided separately.
- Guard fallback: If candidate records excluding validation and training are empty, raise an explicit `ValueError` rather than silently contaminating test metrics with training/validation data.

#### [MODIFY] `scripts/evaluate_r2.py`
- Fix lines 189 & 200: Use safe access `report.threshold_metadata.get("threshold_value", report.threshold_metadata.get("threshold"))`.

---

### Component 2: Leakage Audit Comprehensive Split Disjointness (Issue 3)

#### [MODIFY] `src/forensight/data/audit.py`
- In `audit_manifest_leakage()`:
  - Keep the existing `train ↔ eval_i` disjointness checks.
  - Add pairwise `eval_i ↔ eval_j` disjointness checks for all $i \neq j$ in `named_evals` (checking both `sample_id` and resolved `image_path`).
  - Record any `eval_split_collisions` and add clear violations to `violations`.

---

### Component 3: Aggregated Reports Cohort & Manifest Invariants (Issue 4)

#### [MODIFY] `src/forensight/evaluation/aggregate.py`
- In `aggregate_reports()`:
  - Add consistency checks before declaring a benchmark non-preliminary:
    - Check that all reports have identical `split_version` (if specified in metadata).
    - Check that all reports have matching sets of evaluated splits (`by_split.keys()`).
    - If splits or split versions mismatch, flag `is_preliminary = True` and attach an `aggregation_warning` in `metadata`.

---

### Component 4: Per-Generator Evaluation with Reference Class (Issue 5)

#### [MODIFY] `src/forensight/evaluation/runner.py`
- In `evaluate_predictions()` Section 6 (`by_generator`):
  - Identify the set of real/authentic records in the evaluation scope (e.g., `label == 0` or `generator in NON_GENERATOR_LABELS`).
  - For each synthetic generator $g$: construct a binary evaluation subset consisting of all fake records for $g$ plus all real records from the relevant evaluation context.
  - Compute AUROC and classification metrics on this balanced/semi-balanced binary distribution, restoring meaningful per-generator AUROC values.

---

### Component 5: Kaggle Pipeline Data Hygiene & Provenance (Issues 6, 7, 8)

#### [MODIFY] `deploy/kaggle/main.py` & `deploy/kaggle_eval/main.py`
- In `FastImageDataset.__getitem__`:
  - Remove silent replacement of corrupt images with gray RGB `(128, 128, 128)`.
  - Validate image integrity; log warning and skip corrupted images during manifest preparation, or raise an explicit error.
- In report generation:
  - Integrate `ReproducibilityRecord` schema into output JSON: record `git_commit`, `seed`, `config`, `split_version`, `timestamp` (ISO UTC), and hardware/environment information.
- In documentation `results/r2_kaggle/README.md`:
  - Document the protocol relationship: Kaggle runs serve as fast empirical exploration (ViT-B/32 on Tesla T4), whereas Canonical R2 represents the sealed research protocol (ViT-L/14 on SD1.4 train manifest).

---

## Tasks

### Task 1: Fix Threshold Key Mismatch & Validation Partition Contamination
**Files:**
- Modify: `src/forensight/evaluation/runner.py`
- Modify: `scripts/evaluate_r2.py`
- Test: `tests/evaluation/test_runner.py`

- [x] **Step 1: Write failing tests for threshold key alias and validation partition contamination**
  - Add test asserting `threshold_metadata` contains `"threshold_value"`.
  - Add test asserting that when `val_predictions` is passed separately and `predictions` contains records with `split="val"`, those records are strictly excluded from `overall_result`.
- [x] **Step 2: Run pytest to confirm tests fail**
  - `.venv/bin/pytest tests/evaluation/test_runner.py -k "threshold_value or separate_val_leak"`
- [x] **Step 3: Implement fixes in `runner.py` and `scripts/evaluate_r2.py`**
  - Add `"threshold_value": float(tau_star)` to `threshold_metadata`.
  - Update exclusion logic: `excluded_splits.add(val_split_name)` unconditionally.
  - Update `scripts/evaluate_r2.py` to use safe dictionary access.
- [x] **Step 4: Run pytest to confirm tests pass**
  - `.venv/bin/pytest tests/evaluation/test_runner.py tests/training/test_r2.py`

---

### Task 2: Implement Pairwise Leakage Audit Across Evaluation Splits
**Files:**
- Modify: `src/forensight/data/audit.py`
- Test: `tests/data/test_audit.py`

- [x] **Step 1: Write failing test in `test_audit.py`**
  - Create test where `train` has no overlap, but `val` and `test` share a `sample_id` or `image_path`.
  - Assert that `audit_manifest_leakage()` reports `has_leakage=True` with violation details.
- [x] **Step 2: Run pytest to confirm test fails**
  - `.venv/bin/pytest tests/data/test_audit.py -k "eval_overlap"`
- [x] **Step 3: Implement pairwise evaluation split audit in `src/forensight/data/audit.py`**
  - Iterate over pairs of manifests in `named_evals` and check for intersecting IDs and paths.
- [x] **Step 4: Run pytest to confirm test passes**
  - `.venv/bin/pytest tests/data/test_audit.py`

---

### Task 3: Strengthen Repeated-Run Aggregation Invariants
**Files:**
- Modify: `src/forensight/evaluation/aggregate.py`
- Test: `tests/evaluation/test_aggregate.py`

- [x] **Step 1: Write failing test in `test_aggregate.py`**
  - Attempt to aggregate 3 reports with mismatched `split_version` or disparate split names.
  - Assert that `aggregate_reports()` flags `is_preliminary=True` and populates warning metadata.
- [x] **Step 2: Run pytest to confirm test fails**
  - `.venv/bin/pytest tests/evaluation/test_aggregate.py -k "mismatched_cohort"`
- [x] **Step 3: Implement validation in `src/forensight/evaluation/aggregate.py`**
  - Enforce cohort consistency checks.
- [x] **Step 4: Run pytest to confirm test passes**
  - `.venv/bin/pytest tests/evaluation/test_aggregate.py`

---

### Task 4: Fix Per-Generator AUROC Calculation
**Files:**
- Modify: `src/forensight/evaluation/runner.py`
- Test: `tests/evaluation/test_runner.py`

- [x] **Step 1: Write failing test in `test_runner.py`**
  - Create a `PredictionSet` with real images (`generator="nature"`, `label=0`) and fake images (`generator="midjourney"`, `label=1`).
  - Assert that `report.by_generator["midjourney"].auroc` is NOT None and is a valid float $\in [0, 1]$.
- [x] **Step 2: Run pytest to confirm test fails**
  - `.venv/bin/pytest tests/evaluation/test_runner.py -k "per_generator_auroc"`
- [x] **Step 3: Implement generator + authentic pairing in `runner.py`**
  - Extract authentic images in evaluation scope and combine them with fake images for each generator slice.
- [x] **Step 4: Run pytest to confirm test passes**
  - `.venv/bin/pytest tests/evaluation/test_runner.py`

---

### Task 5: Clean up Kaggle Corrupt Image Handling & Inject Full Provenance
**Files:**
- Modify: `deploy/kaggle/main.py`
- Modify: `deploy/kaggle_eval/main.py`
- Modify: `results/r2_kaggle/README.md`
- Test: `tests/system/test_r2_smoke.py`

- [x] **Step 1: Update `FastImageDataset` in Kaggle scripts**
  - Remove silent replacement with `Image.new("RGB", (224, 224), (128, 128, 128))`.
  - Validate images during dataset preparation; skip unreadable files and log skipped count.
- [x] **Step 2: Add full reproducibility schema to Kaggle report output**
  - Record environment info, commit, seed, config, and timestamp.
- [x] **Step 3: Document Canonical vs Kaggle Protocol in `results/r2_kaggle/README.md`**
  - Add explicit note explaining the protocol mapping (ViT-B/32 on Tesla T4 exploration vs ViT-L/14 canonical spec).


---

## Verification Plan

### Automated Tests
1. **Full Pytest Suite:**
   ```bash
   .venv/bin/pytest
   ```
   Must pass 100% (279+ tests passing, 0 failures).

2. **Leakage & Bias Audit Verification:**
   ```bash
   .venv/bin/python scripts/check_dataset.py --manifest data/manifests/manifest_smoke.jsonl
   ```

3. **Canonical Smoke Test Pipeline:**
   ```bash
   .venv/bin/python scripts/run_smoke_test.py
   ```

### Manual Verification
- Verify that `scripts/evaluate_r2.py --help` runs without import or syntax errors.
- Verify that `results/r2_kaggle/reports/logo_threshold_comparison_report.json` and `README.md` are accurately documented.
