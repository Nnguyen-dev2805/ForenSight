# 2026-09-26 R2 Evaluation & Alignment Fixes Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Khắc phục 5 điểm khiếm khuyết được phát hiện trong đợt rà soát độc lập (generator slicing class balance, TPR sample guard, EER zero-crossing, spatial alignment, checkpoint docstring) nhằm đảm bảo tính toàn vẹn khoa học và độ chính xác toán học của ForenSight R2.

**Architecture:** Plain PyTorch, explicit metric computation, deterministic spatial cropping across multi-modal branches, adhering strictly to AGENTS.md.

**Tech Stack:** Python 3.14, PyTorch, torchvision, scikit-learn, numpy, pytest.

---

### Task 1: Ghép Real theo Split cho lát cắt Generator trong `runner.py`

**Files:**
- Modify: `src/forensight/evaluation/runner.py:851-871`
- Test: `tests/evaluation/test_runner.py`

- [ ] **Step 1: Write unit test verifying generator slice only pairs with reals in same split**
- [ ] **Step 2: Run test to observe failure**
- [ ] **Step 3: Update `runner.py` to filter `matching_reals` by generator splits**
- [ ] **Step 4: Run pytest on `tests/evaluation/test_runner.py` to verify pass**

---

### Task 2: Bổ sung Sample Size Guard cho `calculate_tpr_at_fpr` trong `metrics.py`

**Files:**
- Modify: `src/forensight/evaluation/metrics.py:246-265`
- Test: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Update unit tests in `test_metrics.py` to verify $N_{\text{neg}} < \lceil 1/\text{target\_fpr} \rceil$ returns None**
- [ ] **Step 2: Update `calculate_tpr_at_fpr` with sample size guard**
- [ ] **Step 3: Run pytest on `tests/evaluation/test_metrics.py` to verify pass**

---

### Task 3: Nội suy tuyến tính Zero-crossing cho `calculate_eer` trong `metrics.py`

**Files:**
- Modify: `src/forensight/evaluation/metrics.py:266-285`
- Test: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Write test for exact continuous EER calculation**
- [ ] **Step 2: Implement zero-crossing linear interpolation in `calculate_eer`**
- [ ] **Step 3: Run pytest on `tests/evaluation/test_metrics.py` to verify pass**

---

### Task 4: Đồng bộ Crop không gian giữa 2 nhánh trong `r2_dataset.py`

**Files:**
- Modify: `src/forensight/data/r2_dataset.py:45-63`
- Test: `tests/data/test_r2_dataset.py`

- [ ] **Step 1: Update `build_forensic_input_transform` to use `CenterCrop(image_size)` for both train and eval**
- [ ] **Step 2: Run pytest on `tests/data/test_r2_dataset.py` to verify pass**

---

### Task 5: Đồng bộ Docstring và Checkpoint Criterion trong `train_r2.py`

**Files:**
- Modify: `scripts/train_r2.py:1-10`

- [ ] **Step 1: Update module docstring in `scripts/train_r2.py`**
- [ ] **Step 2: Run canonical smoke test to verify entire pipeline**
