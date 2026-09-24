# ForenSight R2 — Multi-Generator & Leave-One-Out Expansion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the Milestone R2 empirical evaluation from single-generator training (SD 1.5 only) to (1) **Leave-One-Generator-Out (LOGO: Train on 6, Test on 1 unseen)** and (2) **All-In-One Multi-Generator Training (Train on all 7, Test on all 7)** to establish the multi-source generalization capability and empirical upper bound of the R2 architecture.

**Architecture:** Frozen CLIP ViT-B/32 semantic branch + Trainable NPR ResNet18 forensic branch + Concat MLP classifier (512 $\to$ 128 $\to$ 1), calibrated with validation-only threshold $\tau^*$.

**Tech Stack:** PyTorch, torchvision, open_clip_torch, pyarrow, scikit-learn, Kaggle API / GPU (Tesla T4).

**Spec:** `docs/plans/r2-forensic-perception-v1.md`, `docs/roadmap.md` (R2 Gate), and `docs/dataset.md`.

---

## Global Constraints

1. **Zero Generator Leakage:** The held-out test generator in LOGO must NEVER appear in the train or validation partitions.
2. **Strict Threshold Calibration:** Decision threshold $\tau^*$ must be calibrated solely on the validation partition using optimal F1-score and frozen for all test splits.
3. **Fair Sample Budgeting:** Training dataset size must remain balanced across generators (e.g. 500 fake images per generator + 1:1 real images) to isolate the architectural effect of generator diversity from raw dataset scale.
4. **Reproducibility Invariant:** All random operations must use fixed seed (seed = 42). Metrics and confusion matrices must be saved to machine-readable JSON.

---

## 1. Experimental Paradigms

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    3 EXPERIMENTAL PARADIGMS                                      │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Paradigm 1: Single-Source (Completed)                                                            │
│   - Train: SD 1.5 (2,000 fakes + 2,000 reals)                                                    │
│   - Test: In-Domain (SD 1.5: 0.9517) | OOD (6 generators: 0.6435)                                  │
│                                                                                                  │
│ Paradigm 2: Leave-One-Generator-Out (LOGO - This Plan)                                           │
│   - Train: 6 generators (SD 1.5, Wukong, GLIDE, VQDM, BigGAN, ADM) (3,000 fakes + 3,000 reals)    │
│   - Test: Held-out unseen generator (Midjourney: 1,000 images) + In-Domain (Seen 6 generators)       │
│   - Core Question: Does exposure to diverse architectures prevent forensic degradation on OOD?   │
│                                                                                                  │
│ Paradigm 3: All-In-One Upper Bound (This Plan)                                                   │
│   - Train: All 7 generators (3,500 fakes + 3,500 reals = 7,000 images)                           │
│   - Test: All 7 generators in-domain (6,500 test images)                                         │
│   - Core Question: What is the empirical upper bound of R2 when there is zero generator shift?   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. File Structure & Responsibilities

| File | Responsibility |
|---|---|
| `src/forensight/data/tiny_genimage.py` | Core library: partitioning functions for LOGO (`build_logo_splits`) and All-In-One (`build_all_in_one_splits`). |
| `tests/data/test_tiny_genimage.py` | Unit tests: verifying zero leakage, generator disjointness, and correct record counts for new partition modes. |
| `deploy/kaggle/main.py` | Kaggle GPU script: CLI flags `--train-mode [single, logo, all_in_one]` and `--leave-out [midjourney, biggan, ...]`. |
| `scripts/run_on_kaggle.py` | Deployment tool: packaging, kernel pushing, status tracking, and chunked result downloading. |
| `results/r2_kaggle/` | Output storage: `logo_report.json`, `all_in_one_report.json`, and checkpoints. |

---

## Tasks Breakdown

### Task 1: Implement Multi-Generator Partition Functions in `src/forensight/data/tiny_genimage.py`

**Files:**
- Modify: `src/forensight/data/tiny_genimage.py`
- Test: `tests/data/test_tiny_genimage.py`

**Interfaces:**
- `build_logo_splits(records: list[dict], leave_out_gen: str = "midjourney", n_train_per_gen: int = 500, seed: int = 42) -> dict[str, list[dict]]`
- `build_all_in_one_splits(records: list[dict], n_train_per_gen: int = 500, seed: int = 42) -> dict[str, list[dict]]`

- [ ] **Step 1: Write failing unit test for `build_logo_splits` and `build_all_in_one_splits`**
  Add tests in `tests/data/test_tiny_genimage.py` asserting:
  - In LOGO mode, the `leave_out_gen` appears ONLY in `test_<leave_out_gen>` and NOT in `train` or `val`.
  - In All-In-One mode, all 7 generators are present in `train`, `val`, and individual `test_<gen>` splits.
  - Class balance (1:1 Fake:Real) is strictly preserved in all splits.

- [ ] **Step 2: Run pytest to confirm failure**
  ```bash
  .venv/bin/pytest tests/data/test_tiny_genimage.py -k "test_build_logo_splits or test_build_all_in_one_splits"
  ```

- [ ] **Step 3: Implement partition builders in `src/forensight/data/tiny_genimage.py`**
  Implement clean list-comprehension filtering with deterministic shuffling and no sample overlap between train, val, and test.

- [ ] **Step 4: Run pytest and confirm all tests pass**
  ```bash
  .venv/bin/pytest tests/data/test_tiny_genimage.py
  ```

- [ ] **Step 5: Git commit**
  ```bash
  git commit -m "feat(data): add LOGO and all-in-one partition builders for Tiny-GenImage"
  ```

---

### Task 2: Upgrade Kaggle GPU Execution Script (`deploy/kaggle/main.py`)

**Files:**
- Modify: `deploy/kaggle/main.py`

**Interfaces:**
- Add command-line arguments:
  - `--mode`: Choice of `"single"`, `"logo"`, `"all_in_one"`, or `"comprehensive"` (default: runs both LOGO and All-In-One sequentially).
  - `--leave-out`: Generator to hold out for LOGO (default: `"midjourney"`).
  - `--epochs`: Number of training epochs per run (default: `5`).
  - `--batch-size`: Batch size (default: `32`).

- [ ] **Step 1: Refactor `prepare_tiny_genimage` to accept mode and leave-out parameters**
  Allow dynamic partitioning into either:
  - LOGO splits (`train_logo`, `val_logo`, `test_held_out`, `test_in_domain_seen`).
  - All-In-One splits (`train_all`, `val_all`, `test_sd15`, `test_midjourney`, ..., `test_combined`).

- [ ] **Step 2: Add execution routines for LOGO and All-In-One**
  - Train R2 Fusion on the multi-generator train partition.
  - Calibrate threshold $\tau^*$ on multi-generator validation partition.
  - Evaluate across all relevant test sets.
  - Export structured `logo_report.json` and `all_in_one_report.json`.

- [ ] **Step 3: Verify local syntax and mock execution**
  ```bash
  python3 -m py_compile deploy/kaggle/main.py
  ```

- [ ] **Step 4: Git commit**
  ```bash
  git commit -m "feat(deploy): support multi-generator LOGO and all-in-one modes in Kaggle runner"
  ```

---

### Task 3: Execute on Kaggle GPU & Monitor Run

**Files:**
- Use: `scripts/run_on_kaggle.py`

- [ ] **Step 1: Push upgraded kernel to Kaggle**
  ```bash
  .venv/bin/python scripts/run_on_kaggle.py --push-only
  ```

- [ ] **Step 2: Monitor execution status until completion**
  ```bash
  .venv/bin/python scripts/run_on_kaggle.py --status-only
  ```

- [ ] **Step 3: Download reports and checkpoints to `results/r2_kaggle/`**
  ```bash
  .venv/bin/python scripts/run_on_kaggle.py --download-only
  ```

- [ ] **Step 4: Verify downloaded artifacts**
  Ensure `logo_report.json` and `all_in_one_report.json` are present and contain valid AUROC, Accuracy, F1, and confusion matrices.

---

### Task 4: Comparative Scientific Analysis & Synthesis

**Files:**
- Update: `results/r2_kaggle/` analysis
- Update: `walkthrough.md`

- [ ] **Step 1: Construct Comparative Benchmark Table**
  Compare AUROC across all 3 paradigms:
  1. *Single-Source (SD 1.5 Train)*
  2. *Multi-Source LOGO (6 Gens Train, Midjourney Test)*
  3. *All-In-One Upper Bound (All 7 Gens Train)*

- [ ] **Step 2: Analyze Scientific Questions**
  - **Does multi-generator training fix the Midjourney gap?** Compare Midjourney AUROC when trained on SD 1.5 (0.8084) vs trained on 6 diverse generators.
  - **What is the empirical upper bound?** How close to 1.0 does R2 reach across all 7 generators in-domain?
  - **Did BigGAN and ADM improve under in-domain training?** If BigGAN AUROC rises to $>0.90$ in All-In-One, its previous 0.48 was an OOD failure; if it stays low, NPR ResNet18 has a structural blindness to GAN artifacts.

- [ ] **Step 3: Update Walkthrough & Commit**
  Update `walkthrough.md` with final multi-generator tables and conclusions.
