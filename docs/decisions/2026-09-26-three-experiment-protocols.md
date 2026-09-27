# Decision: Three Training and Evaluation Protocols on Kaggle Tiny-GenImage

Date: 2026-09-26
Status: Accepted

## Context

Following the adoption of the 7-generator Kaggle Tiny-GenImage dataset (`yangsangtai/tiny-genimage`), ForenSight requires clear, reproducible experimental regimes to address cross-generator generalization while maintaining rigorous research integrity.

Training strictly on a single generator (`sd15`) addresses the core RQ1 hypothesis (whether dual-branch semantic + forensic representations provide complementary inductive biases for unseen detection). However, single-generator training cannot answer whether multi-generator exposure improves unseen transfer, nor does it quantify the representation capacity ceiling when all target generator architectures are observed.

Conversely, conflating multi-generator training with unseen cross-generator generalization would violate research integrity rules defined in `AGENTS.md` and `docs/research-problem.md`.

## Decision

ForenSight formally defines and supports three distinct, standardized experiment protocols on Kaggle Tiny-GenImage:

### 1. `single` (Single-Generator Unseen Cross-Generator Protocol)
- **Role:** Canonical Stage-1 / RQ1 minimal ablation protocol.
- **Train split:** SD1.5 train partition only (balanced 1:1 real/fake).
- **Validation split:** Held-out SD1.5 validation partition (used solely for early stopping and decision threshold calibration $\tau^*$).
- **In-domain test split:** Disjoint held-out SD1.5 test partition.
- **Out-of-distribution (OOD) test splits:** 6 completely unseen generator architectures (`adm`, `biggan`, `glide`, `midjourney`, `vqdm`, `wukong`).
- **Research Claim:** Unseen cross-generator generalization from a single diffusion source.

### 2. `logo` (Leave-One-Generator-Out Protocol)
- **Role:** Cross-generator transfer under multi-source training across 7 exhaustive folds.
- **Folds:** 7 folds corresponding to each generator (`leave_sd15`, `leave_adm`, `leave_biggan`, `leave_glide`, `leave_midjourney`, `leave_vqdm`, `leave_wukong`).
- **Train split:** Pooled train partitions of the 6 seen generators (1:1 real/fake).
- **Validation split:** Pooled held-out validation partitions from the 6 seen generators (for checkpoint selection and threshold calibration).
- **Test split:** 1 completely unseen held-out generator (1:1 real/fake).
- **Research Claim:** Unseen generator generalization given diverse multi-generator training distributions.

### 3. `all7` (Seen-Generator Multi-Generator Upper Bound Protocol)
- **Role:** Representation capacity upper bound under closed-world generator observation.
- **Train split:** Pooled train partitions across all 7 generators (1:1 real/fake).
- **Validation split:** Pooled held-out validation partitions across all 7 generators.
- **Test splits:** Per-generator held-out test partitions and combined 7-generator test set.
- **STRICT PROTOCOL RULE:** Results from `all7` must **never** be labeled, reported, or claimed as "unseen cross-generator generalization". It is strictly an empirical upper-bound benchmark for representation capacity when all target generators are known at training time.

## Manifest Layout & Leakage Invariants

To avoid collision and ensure complete reproducibility, manifests are partitioned into dedicated subdirectories:
- `data/manifests/single/`
- `data/manifests/logo/leave_<gen>/`
- `data/manifests/all7/`

### Leakage Invariants:
1. **Sample & Path Disjointness:** No image path or sample ID may appear in more than one partition (train, val, test) within an experiment protocol.
2. **Partitioning Real Images Without Replacement:** Evaluation real images are partitioned monotonically without replacement across validation, in-domain test, and all OOD generator test sets. No real sample is ever reused across evaluation splits.
3. **Deterministic Generation:** All splits and random subsampling are deterministically seeded (default `seed=42`).
4. **Unified Training Pipeline:** All three protocols share the identical model implementation (`src/forensight/models/`) and training engine (`src/forensight/training/r2.py`), differing only by manifest inputs and CLI flags.

## Consequences

- Manifest generator `src/forensight/data/tiny_genimage.py` provides dedicated builders (`build_single_generator_manifests`, `build_logo_manifests`, `build_all_logo_manifests`, `build_all7_manifests`, `build_all_experiment_manifests`).
- Training and evaluation entry points (`scripts/train_r2.py`, `scripts/evaluate_r2.py`, `deploy/kaggle/main.py`) support `--experiment {single,logo,all7}` and `--leave-out <gen>`.
- Any comparison across protocols must acknowledge the difference between single-source OOD transfer, multi-source LOGO transfer, and closed-world multi-generator training.
