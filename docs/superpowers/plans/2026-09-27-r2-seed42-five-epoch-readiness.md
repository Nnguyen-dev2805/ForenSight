# R2 Full-Matrix Seed-42 Five-Epoch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one trustworthy, report-ready comparison of Semantic-only, Forensic-only, and Fusion across Single, LOGO leave-Midjourney, and All-7 using seed `42` and exactly `5` epochs.

**Architecture:** Keep the existing R2 architecture and baseline preprocessing unchanged. Fix the data semantics, evaluation cohorts, CLIP pretrained-model compatibility, deterministic execution, and provenance in shared code; then generate canonical manifests and execute the 9-run matrix behind protocol gates.

**Tech Stack:** Python 3.11+, PyTorch, torchvision, OpenCLIP, timm, NumPy, scikit-learn, pytest, Kaggle GPU.

**Spec:** `docs/decisions/2026-09-27-lock-r2-experimental-backbone.md`, `docs/dataset.md`, `docs/evaluation.md`, and `docs/plans/r2-forensic-perception-v1.md`.

## Global Constraints

- Use only seed `42` and exactly `5` training epochs for all three variants.
- Execute exactly `9` runs: Single `3`, LOGO leave-Midjourney `3`, and All-7 `3`.
- Keep baseline preprocessing as `RGB -> Resize(short edge=224) -> CenterCrop(224) -> ToTensor`.
- Do not add JPEG recompression, augmentation, scheduler, early stopping, attention, gating, or new dependencies.
- Keep `AdamW`, batch size `16`, weight decay `1e-4`, and each variant's already-registered learning rate.
- Select checkpoints by validation AUROC and thresholds by validation F1 only.
- Evaluate every test cohort twice: primary threshold `tau*` selected on validation by F1, and secondary fixed threshold `0.5` as a predeclared operational reference.
- Never select between `tau*` and `0.5` using test performance; AUROC, PR-AUC, and EER remain threshold-independent.
- Never tune configuration from in-domain or OOD test results already observed.
- Preserve existing 10-epoch artifacts and label them exploratory; do not overwrite or delete them.
- Do not run additional seeds, JPEG preprocessing experiments, or external benchmarks in this plan.
- Do not commit, push, or publish unless the user separately requests it.

---

## File Structure

- `configs/r2/{semantic,forensic,fusion}.json`: locked seed-42, five-epoch model configurations.
- `src/forensight/data/tiny_genimage.py`: canonical ImageNet class IDs (one shared namespace), evaluation-cohort metadata for Single, LOGO, and All-7, and the expected-generator-set guard on all three builders.
- `src/forensight/data/r2_dataset.py`: expose manifest evaluation metadata to inference.
- `src/forensight/training/r2.py`: deterministic CUDA settings and propagation of evaluation metadata into predictions.
- `src/forensight/evaluation/runner.py`: pair each fake generator with its exact held-out Real cohort.
- `scripts/evaluate_r2.py`: save complete calibrated-threshold and fixed-0.5 evaluation reports from the same prediction scores.
- `src/forensight/models/semantic.py`: canonical OpenCLIP construction; behavior driven by config.
- `scripts/run_on_kaggle.py`: comparable code-payload provenance across variant-specific kernels.
- `docs/decisions/2026-09-27-r2-seed42-five-epoch-reporting-profile.md`: records the reduced reporting scope and limitations.
- `tests/data/test_tiny_genimage.py`: class normalization tests, per-generator cohort-tag tests, and the expected-generator-set guard test.
- `tests/data/test_r2_dataset.py`: metadata propagation test.
- `tests/training/test_r2.py`: deterministic seed and prediction metadata tests.
- `tests/evaluation/test_runner.py`: exact per-generator cohort test.
- `tests/system/test_kaggle_official.py`: seed/epoch/provenance bundle contract.
- `reports/r2_seed42_5epoch_full_matrix.md`: final preliminary comparison for the course report.

---

### Task 1: Lock the Seed-42 Five-Epoch Reporting Profile

**Files:**
- Modify: `configs/r2/semantic.json`
- Modify: `configs/r2/forensic.json`
- Modify: `configs/r2/fusion.json`
- Create: `docs/decisions/2026-09-27-r2-seed42-five-epoch-reporting-profile.md`
- Modify: `docs/decisions/README.md`
- Test: `tests/training/test_r2.py`

**Interfaces:**
- Consumes: existing R2 JSON configuration schema loaded by `load_r2_config(path)`.
- Produces: three configs with `seed=42`, `training.epochs=5`; Semantic and Fusion use `clip_model="ViT-L-14-quickgelu"` with `clip_pretrained="openai"`.

- [x] **Step 1: Write a failing configuration-lock test**

```python
def test_report_configs_lock_seed42_five_epochs_and_openai_quickgelu():
    expected = {
        "semantic": (42, 5, "ViT-L-14-quickgelu"),
        "forensic": (42, 5, "ViT-L-14"),
        "fusion": (42, 5, "ViT-L-14-quickgelu"),
    }
    for variant, (seed, epochs, clip_model) in expected.items():
        config = load_r2_config(Path("configs/r2") / f"{variant}.json")
        assert config["seed"] == seed
        assert config["training"]["epochs"] == epochs
        assert config["model"]["clip_model"] == clip_model
```

- [x] **Step 2: Run the test and confirm the current 10-epoch configs fail**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/training/test_r2.py::test_report_configs_lock_seed42_five_epochs_and_openai_quickgelu -q`

Expected: FAIL because epochs are currently `10` and Semantic/Fusion use `ViT-L-14`.

- [x] **Step 3: Apply the minimal config changes**

Set the following exact values:

```json
{
  "seed": 42,
  "model": {
    "clip_model": "ViT-L-14-quickgelu",
    "clip_pretrained": "openai"
  },
  "training": {
    "epochs": 5
  }
}
```

Keep Forensic's unused `clip_model` value unchanged; it does not instantiate CLIP. Keep all learning rates unchanged: Semantic `1e-3`, Forensic/Fusion `1e-4`.

- [x] **Step 4: Record the reporting profile**

The ADR must state:

```text
Status: Accepted for the course-report run
Protocols: single, LOGO leave-midjourney, all7
Seed: 42
Epochs: 5
Variants: semantic, forensic, fusion
Run count: 9
Claim level: preliminary single-run evidence
Prior 10-epoch results: exploratory only
Test exposure: already observed; no further tuning permitted
Deferred: multi-seed uncertainty, JPEG intervention, external datasets
```

- [x] **Step 5: Run the focused config test**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/training/test_r2.py -q`

Expected: PASS.

---

### Task 2: Canonicalize ImageNet Class IDs ~~and Match Real/Fake Content~~

> **CORRECTED 2026-09-27 — the "match Real/Fake content" half of this task is not
> implementable on this dataset and was rejected.** Measured on the sealed R0 manifests:
>
> - Real and fake class sets overlap at **exactly chance** (ratio 0.94–1.01) in every
>   split and for every generator: there is no real↔fake class pairing to recover.
> - The real/fake class-marginal difference is **sampling noise**: observed total-variation
>   distance sits *below* the null-model mean in all four splits.
> - Class-matching is also arithmetically impossible: `glide` and `vqdm` filenames carry no
>   class field, and `adm` uses a `0` prefix, leaving **0/250** matchable in `val`.
>
> What remained real was a **namespace defect**: real IDs were `n01440764` while fake IDs
> were `c0001`, so `class_balance` warned on a dataset with no such problem. The fix is to
> normalize both into one namespace — which provably cannot change any split, since
> `class_id` feeds only grouping and the audit, never allocation.
>
> See `docs/decisions/2026-09-27-r2-seed42-five-epoch-reporting-profile.md` for the full
> evidence table. Steps 5 and 7 below are **withdrawn**; Steps 1, 4 and 6 stand.

**Files:**
- Modify: `src/forensight/data/tiny_genimage.py:137`
- Modify: `src/forensight/data/tiny_genimage.py:441`
- Test: `tests/data/test_tiny_genimage.py`

**Interfaces:**
- Produces: `_class_id_from_path(path: Path) -> str | None` returning ImageNet WNIDs for both Real and Fake filenames.
- Produces: `_select_class_matched_reals(fake_records, real_records, *, seed) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]` returning selected Real records and the unused Real pool.
- Consumes: `timm.data.imagenet_info.ImageNetInfo`, already installed through OpenCLIP.

- [x] **Step 1: Write failing class-normalization tests**

```python
def test_fake_numeric_class_id_maps_to_imagenet_wnid():
    assert _class_id_from_path(
        Path("imagenet_ai_0424_sdv5/train/ai/001_sdv5_00094.png")
    ) == "n01440764"


def test_real_wnid_class_id_is_preserved():
    assert _class_id_from_path(
        Path("imagenet_ai_0419_biggan/train/nature/n01440764_5969.JPEG")
    ) == "n01440764"
```

- [~] WITHDRAWN — **Step 2: Write a failing class-matched allocation test**

Create two fake classes with counts `2` and `1`, and a larger Real pool containing those classes plus an unrelated class. Assert:

```python
selected, remaining = _select_class_matched_reals(fakes, reals, seed=42)
assert Counter(r["class_id"] for r in selected) == Counter(r["class_id"] for r in fakes)
assert {r["sample_id"] for r in selected}.isdisjoint(
    {r["sample_id"] for r in remaining}
)
```

- [x] **Step 3: Run the tests and verify they fail**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/data/test_tiny_genimage.py -k 'class_id or class_matched' -q`

Expected: numeric fake classes currently return `c0001`, and no class-matched selector exists.

- [x] **Step 4: Implement canonical WNID conversion using the installed dependency**

```python
from functools import lru_cache


@lru_cache(maxsize=1)
def _imagenet_wnids() -> tuple[str, ...]:
    from timm.data.imagenet_info import ImageNetInfo
    return tuple(ImageNetInfo("imagenet-1k").label_names())


def _class_id_from_path(path: Path) -> str | None:
    stem = path.stem
    if stem.startswith("n") and len(stem) > 9 and stem[1:9].isdigit():
        return stem.split("_")[0]
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].isdigit():
        one_based_index = int(parts[0])
        if 1 <= one_based_index <= 1000:
            return _imagenet_wnids()[one_based_index - 1]
    return None
```

- [~] WITHDRAWN — **Step 5: Implement deterministic class-matched Real selection**

Group Real records by canonical `class_id`, shuffle within each class using `random.Random(seed)`, and select exactly the fake count for each class. Raise `ValueError` naming the class when insufficient Real samples exist. Return unused Real records for the next disjoint allocation.

- [~] WITHDRAWN — **Step 6: Use class-matched allocation in all three protocol builders**

Apply it to every Real/Fake cohort produced by:

```text
train SD1.5 fake -> train Real
validation SD1.5 fake -> validation Real
in-domain SD1.5 fake -> in-domain Real
each OOD generator fake cohort -> that generator's Real reference cohort
each LOGO seen-generator train/validation/in-domain cohort
each LOGO held-out-generator test cohort
each All-7 train/validation/per-generator test cohort
```

Maintain no-replacement allocation across validation and all test cohorts.

- [~] WITHDRAWN — **Step 7: Add manifest assertions**

For every Single, LOGO, and All-7 split/cohort, assert Real and Fake `Counter(class_id)` values are equal. Keep existing sample/path/generator leakage assertions.

- [x] **Step 8: Run the data tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/data/test_tiny_genimage.py tests/data/test_audit.py -q`

Expected: PASS; class-balance audit no longer reports synthetic `c####` and Real `n########` as different classes.

---

### Task 3: Preserve Exact Per-Generator Real Cohorts Through Evaluation

**Files:**
- Modify: `src/forensight/data/tiny_genimage.py:517`
- Modify: `src/forensight/data/r2_dataset.py:84`
- Modify: `src/forensight/training/r2.py:315`
- Modify: `src/forensight/evaluation/runner.py:850`
- Test: `tests/data/test_r2_dataset.py`
- Test: `tests/training/test_r2.py`
- Test: `tests/evaluation/test_runner.py`
- Test: `tests/evaluation/test_artifact_contract.py`

**Interfaces:**
- Manifest metadata key: `evaluation_generator: str`.
- Dataset sample key: `evaluation_generator: str`.
- Prediction metadata key: `evaluation_generator: str`.
- Evaluator pairs fake generator `g` only with Real records whose metadata has `evaluation_generator == g`.

- [x] **Step 1: Write a failing manifest-cohort test**

Assert `test_adm` contains 500 Fake ADM and 500 Real records, and every record has:

```python
record.metadata["evaluation_generator"] == "adm"
```

- [x] **Step 2: Write failing metadata propagation tests**

Assert `R2ImageDataset[0]["evaluation_generator"]` reaches:

```python
prediction.metadata["evaluation_generator"]
```

after `predict_to_prediction_set()`.

- [x] **Step 3: Write a failing exact-cohort evaluator test**

Construct predictions for two fake generators, each with two tagged Real references. Assert each generator's confusion-matrix total equals `4`, rather than using all four Real samples and producing `6`.

- [x] **Step 4: Run the failing tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/data/test_r2_dataset.py tests/training/test_r2.py tests/evaluation/test_runner.py -q`

Expected: FAIL because evaluation metadata is currently discarded and the evaluator pools all OOD Real samples.

- [x] **Step 5: Attach cohort metadata during manifest construction**

For each generated cohort, write:

```python
metadata={"evaluation_generator": gen_id}
```

Use `sd15` for Single validation/in-domain records, the held-out generator ID for Single OOD and LOGO held-out pairs, and the evaluated generator ID for each All-7 per-generator pair.

- [x] **Step 6: Propagate the single string through dataset and prediction records**

```python
sample["evaluation_generator"] = str(
    record.metadata.get("evaluation_generator", "")
)
```

and:

```python
metadata={
    "evaluation_generator": str(batch["evaluation_generator"][index])
}
```

- [x] **Step 7: Prefer exact Real cohorts in the evaluator**

```python
matching_reals = [
    record for record in eval_reals
    if record.metadata.get("evaluation_generator") == g
]
```

Raise a clear error for official Tiny-GenImage predictions when a fake generator lacks tagged Real references. Retain the existing same-split fallback only for legacy/synthetic prediction sets without cohort metadata.

- [x] **Step 8: Run the focused tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/data/test_r2_dataset.py tests/training/test_r2.py tests/evaluation/test_runner.py -q`

Expected: PASS; each OOD generator report contains exactly 500 Real + 500 Fake in the actual run.

- [x] **Step 9: Write a failing dual-threshold artifact test**

Run one synthetic evaluation and require both files:

```text
evaluation.json             # tau* selected from validation by F1
evaluation_fixed_0_5.json   # fixed threshold 0.5
```

Assert both reports use identical score/sample cohort hashes, while their threshold metadata is respectively `val_optimal_f1` and `default_0.5`.

- [x] **Step 10: Reuse the evaluator for the fixed-threshold report**

After producing and saving the calibrated report, create a fresh `PredictionSet` copy containing test records only and call `evaluate_predictions()` without a validation cohort and with `default_threshold=0.5`. Save the complete JSON and Markdown outputs as:

```text
evaluation_fixed_0_5.json
evaluation_fixed_0_5.md
```

The fresh copy prevents fixed-threshold evaluation from overwriting the calibrated binary decisions stored in the main predictions artifact.

- [x] **Step 11: Run dual-threshold tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/evaluation/test_runner.py tests/evaluation/test_artifact_contract.py -q`

Expected: PASS; both reports contain overall, per-split, and exact per-generator metrics from the same scores.

---

### Task 4: Make Seed 42 Reproducible and Variant Provenance Comparable

**Files:**
- Modify: `src/forensight/training/r2.py:90`
- Modify: `scripts/run_on_kaggle.py:128`
- Test: `tests/training/test_r2.py`
- Test: `tests/system/test_kaggle_official.py`

**Interfaces:**
- `set_seed(42)` sets Python, NumPy, PyTorch, CUDA, and cuDNN deterministic flags.
- Bundle output includes `code_payload_sha256`, computed only from canonical `src/`, selected `scripts/`, and `configs/`, excluding kernel metadata and run-specific manifests.

- [x] **Step 1: Write a failing CUDA-determinism unit test**

Mock CUDA availability and assert:

```python
set_seed(42)
assert torch.backends.cudnn.deterministic is True
assert torch.backends.cudnn.benchmark is False
```

- [x] **Step 2: Write a failing code-payload provenance test**

Stage Semantic and Forensic bundles from the same checkout and assert:

```python
semantic["code_payload_sha256"] == forensic["code_payload_sha256"]
semantic["source_payload_sha256"] != forensic["source_payload_sha256"]
```

The second assertion confirms run/kernel metadata may differ while canonical code remains identical.

- [x] **Step 3: Implement deterministic cuDNN flags**

```python
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Do not enable `torch.use_deterministic_algorithms(True)` in this reporting cycle because it can reject supported CUDA operations; record this limitation in the ADR.

- [x] **Step 4: Add a code-only bundle digest**

Hash relative paths and bytes for:

```text
src/**
configs/**
scripts/check_dataset.py
scripts/train_r2.py
scripts/evaluate_r2.py
scripts/aggregate_runs.py
```

Keep the existing full `source_payload_sha256` unchanged.

- [x] **Step 5: Run focused provenance tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/training/test_r2.py tests/system/test_kaggle_official.py -q`

Expected: PASS.

---

### Task 5: Verify the Fixed Pipeline Before Spending GPU Time

**Files:**
- Modify only if a verification failure exposes a defect in Tasks 1-4.

**Interfaces:**
- Consumes: all fixes from Tasks 1-4.
- Produces: a green local gate; no training result.

- [x] **Step 1: Run targeted tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/data/test_tiny_genimage.py \
  tests/data/test_r2_dataset.py \
  tests/training/test_r2.py \
  tests/evaluation/test_runner.py \
  tests/system/test_kaggle_official.py -q
```

Expected: PASS.

- [x] **Step 2: Run the full suite**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest`

Expected: all tests pass.

- [x] **Step 3: Run protocol and smoke audits**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/audit_all_protocols.py
UV_CACHE_DIR=.uv-cache uv run python scripts/run_smoke_test.py
```

Expected: all protocol invariants pass and smoke reports `7/7` stages passed.

- [x] **Step 4: Check source hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; pre-existing user changes remain preserved and visible.

---

### Task 6: Run the Three Seed-42 Five-Epoch Single Baselines

**Files:**
- Read: `deploy/kaggle_{semantic,forensic}/kernel-metadata.json`
- Read: `deploy/kaggle/kernel-metadata.json`
- Create via Kaggle download: `results/kaggle/*_seed42_5ep/`

**Interfaces:**
- Consumes: one pinned dataset, one canonical source snapshot, one generated manifest set.
- Produces: Semantic, Forensic, and Fusion checkpoints plus evaluation/reproducibility artifacts.

- [ ] **Step 1: Push and watch Semantic-only**

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/run_on_kaggle.py \
  --push --watch \
  --kernel-dir deploy/kaggle_semantic \
  --kernel-slug truongnhatnguyen2805/forensight-r2-semantic-training \
  --output-dir results/kaggle/semantic_seed42_5ep \
  --experiment single \
  --variant semantic_only \
  --seed 42 \
  --epochs 5 \
  --dataset-ref yangsangtai/tiny-genimage/versions/1 \
  --prepare-manifests
```

- [ ] **Step 2: Inspect Semantic artifacts before continuing**

Require: five history rows, checkpoint selected from epochs `1..5`, no leakage failure, no class-balance scheme warning, and six per-generator cohorts of `N=1000`.
Require both `evaluation.json` at validation-selected `tau*` and `evaluation_fixed_0_5.json` at fixed `0.5`.

- [ ] **Step 3: Push and watch Forensic-only**

Use the same command with:

```text
--kernel-dir deploy/kaggle_forensic
--kernel-slug truongnhatnguyen2805/forensight-r2-forensic-training
--output-dir results/kaggle/forensic_seed42_5ep
--variant forensic_only
```

- [ ] **Step 4: Inspect Forensic artifacts before continuing**

Apply the same gates as Semantic. Do not modify its learning rate, NPR transform, or preprocessing based on the newly observed result.

- [ ] **Step 5: Push and watch Fusion**

Use the same command with:

```text
--kernel-dir deploy/kaggle
--kernel-slug truongnhatnguyen2805/forensight-r2-fusion-training
--output-dir results/kaggle/fusion_seed42_5ep
--variant fusion
```

- [ ] **Step 6: Verify cross-run identity**

Across all three artifacts, require identical:

```text
code_payload_sha256
dataset_revision
split_version
train/val/test manifest SHA-256
seed = 42
epochs = 5
```

Allow variant-specific model name and learning rate differences already registered in config.

---

### Task 7: Run the LOGO Leave-Midjourney Case Study

**Files:**
- Create via Kaggle download: `results/kaggle/logo_seed42_5ep/leave_midjourney/<variant>/`

**Interfaces:**
- Consumes: the exact code payload accepted by Task 6.
- Produces: `1 fold x 3 variants = 3` checkpoints and evaluation reports.

- [ ] **Step 1: Freeze Midjourney as the only held-out generator**

Use this exact protocol:

```text
Train/validation generators: sd15, adm, biggan, glide, vqdm, wukong
Held-out test generator: midjourney
```

- [ ] **Step 2: Verify the generated fold before training**

For `leave_midjourney`, require:

```text
midjourney absent from train and val
midjourney present in test_midjourney
train, val, seen in-domain test, and held-out test are sample/path/SHA disjoint
Real/Fake class counters match within every cohort
seed = 42
```

- [ ] **Step 3: Execute the three runs sequentially**

Run Semantic, Forensic, then Fusion using the same command structure as Task 6 with:

```text
--experiment logo
--leave-out midjourney
--seed 42
--epochs 5
--dataset-ref yangsangtai/tiny-genimage/versions/1
--prepare-manifests
```

Use the existing variant-specific kernel pairs:

```text
semantic_only -> deploy/kaggle_semantic -> forensight-r2-semantic-training
forensic_only -> deploy/kaggle_forensic -> forensight-r2-forensic-training
fusion -> deploy/kaggle -> forensight-r2-fusion-training
```

Download each run into:

```text
results/kaggle/logo_seed42_5ep/leave_midjourney/<variant>/
```

- [ ] **Step 4: Apply the LOGO gate**

Do not start All-7 unless all three variants have:

```text
five training-history rows
checkpoint epoch in 1..5
identical code_payload_sha256
identical train/val/test manifest hashes within the fold
validation-only threshold provenance
complete fixed-threshold-0.5 reference report
no held-out generator leakage
```

- [ ] **Step 5: Build the LOGO comparison table**

For each variant on held-out Midjourney, record:

```text
held-out AUROC
held-out PR-AUC
held-out EER
held-out TPR@1%FPR
seen-generator in-domain AUROC
checkpoint epoch
```

Label this result `LOGO leave-Midjourney case study`. Do not call it complete LOGO or generalize its result to all possible held-out generators.

---

### Task 8: Run the All-7 Closed-World Baselines

**Files:**
- Create via Kaggle download: `results/kaggle/all7_seed42_5ep/<variant>/`

**Interfaces:**
- Consumes: the exact code payload accepted by Tasks 6-7.
- Produces: three All-7 checkpoints and per-generator known-distribution reports.

- [ ] **Step 1: Verify the All-7 manifests**

Require all seven generators in train and val, disjoint sample/path/SHA partitions, class-matched Real/Fake cohorts, and a held-out test cohort for every generator.

- [ ] **Step 2: Run Semantic-only, Forensic-only, and Fusion**

Use Task 6 commands with:

```text
--experiment all7
--seed 42
--epochs 5
--dataset-ref yangsangtai/tiny-genimage/versions/1
--prepare-manifests
```

Download to:

```text
results/kaggle/all7_seed42_5ep/semantic/
results/kaggle/all7_seed42_5ep/forensic/
results/kaggle/all7_seed42_5ep/fusion/
```

- [ ] **Step 3: Verify protocol labeling**

Every report must describe All-7 as a closed-world or known-generator upper bound. It must not use `unseen`, `OOD generalization`, or equivalent language for its seven generator results.

- [ ] **Step 4: Verify cross-run identity**

Require all three variants to share code payload, split version, dataset revision, seed 42, five epochs, and the same All-7 manifests.
Require both threshold reports for every variant.

---

### Task 9: Produce the Full Course-Report Comparison

**Files:**
- Create: `reports/r2_seed42_5epoch_full_matrix.md`

**Interfaces:**
- Consumes: 9 verified `evaluation.json`, `train_history.json`, `config.json`, and `reproducibility.json` artifacts.
- Produces: one transparent seed-42 comparison report covering Single, LOGO, and All-7.

- [ ] **Step 1: Verify report eligibility**

Reject any run with mismatched manifests/code payload inside its comparison group, more or fewer than five epochs, test-tuned threshold, missing cohort metadata, or failed leakage audit.

- [ ] **Step 2: Build the Single comparison table**

Include exactly:

```text
Val AUROC
In-domain AUROC
Cross-generator OOD AUROC
OOD PR-AUC
OOD EER
OOD TPR@1%FPR
OOD Accuracy/F1/Precision/Recall at validation-selected threshold
OOD Accuracy/F1/Precision/Recall at fixed threshold 0.5
Checkpoint epoch
```

- [ ] **Step 3: Build the Single six-generator AUROC table**

For ADM, BigGAN, GLIDE, Midjourney, VQDM, and Wukong, report AUROC from the exact 500 Real + 500 Fake cohort. Do not report `TPR@0.1%FPR` per generator because 500 Real samples cannot resolve that operating point reliably.
For each generator, report Accuracy, F1, Precision, Recall, and confusion matrix at both `tau*` and `0.5`.

- [ ] **Step 4: State the allowed conclusions**

Use this exact claim boundary:

```text
These are preliminary results from one deterministic run per configuration (seed 42).
They compare the three registered baseline pipelines across Single, LOGO, and All-7.
They do not establish statistical superiority or external generalization.
The test partitions had been observed during earlier exploratory runs, so no
hyperparameter or preprocessing change was made in response to those results.
```

- [ ] **Step 5: Add the LOGO leave-Midjourney table**

Report one row per variant for held-out Midjourney. Do not calculate a cross-generator LOGO macro-average from this single fold.

- [ ] **Step 6: Add the All-7 matrix**

Report each known generator separately and label the section `Closed-world known-generator performance`.

- [ ] **Step 7: Compare the three training-diversity regimes**

Use this interpretation boundary:

```text
Single measures transfer from SD1.5 to six unseen generators.
LOGO leave-Midjourney measures transfer from six seen generators specifically to held-out Midjourney.
It does not represent complete seven-fold LOGO performance.
All-7 measures performance when all seven generator families are known during training.
All-7 is not evidence of unseen-generator generalization.
```

- [ ] **Step 8: Record the next research decision without starting another experiment**

Choose one of the following based on the fixed metrics:

```text
Fusion improves or preserves Semantic OOD performance: retain Fusion as the R2 candidate.
Fusion degrades Semantic OOD performance: retain Semantic-only as the stronger baseline and record NPR as non-complementary under this setup.
```

Do not start another seed, JPEG preprocessing, or an external benchmark from this task.

---

## Self-Review

- Spec coverage: class/content balance, exact per-generator cohorts, Single, LOGO leave-Midjourney, All-7, one seed, five epochs, determinism, QuickGELU compatibility, provenance, low-FPR limitations, test exposure, and final reporting are each assigned to a task.
- Placeholder scan: no deferred implementation placeholders are present; explicitly excluded experiments are recorded in Global Constraints.
- Type consistency: `evaluation_generator` is a string from manifest metadata through dataset sample and prediction metadata to evaluator pairing.
- Threshold consistency: both reports reuse identical prediction scores; only the decision threshold changes.
- Scope check: multi-seed aggregation, JPEG intervention, and external validation remain separate future experiments.
