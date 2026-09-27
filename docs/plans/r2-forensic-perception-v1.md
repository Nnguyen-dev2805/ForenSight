# ForenSight R2 — Forensic Perception v1 Implementation Spec

**Status:** IMPLEMENTED — not formally `ACTIVE` until the R0 gate is signed off (see `docs/roadmap.md`).

**Milestone:** R2 — Stage 1: Forensic Perception

**Research question:** Does combining semantic and forensic representations improve cross-generator generalization compared with either representation alone?

> This document specifies the smallest R2 experiment that can answer RQ1. It does not authorize adding extra branches, fusion mechanisms, or explanation/evidence modules before the baseline ablation is understood.

## 1. Source of truth and prerequisites

Before implementation, read:

1. `docs/research-problem.md` — RQ1 and hypotheses;
2. `docs/roadmap.md` — milestone order and R2 gate;
3. `docs/architecture.md` — Stage 1 boundaries and complexity budget;
4. `docs/dataset.md` — active datasets, sealed manifests, split rules;
5. `docs/evaluation.md` — threshold, metrics, repeated runs, reproducibility.

R2 starts only after R0 has produced trustworthy manifests and evaluation plumbing. Existing R1 code may be reused; R1 does not need to be rebuilt before R2 planning, but an official baseline result should eventually be rerun on the sealed R0 protocol for formal comparison.

## 2. Goal

Build and compare exactly three detector variants:

1. **Semantic-only** — frozen CLIP image encoder plus a small classifier.
2. **Forensic-only** — NPR-style low-level representation plus a lightweight ResNet18 encoder and classifier.
3. **Fusion** — projected semantic and forensic features concatenated and classified by a small MLP.

The goal is not to maximize accuracy at any cost. The goal is to isolate whether semantic and low-level forensic representations provide complementary information under generator shift.

## 3. Research hypotheses

### H1 — Complementary signal

Semantic and forensic branches contain complementary signal for AI-generated image detection under cross-generator evaluation.

### H2 — Fusion generalization

Simple feature concatenation improves cross-generator performance compared with either branch alone.

Both hypotheses may be rejected. A negative result is valid and should trigger failure analysis rather than architecture inflation.

## 4. Fixed architecture decision

```text
                              Image
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
        CLIP ViT-L/14                  NPR-style transform
       pretrained, frozen                      |
                 |                             v
                 |                         ResNet18
                 |                         trainable
                 v                             v
        semantic feature               forensic feature
                 |                             |
          Linear -> 256                 Linear -> 256
            LayerNorm                     LayerNorm
                 |                             |
                 +-------------+---------------+
                               |
                               v
                         concatenate
                              512
                               |
                               v
                         MLP classifier
                         512 -> 128 -> 1
                               |
                               v
                         real / fake logit
```

### 4.1 Semantic branch

- Backbone: **CLIP ViT-L/14**.
- Preferred implementation: OpenCLIP using the original OpenAI pretrained weights (`ViT-L-14`, `pretrained="openai"`).
- R2 v1 freezes the CLIP backbone.
- Trainable semantic parameters are limited to the projection/classification layers.
- The branch represents high-level visual/semantic information; it must not be described as a forensic model.

Conceptual forward path:

```python
with torch.no_grad():
    clip_feature = clip.encode_image(clip_input)

semantic = semantic_projection(clip_feature)
semantic = semantic_norm(semantic)
```

### 4.2 Forensic representation

Use an **NPR-inspired low-level residual representation** before the forensic encoder.

Conceptually:

```text
input image
    |
downsample
    |
upsample back to original size
    |
reconstruction
    |
input - reconstruction
    |
low-level residual representation
```

Important constraints:

- deterministic;
- identical policy for real and fake images;
- independent of label and generator;
- no thresholding or handcrafted final detector;
- interpolation and normalization choices are fixed in config and recorded in each experiment.

R2 v1 uses per-channel spatial standard deviation normalization of the residual
(`npr_normalization: spatial_std` in the forensic and fusion configs).

This is a **ForenSight baseline inspired by NPR**, not a claim of reproducing the full NPR paper. The implementation phase should verify the exact transform against the selected reference implementation before training the main experiment.

### 4.3 Forensic encoder

- Backbone: **ResNet18**.
- Input: NPR-style representation, not raw RGB in the required R2 experiment.
- Remove the original classification head and expose the pooled feature vector.
- ResNet18 remains trainable.
- Project the pooled feature to 256 dimensions and apply LayerNorm.

Conceptual forward path:

```python
npr = npr_transform(image)
forensic_feature = forensic_backbone(npr)
forensic = forensic_projection(forensic_feature)
forensic = forensic_norm(forensic)
```

### 4.4 Fusion

R2 v1 uses only concatenation:

```python
fused = torch.cat([semantic, forensic], dim=-1)
logit = classifier(fused)
```

Fixed projected dimensions:

```text
semantic: 256
forensic: 256
fused:    512
```

Classifier:

```text
Linear(512, 128)
ReLU
Dropout
Linear(128, 1)
```

Do not add attention, learned branch weighting, gating, Q-Former, Perceiver, or transformer fusion in R2 v1.

## 5. Required ablation

### A. Semantic-only

```text
Image -> frozen CLIP -> projection -> classifier -> logit
```

Trainable:

- semantic projection;
- classifier.

Frozen:

- CLIP backbone.

### B. Forensic-only

```text
Image -> NPR transform -> ResNet18 -> projection -> classifier -> logit
```

Trainable:

- ResNet18;
- forensic projection;
- classifier.

### C. Fusion

```text
Image -> frozen CLIP ---------> semantic projection ---+
                                                       +-> concat -> MLP -> logit
Image -> NPR -> ResNet18 -----> forensic projection ---+
```

Trainable:

- ResNet18;
- semantic projection;
- forensic projection;
- fusion classifier.

Frozen:

- CLIP backbone.

These three variants are mandatory. R2 is not complete if only the fusion model is trained.

## 6. Data protocol

R2 must reuse sealed manifests from the data subsystem. It must not create ad-hoc train/test splits.

### Experiment Protocols on Kaggle Tiny-GenImage

R2 supports three distinct protocols defined in `docs/dataset.md` and `docs/decisions/2026-09-26-three-experiment-protocols.md`:

1. **`single` (Canonical RQ1 Protocol):**
   - **Training:** Kaggle Tiny-GenImage SD1.5 train only.
   - **Validation:** Kaggle Tiny-GenImage SD1.5 validation (checkpoint selection & threshold calibration $\tau^*$).
   - **In-domain test:** Held-out SD1.5 test samples.
   - **Cross-generator OOD tests:** 6 unseen generators (`adm`, `biggan`, `glide`, `midjourney`, `vqdm`, `wukong`).
   - Manifest dir: `data/manifests/single/`

2. **`logo` (Leave-One-Generator-Out Protocol):**
   - 7 folds (`leave_sd15`, `leave_adm`, etc.).
   - Train on 6 seen generators, validate on 6 seen generators, evaluate on 1 unseen generator.
   - Manifest dir: `data/manifests/logo/leave_<gen>/`

3. **`all7` (Seen-Generator Multi-Generator Upper Bound):**
   - Train on all 7 generators, validate on all 7, test on held-out samples of all 7 generators.
   - Strictly representation capacity upper bound under closed-world generators; **not** unseen generalization.
   - Manifest dir: `data/manifests/all7/`

### Test hierarchy (Canonical `single` protocol)

| Level | Dataset / Generator | Purpose |
|---|---|---|
| In-domain | Tiny-GenImage SD1.5 | Verify learned in-domain discrimination |
| Cross-generator OOD | Midjourney | Unseen generator |
| Cross-generator OOD | ADM | Unseen generator |
| Cross-generator OOD | GLIDE | Unseen generator |
| Cross-generator OOD | Wukong | Unseen generator |
| Cross-generator OOD | VQDM | Unseen generator |
| Cross-generator OOD | BigGAN | Unseen generator / GAN family |
| Modern external | GenImage++ | Modern external generators |

WildRF and Chameleon/AIDE are future benchmarks and are not part of the active R2 v1 runtime protocol unless `docs/dataset.md` is explicitly changed first.

## 7. Preprocessing policy

Load each raw image once, then derive branch-specific model inputs:

```text
Raw image
   |
   +-> CLIP preprocessing required by pretrained CLIP
   |
   +-> NPR transform -> ResNet preprocessing
```

Allowed:

- CLIP's official resize/crop/normalization;
- deterministic NPR transform;
- fixed forensic normalization;
- train-time augmentation only if it is class-independent, documented, and applied equivalently across variants being compared.

Forbidden:

- preprocessing based on label;
- preprocessing based on generator identity;
- real-only or fake-only JPEG recompression;
- tuning preprocessing after inspecting test/OOD results without versioning and rerunning all affected baselines.

The exact preprocessing configuration must be stored in the run config.

## 8. Training contract

### Labels

```text
0 = real
1 = fake
```

### Output

Each model returns one binary logit per image:

```python
logits.shape == [batch_size, 1]
prob_fake = torch.sigmoid(logits)
```

### Loss

```python
torch.nn.BCEWithLogitsLoss()
```

### Optimizer

Default baseline:

```text
AdamW
```

The following must come from experiment configuration rather than being scattered as constants in model code:

- learning rate;
- weight decay;
- batch size;
- epochs;
- dropout;
- seed;
- image size / NPR resize settings;
- early stopping setting (currently **disabled / not used**: training runs the full `epochs`
  and keeps the best checkpoint by val AUROC, falling back to val loss. Add early stopping
  only when a measured training failure motivates it, same as the scheduler);
- threshold strategy.

Do not add a complex scheduler in the first implementation. Add one only after a measured training failure motivates it.

## 9. Threshold and evaluation

Reuse `docs/evaluation.md` and the existing R0 evaluation code.

Required flow:

```text
train
  |
validation predictions
  |
select threshold on validation only
  |
freeze threshold
  |
test / OOD predictions
  |
existing R0 evaluation runner
```

Never tune a threshold on any test or OOD partition.

### Primary metric

- AUROC, following `docs/evaluation.md`.

### Secondary metrics

- Accuracy;
- Precision;
- Recall;
- F1.

Report:

- overall;
- per split;
- per generator;
- per dataset;
- mean cross-generator result over Midjourney, ADM, GLIDE, Wukong, VQDM, and BigGAN.

## 10. Repeated runs and scientific claims

- `smoke` and `pilot` runs are development results only.
- Main comparative claims require the sealed `main` protocol.
- Follow the repeated-run rule in `docs/evaluation.md`; when compute permits, use at least 3 seeds for sealed baselines.
- Do not call fusion "better" based on a single run or a small gain inside run-to-run variance.
- Do not call the model "generalizable" from in-domain performance alone.

## 11. Required R2 result table

At minimum, produce the following table using the same protocol for all three variants:

| Benchmark | Semantic-only | Forensic-only | Fusion |
|---|---:|---:|---:|
| SD1.5 (In-Domain) | — | — | — |
| Midjourney | — | — | — |
| ADM | — | — | — |
| GLIDE | — | — | — |
| Wukong | — | — | — |
| VQDM | — | — | — |
| BigGAN | — | — | — |
| GenImage++ | — | — | — |
| Mean Cross-Generator | — | — | — |

The table must identify which metric it contains; AUROC is the default primary comparison.

## 12. Experiment reproducibility record

Every R2 run must record at least:

```text
experiment_id
model_variant
seed

semantic_backbone
semantic_pretrained_weights
semantic_frozen

forensic_backbone
npr_config

optimizer
learning_rate
weight_decay
batch_size
epochs
dropout

train_manifest
validation_manifest
test_manifest/version

selected_checkpoint
threshold
threshold_source

metrics
predictions
environment
```

Reuse the existing reproducibility/evaluation schema where possible instead of inventing a parallel tracking framework.

## 13. Non-goals for R2 v1

Do not implement:

- CLIP fine-tuning;
- LoRA;
- DCT branch;
- FFT branch;
- SRM branch;
- multi-scale patch ranking;
- attention/gating fusion;
- MLLM integration;
- explanation generation;
- evidence localization;
- candidate evidence extraction;
- R3 intervention logic;
- W&B/MLflow solely for this milestone;
- Lightning/Hydra/custom training framework without an observed need.

## 14. Proposed repository structure

Keep the implementation small and aligned with the current repository:

```text
src/forensight/
├── models/
│   ├── __init__.py
│   ├── semantic.py
│   ├── forensic.py
│   └── fusion.py
├── training/
│   ├── __init__.py
│   └── r2.py
└── evaluation/
    └── existing R0 modules

scripts/
├── train_r2.py
└── evaluate_r2.py

configs/r2/
├── semantic.json
├── forensic.json
└── fusion.json

tests/
├── models/
│   ├── test_semantic.py
│   ├── test_forensic.py
│   └── test_fusion.py
├── training/
│   └── test_r2.py
└── system/
    └── test_r2_smoke.py
```

Use JSON configs because the repository already depends on the Python standard library and does not currently need a YAML/config framework.

### File responsibilities

`semantic.py`

- load CLIP;
- freeze the backbone;
- expose image features;
- project semantic features.

`forensic.py`

- implement deterministic NPR-style transform;
- build ResNet18 feature extractor;
- project forensic features.

`fusion.py`

- compose semantic and forensic branches;
- concatenate projected features;
- classify fused representation.

`training/r2.py`

- shared train/validation step;
- optimizer setup;
- checkpoint save/load;
- prediction export compatible with R0 evaluation.

`train_r2.py`

- parse config/manifest paths;
- seed execution;
- instantiate one of the three variants;
- train and emit artifacts.

`evaluate_r2.py`

- load checkpoint;
- generate prediction records;
- call/reuse the existing R0 evaluation path rather than reimplementing metrics.

## 15. Dependencies at R2 activation

The current project does not yet include PyTorch/CLIP dependencies. When R2 becomes ACTIVE, add only what the chosen implementation requires, expected to be:

- `torch`;
- `torchvision`;
- `open_clip_torch`.

Do not add these dependencies merely to save this planning document. Exact versions should be selected and recorded when the training environment is activated so CPU/CUDA/macOS compatibility is known.

## 16. Implementation phases

### Task R2.1 — Semantic-only baseline

**Files:**

- Create: `src/forensight/models/__init__.py`
- Create: `src/forensight/models/semantic.py`
- Create: `tests/models/test_semantic.py`
- Create: `configs/r2/semantic.json`

**Deliverable:** frozen CLIP feature extractor plus trainable projection/classifier with `[B, 1]` logits.

Verification must cover:

- CLIP parameters have `requires_grad=False`;
- output shape is `[batch, 1]`;
- projection/classifier receive gradients;
- forward pass is finite.

### Task R2.2 — NPR-style transform and forensic-only baseline

**Files:**

- Create: `src/forensight/models/forensic.py`
- Create: `tests/models/test_forensic.py`
- Create: `configs/r2/forensic.json`

**Deliverable:** deterministic NPR transform feeding a trainable ResNet18 feature extractor and binary classifier.

Verification must cover:

- deterministic transform output;
- shape preservation at the transform boundary;
- identical code path for real/fake labels;
- ResNet18 receives gradients;
- output shape is `[batch, 1]`.

Before main training, visually inspect a small fixed sample of NPR outputs as a diagnostic artifact. This inspection is not part of inference or classification logic.

### Task R2.3 — Fusion model

**Files:**

- Create: `src/forensight/models/fusion.py`
- Create: `tests/models/test_fusion.py`
- Create: `configs/r2/fusion.json`

**Deliverable:** 256-d semantic + 256-d forensic projections concatenated into a 512-d vector and classified by `512 -> 128 -> 1` MLP.

Verification must cover:

- branch projection shapes;
- fused shape is 512;
- CLIP stays frozen during backward;
- forensic branch and fusion head receive gradients;
- output shape is `[batch, 1]`.

### Task R2.4 — Shared training pipeline

**Files:**

- Create: `src/forensight/training/__init__.py`
- Create: `src/forensight/training/r2.py`
- Create: `scripts/train_r2.py`
- Create: `tests/training/test_r2.py`

**Deliverable:** one explicit PyTorch training path supporting `semantic`, `forensic`, and `fusion` variants.

Verification must cover:

- one training step produces finite BCE loss;
- optimizer updates trainable parameters only;
- checkpoint save/load reproduces model state;
- seed and config are persisted;
- no test partition is used by the training loop.

Do not create three independent trainers.

### Task R2.5 — Prediction export and R0 evaluation integration

**Files:**

- Create: `scripts/evaluate_r2.py`
- Modify only if required: `src/forensight/evaluation/*`
- Add focused compatibility tests under `tests/evaluation/` or `tests/training/`.

**Deliverable:** prediction files accepted directly by the current evaluation runner.

Required prediction fields must match the existing R0 evaluator, including at least:

```text
sample_id
label
score
split
generator
dataset
```

Verification must prove that validation selects the threshold and test/OOD partitions only consume the frozen threshold.

### Task R2.6 — End-to-end smoke run

**Files:**

- Create: `tests/system/test_r2_smoke.py`
- Reuse the existing smoke manifest/data utilities where possible.

**Deliverable:** a tiny end-to-end run for all three variants:

```text
load samples
-> forward
-> BCE loss
-> backward
-> optimizer step
-> save checkpoint
-> reload checkpoint
-> export predictions
-> R0 evaluator
```

This task proves plumbing only; it must not be reported as a research result.

### Task R2.7 — Pilot ablation

Run all three variants on the sealed `pilot` tier using the same split policy and comparable optimization budget.

Output:

- training curves;
- validation threshold provenance;
- per-split metrics;
- per-generator metrics;
- failure notes;
- one comparison table.

Pilot results may guide debugging and gross hyperparameter choices but are not final research claims.

### Task R2.8 — Main ablation

After the pilot pipeline is stable:

```text
train on SD1.5 main
-> select checkpoint on SD1.5 validation
-> select/freeze threshold on validation
-> evaluate held-out SD1.5 in-domain test
-> evaluate six cross-generator OOD sets
-> evaluate GenImage++
-> repeat required seeds
-> aggregate results
```

Output the final Semantic-only vs Forensic-only vs Fusion table and failure analysis.

## 17. Minimum unit/system tests

R2 implementation is not ready for experiments until these behaviors are covered:

```text
NPR transform deterministic
NPR transform expected shape
CLIP frozen
semantic forward
forensic forward
fusion forward
binary output shape [B, 1]
finite BCE loss
backward path correct
checkpoint round trip
prediction export schema compatible with R0
validation-only threshold calibration
end-to-end smoke flow
```

Do not write tests that merely duplicate third-party library behavior.

## 18. R2 gate

R2 v1 is complete when all of the following are true:

1. Semantic-only is reproducible.
2. Forensic-only is reproducible.
3. Fusion is reproducible.
4. All three use the same sealed R0 data protocol.
5. CLIP is demonstrably frozen.
6. NPR-style forensic branch works independently.
7. Validation selects checkpoints/hyperparameters/thresholds.
8. Test/OOD data never participates in tuning.
9. All three variants use the same evaluation pipeline.
10. Per-generator results are available.
11. Mean cross-generator result is available.
12. Experiment metadata is sufficient to rerun each result.
13. The required three-way ablation table exists.
14. If fusion does not improve, the result is preserved and accompanied by failure analysis instead of silently changing architecture.

## 19. Interpretation rules

Possible outcomes:

### Fusion wins consistently on unseen generators

Evidence supports the hypothesis that the two representations are complementary under the tested protocol.

### Forensic-only wins

The semantic branch may be redundant or may inject content shortcuts. Investigate before changing architecture.

### Semantic-only wins

The selected NPR-style representation or forensic encoder may not add useful generalizable signal. Inspect training dynamics, branch scale, and dataset bias before adding new forensic branches.

### Fusion is mixed by generator

Report the heterogeneity. Do not compress it into a single claim based only on an overall average.

## 20. What comes after R2

R2 answers whether the representations help detection/generalization.

R3 then asks which local features are candidate evidence:

```text
local / patch features
+ coordinates
+ contribution score
-> top-k candidate evidence
```

R2 v1 must therefore avoid pulling evidence localization, projector alignment, or explanation generation forward into this milestone.

## 21. Final decision summary

```text
Semantic encoder:
    CLIP ViT-L/14, OpenAI pretrained weights, frozen

Forensic representation:
    deterministic NPR-inspired low-level residual

Forensic encoder:
    ResNet18, trainable

Projection:
    semantic -> 256
    forensic -> 256

Fusion:
    concatenate

Classifier:
    512 -> 128 -> 1

Loss:
    BCEWithLogitsLoss

Training:
    Kaggle Tiny-GenImage SD1.5 only

Validation:
    Kaggle Tiny-GenImage SD1.5 validation

Evaluation:
    SD1.5
    Midjourney
    ADM
    GLIDE
    Wukong
    VQDM
    BigGAN
    GenImage++

Required ablation:
    Semantic-only
    Forensic-only
    Fusion
```

Guiding principle:

> Implement the smallest architecture capable of testing whether semantic and forensic representations provide complementary information for cross-generator AI-image detection.
