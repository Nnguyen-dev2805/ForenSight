# Dataset Specification & Split Policy

## 1. Overview & Active Datasets

ForenSight investigates generalizable and explainable AI-generated image detection.
To ensure experimental clarity and prevent data snooping, training in Stage 1 is restricted strictly to a single generator (**Stable Diffusion v1.5**). All other generators and external corpuses serve strictly as evaluation benchmarks across distinct axes of generalization.

### Active Stage 1 Runtime Datasets (R0 / R1)

| Dataset | Version / Source | Role | Real Source | Fake Generators | Est. Images | Label Mapping | Local Directory |
|---|---|---|---|---|---|---|---|
| **Tiny-GenImage** | Kaggle [yangsangtai/tiny-genimage](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) | Primary Stage 1 Benchmark | ImageNet (ILSVRC2012) | 7 generators (`sd15`, `midjourney`, `adm`, `glide`, `wukong`, `vqdm`, `biggan`) | 35k (28k train, 7k val) | Real: 0<br>Fake: 1 | `data/raw/tiny_genimage` |
| **GenImage (Full)** | 1.0 (NeurIPS 2023) [Hugging Face](https://huggingface.co/datasets/ENSTA-U2IS/GenImage) | Reference Main Benchmark | ImageNet (ILSVRC2012) | 8 generators (`sd14`, `sd15`, `midjourney`, `adm`, `glide`, `wukong`, `vqdm`, `biggan`) | ~2.6M (~1.33M pairs) | Real: 0<br>Fake: 1 | `data/raw/genimage` |
| **GenImage++** | 1.0 (2024) [Hugging Face](https://huggingface.co/datasets/Lunahera/genimagepp) | Modern External Benchmark (Test-Only) | ImageNet-1k, COCO, Web | Modern models (FLUX.1, SD3, PixArt, Kolors, HunyuanDiT, AuraFlow) | ~100k | Real: 0<br>Fake: 1 | `data/raw/genimagepp` |

### Planned Future Benchmarks (Not in R0/R1 active runtime inventory)

- **WildRF** (2024): Real-world organic social media imagery (~12k images, test-only).
- **Chameleon / AIDE** (2024): Gated diagnostic benchmark for human-hard photorealistic synthetic images (~6k images, test-only).

---

## 2. Generator Roles & Three Experiment Protocols

Kaggle Tiny-GenImage contains seven fake-generator families (`sd15`, `adm`, `biggan`, `glide`, `midjourney`, `vqdm`, `wukong`). To rigorously address cross-generator generalization while maintaining strict research integrity, ForenSight supports three distinct experiment protocols:

### 2.1 Protocol 1: `single` (Single-Generator Unseen Cross-Generator Generalization)
The primary RQ1 protocol. Trains strictly on a single diffusion generator and evaluates zero-shot transfer to unseen architectures:
- **Train (`train.jsonl`):** SD1.5 train partition only (balanced 1:1 real/fake).
- **Validation (`val.jsonl`):** Held-out SD1.5 validation partition for checkpoint selection and threshold calibration ($\tau^*$).
- **In-domain test (`in_domain_test.jsonl`):** Held-out SD1.5 test partition.
- **Combined OOD test (`cross_generator_ood.jsonl`):** All 6 unseen generator architectures pooled into a single benchmark set.
- **Per-generator OOD test sets (`test_<gen>.jsonl`):** 6 unseen generator architectures (`adm`, `biggan`, `glide`, `midjourney`, `vqdm`, `wukong`).
- **Manifest directory:** `data/manifests/single/`

### 2.2 Protocol 2: `logo` (Leave-One-Generator-Out)
Evaluates whether multi-generator training generalizes to an unseen generator architecture across 7 distinct folds:
- **Folds:** `leave_sd15`, `leave_adm`, `leave_biggan`, `leave_glide`, `leave_midjourney`, `leave_vqdm`, `leave_wukong`.
- **Train (`train.jsonl`):** Pooled train partitions of the 6 seen generators (1:1 real/fake).
- **Validation (`val.jsonl`):** Pooled held-out validation partitions of the 6 seen generators.
- **Seen-generator in-domain test (`test_in_domain_seen.jsonl`):** Held-out test partition pooled from the 6 seen generators.
- **Unseen test (`test_<leave_out_gen>.jsonl`):** Completely unseen held-out generator (1:1 real/fake).
- **Manifest directory:** `data/manifests/logo/leave_<gen>/`

### 2.3 Protocol 3: `all7` (Seen-Generator Multi-Generator Upper Bound)
Quantifies representation capacity ceiling when all 7 generators are observed during training:
- **Train (`train.jsonl`):** Pooled train partitions from all 7 generators (1:1 real/fake).
- **Validation (`val.jsonl`):** Pooled held-out validation partitions from all 7 generators.
- **Per-generator tests (`test_<gen>.jsonl`):** Held-out test partitions for each of the 7 generators (including `test_sd15.jsonl`).
- **Combined test (`test_all_combined.jsonl`):** Pooled held-out test partition across all 7 generators.
- **Manifest directory:** `data/manifests/all7/`
- **STRICT RESEARCH RULE:** Results from `all7` must **never** be labeled, reported, or claimed as "unseen cross-generator generalization". It is an empirical upper-bound benchmark for representation capacity under known generator distributions.

---

## 3. Generator-Disjoint Guarantee & Zero Leakage

A fundamental invariant for ForenSight is the strict enforcement of leakage-free protocols:

### Leakage Invariants:
1. **Generator-Disjoint Guarantee in OOD Sets:** For `single` and `logo`, generators in the unseen evaluation split must never appear in the training or validation splits:
   $$\mathcal{G}_{\text{train}} \cap \mathcal{G}_{\text{unseen}} = \emptyset$$
2. **Disjoint Real Allocation Without Replacement:** Evaluation real images are partitioned monotonically without replacement across validation, in-domain test, and all OOD generator test sets. No real sample is ever reused across evaluation splits.
3. **Sample & Path Disjointness:** Sample IDs and image file paths must be strictly disjoint between train, val, and test splits ($\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset$, $\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$, $\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$).
4. **Exact Duplicate Protection:** Exact duplicate detection uses SHA256 checksums (`compute_file_sha256`). Identical file contents between train and test partitions are strictly prohibited.
5. **Threshold Calibration Rule:** Test sets must never be inspected to tune decision thresholds ($\tau^*$) or select hyperparameters. Calibration is performed strictly on `val`.
6. **Automated Enforcement:** Built-in guards `assert_generator_disjoint()` and `validate_no_leakage()` enforce these constraints in code.

---

## 4. Deterministic Scale Tiers

To facilitate rapid development without sacrificing representation across the 1,000 ImageNet categories, ForenSight establishes three canonical scale tiers:

| Tier | Real / Class Limit | Fake / Class Limit | Tiny-GenImage Scope | Intended Purpose |
|---|---|---|---|---|
| **`smoke`** | $\le 1$ | $\le 1$ | $\le 2,000$ images | Fast smoke testing, CI pipelines, pipeline sanity checks (40 images in CI). |
| **`pilot`** | $\le 10$ | $\le 10$ | $\le 20,000$ images | Preliminary experiments, architecture iterations, hyperparameter screening. |
| **`main`** | Full valid set | Full valid set | Full dataset (~35,000 images; ~320k for full GenImage) | Final research conclusions, formal benchmark tables, published results. |

- Subsampling operates per `class_id` using a seeded pseudo-random generator keyed to `(seed, class_id, label)`.
- **Research Integrity:** Conclusions on hypothesis verification, model generalization, or explainability **must** use the sealed `main` manifest tier. `smoke` and `pilot` tiers are strictly development aids.

---

## 5. Storage Conventions & Manifest Schema

### 5.1 Storage Hierarchy
- `data/raw/`: Raw, unmodified downloads. Considered **immutable**.
- `data/manifests/`: Sealed, deterministic split manifests (JSONL/CSV).
- `data/derived/`: Preprocessed, cached, or bias-controlled derived artifacts (with versioning).
- `data/dataset_inventory.json`: Version-controlled machine-readable inventory registry.

### 5.2 ManifestRecord Schema
Each sample is tracked by a `ManifestRecord`:
```python
@dataclass
class ManifestRecord:
    sample_id: str           # Unique identifier for the source image
    image_path: str          # Filepath to image (relative or absolute)
    label: int               # 0 = real, 1 = fake
    dataset: str             # "genimage", "genimage_plus_plus"
    generator: str           # Generator ID ("sd15", "midjourney", etc., or "nature" for real)
    split: str               # "train", "val", "in_domain_test", "cross_generator_ood", etc.
    class_id: str | None     # ImageNet synset ID (e.g. "n01440764") or None
    metadata: dict[str, Any] # Arbitrary metadata (resolution, JPEG QF, etc.)
```

---

## 6. Dataset Download Workflows

ForenSight uses Kaggle (`kagglehub`) as the canonical transport layer for Stage-1 Tiny-GenImage, and Hugging Face as the reference transport layer for full GenImage / GenImage++ benchmarks.

### 6.1 Canonical: Tiny-GenImage (35k images)

Two preparation modes are supported. Both route through `build_protocol_manifests` with
split seed 42 by default, so a given pinned dataset version yields identical splits either
way. The `--seed` on the Kaggle run controls model training, not the sealed split.

**Mode A — prepare locally, then ship the sealed manifests to Kaggle:**

```bash
uv run python scripts/download_tiny_genimage.py \
    --dataset-ref yangsangtai/tiny-genimage/versions/<N> \
    --output-dir data/raw/tiny_genimage --manifest-dir data/manifests
```

**Mode B — keep everything on Kaggle, with no raw data on the development machine:**

```bash
uv run python scripts/run_on_kaggle.py --push \
    --experiment single --variant fusion --seed 42 \
    --dataset-ref yangsangtai/tiny-genimage/versions/<N> \
    --prepare-manifests
```

With `--prepare-manifests` the local manifests are omitted from the kernel bundle; the
kernel downloads the pinned dataset once, scans it, builds the sealed splits, and writes a
`manifest_provenance.json` (scan root, seed, scanned image count, per-split sizes) next to
them. Manifest image paths are relative to the dataset root, so the kernel resolves them
via `--base-dir <downloaded dataset root>` without any raw data being shipped.

Both modes scan the directory tree without rewriting raw files. SD1.5 is used for
train/validation/in-domain evaluation; ADM, BigGAN, GLIDE, Midjourney, VQDM, and Wukong
remain unseen-generator OOD evaluation sets.

### 6.2 Reference only: Full GenImage (Selective Multi-part ZIP)
To selectively download individual generator archives from the full `ENSTA-U2IS/GenImage` benchmark:

```bash
python scripts/download_dataset.py --dataset genimage --generator sd14
```

The downloader resolves `main` (or `--revision`) to an immutable Hub commit SHA, downloads only `stable_diffusion_v_1_4/**`, and records provenance in `data/raw/genimage/_downloads/sd14/source.json`. Extract the archive into `data/raw/genimage/sdv4/` before generating real manifests.

---

## 7. Dataset Confound & Bias Audit

Literature (*Unbiased GenImage*, ICML 2024) identifies two critical confound risks in raw GenImage:
1. **JPEG Quality / Compression Asymmetry:** Real and synthetic images undergo differing compression processes. Detectors risk learning compression artifacts rather than generative synthesis artifacts.
2. **Resolution & Aspect Ratio Asymmetry:** Real images retain diverse photographic dimensions, whereas fake images are generated at fixed square dimensions (512x512 or 256x256).

> [!WARNING]
> Raw GenImage is **not** assumed to be bias-free. Forensic audits (`audit_manifest_images`) quantify compression and resolution distributions before detection scores can be trusted. Raw datasets are immutable; any bias-mitigation preprocessing must produce versioned derived artifacts rather than overwriting raw files.
