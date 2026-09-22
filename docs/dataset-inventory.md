# Dataset Inventory

## 1. Overview & Strategy

This document provides the human-readable summary of the dataset inventory established for **ForenSight Milestone R0**.
The corresponding machine-readable specification is maintained in `data/dataset_inventory.json` and programmatically accessible via `forensight.data.inventory`.

In accordance with the Stage 1 dataset strategy ([docs/decisions/2026-09-18-stage1-dataset-strategy.md](file:///Users/tnhatnguyendev2805/Documents/Projects/ForenSight/docs/decisions/2026-09-18-stage1-dataset-strategy.md)) and the evaluation protocol ([docs/evaluation-protocol.md](file:///Users/tnhatnguyendev2805/Documents/Projects/ForenSight/docs/evaluation-protocol.md)), training in Stage 1 is restricted strictly to a single generator (**Stable Diffusion v1.4**). All other generators and datasets serve strictly as evaluation benchmarks across distinct axes of generalization.

### Summary Table

#### Active Stage 1 Runtime Datasets (R0 / R1)

| Dataset | Version / Source | Evaluation Role | Real Source | Fake Generators | Est. Count | Label Mapping | Local Path | Access |
|---|---|---|---|---|---|---|---|---|
| **GenImage** | 1.0 (NeurIPS 2023) [Hugging Face mirror](https://huggingface.co/datasets/ENSTA-U2IS/GenImage) | Primary Stage 1 Benchmark | ImageNet (ILSVRC2012) | 8 generators (SD1.4, SD1.5, Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN) | ~1.33M pairs (~2.6M images) | Real: 0<br>Fake: 1 | `data/raw/genimage` | Open Research |
| **GenImage++** | 1.0 (2024) [Hugging Face](https://huggingface.co/datasets/Lunahera/genimagepp) | Modern External Benchmark (Test-Only) | ImageNet-1k, COCO, Web | Modern models (FLUX.1, SD3, PixArt, Kolors, HunyuanDiT, AuraFlow) | ~100k images | Real: 0<br>Fake: 1 | `data/raw/genimagepp` | Open Research |

#### Planned Future Benchmarks (Not in R0/R1 runtime inventory)

| Dataset | Version / Source | Evaluation Role | Real Source | Fake Generators | Est. Count | Label Mapping | Local Path | Access |
|---|---|---|---|---|---|---|---|---|
| **WildRF** | 1.0 (2024) [GitHub](https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld) | Real-World External Benchmark (Test-Only) | Social media / open web platforms | In-the-wild generators (Midjourney, DALL-E, SDXL, online tools) | ~12k images | Real: 0<br>Fake: 1 | `data/raw/wildrf` | Open Research |
| **Chameleon** | 1.0 (2024) [GitHub](https://github.com/shilinyan99/AIDE) | Optional External Benchmark (Test-Only) | Professional photography (Unsplash, RAISE) | Human-hard photorealistic synthetic images | ~6k images | Real: 0<br>Fake: 1 | `data/raw/chameleon` | Gated (Application) |

---

## 2. Dataset Profiles

### 2.1 GenImage (Primary Stage 1 Benchmark)

* **Source Reference:** *GenImage: A Million-Scale Dataset for Detecting AI-Generated Image* (Zhu et al., NeurIPS 2023 Datasets & Benchmarks Track).
* **Canonical Project Reference:** [GenImage official repository](https://github.com/GenImage-Dataset/GenImage)
* **ForenSight Download Source:** [Hugging Face mirror `ENSTA-U2IS/GenImage`](https://huggingface.co/datasets/ENSTA-U2IS/GenImage)
* **Real Source:** ImageNet (ILSVRC2012) 1,000 object categories.
* **Label Convention:** `real`: 0, `fake`: 1.
* **License:** GenImage Apache-2.0 / Research Use; ImageNet terms of access for non-commercial educational/research use.

#### Generator Breakdown & Evaluation Hierarchy

GenImage contains 8 distinct generator subsets, each paired with authentic ImageNet photographs from the same classes. Stage 1 assigns them to the following evaluation hierarchy:

| Generator ID | Generator Name | Role in Protocol | Fake Resolution | Real Resolution | Splits & Estimated Images | Local Subdirectory |
|---|---|---|---|---|---|---|
| `sd14` | Stable Diffusion v1.4 | **Train + In-Domain Test** | 512x512 | Variable (ImageNet) | Train: 160k real + 160k fake<br>Val: 20k real + 20k fake | `data/raw/genimage/sdv4` |
| `sd15` | Stable Diffusion v1.5 | **Near-OOD Test** | 512x512 | Variable (ImageNet) | Val: 20k real + 20k fake | `data/raw/genimage/sdv5` |
| `midjourney` | Midjourney | **Cross-Generator OOD** | ~512x512 | Variable (ImageNet) | Val: 20k real + 20k fake | `data/raw/genimage/midjourney` |
| `adm` | ADM (Guided Diffusion) | **Cross-Generator OOD** | 256x256 | Variable (ImageNet) | Val: 20k real + 20k fake | `data/raw/genimage/adm` |
| `glide` | GLIDE | **Cross-Generator OOD** | 256x256 | Variable (ImageNet) | Val: 20k real + 20k fake | `data/raw/genimage/glide` |
| `wukong` | Wukong | **Cross-Generator OOD** | 512x512 | Variable (ImageNet) | Val: 20k real + 20k fake | `data/raw/genimage/wukong` |
| `vqdm` | VQDM (VQ-Diffusion) | **Cross-Generator OOD** | 256x256 | Variable (ImageNet) | Val: 20k real + 20k fake | `data/raw/genimage/vqdm` |
| `biggan` | BigGAN | **Cross-Generator OOD** | 256x256 | Variable (ImageNet) | Val: 20k real + 20k fake | `data/raw/genimage/biggan` |

#### Confound & Shortcut Warnings (Unbiased GenImage Literature)

Literature (e.g. *Unbiased GenImage*, ICML 2024 / arXiv:2402.09459) identifies two critical confound risks in raw GenImage:
1. **JPEG Quality / Compression Asymmetry:** Real ImageNet images and synthetic generated images undergo differing compression processes during generation/collection. Detectors can exploit JPEG quantization artifacts as a trivial shortcut rather than learning semantic or generative synthesis artifacts.
2. **Resolution & Aspect Ratio Asymmetry:** Real images retain diverse original photographic dimensions, whereas fake images are generated at fixed square dimensions (e.g., 512x512 or 256x256).

> [!WARNING]
> Per `docs/decisions/2026-09-18-stage1-dataset-strategy.md`, raw GenImage is **not** assumed to be bias-free. Task 0.3 must audit compression and resolution distributions before detection scores can be trusted. Raw datasets are immutable; any bias-control preprocessing must produce versioned derived artifacts rather than overwriting raw files.

#### Local Directory Structure Convention

Within `data/raw/genimage/<generator>/`:
```text
├── train/
│   ├── nature/          # Real ImageNet images organized by synset (e.g., n01440764/)
│   └── ai/              # Synthetic images generated for corresponding synsets
└── val/
    ├── nature/
    └── ai/
```

#### Hugging Face Download Workflow

ForenSight uses Hugging Face as the transport layer instead of cloning/downloading through GitHub. Start with SD1.4 only:

```bash
pip install -e .
python scripts/download_dataset.py --dataset genimage --generator sd14
```

The downloader resolves `main` (or `--revision`) to an immutable Hub commit SHA, downloads only `stable_diffusion_v_1_4/**`, and records provenance in `data/raw/genimage/_downloads/sd14/source.json`. The mirror stores SD1.4 as a multi-part ZIP archive, so extraction is intentionally separate; extract it into `data/raw/genimage/sdv4/` before generating the real R0 manifests and audit report.

---

### 2.2 GenImage++ (Modern External Benchmark)

* **Source Reference:** GenImage++ Benchmark (2024).
* **Repository:** [https://huggingface.co/datasets/Lunahera/genimagepp](https://huggingface.co/datasets/Lunahera/genimagepp)
* **Role:** Modern external benchmark (Level 4 in evaluation hierarchy). Test-only; never used for training or validation threshold tuning.
* **Real Source:** Curated authentic images from ImageNet-1k, COCO, and high-resolution web photographs.
* **Fake Generators:** Modern architectures post-2023:
  * FLUX.1 (FLUX.1-schnell, FLUX.1-dev)
  * Stable Diffusion 3 (SD3-Medium)
  * PixArt-alpha / PixArt-sigma
  * Kolors
  * HunyuanDiT
  * AuraFlow
* **Image Count:** ~100,000 images (~50k real, ~50k fake).
* **Resolution:** Predominantly 1024x1024 (native output of modern rectified-flow and diffusion models) and 512x512.
* **Format / Compression:** High quality JPEG and PNG.
* **License / Access:** Open research access via Hugging Face.
* **Local Storage Path:** `data/raw/genimagepp`
* **Download Command:**
  ```bash
  huggingface-cli download Lunahera/genimagepp --local-dir data/raw/genimagepp --repo-type dataset
  ```

---

### 2.3 WildRF (Real-World External Benchmark)

* **Source Reference:** *Real-Time Deepfake Detection in the Real World* (Cavia et al., 2024).
* **Repository:** [https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld](https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld)
* **Role:** Real-world external benchmark (Level 5 in evaluation hierarchy). Test-only.
* **Real Source:** Authentic images collected from social platforms (Reddit, Twitter/X, Instagram, news media).
* **Fake Generators:** Unconstrained in-the-wild AI-generated content (Midjourney v5/v6, DALL-E 2/3, SDXL, commercial web tools) uploaded to social networks.
* **Image Count:** ~12,000 images (~6k real, ~6k fake).
* **Resolution:** Highly variable organic web resolutions (400x400 to 2048x2048+).
* **Format / Compression:** Multi-generational social media recompression (JPEG, WebP, PNG) with EXIF metadata stripped.
* **License / Access:** Open academic research access via repository instructions.
* **Local Storage Path:** `data/raw/wildrf`

---

### 2.4 Chameleon / AIDE (Optional External Benchmark)

* **Source Reference:** *AIDE: Artificial Image Detector Evaluation* (Yan et al., 2024).
* **Repository:** [https://github.com/shilinyan99/AIDE](https://github.com/shilinyan99/AIDE)
* **Role:** Optional external benchmark (Level 6 in evaluation hierarchy). Test-only.
* **Real Source:** Professional photography (Unsplash, RAISE dataset).
* **Fake Generators:** Human-hard photorealistic synthetic images curated specifically to test cases where standard visible artifacts are absent.
* **Image Count:** ~6,000 curated images (~3k real, ~3k fake).
* **Resolution:** High-resolution photography (1024x1024 to 2048x2048+).
* **Format / Compression:** High-quality uncorrupted JPEG and PNG.
* **License / Access:** Gated academic research license. Access requires submitting an application to the authors. Development in ForenSight proceeds without waiting for access approval.
* **Local Storage Path:** `data/raw/chameleon`

---

## 3. Storage and Directory Conventions

Per project rules in `AGENTS.md`:
* `data/raw/`: Raw, unmodified downloads. Considered **immutable**.
* `data/manifests/`: Sealed, deterministic split manifests (CSV/JSONL) produced in Task 0.2.
* `data/derived/`: Preprocessed, cached, or bias-controlled artifacts with explicit versioning.
* `data/dataset_inventory.json`: Version-controlled machine-readable inventory manifest.

---

## 4. Programmatic Usage

The Python interface in `forensight.data.inventory` provides full type-checked access and validation for the inventory:

```python
from forensight.data.inventory import (
    build_default_inventory,
    load_inventory,
    save_inventory,
)

# Load existing machine-readable inventory
inventory = load_inventory("data/dataset_inventory.json")

# Validate inventory consistency
errors = inventory.validate()
assert len(errors) == 0

# Access primary training generator (SD1.4)
sd14 = inventory.get_generator_subset("genimage", "sd14")
print(f"SD1.4 role: {sd14.role}")
print(f"SD1.4 local path: {sd14.local_path}")

# Query all cross-generator OOD test sets
cross_ood_subsets = inventory.find_subsets_by_role("cross_generator_ood")
for dataset, subset in cross_ood_subsets:
    print(f"OOD Generator: {subset.generator_name} ({subset.generator_id})")
```
