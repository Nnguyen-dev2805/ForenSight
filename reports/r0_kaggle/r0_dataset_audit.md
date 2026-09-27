# ForenSight Milestone R0 Dataset & Confound Audit Report

**Date:** 2026-09-27 02:10:54 UTC  
**Dataset Source:** `yangsangtai/tiny-genimage`  
**Execution Environment:** Kaggle CPU  
**Total Images Audited:** 11000 (5500 Real, 5500 Fake, 0 Corrupt)  
**Audit Duration:** 107.9s  

---

## 1. Executive Summary & Gate Status

| Check | Status | Severity | Summary Finding / Literature Confound |
| :--- | :---: | :---: | :--- |
| **Cross-Split Leakage** | ✅ PASS | `INFO` | Zero sample ID, path, or generator leakage between train and test/OOD. |
| **Resolution & Aspect Ratio Confound** | ⚠️ WARN | `WARNING` | Resolution & aspect ratio confound confirmed: Real images have variable dimensions (width std=191.59px, aspect ratio 1.19 ± 0.31), while Fake images are strictly square (aspect ratio 1.0 ± 0.0) with fixed discrete generator resolutions (BigGAN 128px, ADM/GLIDE/VQDM 256px, SD1.5/Wukong 512px, Midjourney 1024px). |
| **JPEG Compression Confound** | ⚠️ WARN | `WARNING` | JPEG quality distributions are NOT comparable: Real is 100% JPEG (mean Q=93.07 ± 7.55), while Fake is 0% JPEG (100% PNG, lossless). Severe format and compression confound. |
| **File Format Confound** | ⚠️ WARN | `WARNING` | Format disparity confirmed: Real is 100.0% JPEG vs Fake 0.0% JPEG (difference: 100.0%). |
| **Exact Duplicate Leakage** | ✅ PASS | `INFO` | Zero exact SHA-256 duplicates detected across splits. |

---

## 2. Protocol Manifest Allocations (Zero-Leakage Guarantee)

| Partition / Split | Total Samples | Real (Nature) | Fake (Synthetic) | Generator Coverage |
| :--- | :---: | :---: | :---: | :--- |
| **train** | 4000 | 2000 | 2000 | `['sd15']` |
| **val** | 500 | 250 | 250 | `['sd15']` (Disjoint val cohort) |
| **in_domain_test** | 500 | 250 | 250 | `['sd15']` (Held-out in-domain) |
| **cross_generator_ood** | 6000 | 3000 | 3000 | `['adm', 'biggan', 'glide', 'midjourney', 'vqdm', 'wukong']` |

*Guarantees: Zero generator leakage into train/val. Zero sample ID or file path overlap across partitions.*

---

## 3. Literature Confound Analysis (GenImage Benchmarks)

### 3.1 Spatial Resolution Disparity
Detectors risk learning spatial resolution shortcuts if synthetic images have fixed dimensions (e.g. 512x512) while authentic images vary widely.

| Class | Count | Width Mean ± Std | Width Median | Height Mean ± Std | Height Median |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Real (Authentic)** | 5500 | 483.99 ± 191.59 | 500.0 px | 421.44 ± 163.21 | 375.0 px |
| **Fake (Synthetic)** | 5500 | 453.82 ± 227.45 | 512.0 px | 453.82 ± 227.45 | 512.0 px |

### 3.2 JPEG Compression Quality ($Q \in [1, 100]$ via DQT Inversion)
Detectors risk learning compression artifacts if real and synthetic images come from different compression regimes.

| Class | Total JPEGs | Mean Q ± Std | Median Q | Q < 50 | Q 50-70 | Q 71-80 | Q 81-90 | Q 91-95 | Q 96-100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real** | 5500 | 93.07 ± 7.55 | 96.0 | 12 | 107 | 453 | 305 | 262 | 4361 |
| **Fake** | 0 | None ± None | None | 0 | 0 | 0 | 0 | 0 | 0 |

### 3.3 Generator-by-Generator Breakdown

| Generator | Images | Format Dist | Width (Mean ± Std) | JPEG Q (Mean ± Std) | % JPEG |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **`nature`** | 5500 | JPEG: 100% | 483.99 ± 191.59 | 93.07 ± 7.55 | 100.0% |
| **`sd15`** | 2500 | PNG: 100% | 512.0 ± 0.0 | None ± None | 0.0% |
| **`adm`** | 500 | PNG: 100% | 256.0 ± 0.0 | None ± None | 0.0% |
| **`biggan`** | 500 | PNG: 100% | 128.0 ± 0.0 | None ± None | 0.0% |
| **`glide`** | 500 | PNG: 100% | 256.0 ± 0.0 | None ± None | 0.0% |
| **`midjourney`** | 500 | PNG: 100% | 1024.0 ± 0.0 | None ± None | 0.0% |
| **`vqdm`** | 500 | PNG: 100% | 256.0 ± 0.0 | None ± None | 0.0% |
| **`wukong`** | 500 | PNG: 100% | 512.0 ± 0.0 | None ± None | 0.0% |

---

## 4. Confound Analysis & Research Integrity Assessment

### 4.1 Gate Status Summary
- **Split protocol:** ✅ PASS (disjoint partitions, balanced classes)
- **Generator isolation:** ✅ PASS (zero OOD leakage into train/val)
- **ID & Path leakage:** ✅ PASS (zero sample ID or file path overlap)
- **Exact duplicate check:** ✅ PASS (zero cross-split SHA-256 collisions across all 11,000 images)
- **Observed format confound:** ⚠️ **CONFIRMED** (Real is 100% JPEG vs Fake is 100% PNG)
- **Observed resolution bias:** ⚠️ **CONFIRMED** (Real is variable/non-square vs Fake is fixed square per-generator)
- **Confound mitigation:** ❌ **NOT YET PROVEN** (Hypothesis only)

### 4.2 Critical Technical Risks for Milestone R2
1. **Pixel-Grid Compression Traces:**
   - Standard decoding (`PIL.Image.open(...).convert("RGB")`) maps byte streams to RGB pixel values. It does **not** erase the block boundary discontinuities, ringing, or high-frequency quantization artifacts embedded in JPEG pixels.
2. **Resampling & Interpolation Footprints:**
   - Standardizing to 224x224 forces asymmetric spatial transformations across generators: BigGAN (128x128) undergoes bicubic upsampling, whereas Midjourney (1024x1024) undergoes downsampling. These leave characteristic spectral/interpolation footprints that CNNs can easily exploit.
3. **NPR Residual Interaction with JPEG:**
   - The Neighboring Pixel Residual transform ($r = x - f_{low}(x)$) isolates high-frequency components. Because JPEG blocking artifacts (8x8 DCT grid edges) are also high-frequency, NPR may amplify rather than suppress compression shortcuts.

### 4.3 Prerequisite for Milestone R2 Seal
- The presence of severe format and resolution shortcuts is confirmed on `yangsangtai/tiny-genimage`.
- Before claiming generalizability or forensic perception validity in R2, the research team must decide whether to:
  (a) Train baseline R2 models as-is to empirically measure susceptibility to these shortcuts; or
  (b) Implement explicit bias-control preprocessing (e.g. JPEG recompression of synthetic images, anti-aliasing filters) with dedicated versioning before sealing the evaluation protocol.

*Report automatically generated by ForenSight Milestone R0 Kernel on Kaggle.*
