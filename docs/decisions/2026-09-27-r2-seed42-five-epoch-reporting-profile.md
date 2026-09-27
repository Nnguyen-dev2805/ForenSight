# Decision: Seed-42 Five-Epoch Reporting Profile, and Class-ID Encoding Normalization

Date: 2026-09-27
Status: Accepted for the course-report run

## Context

Two issues surfaced while preparing the seed-42 R2 run.

**1. Reporting scope was undefined.** `docs/decisions/2026-09-27-lock-r2-experimental-backbone.md`
locks an 81-run matrix (3 seeds `42`, `1337`, `2024`; 7 LOGO folds). The same-day plan
`docs/superpowers/plans/2026-09-27-r2-seed42-five-epoch-readiness.md` specifies 9 runs
(1 seed, 1 LOGO fold, 5 epochs). Two incompatible scopes for the same milestone.

**2. The class-balance audit emitted a false warning on every valid dataset.**
Real images are named with ImageNet WNIDs (`n01440764_5969.JPEG`); fake images use a
zero-padded numeric prefix (`001_sdv5_00094.png`). `_class_id_from_path` rendered the
latter as `c0001`, so the two sides occupied provably disjoint namespaces (intersection
size 0 across all four sealed manifests). `audit_manifest_images` then reported
"871 classes missing real samples, 857 missing fake" for a dataset that has no such defect.

## Evidence

**Class encoding is the only defect; there is no real/fake class imbalance.**
Total-variation distance between the real and fake class marginals, against a null model
of independent uniform draws over the same 1000 classes at the observed sample sizes
(200 simulations):

| split | nR | nF | observed TV | null mean | null p95 |
|---|---|---|---|---|---|
| train | 2000 | 2000 | 0.371 | 0.386 | 0.402 |
| val | 250 | 250 | 0.796 | 0.798 | 0.832 |
| in_domain_test | 250 | 250 | 0.820 | 0.800 | 0.832 |
| cross_generator_ood | 3000 | 1999 | 0.355 | 0.360 | 0.376 |

Observed TV is *below* the null mean in every split: the class axis carries no real/fake
signal. Per-class sparsity is likewise expected — sampling 2000 images from 1000 classes
leaves ~117 one-sided classes by chance alone, matching the 120 observed.

**Class-matching Real to Fake is not implementable on this dataset.** Class-set overlap
versus chance is 0.94–1.01 in every split and for every generator, i.e. exactly chance:
no pairing exists to recover. Per-class count correlation for the one fully classifiable
split (`train`) is `r = +0.0502` (1-based) and `+0.0185` (0-based), against a permutation
null with p95 `+0.0518` / `+0.0508` — neither exceeds the null band, so the index
convention is not identifiable from the data. Arithmetically the requirement also fails:
`glide` and `vqdm` filenames carry no class field at all, and `adm` has a `0` prefix that
no 1-based guard accepts, leaving 0/250 matchable in `val`.

**Normalizing the encoding cannot change any split.** Rebuilding the `single` protocol
after normalizing fake prefixes to WNIDs under either convention yields byte-identical
`sample_id` sets in every split, because `class_id` only feeds grouping and the audit,
never allocation.

## Decision

1. **Reporting scope is reduced to the seed-42 five-epoch profile.**
   - Protocols: `single`, LOGO `leave_midjourney`, `all7`
   - Seed: `42`. Epochs: `5`. Variants: `semantic`, `forensic`, `fusion`. Run count: `9`
   - Claim level: **preliminary single-run evidence**
   - The 81-run matrix in `2026-09-27-lock-r2-experimental-backbone.md` is **deferred, not
     superseded**: multi-seed uncertainty remains the target once compute allows.
2. **Class IDs are normalized into one namespace** (`n########`) for real and fake alike:
   WNID filenames pass through; `ILSVRC2012_val_N` resolves via the ImageNet-1k
   val layout (50 images per class, in class order); numeric fake prefixes resolve
   one-based, matching the documented ImageNet ordering (class 1 = `n01440764`).
   A filename carrying no class field returns `None` rather than an invented value.
3. **The `class_balance` finding now tests namespace overlap**, not per-class sparsity.
   It warns only when both sides are present and share no class ID — the actual encoding
   defect — instead of firing on natural sparsity.
4. **Class-matched Real/Fake allocation is rejected.** The premise is false on this
   dataset, and implementing it would fit sampling noise.
5. **The semantic backbone is named `ViT-L-14-quickgelu`.** The OpenAI ViT-L/14 checkpoint
   was trained with QuickGELU; plain `ViT-L-14` uses GELU, so `pretrained="openai"` was
   loading weights into the wrong activation. Both names resolve to identical
   preprocessing, so this changes only the activation function.
6. **`set_seed` pins cuDNN** (`deterministic=True`, `benchmark=False`).
   `torch.use_deterministic_algorithms(True)` is deliberately **not** enabled because it
   rejects CUDA ops this model needs; that limitation is accepted for this cycle.
7. **A fixed-threshold reference report is written alongside the calibrated one**
   (`evaluation_fixed_0_5.json`), from the same prediction scores, so the operational
   reference cannot be selected post hoc on test results.

## Why

The class-balance warning was a permanent false positive that would have blocked the
run's own pre-flight gate and, worse, invited a "fix" (class-matched allocation) that
would have fitted noise. Separating *encoding* from *balance* keeps the audit honest:
it still fails loudly on a genuine namespace mismatch, which is what the old manifests
now do.

The reporting scope is reduced because a single-seed comparison is what the remaining
compute supports; labelling it preliminary is the honest claim level rather than
presenting 9 runs as a sealed benchmark.

## Consequence

- `configs/r2/{semantic,forensic,fusion}.json` now carry `seed=42`, `epochs=5`, and the
  QuickGELU backbone. Learning rates are unchanged (semantic `1e-3`, forensic/fusion `1e-4`).
- **Every R2 result produced before this decision must be re-run** before comparison:
  the backbone activation, the epoch count, the class encoding, and the per-generator
  cohort pairing have all changed.
- Existing 10-epoch artifacts are **exploratory only** and must not be mixed with new runs.
- Test partitions were observed during earlier exploratory runs. No hyperparameter or
  preprocessing change was made in response to those results, and none may be.
- `timm` is now a declared dependency: `forensight.data` imports it directly.
- Deferred: multi-seed uncertainty, JPEG intervention, external datasets.
