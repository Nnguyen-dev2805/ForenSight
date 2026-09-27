# Decision: Adopt Kaggle Tiny-GenImage Seven-Generator Protocol

Date: 2026-09-26
Status: Accepted

## Context

ForenSight previously documented `TheKernel01/Tiny-GenImage` and assumed that
Stable Diffusion v1.4 (`sd14`) was available for canonical Stage-1 training.
Direct verification of the selected Kaggle dataset
`yangsangtai/tiny-genimage` shows seven fake-generator families:

- ADM;
- BigGAN;
- GLIDE;
- Midjourney;
- Stable Diffusion v1.5;
- VQDM;
- Wukong.

The selected Kaggle dataset does not contain SD1.4 samples. Keeping SD1.4 in
the canonical protocol would therefore make the documented experiment
impossible to reproduce from the selected dataset.

## Decision

1. `yangsangtai/tiny-genimage` is the canonical Stage-1 Tiny-GenImage source.
2. SD1.5 is the single training generator for the minimal RQ1 protocol.
3. Validation and in-domain test samples are held-out SD1.5 samples.
4. ADM, BigGAN, GLIDE, Midjourney, VQDM, and Wukong are generator-disjoint
   cross-generator OOD evaluation sets.
5. The previous `near_ood` SD1.4/SD1.5 comparison is removed from protocol v1
   because this dataset does not contain both members of that model-family pair.
6. Threshold calibration remains validation-only and all leakage/bias audit
   requirements remain unchanged.

## Why

This is the smallest protocol change that keeps RQ1 falsifiable while matching
the dataset actually selected by the project. It preserves the single-source
training design and six unseen-generator evaluation sets without inventing data
that is not present.

## Consequence

- Dataset download uses Kaggle (`kagglehub`) rather than the Hugging Face mirror.
- R0/R2 manifests use SD1.5 for train/val/in-domain.
- Any old SD1.4-based results/manifests are legacy artifacts and cannot be mixed
  with the new protocol.
- The earlier Stage-1 dataset decisions are superseded where they require SD1.4.
