# Decision: Align semantic and forensic branch preprocessing geometry

Date: 2026-09-26
Status: Accepted

## Context

`build_forensic_input_transform` used `Resize(scale_size=256) + CenterCrop(224)` while the
semantic branch uses OpenCLIP's own preprocessing (`Resize(224) + CenterCrop(224)`). The
function's docstring claimed the two streams shared "identical spatial framing", but that
claim was never measured.

The forensic branch's purpose is to expose low-level synthesis artifacts, so the scale at
which it samples the image is not a neutral detail: a zoom difference between the two
branches is a confound for RQ1's semantic-vs-forensic comparison.

## Evidence

Measured on a 640x480 image whose pixels encode their own source coordinates, reading the
source window actually observed by each branch:

```
semantic (Resize224+CC224): x[ 77.7 -> 558.8]  y[  0.0 -> 477.1]  span = 481.1 x 477.1
forensic (Resize256+CC224): x[107.8 -> 526.2]  y[ 30.1 -> 447.1]  span = 418.5 x 417.0
```

Both windows are centred, but the forensic branch was zoomed in by `256/224 = 1.14x`.
`Resize(n) + CenterCrop(n)` observes a centred square of `min(W, H)` source pixels; the
older 256 setting observed only `224 * min(W,H) / 256`.

## Decision

1. `build_forensic_input_transform` defaults `scale_size` to `image_size`, so the forensic
   branch uses the same `Resize(224) + CenterCrop(224)` geometry as OpenCLIP.
2. Training and evaluation share a single transform; the `is_train` parameter is removed
   because it was already a no-op (verified: both settings produced bit-identical tensors).
3. `test_forensic_transform_source_window_matches_semantic_geometry` pins the observed
   source window so a larger `scale_size` cannot be reintroduced silently.

## Why

This is the smallest change that removes a measurable confound from RQ1 instead of
documenting it away. It adds no architecture and no dependency.

## Consequence

- Every R2 result produced before this change used the previous geometry and must be
  re-run before it is compared with results produced after it.
- `deploy/kaggle/main.py` and `deploy/kaggle_eval/main.py` still apply their own
  `Resize((224, 224))` (an anisotropic square resize) and are declared non-canonical; they
  must not be used to produce numbers that are compared against `src/` runs.
- No change to `docs/architecture.md`: this decision only touches branch preprocessing.
