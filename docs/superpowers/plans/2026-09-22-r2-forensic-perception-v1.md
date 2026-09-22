# R2 Forensic Perception v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the smallest reproducible R2 experiment that compares frozen CLIP semantic features, NPR-style forensic features, and simple concatenation fusion on the sealed R0 protocol.

**Architecture:** The semantic branch uses frozen OpenCLIP ViT-L/14 and projects its image embedding to 256 dimensions. The forensic branch applies a deterministic NPR-style residual transform, feeds it to a trainable ResNet18, and projects to 256 dimensions. The fusion model concatenates both projections into 512 dimensions and uses a `512 -> 128 -> 1` MLP classifier; all three variants share the same explicit PyTorch training, prediction, and R0 evaluation path.

**Tech Stack:** Python 3.11+, PyTorch, torchvision, open_clip_torch, Pillow, existing ForenSight `Manifest`, `PredictionSet`, `evaluate_predictions`, and reproducibility utilities.

**Spec:** `docs/plans/r2-forensic-perception-v1.md`

## Global Constraints

- Train only on GenImage SD1.4; validation is GenImage SD1.4 validation.
- Evaluate on SD1.4 in-domain, SD1.5 near-OOD, Midjourney/ADM/GLIDE/Wukong/VQDM/BigGAN cross-generator OOD, and GenImage++ modern external.
- WildRF and Chameleon/AIDE are not active R2 v1 runtime datasets.
- CLIP ViT-L/14 uses OpenAI pretrained weights and remains frozen in R2 v1.
- Forensic input is an NPR-inspired deterministic residual, not raw RGB in the required forensic experiment.
- ResNet18 is trainable and starts without ImageNet pretrained weights in this baseline (`weights=None`).
- Semantic projection dimension is 256; forensic projection dimension is 256; fusion dimension is 512.
- Fusion is concatenation only; classifier is `512 -> 128 -> 1` with ReLU and Dropout.
- Loss is `BCEWithLogitsLoss`; optimizer baseline is AdamW; no scheduler is added initially.
- Threshold is selected only from validation predictions and frozen before test/OOD evaluation.
- Required ablation is exactly Semantic-only, Forensic-only, and Fusion.
- Do not add LoRA, CLIP fine-tuning, FFT/DCT/SRM branches, attention/gating fusion, MLLM integration, evidence localization, Lightning, Hydra, W&B, or MLflow in this plan.
- Unit tests must not download CLIP weights or access the network; inject tiny dummy backbones in tests.
- Raw data is immutable; R2 only reads manifests/images and writes checkpoints, predictions, reports, and reproducibility artifacts outside `data/raw/`.

---

## File Structure

Create or modify these files only unless a test exposes a concrete need:

```text
pyproject.toml

src/forensight/
├── data/
│   └── r2_dataset.py
├── models/
│   ├── __init__.py
│   ├── semantic.py
│   ├── forensic.py
│   └── fusion.py
└── training/
    ├── __init__.py
    └── r2.py

configs/r2/
├── semantic.json
├── forensic.json
└── fusion.json

scripts/
├── train_r2.py
└── evaluate_r2.py

tests/
├── data/
│   └── test_r2_dataset.py
├── models/
│   ├── __init__.py
│   ├── test_semantic.py
│   ├── test_forensic.py
│   └── test_fusion.py
├── training/
│   ├── __init__.py
│   └── test_r2.py
└── system/
    └── test_r2_smoke.py
```

Responsibilities:

- `r2_dataset.py`: load R0 manifest records and convert one PIL image into branch-specific tensors while preserving metadata.
- `semantic.py`: frozen CLIP encoder, semantic projection, semantic-only classifier.
- `forensic.py`: NPR transform, trainable ResNet18 encoder, forensic projection, forensic-only classifier.
- `fusion.py`: concatenate semantic/forensic projected features and classify.
- `training/r2.py`: shared seed, forward dispatch, train/validation steps, checkpoint I/O, prediction export, config validation/model assembly.
- `train_r2.py`: CLI orchestration for train + validation checkpoint selection only.
- `evaluate_r2.py`: load a selected checkpoint, produce validation + evaluation predictions, then call the existing R0 evaluator.

---

### Task 1: Add R2 Dependencies and Manifest-backed Image Dataset

**Files:**
- Modify: `pyproject.toml`
- Create: `src/forensight/data/r2_dataset.py`
- Create: `tests/data/test_r2_dataset.py`

**Interfaces:**
- Consumes: `forensight.data.split.Manifest`, `ManifestRecord`.
- Produces: `load_manifest_file(path: str | Path) -> Manifest`, `R2ImageDataset`, `build_forensic_input_transform(image_size: int) -> Callable`.

- [ ] **Step 1: Add failing dataset tests**

Create `tests/data/test_r2_dataset.py` with tests equivalent to:

```python
from pathlib import Path

import torch
from PIL import Image

from forensight.data.r2_dataset import R2ImageDataset, load_manifest_file
from forensight.data.split import Manifest, ManifestRecord


def test_load_manifest_file_jsonl(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    Manifest([
        ManifestRecord(
            sample_id="s1",
            image_path="image.png",
            label=1,
            dataset="genimage",
            generator="sd14",
            split="train",
        )
    ]).to_jsonl(manifest_path)

    loaded = load_manifest_file(manifest_path)
    assert len(loaded) == 1
    assert loaded[0].sample_id == "s1"


def test_dataset_loads_image_once_and_preserves_metadata(tmp_path: Path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (16, 12), color=(100, 110, 120)).save(image_path)
    manifest = Manifest([
        ManifestRecord(
            sample_id="s1",
            image_path=str(image_path),
            label=1,
            dataset="genimage",
            generator="sd14",
            split="train",
        )
    ])

    def clip_transform(image):
        return torch.ones(3, 8, 8)

    def forensic_transform(image):
        return torch.zeros(3, 8, 8)

    dataset = R2ImageDataset(
        manifest,
        clip_transform=clip_transform,
        forensic_transform=forensic_transform,
    )
    sample = dataset[0]

    assert sample["clip_image"].shape == (3, 8, 8)
    assert sample["forensic_image"].shape == (3, 8, 8)
    assert sample["label"].dtype == torch.float32
    assert sample["label"].item() == 1.0
    assert sample["sample_id"] == "s1"
    assert sample["split"] == "train"
    assert sample["generator"] == "sd14"
    assert sample["dataset"] == "genimage"
```

- [ ] **Step 2: Run the tests and verify they fail because the R2 dataset module does not exist**

Run:

```bash
python3 -m pytest tests/data/test_r2_dataset.py -v
```

Expected: collection/import failure for `forensight.data.r2_dataset`.

- [ ] **Step 3: Add the minimum runtime dependencies**

Modify `pyproject.toml` dependencies to include:

```toml
dependencies = [
    "huggingface-hub>=0.34",
    "numpy",
    "pandas",
    "Pillow",
    "scikit-learn",
    "torch",
    "torchvision",
    "open_clip_torch",
]
```

Do not add Lightning, Hydra, timm directly, or experiment tracking libraries. Transitive dependencies are managed by the selected packages.

- [ ] **Step 4: Implement the manifest loader and dataset**

Create `src/forensight/data/r2_dataset.py` with these public interfaces:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from forensight.data.split import Manifest


ImageTransform = Callable[[Image.Image], torch.Tensor]


def load_manifest_file(path: str | Path) -> Manifest:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return Manifest.from_jsonl(path)
    if suffix == ".csv":
        return Manifest.from_csv(path)
    raise ValueError(f"Unsupported manifest format: {path.suffix}")


def build_forensic_input_transform(image_size: int = 224) -> ImageTransform:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


class R2ImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: Manifest,
        *,
        base_dir: str | Path | None = None,
        clip_transform: ImageTransform | None = None,
        forensic_transform: ImageTransform | None = None,
    ) -> None:
        if clip_transform is None and forensic_transform is None:
            raise ValueError("At least one branch transform must be provided.")
        self.manifest = manifest
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.clip_transform = clip_transform
        self.forensic_transform = forensic_transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.manifest[index]
        path = Path(record.image_path)
        if not path.is_absolute() and self.base_dir is not None:
            path = self.base_dir / path

        with Image.open(path) as opened:
            image = opened.convert("RGB")
            sample: dict[str, Any] = {
                "label": torch.tensor(float(record.label), dtype=torch.float32),
                "sample_id": record.sample_id,
                "split": record.split,
                "generator": record.generator,
                "dataset": record.dataset,
            }
            if self.clip_transform is not None:
                sample["clip_image"] = self.clip_transform(image)
            if self.forensic_transform is not None:
                sample["forensic_image"] = self.forensic_transform(image)
            return sample
```

The image is opened once and both transforms operate on the same converted PIL image.

- [ ] **Step 5: Run dataset tests**

Run:

```bash
python3 -m pytest tests/data/test_r2_dataset.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit the dataset adapter**

```bash
git add pyproject.toml src/forensight/data/r2_dataset.py tests/data/test_r2_dataset.py
git commit -m "feat: add R2 manifest image dataset"
```

---

### Task 2: Implement Frozen Semantic Encoder and Semantic-only Detector

**Files:**
- Create: `src/forensight/models/__init__.py`
- Create: `src/forensight/models/semantic.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/test_semantic.py`

**Interfaces:**
- Consumes: a CLIP-like module exposing `encode_image(tensor) -> tensor`.
- Produces: `SemanticEncoder`, `SemanticOnlyDetector`, `load_open_clip_semantic(...) -> tuple[SemanticEncoder, Callable]`.

- [ ] **Step 1: Write failing semantic tests with a dummy CLIP backbone**

Create `tests/models/test_semantic.py`:

```python
import torch
from torch import nn

from forensight.models.semantic import SemanticEncoder, SemanticOnlyDetector


class DummyClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(12, 8)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


def test_semantic_encoder_freezes_backbone_and_projects_to_256():
    backbone = DummyClip()
    encoder = SemanticEncoder(backbone, feature_dim=8, projection_dim=256)
    output = encoder(torch.randn(2, 3, 2, 2))

    assert output.shape == (2, 256)
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_semantic_detector_backpropagates_only_through_trainable_head():
    backbone = DummyClip()
    detector = SemanticOnlyDetector(
        SemanticEncoder(backbone, feature_dim=8, projection_dim=256),
        hidden_dim=128,
        dropout=0.2,
    )
    logits = detector(torch.randn(2, 3, 2, 2))
    logits.sum().backward()

    assert logits.shape == (2, 1)
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert any(
        parameter.grad is not None
        for parameter in detector.classifier.parameters()
    )
```

- [ ] **Step 2: Run the semantic tests and verify RED**

Run:

```bash
python3 -m pytest tests/models/test_semantic.py -v
```

Expected: import failure because `forensight.models.semantic` does not exist.

- [ ] **Step 3: Implement the semantic encoder and detector**

Create `src/forensight/models/semantic.py`:

```python
from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn


class SemanticEncoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: int,
        projection_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.projection = nn.Linear(feature_dim, projection_dim)
        self.norm = nn.LayerNorm(projection_dim)

    def train(self, mode: bool = True) -> "SemanticEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone.encode_image(image)
        return self.norm(self.projection(features.float()))


class SemanticOnlyDetector(nn.Module):
    def __init__(
        self,
        encoder: SemanticEncoder,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        projection_dim = encoder.projection.out_features
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(image))


def load_open_clip_semantic(
    *,
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    projection_dim: int = 256,
) -> tuple[SemanticEncoder, Callable[[Any], torch.Tensor]]:
    import open_clip

    backbone, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    feature_dim = int(backbone.visual.output_dim)
    encoder = SemanticEncoder(
        backbone,
        feature_dim=feature_dim,
        projection_dim=projection_dim,
    )
    return encoder, preprocess
```

Export the three public symbols from `src/forensight/models/__init__.py`.

- [ ] **Step 4: Run semantic tests and verify GREEN**

```bash
python3 -m pytest tests/models/test_semantic.py -v
```

Expected: PASS with no network access.

- [ ] **Step 5: Commit semantic branch**

```bash
git add src/forensight/models tests/models
git commit -m "feat: add frozen semantic detector"
```

---

### Task 3: Implement NPR Transform and Forensic-only Detector

**Files:**
- Create: `src/forensight/models/forensic.py`
- Create: `tests/models/test_forensic.py`
- Modify: `src/forensight/models/__init__.py`

**Interfaces:**
- Produces: `NPRTransform`, `ForensicEncoder`, `ForensicOnlyDetector`, `build_resnet18_forensic(...) -> ForensicEncoder`.
- Later consumed by: `FusionDetector`, shared R2 model builder.

- [ ] **Step 1: Write failing forensic tests**

Create `tests/models/test_forensic.py`:

```python
import torch
from torch import nn

from forensight.models.forensic import (
    ForensicEncoder,
    ForensicOnlyDetector,
    NPRTransform,
)


class DummyForensicBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(2, 3))


def test_npr_transform_is_deterministic_and_shape_preserving():
    transform = NPRTransform(scale_factor=0.5, mode="bilinear")
    image = torch.rand(2, 3, 16, 16)

    first = transform(image)
    second = transform(image)

    assert first.shape == image.shape
    assert torch.equal(first, second)


def test_forensic_detector_outputs_binary_logits_and_trains_backbone():
    backbone = DummyForensicBackbone()
    encoder = ForensicEncoder(
        backbone,
        feature_dim=4,
        projection_dim=256,
        npr=NPRTransform(scale_factor=0.5),
    )
    detector = ForensicOnlyDetector(encoder, hidden_dim=128, dropout=0.2)
    logits = detector(torch.rand(2, 3, 16, 16))
    logits.sum().backward()

    assert logits.shape == (2, 1)
    assert backbone.conv.weight.grad is not None
```

- [ ] **Step 2: Run forensic tests and verify RED**

```bash
python3 -m pytest tests/models/test_forensic.py -v
```

Expected: import failure for `forensight.models.forensic`.

- [ ] **Step 3: Implement the NPR transform and forensic classes**

Create `src/forensight/models/forensic.py` with:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


class NPRTransform(nn.Module):
    def __init__(self, *, scale_factor: float = 0.5, mode: str = "bilinear") -> None:
        super().__init__()
        if not 0.0 < scale_factor < 1.0:
            raise ValueError("scale_factor must be between 0 and 1.")
        if mode not in {"bilinear", "bicubic"}:
            raise ValueError("mode must be 'bilinear' or 'bicubic'.")
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        reduced = F.interpolate(
            image,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=False,
            recompute_scale_factor=False,
        )
        reconstructed = F.interpolate(
            reduced,
            size=image.shape[-2:],
            mode=self.mode,
            align_corners=False,
        )
        return image - reconstructed


class ForensicEncoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: int,
        projection_dim: int = 256,
        npr: NPRTransform | None = None,
    ) -> None:
        super().__init__()
        self.npr = npr or NPRTransform()
        self.backbone = backbone
        self.projection = nn.Linear(feature_dim, projection_dim)
        self.norm = nn.LayerNorm(projection_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residual = self.npr(image)
        features = self.backbone(residual)
        return self.norm(self.projection(features.float()))


class ForensicOnlyDetector(nn.Module):
    def __init__(
        self,
        encoder: ForensicEncoder,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        projection_dim = encoder.projection.out_features
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(image))


def build_resnet18_forensic(
    *,
    projection_dim: int = 256,
    npr_scale_factor: float = 0.5,
    npr_mode: str = "bilinear",
) -> ForensicEncoder:
    backbone = resnet18(weights=None)
    feature_dim = int(backbone.fc.in_features)
    backbone.fc = nn.Identity()
    return ForensicEncoder(
        backbone,
        feature_dim=feature_dim,
        projection_dim=projection_dim,
        npr=NPRTransform(scale_factor=npr_scale_factor, mode=npr_mode),
    )
```

Export the public symbols from `src/forensight/models/__init__.py`.

- [ ] **Step 4: Run forensic tests and verify GREEN**

```bash
python3 -m pytest tests/models/test_forensic.py -v
```

Expected: PASS.

- [ ] **Step 5: Record the NPR baseline definition in code comments**

Add a short module docstring stating that this is the fixed ForenSight NPR-inspired residual baseline (`input - upsample(downsample(input))`), not a claim of exact full-paper reproduction. This prevents later reports from over-claiming the implementation.

- [ ] **Step 6: Commit forensic branch**

```bash
git add src/forensight/models/forensic.py src/forensight/models/__init__.py tests/models/test_forensic.py
git commit -m "feat: add NPR forensic detector"
```

---

### Task 4: Implement Simple Semantic + Forensic Fusion

**Files:**
- Create: `src/forensight/models/fusion.py`
- Create: `tests/models/test_fusion.py`
- Modify: `src/forensight/models/__init__.py`

**Interfaces:**
- Consumes: `SemanticEncoder.forward(clip_image) -> [B, 256]`, `ForensicEncoder.forward(forensic_image) -> [B, 256]`.
- Produces: `FusionDetector.forward(clip_image, forensic_image) -> [B, 1]`.

- [ ] **Step 1: Write the failing fusion test**

Create `tests/models/test_fusion.py` using the dummy backbones from the previous tasks or local equivalents:

```python
import torch
from torch import nn

from forensight.models.forensic import ForensicEncoder, NPRTransform
from forensight.models.fusion import FusionDetector
from forensight.models.semantic import SemanticEncoder


class DummyClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(12, 8)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


class DummyForensic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(2, 3))


def test_fusion_outputs_binary_logit_and_keeps_clip_frozen():
    clip = DummyClip()
    semantic = SemanticEncoder(clip, feature_dim=8, projection_dim=256)
    forensic_backbone = DummyForensic()
    forensic = ForensicEncoder(
        forensic_backbone,
        feature_dim=4,
        projection_dim=256,
        npr=NPRTransform(scale_factor=0.5),
    )
    model = FusionDetector(semantic, forensic, hidden_dim=128, dropout=0.2)

    logits = model(
        torch.randn(2, 3, 2, 2),
        torch.rand(2, 3, 16, 16),
    )
    logits.sum().backward()

    assert logits.shape == (2, 1)
    assert model.fused_dim == 512
    assert all(parameter.grad is None for parameter in clip.parameters())
    assert forensic_backbone.conv.weight.grad is not None
```

- [ ] **Step 2: Run fusion test and verify RED**

```bash
python3 -m pytest tests/models/test_fusion.py -v
```

Expected: import failure for `forensight.models.fusion`.

- [ ] **Step 3: Implement concat-only fusion**

Create `src/forensight/models/fusion.py`:

```python
from __future__ import annotations

import torch
from torch import nn

from forensight.models.forensic import ForensicEncoder
from forensight.models.semantic import SemanticEncoder


class FusionDetector(nn.Module):
    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        forensic_encoder: ForensicEncoder,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.semantic_encoder = semantic_encoder
        self.forensic_encoder = forensic_encoder
        semantic_dim = semantic_encoder.projection.out_features
        forensic_dim = forensic_encoder.projection.out_features
        self.fused_dim = semantic_dim + forensic_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        clip_image: torch.Tensor,
        forensic_image: torch.Tensor,
    ) -> torch.Tensor:
        semantic = self.semantic_encoder(clip_image)
        forensic = self.forensic_encoder(forensic_image)
        fused = torch.cat([semantic, forensic], dim=-1)
        return self.classifier(fused)
```

- [ ] **Step 4: Run all model tests**

```bash
python3 -m pytest tests/models -v
```

Expected: all semantic, forensic, and fusion tests PASS.

- [ ] **Step 5: Commit fusion model**

```bash
git add src/forensight/models/fusion.py src/forensight/models/__init__.py tests/models/test_fusion.py
git commit -m "feat: add R2 concat fusion detector"
```

---

### Task 5: Add R2 Configs and Shared Training Utilities

**Files:**
- Create: `src/forensight/training/__init__.py`
- Create: `src/forensight/training/r2.py`
- Create: `tests/training/__init__.py`
- Create: `tests/training/test_r2.py`
- Create: `configs/r2/semantic.json`
- Create: `configs/r2/forensic.json`
- Create: `configs/r2/fusion.json`

**Interfaces:**
- Produces: `load_r2_config`, `set_seed`, `forward_variant`, `train_one_epoch`, `validate_one_epoch`, `save_checkpoint`, `load_checkpoint`, `predict_to_prediction_set`.
- Consumes: batch dictionaries from `R2ImageDataset`; existing `PredictionRecord`, `PredictionSet`.

- [ ] **Step 1: Write failing training utility tests**

Create `tests/training/test_r2.py` with a tiny detector and tiny batch loader:

```python
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from forensight.training.r2 import (
    forward_variant,
    load_checkpoint,
    predict_to_prediction_set,
    save_checkpoint,
    train_one_epoch,
)


class TinySemantic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(12, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.fc(image.flatten(1))


def tiny_loader() -> DataLoader:
    samples = [
        {
            "clip_image": torch.zeros(3, 2, 2),
            "label": torch.tensor(0.0),
            "sample_id": "real",
            "split": "val",
            "generator": "nature",
            "dataset": "genimage",
        },
        {
            "clip_image": torch.ones(3, 2, 2),
            "label": torch.tensor(1.0),
            "sample_id": "fake",
            "split": "val",
            "generator": "sd14",
            "dataset": "genimage",
        },
    ]
    return DataLoader(samples, batch_size=2, shuffle=False)


def test_train_one_epoch_produces_finite_loss():
    model = TinySemantic()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_one_epoch(
        model,
        tiny_loader(),
        optimizer,
        device=torch.device("cpu"),
        variant="semantic",
    )
    assert torch.isfinite(torch.tensor(loss))


def test_prediction_export_matches_r0_schema():
    predictions = predict_to_prediction_set(
        TinySemantic(),
        tiny_loader(),
        device=torch.device("cpu"),
        variant="semantic",
    )
    assert len(predictions) == 2
    assert predictions[0].sample_id == "real"
    assert predictions[0].split == "val"
    assert 0.0 <= predictions[0].score <= 1.0


def test_checkpoint_round_trip(tmp_path: Path):
    path = tmp_path / "checkpoint.pt"
    model = TinySemantic()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(path, model, optimizer, epoch=3, config={"variant": "semantic"})

    restored = TinySemantic()
    state = load_checkpoint(path, restored, map_location="cpu")
    assert state["epoch"] == 3
    assert state["config"]["variant"] == "semantic"
```

- [ ] **Step 2: Run the training tests and verify RED**

```bash
python3 -m pytest tests/training/test_r2.py -v
```

Expected: import failure for `forensight.training.r2`.

- [ ] **Step 3: Create the three explicit JSON configs**

Create `configs/r2/semantic.json`:

```json
{
  "variant": "semantic",
  "seed": 42,
  "model": {
    "clip_model": "ViT-L-14",
    "clip_pretrained": "openai",
    "projection_dim": 256,
    "hidden_dim": 128,
    "dropout": 0.2,
    "image_size": 224,
    "npr_scale_factor": 0.5,
    "npr_mode": "bilinear"
  },
  "training": {
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "num_workers": 0
  },
  "evaluation": {
    "threshold_strategy": "f1"
  }
}
```

Create `configs/r2/forensic.json` with the same structure but:

```json
{
  "variant": "forensic",
  "seed": 42,
  "model": {
    "clip_model": "ViT-L-14",
    "clip_pretrained": "openai",
    "projection_dim": 256,
    "hidden_dim": 128,
    "dropout": 0.2,
    "image_size": 224,
    "npr_scale_factor": 0.5,
    "npr_mode": "bilinear"
  },
  "training": {
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "num_workers": 0
  },
  "evaluation": {
    "threshold_strategy": "f1"
  }
}
```

Create `configs/r2/fusion.json` with `"variant": "fusion"` and the same optimization values as forensic.

These are starting baselines, not validated hyperparameters.

- [ ] **Step 4: Implement shared training utilities**

Create `src/forensight/training/r2.py` with these exact public signatures:

```python
def load_r2_config(path: str | Path) -> dict[str, Any]: ...

def set_seed(seed: int) -> None: ...

def forward_variant(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    variant: str,
    device: torch.device,
) -> torch.Tensor: ...

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    variant: str,
) -> float: ...

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    variant: str,
) -> float: ...

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    config: dict[str, Any],
) -> None: ...

def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]: ...

@torch.no_grad()
def predict_to_prediction_set(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    variant: str,
) -> PredictionSet: ...
```

Core `forward_variant` behavior:

```python
if variant == "semantic":
    return model(batch["clip_image"].to(device))
if variant == "forensic":
    return model(batch["forensic_image"].to(device))
if variant == "fusion":
    return model(
        batch["clip_image"].to(device),
        batch["forensic_image"].to(device),
    )
raise ValueError(f"Unsupported R2 variant: {variant}")
```

Training loss behavior:

```python
criterion = nn.BCEWithLogitsLoss()
labels = batch["label"].to(device).float().view(-1, 1)
logits = forward_variant(model, batch, variant=variant, device=device)
loss = criterion(logits, labels)
```

Prediction conversion behavior:

```python
scores = torch.sigmoid(logits).squeeze(1).cpu().tolist()
records.append(PredictionRecord(
    sample_id=batch["sample_id"][index],
    label=int(batch["label"][index].item()),
    score=float(scores[index]),
    split=batch["split"][index],
    generator=batch["generator"][index],
    dataset=batch["dataset"][index],
))
```

`load_r2_config` must reject variants outside `{semantic, forensic, fusion}` and require top-level `model`, `training`, and `evaluation` dictionaries.

- [ ] **Step 5: Run training utility tests and all R2 unit tests**

```bash
python3 -m pytest tests/data/test_r2_dataset.py tests/models tests/training/test_r2.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit shared training utilities and configs**

```bash
git add src/forensight/training tests/training configs/r2
git commit -m "feat: add shared R2 training utilities"
```

---

### Task 6: Add Model Assembly and Training CLI

**Files:**
- Modify: `src/forensight/training/r2.py`
- Create: `scripts/train_r2.py`
- Modify: `tests/training/test_r2.py`

**Interfaces:**
- Produces: `build_r2_model(config) -> tuple[nn.Module, Callable | None, Callable | None]`, `select_device() -> torch.device`.
- CLI consumes: config path, train manifest, val manifest, base dir, output dir.
- CLI produces: `best.pt`, `history.json`, copied resolved `config.json`.

- [ ] **Step 1: Add failing model assembly tests using monkeypatch**

Add tests that monkeypatch `load_open_clip_semantic` and `build_resnet18_forensic` so no weights are downloaded. Assert:

```python
model, clip_transform, forensic_transform = build_r2_model(config)
assert model is not None

if config["variant"] == "semantic":
    assert clip_transform is not None
    assert forensic_transform is None
elif config["variant"] == "forensic":
    assert clip_transform is None
    assert forensic_transform is not None
else:
    assert clip_transform is not None
    assert forensic_transform is not None
```

Use local dummy encoders in the monkeypatch return values; do not instantiate real OpenCLIP in unit tests.

- [ ] **Step 2: Run targeted assembly test and verify RED**

```bash
python3 -m pytest tests/training/test_r2.py -k build_r2_model -v
```

Expected: FAIL because `build_r2_model` does not exist.

- [ ] **Step 3: Implement simple model assembly**

Add to `src/forensight/training/r2.py`:

```python
def build_r2_model(
    config: dict[str, Any],
) -> tuple[nn.Module, Callable | None, Callable | None]:
    variant = config["variant"]
    model_cfg = config["model"]
    clip_transform = None
    forensic_transform = None

    if variant in {"semantic", "fusion"}:
        semantic_encoder, clip_transform = load_open_clip_semantic(
            model_name=model_cfg["clip_model"],
            pretrained=model_cfg["clip_pretrained"],
            projection_dim=model_cfg["projection_dim"],
        )

    if variant in {"forensic", "fusion"}:
        forensic_encoder = build_resnet18_forensic(
            projection_dim=model_cfg["projection_dim"],
            npr_scale_factor=model_cfg["npr_scale_factor"],
            npr_mode=model_cfg["npr_mode"],
        )
        forensic_transform = build_forensic_input_transform(model_cfg["image_size"])

    if variant == "semantic":
        model = SemanticOnlyDetector(
            semantic_encoder,
            hidden_dim=model_cfg["hidden_dim"],
            dropout=model_cfg["dropout"],
        )
    elif variant == "forensic":
        model = ForensicOnlyDetector(
            forensic_encoder,
            hidden_dim=model_cfg["hidden_dim"],
            dropout=model_cfg["dropout"],
        )
    elif variant == "fusion":
        model = FusionDetector(
            semantic_encoder,
            forensic_encoder,
            hidden_dim=model_cfg["hidden_dim"],
            dropout=model_cfg["dropout"],
        )
    else:
        raise ValueError(f"Unsupported R2 variant: {variant}")

    return model, clip_transform, forensic_transform
```

Keep this as a single explicit conditional; do not introduce a model registry/factory framework.

- [ ] **Step 4: Implement `scripts/train_r2.py` with validation-selected checkpointing**

The CLI arguments must be:

```text
--config
--train-manifest
--val-manifest
--base-dir
--output-dir
```

The script must:

1. load config;
2. seed Python/NumPy/PyTorch;
3. build model + transforms;
4. create `R2ImageDataset` for train and val only;
5. create DataLoaders;
6. construct `AdamW` from trainable parameters only;
7. train for configured epochs;
8. compute validation loss every epoch;
9. save `best.pt` only when validation loss improves;
10. save `history.json` containing `epoch`, `train_loss`, `val_loss`;
11. save the resolved config as `config.json` beside the checkpoint.

The optimizer construction must be:

```python
trainable_parameters = [
    parameter for parameter in model.parameters() if parameter.requires_grad
]
optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=config["training"]["learning_rate"],
    weight_decay=config["training"]["weight_decay"],
)
```

The script must have no `--test-manifest` argument.

- [ ] **Step 5: Add CLI parsing test**

Import `scripts.train_r2` in the test and verify parsing accepts train/val inputs but exposes no test tuning input. Example:

```python
args = train_r2.parse_args([
    "--config", "configs/r2/semantic.json",
    "--train-manifest", "train.jsonl",
    "--val-manifest", "val.jsonl",
    "--output-dir", "results/r2/semantic/seed42",
])
assert args.train_manifest == "train.jsonl"
assert args.val_manifest == "val.jsonl"
assert not hasattr(args, "test_manifest")
```

- [ ] **Step 6: Run training and CLI tests**

```bash
python3 -m pytest tests/training/test_r2.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit model assembly and training CLI**

```bash
git add src/forensight/training/r2.py scripts/train_r2.py tests/training/test_r2.py
git commit -m "feat: add R2 training CLI"
```

---

### Task 7: Integrate Prediction Export with the Existing R0 Evaluator

**Files:**
- Create: `scripts/evaluate_r2.py`
- Modify: `tests/training/test_r2.py` or create a focused `tests/system/test_r2_evaluation.py`

**Interfaces:**
- Consumes: one selected checkpoint, one validation manifest, one or more evaluation manifests.
- Produces: combined `PredictionSet` JSONL, `EvaluationReport` JSON/Markdown, and `ReproducibilityRecord` JSON.
- Reuses: `evaluate_predictions`, `create_reproducibility_record` without reimplementing metrics or threshold search.

- [ ] **Step 1: Write a failing R0-compatibility test**

Build a small `PredictionSet` with validation and test records and verify the R2 evaluation helper calls the existing evaluator with validation calibration. The core assertion must be:

```python
report = evaluate_predictions(
    predictions,
    val_split_name="val",
    threshold_strategy="f1",
)
assert report.threshold_metadata["threshold_source"].startswith("val_")
```

The R2 layer must not implement its own threshold selector.

- [ ] **Step 2: Implement `scripts/evaluate_r2.py`**

Required CLI:

```text
--config
--checkpoint
--val-manifest
--eval-manifest <path> [<path> ...]
--base-dir
--predictions-out
--report-json
--report-md
--repro-json
--split-version
```

Execution flow:

```python
config = load_r2_config(args.config)
model, clip_transform, forensic_transform = build_r2_model(config)
load_checkpoint(args.checkpoint, model, map_location=device)

val_predictions = predict_manifest(...args.val_manifest...)
eval_predictions = [predict_manifest(...path...) for path in args.eval_manifest]
combined = PredictionSet([
    *val_predictions.records,
    *[record for prediction_set in eval_predictions for record in prediction_set.records],
])
combined.to_jsonl(args.predictions_out)

report = evaluate_predictions(
    combined,
    val_split_name="val",
    threshold_strategy=config["evaluation"]["threshold_strategy"],
    seed=config["seed"],
)
report.save_json(args.report_json)
report.save_markdown(args.report_md)
```

Then create a reproducibility record using the existing `create_reproducibility_record(...)` and save it to `args.repro_json`.

Add a small local helper `predict_manifest(...)` inside `evaluate_r2.py`; do not create a second evaluation framework.

- [ ] **Step 3: Verify prediction fields exactly match R0 expectations**

Each exported record must contain:

```json
{
  "sample_id": "...",
  "label": 0,
  "score": 0.123,
  "split": "in_domain_test",
  "generator": "sd14",
  "dataset": "genimage",
  "metadata": {}
}
```

No separate `predicted_label` field is needed; the existing evaluator applies the frozen validation threshold to `score`.

- [ ] **Step 4: Run evaluator compatibility tests and existing R0 evaluation tests**

```bash
python3 -m pytest tests/training/test_r2.py tests/evaluation -v
```

Expected: PASS, proving R2 did not regress R0 evaluation behavior.

- [ ] **Step 5: Commit R0 evaluation integration**

```bash
git add scripts/evaluate_r2.py tests/training/test_r2.py
git commit -m "feat: integrate R2 with R0 evaluation"
```

---

### Task 8: Add Network-free End-to-End R2 Smoke Test

**Files:**
- Create: `tests/system/test_r2_smoke.py`

**Interfaces:**
- Exercises: dataset -> model -> loss -> backward -> optimizer -> checkpoint -> reload -> prediction export -> R0 evaluator.
- Does not use real CLIP weights or real GenImage data.

- [ ] **Step 1: Build a synthetic two-class image manifest in the test**

The test creates four tiny PNGs and a manifest containing train, val, and in-domain test records. Use deterministic colors/patterns and `tmp_path`; do not depend on repository datasets.

- [ ] **Step 2: Inject dummy semantic and forensic backbones**

Use the same dummy modules from unit tests to build Semantic-only, Forensic-only, and Fusion models without OpenCLIP downloads.

- [ ] **Step 3: Execute one optimizer step for every variant**

For each of `semantic`, `forensic`, `fusion`:

```text
DataLoader
-> forward_variant
-> BCEWithLogitsLoss
-> backward
-> AdamW.step
-> save_checkpoint
-> load_checkpoint
-> predict_to_prediction_set
```

Assert loss is finite and each model emits `[B, 1]` logits.

- [ ] **Step 4: Run predictions through the R0 evaluator**

Combine validation and test predictions, call:

```python
report = evaluate_predictions(
    predictions,
    val_split_name="val",
    threshold_strategy="f1",
)
```

Assert:

```python
assert report.threshold_metadata["calibrated"] is True
assert "in_domain_test" in report.by_split
```

- [ ] **Step 5: Run the system smoke test**

```bash
python3 -m pytest tests/system/test_r2_smoke.py -v
```

Expected: PASS entirely offline.

- [ ] **Step 6: Run the complete test suite**

```bash
python3 -m pytest
```

Expected: all existing R0 tests plus new R2 tests PASS.

- [ ] **Step 7: Commit R2 smoke coverage**

```bash
git add tests/system/test_r2_smoke.py
git commit -m "test: add R2 end-to-end smoke coverage"
```

---

### Task 9: Run the Pilot Ablation Without Making Research Claims

**Files:**
- No code changes unless the smoke test exposes a concrete defect.
- Runtime artifacts: `results/r2/<variant>/seed42/` and prediction/report files, which must remain uncommitted if covered by `.gitignore`.

**Interfaces:**
- Consumes: sealed R0 pilot train/val/evaluation manifests.
- Produces: one preliminary report for each of Semantic-only, Forensic-only, Fusion.

- [ ] **Step 1: Confirm pilot manifests pass R0 leakage/audit checks**

Run the existing audit command for the selected pilot manifest(s):

```bash
python3 scripts/check_dataset.py --manifest <pilot-manifest.jsonl>
```

Do not continue if generator/sample leakage is reported as a critical failure.

- [ ] **Step 2: Train Semantic-only**

```bash
python3 scripts/train_r2.py \
  --config configs/r2/semantic.json \
  --train-manifest <sd14-pilot-train.jsonl> \
  --val-manifest <sd14-pilot-val.jsonl> \
  --base-dir <dataset-root> \
  --output-dir results/r2/semantic/seed42
```

- [ ] **Step 3: Train Forensic-only**

```bash
python3 scripts/train_r2.py \
  --config configs/r2/forensic.json \
  --train-manifest <sd14-pilot-train.jsonl> \
  --val-manifest <sd14-pilot-val.jsonl> \
  --base-dir <dataset-root> \
  --output-dir results/r2/forensic/seed42
```

- [ ] **Step 4: Train Fusion**

```bash
python3 scripts/train_r2.py \
  --config configs/r2/fusion.json \
  --train-manifest <sd14-pilot-train.jsonl> \
  --val-manifest <sd14-pilot-val.jsonl> \
  --base-dir <dataset-root> \
  --output-dir results/r2/fusion/seed42
```

- [ ] **Step 5: Evaluate each selected checkpoint with the same hierarchy**

For each variant, run `scripts/evaluate_r2.py` using:

```text
validation: SD1.4 val
evaluation: SD1.4 in-domain, SD1.5, Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN, GenImage++
```

Use the same manifest versions for all variants.

- [ ] **Step 6: Produce a preliminary comparison table**

Record at least AUROC, Accuracy, F1, Precision, Recall per split/generator and calculate mean cross-generator AUROC over the six unseen GenImage generators.

Label the table **Preliminary / pilot / single-seed**. Do not claim one variant is generally better from this run.

- [ ] **Step 7: Write failure notes before changing architecture**

For each model record:

```text
training stability
validation gap
in-domain vs OOD gap
weakest generator
strongest generator
obvious data/preprocessing anomaly
```

Only a measured failure can motivate a change to R2 v1.

---

### Task 10: Run the Main Three-way Ablation and Seal R2 Results

**Files:**
- No architecture changes.
- Runtime artifacts only: main checkpoints, predictions, reports, reproducibility records, aggregated reports.

**Interfaces:**
- Consumes: sealed `main` R0 manifests and the code that passed Tasks 1-9.
- Produces: reproducible multi-seed Semantic-only vs Forensic-only vs Fusion results.

- [ ] **Step 1: Freeze the implementation and config before reading main OOD results**

Record:

```text
git commit SHA
manifest/split version
config files
preprocessing parameters
seed list
```

Use seeds `42`, `43`, and `44` unless compute constraints are explicitly documented.

- [ ] **Step 2: Train all three variants for each seed on SD1.4 main**

Run `train_r2.py` for each `(variant, seed)` pair. Change only the seed field between repeated runs unless a new experiment version is declared.

- [ ] **Step 3: Evaluate each checkpoint with validation-calibrated threshold**

For every run, evaluate the same hierarchy and save:

```text
predictions.jsonl
report.json
report.md
reproducibility.json
```

- [ ] **Step 4: Aggregate repeated runs with the existing aggregator**

For each variant:

```bash
python3 scripts/aggregate_runs.py \
  --reports <seed42-report.json> <seed43-report.json> <seed44-report.json> \
  --output-json <variant>-aggregate.json \
  --output-md <variant>-aggregate.md
```

- [ ] **Step 5: Build the required final table**

The primary table uses AUROC mean ± std:

```text
Benchmark              Semantic-only    Forensic-only    Fusion
SD1.4                  mean ± std       mean ± std       mean ± std
SD1.5                  mean ± std       mean ± std       mean ± std
Midjourney             mean ± std       mean ± std       mean ± std
ADM                    mean ± std       mean ± std       mean ± std
GLIDE                  mean ± std       mean ± std       mean ± std
Wukong                 mean ± std       mean ± std       mean ± std
VQDM                   mean ± std       mean ± std       mean ± std
BigGAN                 mean ± std       mean ± std       mean ± std
GenImage++             mean ± std       mean ± std       mean ± std
Mean Cross-Generator   mean ± std       mean ± std       mean ± std
```

Secondary metrics remain available in reports; do not collapse all analysis into one average.

- [ ] **Step 6: Apply the R2 interpretation rules**

Write the conclusion in terms of observed evidence:

- if Fusion consistently exceeds both branches outside run variance on unseen generators, H2 is supported under this protocol;
- if Semantic-only wins, report that the current forensic representation did not add generalizable value;
- if Forensic-only wins, investigate semantic/content shortcut risk;
- if results differ by generator, preserve that heterogeneity instead of claiming a universal winner.

- [ ] **Step 7: Verify the R2 gate from the spec line by line**

Confirm all 14 gate conditions in `docs/plans/r2-forensic-perception-v1.md` are satisfied before marking R2 complete.

---

## Final Verification Before R2 Implementation Is Considered Complete

Run fresh:

```bash
python3 -m pytest
python3 scripts/run_smoke_test.py
git diff --check
```

Then verify the R2-specific offline smoke:

```bash
python3 -m pytest tests/system/test_r2_smoke.py -v
```

Finally inspect the working tree:

```bash
git status --short
```

There must be no committed raw dataset, downloaded model weights, checkpoints, secrets, or generated result caches.

## Execution Order

```text
Task 1  Data adapter + dependencies
  -> Task 2  Semantic-only
  -> Task 3  Forensic-only
  -> Task 4  Fusion
  -> Task 5  Shared training utilities/configs
  -> Task 6  Training CLI/model assembly
  -> Task 7  R0 evaluation integration
  -> Task 8  Offline end-to-end smoke
  -> Task 9  Pilot three-way ablation
  -> Task 10 Main multi-seed ablation + R2 gate
```

Do not parallelize Tasks 2-7 unless their declared interfaces are frozen first; they share model/training contracts. Pilot and main experiments begin only after the full offline test suite passes.

