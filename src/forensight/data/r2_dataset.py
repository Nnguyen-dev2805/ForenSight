"""R2 dataset adapter for ForenSight.

Loads images and records from sealed R0 manifests, converting each sample
into branch-specific tensors (CLIP semantic and NPR forensic) while preserving
evaluation metadata.
"""

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
    """Load a Manifest instance from a .jsonl or .csv file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return Manifest.from_jsonl(path)
    if suffix == ".csv":
        return Manifest.from_csv(path)
    raise ValueError(f"Unsupported manifest format: {path.suffix}")


def build_forensic_input_transform(image_size: int = 224) -> ImageTransform:
    """Standard RGB tensor transform for forensic branch input."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


class R2ImageDataset(Dataset[dict[str, Any]]):
    """Manifest-backed PyTorch dataset for ForenSight R2 experiments.

    Opens each raw image exactly once and applies configured branch transforms
    (clip_transform, forensic_transform) without creating redundant file reads.
    """

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
