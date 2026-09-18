"""Deterministic split manifests, generator-disjointness assertions, and scale tiers.

This module implements Task 0.2 of R0:
- ManifestRecord dataclass for atomic sample representation.
- Manifest container for collection operations, filtering, serialization (CSV/JSONL),
  and leakage validation.
- assert_generator_disjoint helper guaranteeing cross-generator test independence.
- Deterministic subsampling across ImageNet classes for smoke/pilot scale tiers.
- Protocol v1 split specifications and generator membership summaries.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable, Iterator
import pandas as pd


VALID_SPLIT_NAMES = {
    "train",
    "val",
    "validation",
    "test",
    "in_domain_test",
    "near_ood",
    "cross_generator_ood",
    "cross_dataset_test",
    "modern_external",
    "real_world_external",
    "optional_external",
}

NON_GENERATOR_LABELS = {"nature", "real", "none", ""}

PROTOCOL_V1_SPLITS: dict[str, dict[str, Any]] = {
    "train": {
        "dataset": "genimage",
        "generators": ["sd14"],
        "split_role": "train",
        "description": "GenImage SD1.4 official train split for model optimization",
    },
    "val": {
        "dataset": "genimage",
        "generators": ["sd14"],
        "split_role": "val",
        "description": "GenImage SD1.4 held-out official validation split for threshold and hyperparameter tuning",
    },
    "in_domain_test": {
        "dataset": "genimage",
        "generators": ["sd14"],
        "split_role": "test",
        "description": "GenImage SD1.4 held-out official test split for final in-domain benchmark",
    },
    "near_ood": {
        "dataset": "genimage",
        "generators": ["sd15"],
        "split_role": "near_ood",
        "description": "GenImage SD1.5 subset (same model family with updated weights)",
    },
    "cross_generator_ood": {
        "dataset": "genimage",
        "generators": ["midjourney", "adm", "glide", "wukong", "vqdm", "biggan"],
        "split_role": "cross_generator_ood",
        "description": "GenImage unseen generator architectures for cross-generator generalization benchmark",
    },
    "modern_external": {
        "dataset": "genimage_plus_plus",
        "generators": ["flux_sd3_modern"],
        "split_role": "modern_external",
        "description": "GenImage++ modern flow-matching and advanced diffusion models (FLUX.1, SD3, etc.)",
    },
    "real_world_external": {
        "dataset": "wildrf",
        "generators": ["social_wild"],
        "split_role": "real_world_external",
        "description": "WildRF real-world in-the-wild social media deepfake benchmark",
    },
}

SCALE_TIERS: dict[str, dict[str, Any]] = {
    "smoke": {
        "max_per_class_real": 1,
        "max_per_class_fake": 1,
        "description": "Max 1 real + 1 fake per ImageNet class from SD1.4 train (pipeline sanity check)",
    },
    "pilot": {
        "max_per_class_real": 10,
        "max_per_class_fake": 10,
        "description": "Max 10 real + 10 fake per ImageNet class from SD1.4 train (preliminary experiments)",
    },
    "main": {
        "max_per_class_real": None,
        "max_per_class_fake": None,
        "description": "Full valid dataset within compute budget (canonical research conclusions)",
    },
}


@dataclass
class ManifestRecord:
    """Atomic sample record for ForenSight dataset manifests.

    Attributes:
        sample_id: Unique identifier for the sample.
        image_path: Relative or absolute path to the image file.
        label: Binary label (0 = real, 1 = fake).
        dataset: Source dataset name (e.g., 'genimage', 'wildrf').
        generator: Generative model identifier (e.g., 'sd14', 'midjourney', 'nature').
        split: Assigned evaluation partition (e.g., 'train', 'val', 'cross_generator_ood').
        class_id: Optional semantic class identifier (e.g. ImageNet synset 'n01440764').
        metadata: Arbitrary additional metadata (e.g., resolution, quality factor, prompt).
    """

    sample_id: str
    image_path: str
    label: int
    dataset: str
    generator: str
    split: str
    class_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(
                f"ManifestRecord label must be 0 (real) or 1 (fake), got: {self.label}"
            )
        if not self.sample_id:
            raise ValueError("ManifestRecord sample_id must be non-empty string.")
        if not self.image_path:
            raise ValueError("ManifestRecord image_path must be non-empty string.")
        if not self.dataset:
            raise ValueError("ManifestRecord dataset must be non-empty string.")
        if not self.generator:
            raise ValueError("ManifestRecord generator must be non-empty string.")
        if not self.split:
            raise ValueError("ManifestRecord split must be non-empty string.")
        if not isinstance(self.metadata, dict):
            raise ValueError(
                f"ManifestRecord metadata must be a dictionary, got: {type(self.metadata)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestRecord:
        """Create ManifestRecord from dictionary representation."""
        return cls(
            sample_id=str(data["sample_id"]),
            image_path=str(data["image_path"]),
            label=int(data["label"]),
            dataset=str(data["dataset"]),
            generator=str(data["generator"]),
            split=str(data["split"]),
            class_id=data.get("class_id"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Manifest:
    """Collection of ManifestRecord entries supporting filtering, I/O, and leakage checks."""

    records: list[ManifestRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> ManifestRecord:
        return self.records[idx]

    def __iter__(self) -> Iterator[ManifestRecord]:
        return iter(self.records)

    def append(self, record: ManifestRecord) -> None:
        """Add a record to the manifest."""
        self.records.append(record)

    def filter(
        self,
        predicate: Callable[[ManifestRecord], bool] | None = None,
        **kwargs: Any,
    ) -> Manifest:
        """Filter manifest records by matching keyword attributes and/or custom predicate.

        Example:
            reals = manifest.filter(label=0)
            train_sd14 = manifest.filter(split="train", generator="sd14")
            specific = manifest.filter(predicate=lambda r: r.class_id == "n01440764")
        """
        filtered = []
        for r in self.records:
            match = True
            for k, v in kwargs.items():
                if getattr(r, k, None) != v:
                    match = False
                    break
            if match and (predicate is None or predicate(r)):
                filtered.append(r)
        return Manifest(filtered)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert manifest to a pandas DataFrame."""
        data = [r.to_dict() for r in self.records]
        return pd.DataFrame(data, columns=[
            "sample_id",
            "image_path",
            "label",
            "dataset",
            "generator",
            "split",
            "class_id",
            "metadata",
        ])

    def to_jsonl(self, path: str | Path) -> None:
        """Serialize manifest to JSON Lines file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r.to_dict()) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Manifest:
        """Load manifest from JSON Lines file."""
        target = Path(path)
        records = []
        with target.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(ManifestRecord.from_dict(json.loads(line)))
        return cls(records)

    def to_csv(self, path: str | Path) -> None:
        """Serialize manifest to CSV file, serializing metadata as JSON string."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample_id",
            "image_path",
            "label",
            "dataset",
            "generator",
            "split",
            "class_id",
            "metadata",
        ]
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                row = r.to_dict()
                row["metadata"] = json.dumps(row["metadata"])
                if row["class_id"] is None:
                    row["class_id"] = ""
                writer.writerow(row)

    @classmethod
    def from_csv(cls, path: str | Path) -> Manifest:
        """Load manifest from CSV file."""
        target = Path(path)
        records = []
        with target.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta_raw = row.get("metadata", "")
                metadata = json.loads(meta_raw) if meta_raw else {}
                class_id = row.get("class_id") or None
                if class_id == "":
                    class_id = None
                records.append(
                    ManifestRecord(
                        sample_id=row["sample_id"],
                        image_path=row["image_path"],
                        label=int(row["label"]),
                        dataset=row["dataset"],
                        generator=row["generator"],
                        split=row["split"],
                        class_id=class_id,
                        metadata=metadata,
                    )
                )
        return cls(records)

    def validate_no_leakage(
        self,
        test_manifest: Manifest,
        check_generators: bool = False,
        strict: bool = True,
    ) -> list[str]:
        """Validate that there is zero sample or generator leakage between self and test_manifest."""
        return validate_no_leakage(
            train_manifest=self,
            test_manifest=test_manifest,
            check_generators=check_generators,
            strict=strict,
        )


def validate_no_leakage(
    train_manifest: Manifest,
    test_manifest: Manifest,
    check_generators: bool = False,
    strict: bool = True,
) -> list[str]:
    """Check for sample_id, image_path, and generator leakage between train and test manifests.

    Args:
        train_manifest: The training manifest.
        test_manifest: The evaluation or test manifest.
        check_generators: If True, or if test_manifest contains cross-generator splits,
            verify that fake generators do not overlap between train and test.
        strict: If True, raises ValueError upon detecting any leakage violation.

    Returns:
        List of formatted error messages describing detected leakages (empty if clean).
    """
    errors: list[str] = []

    # 1. Check sample_id collisions
    train_ids = {r.sample_id for r in train_manifest}
    test_ids = {r.sample_id for r in test_manifest}
    id_overlap = train_ids & test_ids
    if id_overlap:
        sample_sample = sorted(list(id_overlap))[:5]
        errors.append(
            f"sample_id leakage detected: {len(id_overlap)} samples shared between splits "
            f"(examples: {sample_sample})."
        )

    # 2. Check image_path collisions
    train_paths = {Path(r.image_path).resolve() for r in train_manifest}
    test_paths = {Path(r.image_path).resolve() for r in test_manifest}
    path_overlap = train_paths & test_paths
    if path_overlap:
        sample_paths = [str(p) for p in sorted(list(path_overlap))[:5]]
        errors.append(
            f"image_path leakage detected: {len(path_overlap)} identical image paths shared between splits "
            f"(examples: {sample_paths})."
        )

    # 3. Check generator disjointness if requested or if split indicates cross-generator OOD
    test_splits = {r.split for r in test_manifest}
    should_check_generators = check_generators or any(
        s in ("cross_generator_ood", "cross_generator_test") for s in test_splits
    )

    if should_check_generators:
        train_fake_gens = {
            r.generator
            for r in train_manifest
            if r.label == 1 and r.generator.lower() not in NON_GENERATOR_LABELS
        }
        test_fake_gens = {
            r.generator
            for r in test_manifest
            if r.label == 1 and r.generator.lower() not in NON_GENERATOR_LABELS
        }
        gen_overlap = train_fake_gens & test_fake_gens
        if gen_overlap:
            errors.append(
                f"Generator leakage detected: train and test share fake generator(s) "
                f"{sorted(gen_overlap)} in a cross-generator partition."
            )

    if strict and errors:
        raise ValueError(
            f"Leakage detected between manifests ({len(errors)} violation(s)):\n"
            + "\n".join(f" - {err}" for err in errors)
        )

    return errors


def assert_generator_disjoint(
    train_manifest: Manifest,
    ood_manifests: Manifest | list[Manifest] | dict[str, Manifest],
) -> None:
    """Assert that train manifest and OOD manifests are strictly generator-disjoint for fake samples.

    Raises:
        AssertionError: If any fake generator from train_manifest is present in an OOD manifest.
    """
    if isinstance(ood_manifests, Manifest):
        manifest_list = [ood_manifests]
    elif isinstance(ood_manifests, dict):
        manifest_list = list(ood_manifests.values())
    else:
        manifest_list = list(ood_manifests)

    train_gens = {
        r.generator
        for r in train_manifest
        if r.label == 1 and r.generator.lower() not in NON_GENERATOR_LABELS
    }

    for idx, ood_m in enumerate(manifest_list):
        ood_gens = {
            r.generator
            for r in ood_m
            if r.label == 1 and r.generator.lower() not in NON_GENERATOR_LABELS
        }
        overlap = train_gens & ood_gens
        if overlap:
            raise AssertionError(
                f"Generator leakage detected: train and OOD manifest (index {idx}) share "
                f"generator(s): {sorted(overlap)}."
            )


def subsample_manifest_by_class(
    manifest: Manifest,
    max_per_class_real: int | None,
    max_per_class_fake: int | None,
    seed: int = 42,
) -> Manifest:
    """Subsample manifest deterministically while preserving balanced representation across classes.

    Args:
        manifest: Source Manifest.
        max_per_class_real: Maximum real samples (label=0) per class_id (None for all).
        max_per_class_fake: Maximum fake samples (label=1) per class_id (None for all).
        seed: Base random seed for reproducible sampling.

    Returns:
        New Manifest with deterministically sampled records, ordered deterministically by sample_id.
    """
    if max_per_class_real is None and max_per_class_fake is None:
        return Manifest(manifest.records.copy())

    # Group records by class_id
    classes: dict[str, list[ManifestRecord]] = {}
    for r in manifest.records:
        cid = r.class_id if r.class_id is not None else "__none__"
        classes.setdefault(cid, []).append(r)

    retained_records: list[ManifestRecord] = []

    # Sort class IDs for deterministic iteration
    sorted_class_ids = sorted(classes.keys())

    for cid in sorted_class_ids:
        class_records = classes[cid]
        reals = [r for r in class_records if r.label == 0]
        fakes = [r for r in class_records if r.label == 1]

        # Sort deterministically by sample_id before sampling
        reals.sort(key=lambda r: r.sample_id)
        fakes.sort(key=lambda r: r.sample_id)

        # Subsample real
        if max_per_class_real is not None and len(reals) > max_per_class_real:
            # Deterministic RNG keyed to seed and class_id
            class_seed_real = int(
                hashlib.sha256(f"{seed}:{cid}:real".encode("utf-8")).hexdigest()[:15], 16
            )
            rng_real = random.Random(class_seed_real)
            sampled_reals = rng_real.sample(reals, max_per_class_real)
        else:
            sampled_reals = reals

        # Subsample fake
        if max_per_class_fake is not None and len(fakes) > max_per_class_fake:
            class_seed_fake = int(
                hashlib.sha256(f"{seed}:{cid}:fake".encode("utf-8")).hexdigest()[:15], 16
            )
            rng_fake = random.Random(class_seed_fake)
            sampled_fakes = rng_fake.sample(fakes, max_per_class_fake)
        else:
            sampled_fakes = fakes

        retained_records.extend(sampled_reals)
        retained_records.extend(sampled_fakes)

    # Sort final retained records deterministically by sample_id
    retained_records.sort(key=lambda r: r.sample_id)
    return Manifest(retained_records)


def create_scale_manifest(
    manifest: Manifest,
    tier: str,
    seed: int = 42,
) -> Manifest:
    """Create a scale tier manifest (smoke, pilot, or main) from a base manifest.

    Args:
        manifest: Base manifest (typically SD1.4 train).
        tier: One of 'smoke', 'pilot', 'main'.
        seed: Random seed for deterministic subsampling.

    Returns:
        Subsampled or complete Manifest according to the tier definition.
    """
    if tier not in SCALE_TIERS:
        raise ValueError(
            f"Unknown scale tier '{tier}'. Expected one of: {list(SCALE_TIERS.keys())}"
        )

    tier_cfg = SCALE_TIERS[tier]
    return subsample_manifest_by_class(
        manifest=manifest,
        max_per_class_real=tier_cfg["max_per_class_real"],
        max_per_class_fake=tier_cfg["max_per_class_fake"],
        seed=seed,
    )


def get_generator_membership_summary() -> dict[str, Any]:
    """Return a comprehensive summary of generator membership across Protocol v1 splits."""
    return {
        "protocol_version": "v1",
        "train_generators": PROTOCOL_V1_SPLITS["train"]["generators"],
        "in_domain_generators": PROTOCOL_V1_SPLITS["in_domain_test"]["generators"],
        "near_ood_generators": PROTOCOL_V1_SPLITS["near_ood"]["generators"],
        "cross_generator_ood": PROTOCOL_V1_SPLITS["cross_generator_ood"]["generators"],
        "modern_external": PROTOCOL_V1_SPLITS["modern_external"]["generators"],
        "real_world_external": PROTOCOL_V1_SPLITS["real_world_external"]["generators"],
        "disjoint_guarantees": [
            "G_train ∩ G_cross_generator_ood = ∅",
            "G_train ∩ G_near_ood = ∅",
        ],
        "scale_tiers": list(SCALE_TIERS.keys()),
    }
