"""Dataset inventory schema, validation, serialization, and default metadata.

This module provides data structures to document, validate, serialize, and load
dataset inventory records for ForenSight in accordance with the Stage 1 dataset
strategy (docs/decisions/2026-09-18-stage1-dataset-strategy.md) and R0 plan
(docs/plans/r0-dataset-evaluation.md).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator


VALID_SUBSET_ROLES = {
    "train_and_in_domain",
    "near_ood",
    "cross_generator_ood",
    "modern_external",
    "real_world_external",
    "optional_external",
}


@dataclass
class GeneratorSubset:
    """Metadata for an individual generator subset within a dataset."""

    generator_id: str
    generator_name: str
    role: str
    resolution: str
    estimated_count: dict[str, int | str] = field(default_factory=dict)
    local_path: str | None = None
    download_urls: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert subset metadata to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneratorSubset:
        """Instantiate GeneratorSubset from a dictionary."""
        return cls(
            generator_id=data["generator_id"],
            generator_name=data["generator_name"],
            role=data["role"],
            resolution=data["resolution"],
            estimated_count=dict(data.get("estimated_count", {})),
            local_path=data.get("local_path"),
            download_urls=list(data.get("download_urls", [])),
            notes=data.get("notes", ""),
        )


@dataclass
class DatasetEntry:
    """Metadata entry for a dataset in the ForenSight inventory."""

    name: str
    display_name: str
    version: str
    source: str
    evaluation_role: str
    real_source: str
    fake_generators: list[str]
    image_count: dict[str, int | str]
    resolution_distribution: str
    format_compression: str
    label_mapping: dict[str, int] = field(default_factory=lambda: {"real": 0, "fake": 1})
    license: str = ""
    access_requirements: str = ""
    local_path: str | None = None
    generator_subsets: list[GeneratorSubset] = field(default_factory=list)
    download_instructions: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert dataset entry to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetEntry:
        """Instantiate DatasetEntry from a dictionary."""
        subsets = [
            GeneratorSubset.from_dict(sub)
            for sub in data.get("generator_subsets", [])
        ]
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            version=data["version"],
            source=data["source"],
            evaluation_role=data["evaluation_role"],
            real_source=data["real_source"],
            fake_generators=list(data.get("fake_generators", [])),
            image_count=dict(data.get("image_count", {})),
            resolution_distribution=data["resolution_distribution"],
            format_compression=data["format_compression"],
            label_mapping=dict(data.get("label_mapping", {"real": 0, "fake": 1})),
            license=data.get("license", ""),
            access_requirements=data.get("access_requirements", ""),
            local_path=data.get("local_path"),
            generator_subsets=subsets,
            download_instructions=list(data.get("download_instructions", [])),
            notes=data.get("notes", ""),
        )


@dataclass
class DatasetInventory:
    """Collection of dataset entries with validation, querying, and I/O."""

    schema_version: str = "1.0.0"
    updated_at: str = "2026-09-18"
    datasets: dict[str, DatasetEntry] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.datasets)

    def __getitem__(self, name: str) -> DatasetEntry:
        return self.datasets[name]

    def __contains__(self, name: str) -> bool:
        return name in self.datasets

    def __iter__(self) -> Iterator[str]:
        return iter(self.datasets)

    def items(self):
        return self.datasets.items()

    def values(self):
        return self.datasets.values()

    def keys(self):
        return self.datasets.keys()

    def get(self, name: str, default: DatasetEntry | None = None) -> DatasetEntry | None:
        """Retrieve dataset entry by name, returning default if not found."""
        return self.datasets.get(name, default)

    def add_entry(self, entry: DatasetEntry) -> None:
        """Add or update a dataset entry."""
        self.datasets[entry.name] = entry

    def list_dataset_names(self) -> list[str]:
        """Return list of dataset names registered in the inventory."""
        return list(self.datasets.keys())

    def get_generator_subset(
        self, dataset_name: str, generator_id: str
    ) -> GeneratorSubset | None:
        """Find a specific generator subset by dataset name and generator ID."""
        dataset = self.get(dataset_name)
        if not dataset:
            return None
        for subset in dataset.generator_subsets:
            if subset.generator_id == generator_id:
                return subset
        return None

    def find_subsets_by_role(
        self, role: str
    ) -> list[tuple[DatasetEntry, GeneratorSubset]]:
        """Find all generator subsets across datasets assigned to a specific role."""
        matches: list[tuple[DatasetEntry, GeneratorSubset]] = []
        for dataset in self.datasets.values():
            for subset in dataset.generator_subsets:
                if subset.role == role:
                    matches.append((dataset, subset))
        return matches

    def validate(self, strict: bool = False) -> list[str]:
        """Validate inventory entries against required fields and consistency rules."""
        errors: list[str] = []

        if not self.datasets:
            errors.append("Inventory contains no dataset entries.")

        for name, entry in self.datasets.items():
            if not entry.name:
                errors.append(f"Entry with key '{name}' has empty name.")
            if not entry.display_name:
                errors.append(f"Entry '{name}' has empty display_name.")
            if not entry.version:
                errors.append(f"Entry '{name}' has empty version.")
            if not entry.source:
                errors.append(f"Entry '{name}' has empty source URL/reference.")
            if not entry.evaluation_role:
                errors.append(f"Entry '{name}' has empty evaluation_role.")
            if not entry.real_source:
                errors.append(f"Entry '{name}' has empty real_source.")
            if not entry.fake_generators:
                errors.append(f"Entry '{name}' specifies no fake_generators.")
            if not entry.image_count:
                errors.append(f"Entry '{name}' has empty image_count.")
            if not entry.resolution_distribution:
                errors.append(f"Entry '{name}' has empty resolution_distribution.")
            if not entry.format_compression:
                errors.append(f"Entry '{name}' has empty format_compression.")
            if not entry.license:
                errors.append(f"Entry '{name}' has empty license.")
            if not entry.access_requirements:
                errors.append(f"Entry '{name}' has empty access_requirements.")

            # Label mapping check
            if set(entry.label_mapping.keys()) != {"real", "fake"}:
                errors.append(
                    f"Entry '{name}' label_mapping must have exactly keys {{'real', 'fake'}}, got: {list(entry.label_mapping.keys())}"
                )
            elif entry.label_mapping["real"] != 0 or entry.label_mapping["fake"] != 1:
                errors.append(
                    f"Entry '{name}' label_mapping must map real->0 and fake->1, got: {entry.label_mapping}"
                )


            # Validate generator subsets if present
            for subset in entry.generator_subsets:
                if not subset.generator_id:
                    errors.append(f"Entry '{name}' contains a subset with empty generator_id.")
                if not subset.generator_name:
                    errors.append(f"Entry '{name}' subset '{subset.generator_id}' has empty generator_name.")
                if subset.role not in VALID_SUBSET_ROLES:
                    errors.append(
                        f"Entry '{name}' subset '{subset.generator_id}' has unrecognized role '{subset.role}'."
                    )

        if strict and errors:
            raise ValueError(
                f"Inventory validation failed with {len(errors)} error(s):\n"
                + "\n".join(f" - {err}" for err in errors)
            )

        return errors

    def is_valid(self) -> bool:
        """Check whether inventory has zero validation errors."""
        return len(self.validate(strict=False)) == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize inventory to a dictionary."""
        return {
            "schema_version": self.schema_version,
            "updated_at": self.updated_at,
            "datasets": {name: entry.to_dict() for name, entry in self.datasets.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize inventory to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path, indent: int = 2) -> None:
        """Save inventory to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=indent), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetInventory:
        """Instantiate DatasetInventory from serialized dictionary."""
        datasets = {
            name: DatasetEntry.from_dict(entry_data)
            for name, entry_data in data.get("datasets", {}).items()
        }
        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            updated_at=data.get("updated_at", ""),
            datasets=datasets,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> DatasetInventory:
        """Load DatasetInventory from a JSON file."""
        target = Path(path)
        content = target.read_text(encoding="utf-8")
        data = json.loads(content)
        return cls.from_dict(data)


def build_default_inventory() -> DatasetInventory:
    """Build canonical Stage 1 dataset inventory for ForenSight.

    Covers:
      1. GenImage (8 generators: SD1.4 [train + in-domain], SD1.5 [near-OOD],
         Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN [cross-generator OOD])
      2. GenImage++ (modern external, test-only: FLUX.1, SD3, etc.)
      3. WildRF (real-world external, test-only)
      4. Chameleon (optional external, test-only)
    """
    genimage_subsets = [
        GeneratorSubset(
            generator_id="sd14",
            generator_name="Stable Diffusion v1.4",
            role="train_and_in_domain",
            resolution="512x512 for fake; variable (ImageNet) for real",
            estimated_count={
                "train_real": 160000,
                "train_fake": 160000,
                "val_real": 20000,
                "val_fake": 20000,
                "total": 360000,
            },
            local_path="data/raw/genimage/sdv4",
            download_urls=[
                "https://github.com/GenImage-Dataset/GenImage",
                "https://huggingface.co/datasets/GenImage/GenImage",
            ],
            notes="Primary training source and in-domain held-out validation/test split for Stage 1.",
        ),
        GeneratorSubset(
            generator_id="sd15",
            generator_name="Stable Diffusion v1.5",
            role="near_ood",
            resolution="512x512 for fake; variable (ImageNet) for real",
            estimated_count={
                "val_real": 20000,
                "val_fake": 20000,
                "total": 40000,
            },
            local_path="data/raw/genimage/sdv5",
            download_urls=["https://github.com/GenImage-Dataset/GenImage"],
            notes="Near-OOD evaluation: same model family and architecture with updated weights.",
        ),
        GeneratorSubset(
            generator_id="midjourney",
            generator_name="Midjourney",
            role="cross_generator_ood",
            resolution="Variable / approx 512x512 for fake; variable (ImageNet) for real",
            estimated_count={
                "val_real": 20000,
                "val_fake": 20000,
                "total": 40000,
            },
            local_path="data/raw/genimage/midjourney",
            download_urls=["https://github.com/GenImage-Dataset/GenImage"],
            notes="Cross-generator OOD: closed-source commercial text-to-image generator.",
        ),
        GeneratorSubset(
            generator_id="adm",
            generator_name="ADM (Guided Diffusion)",
            role="cross_generator_ood",
            resolution="256x256 for fake; variable (ImageNet) for real",
            estimated_count={
                "val_real": 20000,
                "val_fake": 20000,
                "total": 40000,
            },
            local_path="data/raw/genimage/adm",
            download_urls=["https://github.com/GenImage-Dataset/GenImage"],
            notes="Cross-generator OOD: pixel-space diffusion model with classifier guidance.",
        ),
        GeneratorSubset(
            generator_id="glide",
            generator_name="GLIDE",
            role="cross_generator_ood",
            resolution="256x256 for fake; variable (ImageNet) for real",
            estimated_count={
                "val_real": 20000,
                "val_fake": 20000,
                "total": 40000,
            },
            local_path="data/raw/genimage/glide",
            download_urls=["https://github.com/GenImage-Dataset/GenImage"],
            notes="Cross-generator OOD: filtered pixel-space diffusion model.",
        ),
        GeneratorSubset(
            generator_id="wukong",
            generator_name="Wukong",
            role="cross_generator_ood",
            resolution="512x512 for fake; variable (ImageNet) for real",
            estimated_count={
                "val_real": 20000,
                "val_fake": 20000,
                "total": 40000,
            },
            local_path="data/raw/genimage/wukong",
            download_urls=["https://github.com/GenImage-Dataset/GenImage"],
            notes="Cross-generator OOD: Chinese text-to-image diffusion model.",
        ),
        GeneratorSubset(
            generator_id="vqdm",
            generator_name="VQDM (VQ-Diffusion)",
            role="cross_generator_ood",
            resolution="256x256 for fake; variable (ImageNet) for real",
            estimated_count={
                "val_real": 20000,
                "val_fake": 20000,
                "total": 40000,
            },
            local_path="data/raw/genimage/vqdm",
            download_urls=["https://github.com/GenImage-Dataset/GenImage"],
            notes="Cross-generator OOD: vector-quantized discrete diffusion model.",
        ),
        GeneratorSubset(
            generator_id="biggan",
            generator_name="BigGAN",
            role="cross_generator_ood",
            resolution="256x256 for fake; variable (ImageNet) for real",
            estimated_count={
                "val_real": 20000,
                "val_fake": 20000,
                "total": 40000,
            },
            local_path="data/raw/genimage/biggan",
            download_urls=["https://github.com/GenImage-Dataset/GenImage"],
            notes="Cross-generator OOD: non-diffusion GAN architecture baseline.",
        ),
    ]

    genimage_entry = DatasetEntry(
        name="genimage",
        display_name="GenImage",
        version="1.0 (NeurIPS 2023)",
        source="https://github.com/GenImage-Dataset/GenImage",
        evaluation_role="Primary Stage 1 benchmark (train, in-domain val/test, near-OOD, cross-generator OOD)",
        real_source="ImageNet (ILSVRC2012)",
        fake_generators=[
            "Stable Diffusion v1.4",
            "Stable Diffusion v1.5",
            "Midjourney",
            "ADM",
            "GLIDE",
            "Wukong",
            "VQDM",
            "BigGAN",
        ],
        image_count={
            "train_real_sd14": 160000,
            "train_fake_sd14": 160000,
            "val_real_per_generator": 20000,
            "val_fake_per_generator": 20000,
            "total_benchmark_pairs": "~1,331,167",
        },
        resolution_distribution=(
            "Fake images have model-native resolutions: 512x512 for SD1.4, SD1.5, Wukong; "
            "256x256 for ADM, GLIDE, VQDM, BigGAN; variable/512+ for Midjourney. "
            "Real ImageNet images have variable dimensions."
        ),
        format_compression=(
            "Real images from ImageNet JPEG; fake images generated JPEG/PNG. "
            "Confound warning: Unbiased GenImage documented significant JPEG quality factor "
            "and resolution disparity between real and fake, requiring Task 0.3 audit before interpreting detection scores."
        ),
        label_mapping={"real": 0, "fake": 1},
        license="GenImage: Apache-2.0 / Research Use; ImageNet: Non-commercial educational and research terms.",
        access_requirements="Open research download via official GenImage repository (Google Drive / Baidu Netdisk / Hugging Face mirrors).",
        local_path="data/raw/genimage",
        generator_subsets=genimage_subsets,
        download_instructions=[
            "Clone or visit official repo: https://github.com/GenImage-Dataset/GenImage",
            "Download SD1.4 (imagenet_ai_0419_sdv4) into data/raw/genimage/sdv4",
            "Download remaining generators into corresponding data/raw/genimage/<generator> directories as needed for cross-generator testing",
        ],
        notes=(
            "Primary benchmark for Stage 1. Training is strictly restricted to SD1.4 subset to test cross-generator generalization. "
            "SD1.4 contains official train/nature, train/ai, val/nature, val/ai splits."
        ),
    )

    genimage_plus_plus_subsets = [
        GeneratorSubset(
            generator_id="flux_sd3_modern",
            generator_name="Modern Generative Models (FLUX.1, SD3, PixArt, Kolors)",
            role="modern_external",
            resolution="1024x1024 / 512x512",
            estimated_count={"total": 100000},
            local_path="data/raw/genimagepp",
            download_urls=["https://huggingface.co/datasets/Lunahera/genimagepp"],
            notes="Modern rectified-flow and advanced diffusion models with reduced high-frequency spectral artifacts.",
        )
    ]

    genimage_plus_plus_entry = DatasetEntry(
        name="genimage_plus_plus",
        display_name="GenImage++",
        version="1.0 (2024)",
        source="https://huggingface.co/datasets/Lunahera/genimagepp",
        evaluation_role="Modern external benchmark (test-only, evaluation hierarchy level 4)",
        real_source="ImageNet-1k, COCO, and high-quality web photographs",
        fake_generators=[
            "FLUX.1 (FLUX.1-schnell, FLUX.1-dev)",
            "Stable Diffusion 3 (SD3-Medium)",
            "PixArt-alpha / PixArt-sigma",
            "Kolors",
            "HunyuanDiT",
            "AuraFlow",
        ],
        image_count={"real": 50000, "fake": 50000, "total": 100000},
        resolution_distribution="Predominantly 1024x1024 for modern generative models; some 512x512 subsets.",
        format_compression="JPEG and PNG; modern generation pipelines with varying post-processing quality.",
        label_mapping={"real": 0, "fake": 1},
        license="Research and educational use per Hugging Face repository terms (CC-BY / OpenRAIL derivatives).",
        access_requirements="Open Hugging Face dataset download via huggingface-cli or datasets library.",
        local_path="data/raw/genimagepp",
        generator_subsets=genimage_plus_plus_subsets,
        download_instructions=[
            "Install huggingface_hub: pip install huggingface_hub",
            "Download dataset: huggingface-cli download Lunahera/genimagepp --local-dir data/raw/genimagepp --repo-type dataset",
        ],
        notes="Evaluation-only. Used to test generalization to latest generative architectures (FLUX, SD3) without tuning hyperparameters.",
    )

    wildrf_subsets = [
        GeneratorSubset(
            generator_id="social_wild",
            generator_name="Social Media In-The-Wild AI-Generated Images",
            role="real_world_external",
            resolution="Variable web resolutions (400x400 to 2048x2048+)",
            estimated_count={"total": 12000},
            local_path="data/raw/wildrf",
            download_urls=["https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld"],
            notes="Real-world distribution subjected to social platform recompression, resizing, and screenshotting.",
        )
    ]

    wildrf_entry = DatasetEntry(
        name="wildrf",
        display_name="WildRF",
        version="1.0 (2024)",
        source="https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld",
        evaluation_role="Real-world external benchmark (test-only, evaluation hierarchy level 5)",
        real_source="Social media and open web platforms (Reddit, Twitter/X, Instagram, news media)",
        fake_generators=[
            "In-the-wild AI generators (Midjourney v5/v6, DALL-E 2/3, Stable Diffusion XL, Firefly, online tools)",
        ],
        image_count={"real": 6000, "fake": 6000, "total": 12000},
        resolution_distribution="Highly variable resolutions and aspect ratios matching organic user uploads.",
        format_compression="JPEG, WebP, PNG; multiple generations of platform lossy compression and metadata stripping.",
        label_mapping={"real": 0, "fake": 1},
        license="Academic research use only.",
        access_requirements="Publicly accessible via repository download links.",
        local_path="data/raw/wildrf",
        generator_subsets=wildrf_subsets,
        download_instructions=[
            "Refer to https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld for download scripts and Google Drive archive links.",
            "Extract contents into data/raw/wildrf.",
        ],
        notes="Evaluation-only benchmark to diagnose robustness against uncontrolled real-world social media artifacts.",
    )

    chameleon_subsets = [
        GeneratorSubset(
            generator_id="aide_human_hard",
            generator_name="AIDE/Chameleon Human-Hard Deepfakes",
            role="optional_external",
            resolution="High resolution (1024x1024 to 2048x2048+)",
            estimated_count={"total": 6000},
            local_path="data/raw/chameleon",
            download_urls=["https://github.com/shilinyan99/AIDE"],
            notes="Curated challenging synthetic images where visual artifacts are subtle or imperceptible.",
        )
    ]

    chameleon_entry = DatasetEntry(
        name="chameleon",
        display_name="Chameleon (AIDE Benchmark)",
        version="1.0 (2024)",
        source="https://github.com/shilinyan99/AIDE",
        evaluation_role="Optional external benchmark (test-only, evaluation hierarchy level 6)",
        real_source="High-quality professional photography (Unsplash, RAISE, photographic datasets)",
        fake_generators=[
            "Human-hard photorealistic modern generative models curated to fool human observers",
        ],
        image_count={"real": 3000, "fake": 3000, "total": 6000},
        resolution_distribution="High resolution (1024x1024 and variable high-res photography).",
        format_compression="High quality JPEG and PNG with minimal artifact distortion.",
        label_mapping={"real": 0, "fake": 1},
        license="Restricted research license (agreement required upon request).",
        access_requirements="Requires formal access request via Google Form / author contact as specified in the AIDE repository.",
        local_path="data/raw/chameleon",
        generator_subsets=chameleon_subsets,
        download_instructions=[
            "Submit access request form via https://github.com/shilinyan99/AIDE.",
            "Upon approval, download archives and extract into data/raw/chameleon.",
        ],
        notes="Optional benchmark in Stage 1; development and verification can proceed without blocking on access approval.",
    )

    inventory = DatasetInventory(
        schema_version="1.0.0",
        updated_at="2026-09-18",
        datasets={
            "genimage": genimage_entry,
            "genimage_plus_plus": genimage_plus_plus_entry,
            "wildrf": wildrf_entry,
            "chameleon": chameleon_entry,
        },
    )
    return inventory


def get_default_inventory() -> DatasetInventory:
    """Convenience alias for build_default_inventory."""
    return build_default_inventory()


def load_inventory(path: str | Path) -> DatasetInventory:
    """Load a DatasetInventory instance from a JSON file."""
    return DatasetInventory.from_json(path)


def save_inventory(inventory: DatasetInventory, path: str | Path, indent: int = 2) -> None:
    """Save a DatasetInventory instance to a JSON file."""
    inventory.save_json(path, indent=indent)
