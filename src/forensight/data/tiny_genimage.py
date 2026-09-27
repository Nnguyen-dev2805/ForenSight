"""Dataset utilities for the Kaggle Tiny-GenImage benchmark.

This module handles:
1. Downloading ``yangsangtai/tiny-genimage`` from Kaggle.
2. Scanning the original Kaggle directory tree without rewriting raw images.
3. Generating sealed ForenSight manifests guaranteeing zero generator leakage into train/val.

Legacy Hugging Face Parquet helpers are kept for compatibility with old artifacts,
but the canonical Stage-1 source is the Kaggle dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import io
import json
import logging
from pathlib import Path
import random
import re
from typing import Any, Callable, Sequence

import pyarrow.parquet as pq

from forensight.data.split import (
    Manifest,
    ManifestRecord,
    assert_generator_disjoint,
)

logger = logging.getLogger(__name__)

DEFAULT_TINY_GENIMAGE_REPO_ID = "TheKernel01/Tiny-GenImage"
DEFAULT_TINY_GENIMAGE_KAGGLE_DATASET = "yangsangtai/tiny-genimage"

KAGGLE_GENERATOR_ALIASES: dict[str, str] = {
    "imagenet_ai_0508_adm": "adm",
    "adm": "adm",
    "imagenet_ai_0419_biggan": "biggan",
    "biggan": "biggan",
    "imagenet_glide": "glide",
    "glide": "glide",
    "imagenet_midjourney": "midjourney",
    "midjourney": "midjourney",
    "imagenet_ai_0424_sdv5": "sd15",
    "stable_diffusion_v_1_5": "sd15",
    "sd15": "sd15",
    "imagenet_ai_0419_vqdm": "vqdm",
    "vqdm": "vqdm",
    "imagenet_ai_0424_wukong": "wukong",
    "wukong": "wukong",
}

TINY_GENIMAGE_GENERATOR_MAP: dict[int, str] = {
    0: "nature",
    1: "adm",
    2: "biggan",
    3: "glide",
    4: "midjourney",
    5: "sd14",
    6: "sd15",
    7: "vqdm",
    8: "wukong",
}

CROSS_GENERATOR_OOD_IDS = [
    "adm",
    "biggan",
    "glide",
    "midjourney",
    "vqdm",
    "wukong",
]

TINY_GENIMAGE_SEVEN_GENERATORS: tuple[str, ...] = (
    "sd15",
    "adm",
    "biggan",
    "glide",
    "midjourney",
    "vqdm",
    "wukong",
)

KAGGLE_TINY_GENIMAGE_GENERATORS: list[str] = list(TINY_GENIMAGE_SEVEN_GENERATORS)


def _assert_expected_generator_set(records: Sequence[dict[str, Any]], *, builder: str) -> None:
    """Fail fast if the input records do not contain exactly the seven canonical generators.

    Deriving the generator set from the input would let a partial or misspelled dataset
    silently redefine the protocol (e.g. a 6-generator `all7`, or a LOGO fold that trains
    on fewer sources than documented). Both builders therefore pin the exact set.
    """
    found = sorted({r["generator"] for r in records if r.get("label") == 1})
    expected = sorted(KAGGLE_TINY_GENIMAGE_GENERATORS)
    if found != expected:
        raise ValueError(
            f"{builder} requires exactly the canonical generator set "
            f"{expected}; got {found}. Refusing to build a protocol from a different "
            "generator set."
        )


def download_kaggle_tiny_genimage(
    dataset_ref: str = DEFAULT_TINY_GENIMAGE_KAGGLE_DATASET,
    output_dir: str | Path = "data/raw/tiny_genimage",
    *,
    force_download: bool = False,
) -> Path:
    """Download the canonical Tiny-GenImage dataset from Kaggle via kagglehub."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required to download Tiny-GenImage from Kaggle. "
            "Install project dependencies first: pip install -e ."
        ) from exc

    target = Path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = kagglehub.dataset_download(
        dataset_ref,
        output_dir=str(target),
        force_download=force_download,
    )
    return Path(downloaded)


def _generator_from_path(parts: Sequence[str]) -> str | None:
    for part in reversed(parts):
        normalized = part.lower()
        if normalized in KAGGLE_GENERATOR_ALIASES:
            return KAGGLE_GENERATOR_ALIASES[normalized]
    return None


@lru_cache(maxsize=1)
def _imagenet_wnids() -> tuple[str, ...]:
    """The 1000 ImageNet-1k WNIDs in class order (class 1 first).

    Imported lazily: `timm` arrives with OpenCLIP, so this adds no dependency,
    and deferring the import keeps `forensight.data` importable without it.
    """
    from timm.data.imagenet_info import ImageNetInfo

    return tuple(ImageNetInfo("imagenet-1k").label_names())


# ImageNet-1k validation holds 50 images per class, in class order.
_IMAGENET_VAL_PER_CLASS = 50


def _class_id_from_path(path: Path) -> str | None:
    """Resolve an ImageNet WNID from a Tiny-GenImage filename.

    Two encodings appear in the same dataset, and both must land in ONE
    namespace or the class-balance audit reports a phantom real/fake imbalance:

      - real, train:      `n01440764_5969.JPEG`      -> WNID already
      - real, val/test:   `ILSVRC2012_val_00000001`  -> WNID by running index
      - fake:             `001_sdv5_00094.png`       -> WNID by 1-based index

    The 1-based reading of the fake prefix matches the documented ImageNet
    ordering (class 1 = `n01440764`). Returns None when no class is encoded,
    which is the honest answer for generators such as GLIDE and VQDM whose
    filenames carry no class field.
    """
    stem = path.stem

    # Real images that already carry a WNID.
    if stem.startswith("n") and len(stem) > 9 and stem[1:9].isdigit():
        return stem.split("_")[0]
    for part in path.parts:
        if part.startswith("n") and len(part) == 9 and part[1:].isdigit():
            return part

    # Real validation images named by their position in the ImageNet val split.
    val_match = re.fullmatch(r"ILSVRC2012_val_(\d+)", stem)
    if val_match:
        index = int(val_match.group(1))
        if index >= 1:
            class_index = (index - 1) // _IMAGENET_VAL_PER_CLASS
            wnids = _imagenet_wnids()
            if class_index < len(wnids):
                return wnids[class_index]
        return None

    # Fakes named with a 1-based class prefix.
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].isdigit():
        one_based = int(parts[0])
        wnids = _imagenet_wnids()
        if 1 <= one_based <= len(wnids):
            return wnids[one_based - 1]
    return None


def _raw_split_from_path(parts: Sequence[str]) -> str | None:
    lowered = {part.lower() for part in parts}
    if "train" in lowered:
        return "train"
    if "val" in lowered or "validation" in lowered:
        return "validation"
    return None


def scan_kaggle_tiny_genimage(dataset_root: str | Path) -> list[dict[str, Any]]:
    """Scan the Kaggle Tiny-GenImage tree into canonical extracted-record metadata.

    The scanner accepts GenImage-style trees such as
    ``<generator>/<train|val>/<ai|nature>/...`` and flattened mirrors with
    top-level generator/Nature folders. Raw files are never copied or modified.
    """
    root = Path(dataset_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Tiny-GenImage dataset root does not exist: {root}")

    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    records: list[dict[str, Any]] = []

    image_files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in image_suffixes
    )
    for image_path in image_files:
        rel = image_path.relative_to(root)
        parts = rel.parts[:-1]
        lowered = {part.lower() for part in parts}
        raw_split = _raw_split_from_path(parts)
        if raw_split is None:
            continue

        generator = _generator_from_path(parts)
        is_real = bool(lowered & {"nature", "real"})
        is_fake = bool(lowered & {"ai", "fake", "synthetic"}) or generator is not None

        if is_real:
            label = 0
            generator_id = "nature"
        elif is_fake and generator is not None:
            label = 1
            generator_id = generator
        else:
            continue

        if label == 1 and generator_id not in KAGGLE_TINY_GENIMAGE_GENERATORS:
            continue

        class_id = _class_id_from_path(image_path)

        records.append(
            {
                "sample_id": f"tiny_genimage_{raw_split}_{len(records):06d}",
                "image_path": rel.as_posix(),
                "label": label,
                "generator": generator_id,
                "raw_split": raw_split,
                "class_id": class_id,
                "index": len(records),
            }
        )

    fake_generators = {r["generator"] for r in records if r["label"] == 1}
    missing = set(KAGGLE_TINY_GENIMAGE_GENERATORS) - fake_generators
    if missing:
        raise ValueError(
            "Kaggle Tiny-GenImage scan is missing expected fake generators: "
            f"{sorted(missing)}. Found: {sorted(fake_generators)}"
        )
    if not any(r["label"] == 0 for r in records):
        raise ValueError("Kaggle Tiny-GenImage scan found no authentic/nature images.")
    return records


def extract_image_bytes(image_val: Any) -> bytes:
    """Extract raw image bytes from a PyArrow or Hugging Face dataset image entry."""
    if hasattr(image_val, "as_py"):
        image_val = image_val.as_py()

    if isinstance(image_val, dict):
        raw = image_val.get("bytes")
        if raw is not None:
            return bytes(raw)
        path = image_val.get("path")
        if path and Path(path).exists():
            return Path(path).read_bytes()

    if isinstance(image_val, (bytes, bytearray)):
        return bytes(image_val)

    raise ValueError(f"Unable to extract image bytes from object of type {type(image_val)}: {image_val!r}")


def detect_image_extension(img_bytes: bytes) -> str:
    """Detect image file extension from byte signatures."""
    if img_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if img_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if img_bytes.startswith(b"RIFF") and b"WEBP" in img_bytes[:16]:
        return ".webp"
    return ".png"


def download_tiny_genimage_parquets(
    repo_id: str = DEFAULT_TINY_GENIMAGE_REPO_ID,
    output_dir: str | Path = "data/raw/tiny_genimage/_parquets",
    token: str | None = None,
) -> dict[str, list[Path]]:
    """Download Parquet shards for Tiny-GenImage from Hugging Face Hub.

    Returns:
        Dictionary mapping split name ('train', 'validation') to lists of local Parquet file paths.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required to download Tiny-GenImage. "
            "Install project dependencies first: pip install -e ."
        ) from exc

    api = HfApi(token=token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[Path]] = {"train": [], "validation": []}

    for filename in sorted(repo_files):
        if not filename.endswith(".parquet"):
            continue

        split_name = None
        if "train" in filename:
            split_name = "train"
        elif "validation" in filename:
            split_name = "validation"

        if split_name is None:
            continue

        local_file = target_dir / Path(filename).name
        if not local_file.exists() or local_file.stat().st_size == 0:
            logger.info("Downloading %s from %s...", filename, repo_id)
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=token,
                local_dir=str(target_dir),
            )
            local_file = Path(downloaded)

        splits[split_name].append(local_file)

    return splits


def extract_parquet_images(
    parquet_path: str | Path,
    output_root: str | Path,
    raw_split: str,
    start_index: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Read images and metadata from a Parquet file and write images to disk.

    Writes images to:
        `<output_root>/<generator>/<raw_split>/<generator>_<raw_split>_<counter:06d>.<ext>`

    Returns:
        Tuple of (list of extracted metadata dicts, next_index).
    """
    parquet_path = Path(parquet_path)
    output_root = Path(output_root)
    table = pq.read_table(parquet_path)

    images_col = table.column("image")
    labels_col = table.column("label")
    generators_col = table.column("generator")

    extracted: list[dict[str, Any]] = []
    current_idx = start_index

    num_rows = len(table)
    for i in range(num_rows):
        img_bytes = extract_image_bytes(images_col[i])
        label = int(labels_col[i].as_py())
        gen_int = int(generators_col[i].as_py())
        generator = TINY_GENIMAGE_GENERATOR_MAP.get(gen_int, f"gen_{gen_int}")

        ext = detect_image_extension(img_bytes)
        filename = f"{generator}_{raw_split}_{current_idx:06d}{ext}"
        rel_dir = Path(generator) / raw_split
        target_dir = output_root / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / filename
        if not target_file.exists() or target_file.stat().st_size == 0:
            target_file.write_bytes(img_bytes)

        rel_path = str(rel_dir / filename)
        sample_id = f"tiny_genimage_{generator}_{raw_split}_{current_idx:06d}"

        extracted.append({
            "sample_id": sample_id,
            "image_path": rel_path,
            "label": label,
            "generator": generator,
            "raw_split": raw_split,
            "index": current_idx,
        })
        current_idx += 1

    return extracted, current_idx


def interleave_records_by_class(
    records: Sequence[dict[str, Any]],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministically shuffle records within each class and interleave them across classes.

    Ensures that any sequential slice taken from the returned list has maximal class balance
    while being fully governed by the random seed.
    """
    if not records:
        return []
    rng = random.Random(seed)
    sorted_recs = sorted(records, key=lambda r: str(r.get("sample_id", "")))
    by_class: dict[str, list[dict[str, Any]]] = {}
    for r in sorted_recs:
        c = str(r.get("class_id") or "unknown")
        by_class.setdefault(c, []).append(r)

    sorted_classes = sorted(by_class.keys())
    rng.shuffle(sorted_classes)

    for c in sorted_classes:
        rng.shuffle(by_class[c])

    interleaved: list[dict[str, Any]] = []
    max_len = max(len(items) for items in by_class.values())
    for idx in range(max_len):
        for c in sorted_classes:
            items = by_class[c]
            if idx < len(items):
                interleaved.append(items[idx])
    return interleaved


def split_records_by_class_and_seed(
    records: Sequence[dict[str, Any]],
    val_ratio: float = 0.5,
    n_val: int | None = None,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into (val_records, test_records) with class balancing and seed governance.

    Args:
        records: Collection of record dictionaries (must contain 'sample_id', may contain 'class_id').
        val_ratio: Fraction of records to allocate to validation if n_val is None.
        n_val: Absolute count of records to allocate to validation. Overrides val_ratio if set.
        seed: Random seed governing shuffling and partition assignments.

    Returns:
        (val_records, test_records) tuple of mutually disjoint record lists.
    """
    if not records:
        return [], []

    n_total = len(records)
    if n_val is not None:
        target_val = min(max(0, n_val), n_total)
    else:
        target_val = min(max(0, int(round(n_total * val_ratio))), n_total)

    if target_val == 0:
        return [], list(records)
    if target_val == n_total:
        return list(records), []

    interleaved = interleave_records_by_class(records, seed=seed)
    val_records = interleaved[:target_val]
    test_records = interleaved[target_val:]
    return val_records, test_records


def build_single_generator_manifests(
    extracted_records: Sequence[dict[str, Any]],
    manifest_dir: str | Path | None = None,
    train_generator: str = "sd15",
    dataset_name: str = "genimage",
    val_in_domain_ratio: float = 0.5,
    seed: int = 42,
) -> dict[str, Manifest]:
    """Construct Single-Generator Training & Cross-Generator OOD Manifests.

    Protocol 1 (single):
      - Train fake: only `train_generator` (default 'sd15') from raw train.
      - Train real: matched 1:1 with train fake from raw train.
      - Val: `val_in_domain_ratio` of validation fake from `train_generator` + balanced val reals.
      - In-domain test: remaining validation fake from `train_generator` + balanced val reals.
      - Cross-generator OOD: all remaining generators from validation + balanced val reals.
      - Per-generator OOD: individual test_<gen>.jsonl manifests.

    Guarantees:
      - Zero generator leakage: unseen generators NEVER appear in train or val.
      - Zero sample leakage: train, val, in_domain_test, and cross_generator_ood are mutually disjoint.
      - Mutually disjoint real image allocation across all evaluation splits.
      - Deterministic ordering based on seed.
    """
    if train_generator not in KAGGLE_TINY_GENIMAGE_GENERATORS:
        raise ValueError(
            f"Invalid train_generator '{train_generator}'. Must be one of: {list(KAGGLE_TINY_GENIMAGE_GENERATORS)}"
        )

    sorted_records = sorted(extracted_records, key=lambda r: str(r.get("sample_id", "")))
    _assert_expected_generator_set(sorted_records, builder="build_single_generator_manifests")

    # Separate train and validation raw records
    train_reals = [r for r in sorted_records if r.get("raw_split") == "train" and r.get("label") == 0]
    train_fakes = [
        r for r in sorted_records
        if r.get("raw_split") == "train" and r.get("generator") == train_generator and r.get("label") == 1
    ]

    val_reals = [r for r in sorted_records if r.get("raw_split") == "validation" and r.get("label") == 0]
    val_fakes_train_gen = [
        r for r in sorted_records
        if r.get("raw_split") == "validation" and r.get("generator") == train_generator and r.get("label") == 1
    ]

    # Balance train: pair train fakes with equal number of train real images
    n_train_pairs = min(len(train_fakes), len(train_reals))
    selected_train_fakes = train_fakes[:n_train_pairs]
    selected_train_reals = train_reals[:n_train_pairs]

    train_records = [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=1,
            dataset=dataset_name,
            generator=r["generator"],
            split="train",
            class_id=r.get("class_id"),
            metadata={"evaluation_generator": train_generator},
        )
        for r in selected_train_fakes
    ] + [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=0,
            dataset=dataset_name,
            generator="nature",
            split="train",
            class_id=r.get("class_id"),
            metadata={"evaluation_generator": train_generator},
        )
        for r in selected_train_reals
    ]

    shuffled_val_reals = interleave_records_by_class(val_reals, seed=seed)
    real_offset = 0

    def allocate_reals(n: int, split_name: str, cohort: str) -> list[ManifestRecord]:
        """Take `n` reals without replacement and tag them with their cohort.

        `cohort` names the generator whose fakes these reals serve as the negative
        reference for, so the evaluator can pair each generator with exactly its
        own held-out reals instead of the whole pooled real set.
        """
        nonlocal real_offset
        allocated = shuffled_val_reals[real_offset : real_offset + n]
        real_offset += len(allocated)
        return [
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=0,
                dataset=dataset_name,
                generator="nature",
                split=split_name,
                class_id=r.get("class_id"),
                metadata={"evaluation_generator": cohort},
            )
            for r in allocated
        ]

    # 1. SD1.5 val vs in_domain_test (seed-governed and class-balanced)
    val_part_fakes, test_part_fakes = split_records_by_class_and_seed(
        val_fakes_train_gen, val_ratio=val_in_domain_ratio, seed=seed
    )

    val_allocated_reals = allocate_reals(len(val_part_fakes), "val", train_generator)
    if len(val_allocated_reals) < len(val_part_fakes):
        val_part_fakes = val_part_fakes[: len(val_allocated_reals)]

    test_allocated_reals = allocate_reals(len(test_part_fakes), "in_domain_test", train_generator)
    if len(test_allocated_reals) < len(test_part_fakes):
        test_part_fakes = test_part_fakes[: len(test_allocated_reals)]

    val_records = [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=1,
            dataset=dataset_name,
            generator=r["generator"],
            split="val",
            class_id=r.get("class_id"),
            metadata={"evaluation_generator": train_generator},
        )
        for r in val_part_fakes
    ] + val_allocated_reals

    in_domain_records = [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=1,
            dataset=dataset_name,
            generator=r["generator"],
            split="in_domain_test",
            class_id=r.get("class_id"),
            metadata={"evaluation_generator": train_generator},
        )
        for r in test_part_fakes
    ] + test_allocated_reals

    # 2. Cross-Generator OOD (all generators except train_generator)
    ood_gens = [g for g in KAGGLE_TINY_GENIMAGE_GENERATORS if g != train_generator]
    cross_ood_records: list[ManifestRecord] = []
    per_gen_manifests: dict[str, Manifest] = {}

    for gen_id in ood_gens:
        val_gen_fakes = [
            r for r in sorted_records
            if r.get("raw_split") == "validation" and r.get("generator") == gen_id and r.get("label") == 1
        ]
        assigned_reals = allocate_reals(len(val_gen_fakes), "cross_generator_ood", gen_id)
        if len(assigned_reals) < len(val_gen_fakes):
            val_gen_fakes = val_gen_fakes[: len(assigned_reals)]

        cur_gen_records = [
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=1,
                dataset=dataset_name,
                generator=r["generator"],
                split="cross_generator_ood",
                class_id=r.get("class_id"),
                metadata={"evaluation_generator": gen_id},
            )
            for r in val_gen_fakes
        ] + assigned_reals

        cross_ood_records.extend(cur_gen_records)
        per_gen_manifests[f"test_{gen_id}"] = Manifest(cur_gen_records)

    train_manifest = Manifest(train_records)
    val_manifest = Manifest(val_records)
    in_domain_manifest = Manifest(in_domain_records)
    cross_ood_manifest = Manifest(cross_ood_records)

    # Invariant checks: zero generator leakage into train or val
    if len(train_manifest) > 0 and len(cross_ood_manifest) > 0:
        assert_generator_disjoint(train_manifest, cross_ood_manifest)
    if len(val_manifest) > 0 and len(cross_ood_manifest) > 0:
        assert_generator_disjoint(val_manifest, cross_ood_manifest)

    manifests: dict[str, Manifest] = {
        "train": train_manifest,
        "val": val_manifest,
        "in_domain_test": in_domain_manifest,
        "cross_generator_ood": cross_ood_manifest,
    }
    manifests.update(per_gen_manifests)

    if manifest_dir is not None:
        target_dir = Path(manifest_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, manifest in manifests.items():
            out_path = target_dir / f"{name}.jsonl"
            manifest.to_jsonl(out_path)
            logger.info("Saved %s with %d records to %s", name, len(manifest), out_path)

    return manifests


def build_tiny_genimage_manifests(
    extracted_records: Sequence[dict[str, Any]],
    manifest_dir: str | Path,
    dataset_name: str = "genimage",
    val_in_domain_ratio: float = 0.5,
    seed: int = 42,
) -> dict[str, Manifest]:
    """Compatibility wrapper for build_single_generator_manifests with legacy filenames.

    Writes standard and legacy `tiny_genimage_*` filenames to `manifest_dir`
    and returns a dictionary with both sets of keys.
    """
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifests = build_single_generator_manifests(
        extracted_records=extracted_records,
        manifest_dir=manifest_dir,
        train_generator="sd15",
        dataset_name=dataset_name,
        val_in_domain_ratio=val_in_domain_ratio,
        seed=seed,
    )

    # Add legacy names for backwards compatibility
    legacy_map: dict[str, Manifest] = {
        "tiny_genimage_train": manifests["train"],
        "tiny_genimage_val": manifests["val"],
        "tiny_genimage_in_domain_test": manifests["in_domain_test"],
        "tiny_genimage_cross_generator_ood": manifests["cross_generator_ood"],
    }
    for name, manifest in manifests.items():
        if name.startswith("test_"):
            legacy_map[f"tiny_genimage_{name}"] = manifest

    for name, manifest in legacy_map.items():
        out_path = manifest_dir / f"{name}.jsonl"
        manifest.to_jsonl(out_path)

    combined_dict = dict(manifests)
    combined_dict.update(legacy_map)
    return combined_dict


def build_logo_manifests(
    extracted_records: Sequence[dict[str, Any]],
    manifest_dir: str | Path | None = None,
    leave_out_gen: str = "midjourney",
    dataset_name: str = "genimage",
    n_val_per_gen: int = 100,
    seed: int = 42,
) -> dict[str, Manifest]:
    """Construct Leave-One-Generator-Out (LOGO) manifests.

    Protocol 2 (logo):
      - 6 seen generators -> train & val & in-domain seen test.
      - 1 leave_out_gen -> held-out test split (test_<leave_out_gen>.jsonl).

    Guarantees:
      - `leave_out_gen` appears ONLY in `test_<leave_out_gen>` (held-out test split).
      - `leave_out_gen` NEVER appears in train or val splits (zero generator leakage).
      - Strict 1:1 real:fake balance across all splits.
      - Mutually disjoint real image allocation.
      - Deterministic ordering based on seed.
    """
    if leave_out_gen not in KAGGLE_TINY_GENIMAGE_GENERATORS:
        raise ValueError(
            f"Invalid leave_out_gen '{leave_out_gen}'. Must be one of: {list(KAGGLE_TINY_GENIMAGE_GENERATORS)}"
        )

    rng = random.Random(seed)
    sorted_records = sorted(extracted_records, key=lambda r: str(r.get("sample_id", "")))
    _assert_expected_generator_set(sorted_records, builder="build_logo_manifests")

    all_fake_gens = sorted(list({r["generator"] for r in sorted_records if r.get("label") == 1}))
    if leave_out_gen not in all_fake_gens:
        raise ValueError(f"leave_out_gen '{leave_out_gen}' not found in extracted records: {all_fake_gens}")

    seen_gens = [g for g in KAGGLE_TINY_GENIMAGE_GENERATORS if g != leave_out_gen and g in all_fake_gens]

    train_reals = [r for r in sorted_records if r.get("raw_split") == "train" and r.get("label") == 0]
    val_reals = [r for r in sorted_records if r.get("raw_split") == "validation" and r.get("label") == 0]

    # Build Train: all train fakes from seen_gens
    train_fakes = [
        r for r in sorted_records
        if r.get("raw_split") == "train" and r.get("generator") in seen_gens and r.get("label") == 1
    ]
    n_train_reals = min(len(train_fakes), len(train_reals))
    selected_train_fakes = train_fakes[:n_train_reals]
    selected_train_reals = train_reals[:n_train_reals]

    train_records = [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=1,
            dataset=dataset_name,
            generator=r["generator"],
            split="train",
            class_id=r.get("class_id"),
            metadata={"evaluation_generator": r["generator"]},
        )
        for r in selected_train_fakes
    ] + [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=0,
            dataset=dataset_name,
            generator="nature",
            split="train",
            class_id=r.get("class_id"),
            metadata={"evaluation_generator": "__train__"},
        )
        for r in selected_train_reals
    ]

    shuffled_val_reals = interleave_records_by_class(val_reals, seed=seed)
    real_offset = 0

    def allocate_reals(n: int, split_name: str, cohort: str) -> list[ManifestRecord]:
        """Take `n` reals without replacement, tagged with the generator they serve."""
        nonlocal real_offset
        allocated = shuffled_val_reals[real_offset : real_offset + n]
        real_offset += len(allocated)
        return [
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=0,
                dataset=dataset_name,
                generator="nature",
                split=split_name,
                class_id=r.get("class_id"),
                metadata={"evaluation_generator": cohort},
            )
            for r in allocated
        ]

    val_records: list[ManifestRecord] = []
    test_seen_records: list[ManifestRecord] = []

    for gen in seen_gens:
        gen_val_fakes = [
            r for r in sorted_records
            if r.get("raw_split") == "validation" and r.get("generator") == gen and r.get("label") == 1
        ]
        n_val = min(n_val_per_gen, len(gen_val_fakes) // 2) if len(gen_val_fakes) >= 2 else len(gen_val_fakes)
        cur_val_fakes, cur_test_fakes = split_records_by_class_and_seed(
            gen_val_fakes, n_val=n_val, seed=seed
        )

        val_records.extend([
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=1,
                dataset=dataset_name,
                generator=gen,
                split="val",
                class_id=r.get("class_id"),
                metadata={"evaluation_generator": gen},
            )
            for r in cur_val_fakes
        ])
        val_records.extend(allocate_reals(len(cur_val_fakes), "val", gen))

        test_seen_records.extend([
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=1,
                dataset=dataset_name,
                generator=gen,
                split="in_domain_test",
                class_id=r.get("class_id"),
                metadata={"evaluation_generator": gen},
            )
            for r in cur_test_fakes
        ])
        test_seen_records.extend(allocate_reals(len(cur_test_fakes), "in_domain_test", gen))

    # Build Held-out Test: fakes of leave_out_gen paired with available held-out validation reals.
    # Restrict to raw validation fakes so the held-out split matches the real side (which is
    # drawn from validation reals): a train/validation mismatch would let the model separate
    # the classes by raw-split source distribution instead of by generator.
    held_out_fakes = [
        r for r in sorted_records
        if r.get("raw_split") == "validation"
        and r.get("generator") == leave_out_gen
        and r.get("label") == 1
    ]
    remaining_val_reals_count = max(0, len(val_reals) - real_offset)
    n_alloc = min(len(held_out_fakes), remaining_val_reals_count)
    held_out_reals = allocate_reals(n_alloc, "cross_generator_ood", leave_out_gen)
    held_out_fakes = held_out_fakes[: len(held_out_reals)]

    test_held_out_records = [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=1,
            dataset=dataset_name,
            generator=leave_out_gen,
            split="cross_generator_ood",
            class_id=r.get("class_id"),
            metadata={"evaluation_generator": leave_out_gen},
        )
        for r in held_out_fakes
    ] + held_out_reals

    train_manifest = Manifest(train_records)
    val_manifest = Manifest(val_records)
    test_held_out_manifest = Manifest(test_held_out_records)
    test_seen_manifest = Manifest(test_seen_records)

    # Invariant assertions
    if len(train_manifest) > 0 and len(test_held_out_manifest) > 0:
        assert_generator_disjoint(train_manifest, test_held_out_manifest)
    if len(val_manifest) > 0 and len(test_held_out_manifest) > 0:
        assert_generator_disjoint(val_manifest, test_held_out_manifest)

    manifests = {
        "train": train_manifest,
        "val": val_manifest,
        "in_domain_test": test_seen_manifest,
        "test_in_domain_seen": test_seen_manifest,
        f"test_{leave_out_gen}": test_held_out_manifest,
    }

    if manifest_dir is not None:
        target_dir = Path(manifest_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, m in manifests.items():
            out_file = target_dir / f"{name}.jsonl"
            m.to_jsonl(out_file)

    return manifests


def build_all_logo_manifests(
    extracted_records: Sequence[dict[str, Any]],
    manifest_dir: str | Path,
    dataset_name: str = "genimage",
    n_val_per_gen: int = 100,
    seed: int = 42,
) -> dict[str, dict[str, Manifest]]:
    """Generate all 7 Leave-One-Generator-Out (LOGO) folds into manifest_dir / leave_<gen>/."""
    base_dir = Path(manifest_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    all_folds: dict[str, dict[str, Manifest]] = {}
    for gen in KAGGLE_TINY_GENIMAGE_GENERATORS:
        fold_dir = base_dir / f"leave_{gen}"
        fold_manifests = build_logo_manifests(
            extracted_records=extracted_records,
            manifest_dir=fold_dir,
            leave_out_gen=gen,
            dataset_name=dataset_name,
            n_val_per_gen=n_val_per_gen,
            seed=seed,
        )
        all_folds[gen] = fold_manifests
    return all_folds


def build_all7_manifests(
    extracted_records: Sequence[dict[str, Any]],
    manifest_dir: str | Path | None = None,
    dataset_name: str = "genimage",
    n_val_per_gen: int = 100,
    seed: int = 42,
) -> dict[str, Manifest]:
    """Construct All-7 Multi-Generator Training & Evaluation Manifests.

    Protocol 3 (all7):
      - Train on all 7 generators: sd15, adm, biggan, glide, midjourney, vqdm, wukong.
      - Val: n_val_per_gen fakes from each generator + balanced val reals.
      - Per-generator in-domain test: remaining validation fakes from each generator + balanced val reals.
      - Combined test: all per-generator test sets combined into `test_all_combined.jsonl`.
      - Individual tests: `test_<gen>.jsonl` for all 7 generators.

    Note:
      This is a multi-generator upper bound on seen generators, NOT an unseen-generator
      generalization experiment.
    """
    rng = random.Random(seed)
    sorted_records = sorted(extracted_records, key=lambda r: str(r.get("sample_id", "")))
    _assert_expected_generator_set(sorted_records, builder="build_all7_manifests")

    all_fake_gens = sorted(list({r["generator"] for r in sorted_records if r.get("label") == 1}))

    train_reals = [r for r in sorted_records if r.get("raw_split") == "train" and r.get("label") == 0]
    val_reals = [r for r in sorted_records if r.get("raw_split") == "validation" and r.get("label") == 0]

    train_fakes = [r for r in sorted_records if r.get("raw_split") == "train" and r.get("label") == 1]
    n_train_pairs = min(len(train_fakes), len(train_reals))
    selected_train_fakes = train_fakes[:n_train_pairs]
    selected_train_reals = train_reals[:n_train_pairs]

    train_records = [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=1,
            dataset=dataset_name,
            generator=r["generator"],
            split="train",
            class_id=r.get("class_id"),
            metadata={"evaluation_generator": r["generator"]},
        )
        for r in selected_train_fakes
    ] + [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=0,
            dataset=dataset_name,
            generator="nature",
            split="train",
            class_id=r.get("class_id"),
            metadata={"evaluation_generator": "__train__"},
        )
        for r in selected_train_reals
    ]

    shuffled_val_reals = interleave_records_by_class(val_reals, seed=seed)
    real_offset = 0

    def allocate_reals(n: int, split_name: str, cohort: str) -> list[ManifestRecord]:
        """Take `n` reals without replacement, tagged with the generator they serve."""
        nonlocal real_offset
        allocated = shuffled_val_reals[real_offset : real_offset + n]
        real_offset += len(allocated)
        return [
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=0,
                dataset=dataset_name,
                generator="nature",
                split=split_name,
                class_id=r.get("class_id"),
                metadata={"evaluation_generator": cohort},
            )
            for r in allocated
        ]

    val_records: list[ManifestRecord] = []
    per_gen_test_manifests: dict[str, Manifest] = {}
    combined_test_records: list[ManifestRecord] = []

    for gen in all_fake_gens:
        gen_val_fakes = [
            r for r in sorted_records
            if r.get("raw_split") == "validation" and r.get("generator") == gen and r.get("label") == 1
        ]
        n_val = min(n_val_per_gen, len(gen_val_fakes) // 2) if len(gen_val_fakes) >= 2 else len(gen_val_fakes)
        cur_val_fakes, cur_test_fakes = split_records_by_class_and_seed(
            gen_val_fakes, n_val=n_val, seed=seed
        )

        val_records.extend([
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=1,
                dataset=dataset_name,
                generator=gen,
                split="val",
                class_id=r.get("class_id"),
                metadata={"evaluation_generator": gen},
            )
            for r in cur_val_fakes
        ])
        val_records.extend(allocate_reals(len(cur_val_fakes), "val", gen))

        cur_test_reals = allocate_reals(len(cur_test_fakes), "in_domain_test", gen)
        cur_test_records = [
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=1,
                dataset=dataset_name,
                generator=gen,
                split="in_domain_test",
                class_id=r.get("class_id"),
                metadata={"evaluation_generator": gen},
            )
            for r in cur_test_fakes
        ] + cur_test_reals

        per_gen_test_manifests[f"test_{gen}"] = Manifest(cur_test_records)
        combined_test_records.extend(cur_test_records)

    train_manifest = Manifest(train_records)
    val_manifest = Manifest(val_records)
    combined_test_manifest = Manifest(combined_test_records)

    manifests = {
        "train": train_manifest,
        "val": val_manifest,
        "test_all_combined": combined_test_manifest,
    }
    manifests.update(per_gen_test_manifests)

    if manifest_dir is not None:
        target_dir = Path(manifest_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, m in manifests.items():
            out_file = target_dir / f"{name}.jsonl"
            m.to_jsonl(out_file)

    return manifests


build_all_in_one_manifests = build_all7_manifests


def build_all_experiment_manifests(
    extracted_records: Sequence[dict[str, Any]],
    output_dir: str | Path = "data/manifests",
    train_generator: str = "sd15",
    dataset_name: str = "genimage",
    val_in_domain_ratio: float = 0.5,
    n_val_per_gen: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Construct complete manifest hierarchy for all 3 ForenSight protocols:
        <output_dir>/
        ├── single/
        │   ├── train.jsonl
        │   ├── val.jsonl
        │   ├── in_domain_test.jsonl
        │   ├── cross_generator_ood.jsonl
        │   └── test_<gen>.jsonl (6 OOD generators)
        ├── logo/
        │   ├── leave_sd15/
        │   ├── leave_adm/
        │   ├── leave_biggan/
        │   ├── leave_glide/
        │   ├── leave_midjourney/
        │   ├── leave_vqdm/
        │   └── leave_wukong/
        └── all7/
            ├── train.jsonl
            ├── val.jsonl
            ├── test_all_combined.jsonl
            └── test_<gen>.jsonl (all 7 generators)
    """
    root = Path(output_dir)
    single_dir = root / "single"
    logo_dir = root / "logo"
    all7_dir = root / "all7"

    single_manifests = build_single_generator_manifests(
        extracted_records=extracted_records,
        manifest_dir=single_dir,
        train_generator=train_generator,
        dataset_name=dataset_name,
        val_in_domain_ratio=val_in_domain_ratio,
        seed=seed,
    )
    logo_folds = build_all_logo_manifests(
        extracted_records=extracted_records,
        manifest_dir=logo_dir,
        dataset_name=dataset_name,
        n_val_per_gen=n_val_per_gen,
        seed=seed,
    )
    all7_manifests = build_all7_manifests(
        extracted_records=extracted_records,
        manifest_dir=all7_dir,
        dataset_name=dataset_name,
        n_val_per_gen=n_val_per_gen,
        seed=seed,
    )
    return {
        "single": single_manifests,
        "logo": logo_folds,
        "all7": all7_manifests,
    }


def build_protocol_manifests(
    extracted_records: Sequence[dict[str, Any]],
    manifest_dir: str | Path = "data/manifests",
    *,
    experiment: str = "single",
    seed: int = 42,
    leave_out_gen: str | None = None,
    val_in_domain_ratio: float = 0.5,
    dataset_name: str = "genimage",
    n_val_per_gen: int = 100,
) -> dict[str, Manifest]:
    """Build the sealed manifests for a single protocol under its canonical subdirectory.

    This is the one place that knows the per-protocol builder parameters. Both the local
    preparation script and the in-kernel Kaggle entrypoint go through it, so building the
    manifests from the same pinned dataset version with the same seed yields identical
    splits regardless of where the work happens.

    Output layout:
        single -> <manifest_dir>/single/
        logo   -> <manifest_dir>/logo/leave_<generator>/
        all7   -> <manifest_dir>/all7/
    """
    manifest_dir = Path(manifest_dir)

    if experiment == "single":
        return build_single_generator_manifests(
            extracted_records=extracted_records,
            manifest_dir=manifest_dir / "single",
            train_generator="sd15",
            dataset_name=dataset_name,
            val_in_domain_ratio=val_in_domain_ratio,
            seed=seed,
        )

    if experiment == "logo":
        if leave_out_gen not in KAGGLE_TINY_GENIMAGE_GENERATORS:
            raise ValueError(
                f"LOGO requires a held-out generator from {list(KAGGLE_TINY_GENIMAGE_GENERATORS)}, "
                f"got: {leave_out_gen!r}"
            )
        return build_logo_manifests(
            extracted_records=extracted_records,
            manifest_dir=manifest_dir / "logo" / f"leave_{leave_out_gen}",
            leave_out_gen=leave_out_gen,
            dataset_name=dataset_name,
            n_val_per_gen=n_val_per_gen,
            seed=seed,
        )

    if experiment == "all7":
        return build_all7_manifests(
            extracted_records=extracted_records,
            manifest_dir=manifest_dir / "all7",
            dataset_name=dataset_name,
            n_val_per_gen=n_val_per_gen,
            seed=seed,
        )

    raise ValueError(f"Unsupported experiment protocol: {experiment!r}")
