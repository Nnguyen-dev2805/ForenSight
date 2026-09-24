"""Dataset utilities for TheKernel01/Tiny-GenImage.

This module handles:
1. Downloading Parquet dataset shards from Hugging Face Hub (TheKernel01/Tiny-GenImage).
2. Extracting image files into local directories with directory-per-generator structure.
3. Generating sealed ForenSight manifests guaranteeing zero generator leakage into train/val.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import pyarrow.parquet as pq

from forensight.data.split import (
    Manifest,
    ManifestRecord,
    assert_generator_disjoint,
)

logger = logging.getLogger(__name__)

DEFAULT_TINY_GENIMAGE_REPO_ID = "TheKernel01/Tiny-GenImage"

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


def build_tiny_genimage_manifests(
    extracted_records: Sequence[dict[str, Any]],
    manifest_dir: str | Path,
    dataset_name: str = "genimage",
    val_in_domain_ratio: float = 0.5,
) -> dict[str, Manifest]:
    """Construct ForenSight manifests with zero generator leakage.

    Splits produced:
    1. train: SD1.4 fake + nature (real) from raw train. (split='train')
    2. val: 50% SD1.4 fake + nature from raw validation. (split='val')
    3. in_domain_test: 50% SD1.4 fake + nature from raw validation. (split='in_domain_test')
    4. near_ood: SD1.5 fake + nature from raw validation. (split='near_ood')
    5. cross_generator_ood: ADM, BigGAN, GLIDE, Midjourney, VQDM, Wukong fake + nature from raw validation. (split='cross_generator_ood')
    6. Individual OOD manifests for granular per-generator reporting: test_<generator>.jsonl.

    All real images used in evaluation are mutually disjoint to prevent sample reuse.
    """
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Separate train and validation raw records
    train_reals = [r for r in extracted_records if r["raw_split"] == "train" and r["label"] == 0]
    train_sd14 = [r for r in extracted_records if r["raw_split"] == "train" and r["generator"] == "sd14" and r["label"] == 1]

    val_reals = [r for r in extracted_records if r["raw_split"] == "validation" and r["label"] == 0]
    val_sd14 = [r for r in extracted_records if r["raw_split"] == "validation" and r["generator"] == "sd14" and r["label"] == 1]
    val_sd15 = [r for r in extracted_records if r["raw_split"] == "validation" and r["generator"] == "sd15" and r["label"] == 1]

    # Balance train: pair all SD1.4 fake with equal number of train real images
    n_train_pairs = min(len(train_sd14), len(train_reals))
    selected_train_sd14 = train_sd14[:n_train_pairs]
    selected_train_reals = train_reals[:n_train_pairs]

    train_records = [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=1,
            dataset=dataset_name,
            generator=r["generator"],
            split="train",
        )
        for r in selected_train_sd14
    ] + [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=0,
            dataset=dataset_name,
            generator="nature",
            split="train",
        )
        for r in selected_train_reals
    ]

    # Partition validation reals across eval splits to ensure zero sample leakage
    real_offset = 0

    def allocate_reals(n: int) -> list[dict[str, Any]]:
        nonlocal real_offset
        allocated = val_reals[real_offset : real_offset + n]
        real_offset += len(allocated)
        return allocated

    # 1. SD1.4 val vs in_domain_test
    n_val_sd14 = int(len(val_sd14) * val_in_domain_ratio)
    val_part_sd14 = val_sd14[:n_val_sd14]
    test_part_sd14 = val_sd14[n_val_sd14:]

    val_part_reals = allocate_reals(len(val_part_sd14))
    test_part_reals = allocate_reals(len(test_part_sd14))

    val_records = [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=1,
            dataset=dataset_name,
            generator=r["generator"],
            split="val",
        )
        for r in val_part_sd14
    ] + [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=0,
            dataset=dataset_name,
            generator="nature",
            split="val",
        )
        for r in val_part_reals
    ]

    in_domain_records = [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=1,
            dataset=dataset_name,
            generator=r["generator"],
            split="in_domain_test",
        )
        for r in test_part_sd14
    ] + [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=0,
            dataset=dataset_name,
            generator="nature",
            split="in_domain_test",
        )
        for r in test_part_reals
    ]

    # 2. Near-OOD (SD1.5)
    near_ood_reals = allocate_reals(len(val_sd15))
    near_ood_records = [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=1,
            dataset=dataset_name,
            generator=r["generator"],
            split="near_ood",
        )
        for r in val_sd15
    ] + [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=0,
            dataset=dataset_name,
            generator="nature",
            split="near_ood",
        )
        for r in near_ood_reals
    ]

    # 3. Cross-Generator OOD (ADM, BigGAN, GLIDE, Midjourney, VQDM, Wukong)
    cross_ood_records: list[ManifestRecord] = []
    per_gen_manifests: dict[str, Manifest] = {}

    for gen_id in CROSS_GENERATOR_OOD_IDS:
        val_gen_fakes = [
            r for r in extracted_records
            if r["raw_split"] == "validation" and r["generator"] == gen_id and r["label"] == 1
        ]
        assigned_reals = allocate_reals(len(val_gen_fakes))

        cur_gen_records = [
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=1,
                dataset=dataset_name,
                generator=r["generator"],
                split="cross_generator_ood",
            )
            for r in val_gen_fakes
        ] + [
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=0,
                dataset=dataset_name,
                generator="nature",
                split="cross_generator_ood",
            )
            for r in assigned_reals
        ]
        cross_ood_records.extend(cur_gen_records)
        per_gen_manifests[f"test_{gen_id}"] = Manifest(cur_gen_records)

    train_manifest = Manifest(train_records)
    val_manifest = Manifest(val_records)
    in_domain_manifest = Manifest(in_domain_records)
    near_ood_manifest = Manifest(near_ood_records)
    cross_ood_manifest = Manifest(cross_ood_records)

    # Invariant checks: zero generator leakage into train or val
    if len(train_manifest) > 0 and len(cross_ood_manifest) > 0:
        assert_generator_disjoint(train_manifest, cross_ood_manifest)
        assert_generator_disjoint(train_manifest, near_ood_manifest)
    if len(val_manifest) > 0 and len(cross_ood_manifest) > 0:
        assert_generator_disjoint(val_manifest, cross_ood_manifest)
        assert_generator_disjoint(val_manifest, near_ood_manifest)

    # Save to disk
    manifests: dict[str, Manifest] = {
        "tiny_genimage_train": train_manifest,
        "tiny_genimage_val": val_manifest,
        "tiny_genimage_in_domain_test": in_domain_manifest,
        "tiny_genimage_near_ood": near_ood_manifest,
        "tiny_genimage_cross_generator_ood": cross_ood_manifest,
    }
    for name, manifest in per_gen_manifests.items():
        manifests[f"tiny_genimage_{name}"] = manifest

    for name, manifest in manifests.items():
        out_path = manifest_dir / f"{name}.jsonl"
        manifest.to_jsonl(out_path)
        logger.info("Saved %s with %d records to %s", name, len(manifest), out_path)

    return manifests


def build_logo_manifests(
    extracted_records: Sequence[dict[str, Any]],
    manifest_dir: str | Path | None = None,
    leave_out_gen: str = "midjourney",
    dataset_name: str = "genimage",
    n_val_per_gen: int = 100,
    seed: int = 42,
) -> dict[str, Manifest]:
    """Construct Leave-One-Generator-Out (LOGO) manifests.

    Guarantees:
    - `leave_out_gen` appears ONLY in `test_<leave_out_gen>` (held-out test split).
    - `leave_out_gen` NEVER appears in train or val splits (zero generator leakage).
    - All other generators are used for training and in-domain validation/testing.
    - Strict 1:1 real:fake balance across all splits.
    - Mutually disjoint real image allocation.
    """
    if manifest_dir is not None:
        manifest_dir = Path(manifest_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)

    all_fake_gens = sorted(list({r["generator"] for r in extracted_records if r["label"] == 1}))
    if leave_out_gen not in all_fake_gens:
        raise ValueError(f"leave_out_gen '{leave_out_gen}' not found in records: {all_fake_gens}")

    seen_gens = [g for g in all_fake_gens if g != leave_out_gen]

    train_reals = [r for r in extracted_records if r["raw_split"] == "train" and r["label"] == 0]
    val_reals = [r for r in extracted_records if r["raw_split"] == "validation" and r["label"] == 0]

    # Build Train: all train fakes from seen_gens
    train_fakes = [r for r in extracted_records if r["raw_split"] == "train" and r["generator"] in seen_gens and r["label"] == 1]
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
        )
        for r in selected_train_reals
    ]

    real_offset = 0

    def allocate_reals(n: int, split_name: str) -> list[ManifestRecord]:
        nonlocal real_offset
        allocated = val_reals[real_offset : real_offset + n]
        real_offset += len(allocated)
        return [
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=0,
                dataset=dataset_name,
                generator="nature",
                split=split_name,
            )
            for r in allocated
        ]

    val_records: list[ManifestRecord] = []
    test_seen_records: list[ManifestRecord] = []

    for gen in seen_gens:
        gen_val_fakes = [r for r in extracted_records if r["raw_split"] == "validation" and r["generator"] == gen and r["label"] == 1]
        n_val = min(n_val_per_gen, len(gen_val_fakes) // 2) if len(gen_val_fakes) >= 2 else len(gen_val_fakes)
        cur_val_fakes = gen_val_fakes[:n_val]
        cur_test_fakes = gen_val_fakes[n_val:]

        val_records.extend([
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=1,
                dataset=dataset_name,
                generator=gen,
                split="val",
            )
            for r in cur_val_fakes
        ])
        val_records.extend(allocate_reals(len(cur_val_fakes), "val"))

        test_seen_records.extend([
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=1,
                dataset=dataset_name,
                generator=gen,
                split="in_domain_test",
            )
            for r in cur_test_fakes
        ])
        test_seen_records.extend(allocate_reals(len(cur_test_fakes), "in_domain_test"))

    # Build Held-out Test: all available fakes of leave_out_gen
    held_out_fakes = [r for r in extracted_records if r["generator"] == leave_out_gen and r["label"] == 1]
    n_held_out = len(held_out_fakes)
    remaining_val_reals_count = len(val_reals) - real_offset
    held_out_reals: list[ManifestRecord] = []

    if remaining_val_reals_count >= n_held_out:
        held_out_reals = allocate_reals(n_held_out, "cross_generator_ood")
    else:
        held_out_reals.extend(allocate_reals(remaining_val_reals_count, "cross_generator_ood"))
        needed_from_train = n_held_out - len(held_out_reals)
        unused_train_reals = train_reals[n_train_reals : n_train_reals + needed_from_train]
        held_out_reals.extend([
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=0,
                dataset=dataset_name,
                generator="nature",
                split="cross_generator_ood",
            )
            for r in unused_train_reals
        ])

    if len(held_out_reals) < len(held_out_fakes):
        held_out_fakes = held_out_fakes[:len(held_out_reals)]

    test_held_out_records = [
        ManifestRecord(
            sample_id=r["sample_id"],
            image_path=r["image_path"],
            label=1,
            dataset=dataset_name,
            generator=leave_out_gen,
            split="cross_generator_ood",
        )
        for r in held_out_fakes
    ] + held_out_reals

    train_manifest = Manifest(train_records)
    val_manifest = Manifest(val_records)
    test_held_out_manifest = Manifest(test_held_out_records)
    test_seen_manifest = Manifest(test_seen_records)

    manifests = {
        "train": train_manifest,
        "val": val_manifest,
        f"test_{leave_out_gen}": test_held_out_manifest,
        "test_in_domain_seen": test_seen_manifest,
    }

    if manifest_dir is not None:
        for name, m in manifests.items():
            out_file = manifest_dir / f"{name}.jsonl"
            m.to_jsonl(out_file)

    return manifests


def build_all_in_one_manifests(
    extracted_records: Sequence[dict[str, Any]],
    manifest_dir: str | Path | None = None,
    dataset_name: str = "genimage",
    n_val_per_gen: int = 100,
    seed: int = 42,
) -> dict[str, Manifest]:
    """Construct All-In-One multi-generator manifests to measure empirical upper bound.

    All available generators are present in train, val, and per-generator test sets:
    - train: all available train fakes (from raw train) + balanced train reals.
    - val: n_val_per_gen fakes from raw validation + balanced val reals (for tau* calibration).
    - test_<gen>: remaining val fakes + balanced val reals.
    - test_all_combined: all per-generator test sets combined.
    """
    if manifest_dir is not None:
        manifest_dir = Path(manifest_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)

    all_fake_gens = sorted(list({r["generator"] for r in extracted_records if r["label"] == 1}))

    train_reals = [r for r in extracted_records if r["raw_split"] == "train" and r["label"] == 0]
    val_reals = [r for r in extracted_records if r["raw_split"] == "validation" and r["label"] == 0]

    train_fakes = [r for r in extracted_records if r["raw_split"] == "train" and r["label"] == 1]
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
        )
        for r in selected_train_reals
    ]

    real_offset = 0

    def allocate_reals(n: int, split_name: str) -> list[ManifestRecord]:
        nonlocal real_offset
        allocated = val_reals[real_offset : real_offset + n]
        real_offset += len(allocated)
        return [
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=0,
                dataset=dataset_name,
                generator="nature",
                split=split_name,
            )
            for r in allocated
        ]

    val_records: list[ManifestRecord] = []
    per_gen_test_manifests: dict[str, Manifest] = {}
    combined_test_records: list[ManifestRecord] = []

    for gen in all_fake_gens:
        gen_val_fakes = [r for r in extracted_records if r["raw_split"] == "validation" and r["generator"] == gen and r["label"] == 1]
        n_val = min(n_val_per_gen, len(gen_val_fakes) // 2) if len(gen_val_fakes) >= 2 else len(gen_val_fakes)
        cur_val_fakes = gen_val_fakes[:n_val]
        cur_test_fakes = gen_val_fakes[n_val:]

        val_records.extend([
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=1,
                dataset=dataset_name,
                generator=gen,
                split="val",
            )
            for r in cur_val_fakes
        ])
        val_records.extend(allocate_reals(len(cur_val_fakes), "val"))

        cur_test_reals = allocate_reals(len(cur_test_fakes), "in_domain_test")
        cur_test_records = [
            ManifestRecord(
                sample_id=r["sample_id"],
                image_path=r["image_path"],
                label=1,
                dataset=dataset_name,
                generator=gen,
                split="in_domain_test",
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
        for name, m in manifests.items():
            out_file = manifest_dir / f"{name}.jsonl"
            m.to_jsonl(out_file)

    return manifests
